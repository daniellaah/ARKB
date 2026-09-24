"""LoRA on the 4B student, with the loss mask the dataset carries.

`mlx_lm.lora` masks a single prefix per sample: everything before an offset is
environment, everything after is the model's. That is the right shape for a
prompt-completion pair and this pilot's samples are exactly that (see
`build_dataset`), but the mask is still carried per token and used per token
here, so a later multi-turn sample -- one sequence with tool observations
between assistant turns -- trains correctly without a second training path.
Everything else (optimizer loop, checkpointing, validation, reporting) is
mlx_lm's.

    python -m training.train_lora --data training/data/sft-pilot --adapter training/models/sft-pilot-4b
"""

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

from training import chat_format


def keep_frozen_blocks_on_the_kernel(model, trainable_layers):
    """Only the blocks that are trained pay for the training path.

    Qwen3.5 is a hybrid: three of every four blocks run a gated delta rule whose
    Metal kernel has no backward pass, so mlx_lm substitutes a sequential
    reference implementation for any module in training mode. Below the lowest
    LoRA layer no gradient is needed, and leaving those blocks in training mode
    builds a per-timestep graph nothing ever reads: at 4,096 tokens that is the
    difference between 110 GB and 30 GB. The trainer re-enters training mode
    after every validation pass, so the rule is attached to the model's own
    `train`, not applied once.
    """
    frozen = list(model.layers[:-trainable_layers])
    model_class = type(model)
    base = model_class.train

    def train(self, mode: bool = True):
        base(self, mode)
        if mode:
            for block in frozen:
                block.eval()
        return self

    model_class.train = train
    return model


class MaskedDataset:
    """Samples already tokenized and masked; nothing is templated at training time."""

    def __init__(self, path):
        self.samples = [json.loads(line) for line in Path(path).read_text().split('\n') if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def masked_loss(model, batch, mask):
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model(inputs)
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


def iterate_batches(dataset, batch_size, max_seq_length, loop=False, seed=0, comm_group=None):
    """Length-sorted batches padded to a multiple of 32, as mlx_lm does, plus the mask."""
    order = sorted(range(len(dataset)), key=lambda i: len(dataset[i]['tokens']))
    batches = [order[i:i + batch_size] for i in range(0, len(order) - batch_size + 1, batch_size)]
    if not batches:
        raise ValueError('Not enough samples for one batch.')
    rng = np.random.default_rng(seed)
    while True:
        for index in rng.permutation(len(batches)):
            picked = [dataset[i] for i in batches[index]]
            lengths = [min(len(s['tokens']), max_seq_length) for s in picked]
            width = 1 + 32 * ((max(lengths) + 31) // 32)
            tokens = np.zeros((len(picked), width), np.int32)
            mask = np.zeros((len(picked), width - 1), np.float32)
            for row, (sample, length) in enumerate(zip(picked, lengths)):
                tokens[row, :length] = sample['tokens'][:length]
                # The mask aligns with the targets, which are the inputs shifted by one.
                mask[row, :length - 1] = sample['loss_mask'][1:length]
            yield mx.array(tokens), mx.array(mask)
        if not loop:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--model', default=chat_format.MODEL_PATH)
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--iters', type=int, default=0, help='overrides --epochs when set')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accumulation-steps', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--rank', type=int, default=32)
    # mlx_lm's default is 20, which is alpha/r = 20 and pairs with its 1e-5
    # learning rate; 2 is the alpha = 2r convention the quoted LoRA learning
    # rates assume, and the one this pilot's 1e-4 belongs to.
    parser.add_argument('--scale', type=float, default=2.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=20, help='optimizer updates, not iterations')
    parser.add_argument('--max-seq-length', type=int, default=12288)
    parser.add_argument('--steps-per-report', type=int, default=10)
    parser.add_argument('--steps-per-eval', type=int, default=400)
    parser.add_argument('--val-batches', type=int, default=12, help='-1 uses the whole validation set')
    parser.add_argument('--seed', type=int, default=0)
    # Checkpointing trades memory for a second forward pass. Training one block
    # needs neither: the frozen blocks below it keep no graph at all.
    parser.add_argument('--grad-checkpoint', action='store_true')
    args = parser.parse_args()

    mx.random.seed(args.seed)
    train_set = MaskedDataset(args.data / 'train.jsonl')
    valid_set = MaskedDataset(args.data / 'valid.jsonl')
    steps_per_epoch = max(1, len(train_set) // (args.batch_size * args.grad_accumulation_steps))
    iters = args.iters or max(1, math.ceil(steps_per_epoch * args.epochs)) * args.grad_accumulation_steps

    model, _ = load(args.model)
    model.freeze()
    lora_parameters = {'rank': args.rank, 'scale': args.scale, 'dropout': args.dropout}
    linear_to_lora_layers(model, args.num_layers, lora_parameters)
    keep_frozen_blocks_on_the_kernel(model, args.num_layers)
    print_trainable_parameters(model)

    args.adapter.mkdir(parents=True, exist_ok=True)
    (args.adapter / 'adapter_config.json').write_text(json.dumps({
        'fine_tune_type': 'lora', 'num_layers': args.num_layers, 'lora_parameters': lora_parameters,
        'model': args.model, 'data': str(args.data), 'train_samples': len(train_set),
        'valid_samples': len(valid_set), 'iters': iters, 'batch_size': args.batch_size,
        'grad_accumulation_steps': args.grad_accumulation_steps, 'learning_rate': args.learning_rate,
        'warmup': args.warmup, 'optimizer_updates': max(1, iters // args.grad_accumulation_steps),
        'max_seq_length': args.max_seq_length, 'seed': args.seed}, indent=1) + '\n')

    # The schedule advances once per optimizer update, not once per micro-batch,
    # so its horizon is the number of updates. Sizing it in iterations instead
    # would leave the cosine a quarter travelled when training ends.
    updates = max(1, iters // args.grad_accumulation_steps)
    schedule = optim.join_schedules(
        [optim.linear_schedule(0, args.learning_rate, args.warmup),
         optim.cosine_decay(args.learning_rate, max(1, updates - args.warmup), args.learning_rate * 0.1)],
        [args.warmup])
    train(model=model, optimizer=optim.AdamW(learning_rate=schedule), train_dataset=train_set,
          val_dataset=valid_set, loss=masked_loss, iterate_batches=iterate_batches,
          args=TrainingArgs(batch_size=args.batch_size, iters=iters, val_batches=args.val_batches,
                            steps_per_report=args.steps_per_report, steps_per_eval=args.steps_per_eval,
                            steps_per_save=max(200, iters // 4), adapter_file=args.adapter / 'adapters.safetensors',
                            max_seq_length=args.max_seq_length, grad_checkpoint=args.grad_checkpoint,
                            grad_accumulation_steps=args.grad_accumulation_steps))


if __name__ == '__main__':
    main()

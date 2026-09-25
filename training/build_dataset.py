"""Turn saved trajectories into SFT samples, one per model request.

A sample is the prompt the agent loop actually sent for one turn, plus the
assistant turn the teacher produced for it. That is deliberately not one
sample per trajectory. Qwen3.5's template keeps a `<think>` block on every
assistant message after the last real user message, while the loop strips
thinking before replaying a message; so within a single rendered trajectory
the same assistant turn would have to appear both with its reasoning (as the
target) and without it (as history). One sample per request has no such
conflict: the prompt is the saved `request.messages`, which is the exact list
the loop sent, and the target is the saved `response.message`.

The loss mask follows from that split and is computed, not asserted: the
prompt is rendered with the generation prompt, the prompt plus the assistant
turn is rendered without it, and the second must begin with the first. Every
token of the prefix is environment input and is masked; every token after it
is the model's own output -- its reasoning, its tool calls, its answer -- and
carries loss. A sample whose full rendering does not extend its prompt
rendering, as text, is dropped and counted, because a silently shifted
boundary is the failure this file is written to make impossible.

The two renderings usually nest as tokens too, and then the sample is that
exact token sequence. Sometimes they do not: the generation prompt ends with
`<think>\n`, and when the turn carries no reasoning the next character is
another newline, which the tokenizer merges into one token that the prompt's
single newline is not a prefix of. There the continuation is tokenized on its
own and appended, which is what generation from that prompt would produce
anyway. Both counts are reported, because the first is the health of the
rendering and the second is where it is being patched.

    python -m training.build_dataset evaluation/results/... --out training/data/sft-pilot
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from training import chat_format

ROOT = Path(__file__).resolve().parents[1]


def load_tokenizer(model_path=chat_format.MODEL_PATH):
    from mlx_lm import load
    return load(model_path)[1]


def build_sample(tokenizer, request, message, dialect=chat_format.QWEN3):
    """One turn as tokens and a mask, or None when the two renderings do not nest."""
    tools = chat_format.wire_tools(request.get('tools'), dialect)
    format = request.get('format')
    thinking = not format  # the reserved finalization runs with thinking off
    # The schema instruction belongs at the end of the prompt, so the assistant
    # turn is appended to the wire conversation, never to the raw one.
    prompt_messages = chat_format.to_wire(request['messages'], format, dialect)
    turn = {'role': 'assistant', 'content': message.get('content') or ''}
    # A finalization prompt ends with a closed, empty think block, so the model
    # cannot reason there and the target must not either. deepseek-reasoner
    # reasons regardless -- thinking is a property of its name, not of the
    # request -- and that reasoning is dropped rather than taught.
    if message.get('thinking') and thinking:
        turn['thinking'] = message['thinking']
    if message.get('tool_calls'):
        turn['tool_calls'] = message['tool_calls']
    full_messages = [*prompt_messages, *chat_format.to_wire([turn], None, dialect)]
    prompt = chat_format.render(tokenizer, prompt_messages, tools,
                                add_generation_prompt=True, enable_thinking=thinking)
    full = chat_format.render(tokenizer, full_messages, tools,
                              add_generation_prompt=False, enable_thinking=thinking)
    if len(full) > len(prompt) and full[:len(prompt)] == prompt:
        return {'tokens': full, 'loss_mask': [0] * len(prompt) + [1] * (len(full) - len(prompt)),
                'prompt_tokens': len(prompt), 'boundary': 'tokens'}
    prompt_text = chat_format.render_text(tokenizer, prompt_messages, tools,
                                          add_generation_prompt=True, enable_thinking=thinking)
    full_text = chat_format.render_text(tokenizer, full_messages, tools,
                                        add_generation_prompt=False, enable_thinking=thinking)
    if not full_text.startswith(prompt_text) or len(full_text) == len(prompt_text):
        return None
    continuation = tokenizer.encode(full_text[len(prompt_text):], add_special_tokens=False)
    return {'tokens': [*prompt, *continuation], 'loss_mask': [0] * len(prompt) + [1] * len(continuation),
            'prompt_tokens': len(prompt), 'boundary': 'text'}


def trajectory_samples(tokenizer, row, *, max_tokens, dialect=chat_format.QWEN3):
    """Every model request in one saved trajectory that rendered cleanly."""
    kept, dropped, oversized = [], 0, 0
    observation = ((row.get('result') or {}).get('observation') or {})
    for record in observation.get('models') or []:
        request, response = record.get('request'), record.get('response')
        if record.get('status') == 'recoverable_error' or not request or not response:
            continue
        if record.get('phase') == 'citation_verification':
            continue  # not a step of the trajectory: a separate check on the finished answer
        sample = build_sample(tokenizer, request, response.get('message') or {}, dialect)
        if sample is None:
            dropped += 1
            continue
        if len(sample['tokens']) > max_tokens:
            oversized += 1
            continue
        sample.update(question_id=row['question']['id'], turn=record.get('turn'), phase=record.get('phase'))
        kept.append(sample)
    return kept, dropped, oversized


def render_for_review(tokenizer, sample, *, head=1200, tail=1200):
    """The sample as text with the mask boundary marked, for a human to read.

    Printing the dataset is the only check that catches a mask off by one
    message: a wrong boundary trains the model on the tool output it is
    supposed to be reading, and nothing in the loss curve says so.
    """
    cut = sample['prompt_tokens']
    masked = tokenizer.decode(sample['tokens'][:cut])
    trained = tokenizer.decode(sample['tokens'][cut:])
    if len(masked) > head + tail:
        masked = masked[:head] + f'\n\n... [{len(masked) - head - tail} characters of masked prompt] ...\n\n' + masked[-tail:]
    return (f"question {sample.get('question_id')} turn {sample.get('turn')} phase {sample.get('phase')} "
            f"boundary {sample.get('boundary')}\n"
            f"tokens {len(sample['tokens'])}, masked {cut}, trained {len(sample['tokens']) - cut}\n"
            f"---------------- MASKED: environment input, loss_mask 0 ----------------\n{masked}\n"
            f"---------------- BOUNDARY: everything below is the model's own output, loss_mask 1 ----------------\n"
            f"{trained}\n---------------- end ----------------")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rollouts', type=Path, help='a directory holding results.jsonl')
    parser.add_argument('--out', type=Path, required=True, help='directory for train.jsonl and valid.jsonl')
    parser.add_argument('--max-tokens', type=int, default=12288)
    parser.add_argument('--valid-fraction', type=float, default=0.1)
    parser.add_argument('--samples-per-question', type=int, default=0,
                        help='keep at most this many trajectories per question; 0 keeps all')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', default=chat_format.MODEL_PATH)
    parser.add_argument('--show', type=int, default=0, help='print this many rendered samples for review')
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model)
    # See the note in rollout.py: splitlines() breaks on characters a note may contain.
    rows = [json.loads(line) for line in (args.rollouts / 'results.jsonl').read_text().split('\n') if line.strip()]
    if args.samples_per_question:
        per, kept_rows = Counter(), []
        for row in rows:
            per[row['question']['id']] += 1
            if per[row['question']['id']] <= args.samples_per_question:
                kept_rows.append(row)
        rows = kept_rows
    samples, dropped, oversized = [], 0, 0
    for row in rows:
        kept, lost, big = trajectory_samples(tokenizer, row, max_tokens=args.max_tokens,
                                             dialect=chat_format.dialect_for(args.model))
        samples += kept
        dropped += lost
        oversized += big

    # Split by question, not by sample: turns of one trajectory share a prompt
    # prefix, so splitting by sample would put near-duplicates on both sides.
    questions = sorted({s['question_id'] for s in samples})
    random.Random(args.seed).shuffle(questions)
    held = set(questions[:max(1, int(len(questions) * args.valid_fraction))])
    train = [s for s in samples if s['question_id'] not in held]
    valid = [s for s in samples if s['question_id'] in held]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in (('train', train), ('valid', valid)):
        (args.out / f'{name}.jsonl').write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in part))
    report = {'trajectories': len(rows), 'samples': len(samples), 'train': len(train), 'valid': len(valid),
              'boundary': dict(Counter(s['boundary'] for s in samples)),
              'dropped_not_prefix': dropped, 'dropped_over_max_tokens': oversized,
              'max_tokens': args.max_tokens, 'model': args.model,
              'trained_tokens': sum(sum(s['loss_mask']) for s in samples),
              'total_tokens': sum(len(s['tokens']) for s in samples)}
    (args.out / 'dataset.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))
    for sample in train[:args.show]:
        print('\n' + render_for_review(tokenizer, sample))


if __name__ == '__main__':
    main()

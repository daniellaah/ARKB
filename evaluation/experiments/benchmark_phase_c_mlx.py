"""Isolated native MLX model probe on the fixed corpus-only embedding sample.

Use unchanged Qwen3/base modules from mlx-embeddings and strict loading of the
official cached weights, cast to float16. Explicit float32 last-token L2 output
matches the reference pooling semantics. This is a different precision/backend
from production Q8_0, with no production cache writes or retrieval calls.
"""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import statistics
import sys
from time import perf_counter

import numpy as np
from ollama import Client
from tokenizers import Tokenizer

from arkb.evaluation.external import digest, write_json
from arkb.knowledge.embeddings import load_tokenizer, resolve_embedding_spec, validate_vectors
from arkb.knowledge.sqlite import SQLiteStorage

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--site-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tei-reference', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if (out / 'protocol.json').exists():
        raise FileExistsError('Use a fresh probe output; preserve existing results.')
    # Import the unchanged model implementation independently of the package's
    # unrelated vision loaders; no production or installed package is patched.
    package = out / 'probe_qwen'
    package.mkdir()
    (package / '__init__.py').write_text('')
    for name in ('base.py', 'qwen3.py'):
        shutil.copyfile(args.site_dir / 'mlx_embeddings/models' / name, package / name)
    sys.path[:0] = [str(args.site_dir.resolve()), str(out)]
    import mlx.core as mx
    module = importlib.import_module('probe_qwen.qwen3')
    mx.set_default_device(mx.gpu)
    items = json.loads((ROOT / 'evaluation/results/phase-c-v1/parallel-probe/inputs.json').read_text())
    assert len(items) == 256
    write_json(out / 'inputs.json', items)
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    original = load_tokenizer(cache_dir=ROOT / '.uv-cache/tokenizers', local_files_only=True)
    assert all(tokenizer.encode(x['text']).ids == original.encode(x['text']).ids for x in items)
    database = Path('/Volumes/ARKBPhaseC/validation/fiqa-r2/index.sqlite')
    before = digest(database)
    with SQLiteStorage(database, read_only=True) as storage:
        spec = storage.active_manifest('fiqa').embedding_spec
        expected = np.asarray([storage.get_embedding(spec, x['text']) for x in items])
    tei_reference = np.load(args.tei_reference)
    groups, current, tokens = [], [], 0
    for item in items:
        if current and (len(current) == 32 or tokens + item['tokens'] > 8192):
            groups.append(current)
            current, tokens = [], 0
        current.append(item)
        tokens += item['tokens']
    if current:
        groups.append(current)
    protocol = {'scope': __doc__, 'source_sha256': digest(__file__),
                'justification': 'User requested GPU embedding acceleration. Test a second native batching implementation independently of the unchanged Phase C workflow.',
                'package_versions': json.loads((out / 'package-metadata.json').read_text()),
                'model_modules': {p.name: digest(p) for p in package.glob('*.py')},
                'hf_model_revision': args.model.name, 'hf_weights_sha256': digest(args.model / 'model.safetensors'),
                'hf_config_sha256': digest(args.model / 'config.json'), 'dtype': 'float16', 'normalization': 'float32 last-token L2',
                'input_sha256': digest(out / 'inputs.json'), 'token_ids_equal': True,
                'max_batch_inputs': 32, 'max_batch_tokens': 8192, 'max_input_tokens': 8192,
                'padding': 'right; pad ID 151643 masked; no truncation; no added prompt',
                'tei_reference_sha256': digest(args.tei_reference),
                'arms': ['ollama_q8', 'mlx_f16_single', 'mlx_f16_batch'], 'repetitions': 3,
                'timing': 'Tokenization, direct MLX computation with explicit synchronization, float32 pooling/L2, host transfer and validation; Ollama includes persistent-client HTTP. Model load/full-sample warmup excluded.',
                'limitations': 'Different precision/backend; includes no MLX HTTP server. CPU preparation shares host. No proof of retrieval quality or whole-corpus equivalence.'}
    write_json(out / 'protocol.json', protocol)
    meta = {'status': 'loading', 'protocol_sha256': digest(out / 'protocol.json'), 'rounds': []}
    write_json(out / 'audit.json', meta)

    def compare(matrix, reference):
        cosine = np.sum(matrix * reference, axis=1) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(reference, axis=1))
        return {'exact': bool(np.array_equal(matrix, reference)), 'max_abs_difference': float(abs(matrix - reference).max()),
                'min_cosine': float(cosine.min()), 'mean_cosine': float(cosine.mean())}

    def check_background():
        state = json.loads(Path('/Volumes/ARKBPhaseC/cache/browsecomp-plus/status.json').read_text())
        if state['status'] != 'preparing':
            raise RuntimeError('Full background job advanced; stop probe to avoid competing embedding workloads.')

    try:
        check_background()
        config = module.ModelArgs.from_dict(json.loads((args.model / 'config.json').read_text()))
        model = module.Model(config)
        weights = model.sanitize(mx.load(str(args.model / 'model.safetensors')))
        model.load_weights([(key, value.astype(mx.float16)) for key, value in weights.items()], strict=True)
        del weights
        model.eval()
        mx.eval(model.parameters())
        with Client(host='http://127.0.0.1:11434', timeout=180) as client:
            assert resolve_embedding_spec(client, spec.model, context_length=8192) == spec
            def execute(arm):
                batches = [[x] for x in items] if arm == 'mlx_f16_single' else groups
                vectors = []
                for batch in batches:
                    texts = [x['text'] for x in batch]
                    if arm == 'ollama_q8':
                        raw = client.embed(model=spec.model, input=texts, truncate=False,
                                           options={'num_ctx': 8192}).embeddings
                    else:
                        encoded = tokenizer.encode_batch(texts)
                        lengths = [len(x.ids) for x in encoded]
                        assert all(0 < n <= 8192 for n in lengths)
                        ids = np.full((len(batch), max(lengths)), 151643, dtype=np.int32)
                        mask = np.zeros(ids.shape, dtype=np.int32)
                        for row, encoding in enumerate(encoded):
                            ids[row, :lengths[row]], mask[row, :lengths[row]] = encoding.ids, 1
                        output = model(mx.array(ids), attention_mask=mx.array(mask))
                        pooled = output.last_hidden_state[mx.arange(len(batch)), mx.array(lengths) - 1, :].astype(mx.float32)
                        normalized = pooled / mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True))
                        mx.eval(normalized)
                        raw = np.asarray(normalized, dtype=np.float64)
                    vectors.append(validate_vectors(raw, rows=len(batch), dimensions=1024,
                                                    dtype='float64', normalization='l2'))
                return np.concatenate(vectors)
            references = {}
            for arm in protocol['arms']:
                check_background()
                references[arm] = execute(arm)
                np.save(out / (arm + '-warmup.npy'), references[arm])
                agreement = compare(references[arm], tei_reference)
                if arm.startswith('mlx') and agreement['min_cosine'] < 0.99:
                    raise ValueError('MLX reference agreement failed: ' + json.dumps(agreement))
                print(arm, 'full sample warmed', agreement, flush=True)
            meta['status'] = 'timing'
            for repetition in range(3):
                for arm in protocol['arms'][repetition:] + protocol['arms'][:repetition]:
                    check_background()
                    start = perf_counter()
                    matrix = execute(arm)
                    seconds = perf_counter() - start
                    check_background()
                    row = {'arm': arm, 'repetition': repetition, 'seconds': seconds, 'inputs_per_second': len(items) / seconds,
                           'vs_original_cache': compare(matrix, expected), 'vs_tei_reference': compare(matrix, tei_reference),
                           'vs_mlx_single': compare(matrix, references['mlx_f16_single'])}
                    meta['rounds'].append(row)
                    np.save(out / f'{arm}-{repetition}.npy', matrix)
                    write_json(out / 'audit.json', meta)
                    print(json.dumps(row), flush=True)
            meta['summary'] = {arm: {'mean_inputs_per_second': statistics.mean(r['inputs_per_second'] for r in meta['rounds']
                                                                                 if r['arm'] == arm)} for arm in protocol['arms']}
            meta['status'] = 'completed'
    except BaseException as error:
        meta.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        meta['source_sqlite_unchanged'] = digest(database) == before
        write_json(out / 'audit.json', meta)


if __name__ == '__main__':
    main()

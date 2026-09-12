"""Small embedding-only throughput/equivalence audit; never updates a cache/index.

Compare one synchronous client, two clients sharing one Ollama instance, and
two independent Ollama instances. All arms use identical stored production
inputs, model revision, context, 32-input/8192-token request limits and ordering.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np
from ollama import Client

from arkb.evaluation.external import digest, write_json
from arkb.knowledge.embeddings import (
    load_tokenizer, prepare_document, validate_input_tokens, validate_vectors,
    resolve_embedding_spec,
)
from arkb.knowledge.models import Chunk, ChunkRecord
from arkb.knowledge.sqlite import SQLiteStorage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replica-url', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = load_tokenizer(cache_dir=Path('.uv-cache/tokenizers').resolve(), local_files_only=True)
    database = Path('/Volumes/ARKBPhaseC/validation/fiqa-r2/index.sqlite')
    items, cached = [], []
    with SQLiteStorage(database, read_only=True) as storage:
        manifest = storage.active_manifest('fiqa')
        spec = manifest.embedding_spec
        for (serialized,) in storage.connection.execute(
            'SELECT record FROM snapshot_chunks WHERE version=? ORDER BY ordinal LIMIT 5000',
            (manifest.index_version,),
        ):
            data = json.loads(serialized)
            data['chunk'] = Chunk(**data['chunk'])
            record = ChunkRecord(**data)
            text = prepare_document(record.chunk, document_template=spec.document_template)
            tokens = validate_input_tokens(text, tokenizer=tokenizer, max_tokens=8192)
            if 400 <= tokens <= 512:
                items.append({'chunk_id': record.chunk_id, 'text': text, 'tokens': tokens})
                cached.append(storage.get_embedding(spec, text))
            if len(items) == 256:
                break
    if len(items) != 256:
        raise ValueError('Insufficient fixed length-stratified sample.')
    write_json(args.output / 'inputs.json', items)
    expected = np.asarray(cached)
    batches, start = [], 0
    while start < len(items):
        end, tokens = start, 0
        while end < len(items) and end - start < 32 and tokens + items[end]['tokens'] <= 8192:
            tokens += items[end]['tokens']
            end += 1
        batches.append(items[start:end])
        start = end
    urls = ['http://127.0.0.1:11434', args.replica_url]
    for url in urls:
        with Client(host=url, timeout=120) as client:
            if resolve_embedding_spec(client, spec.model, context_length=8192) != spec:
                raise ValueError('Replica model identity mismatch.')
    arms = {'serial_one_instance': [urls[0]], 'two_clients_one_instance': [urls[0], urls[0]],
            'two_independent_instances': urls}
    output = {'status': 'running', 'scope': 'embedding throughput only; no retrieval, qrels, cache writes or policy selection',
              'sample': {'dataset': 'FiQA saved corpus chunks', 'index_version': manifest.index_version,
                         'selection': 'First 256 chunks of 400–512 prepared-input tokens in stored ordinal order; inspect at most first 5000 chunks.',
                         'inputs': len(items), 'tokens': sum(x['tokens'] for x in items), 'request_batches': len(batches),
                         'inputs_sha256': digest(args.output / 'inputs.json')},
              'embedding_spec': asdict(spec), 'context': 8192, 'max_batch_inputs': 32, 'max_batch_tokens': 8192,
              'source_sha256': digest(__file__), 'repetitions': 3, 'rounds': []}
    reference = None

    def request(url, batch):
        with Client(host=url, timeout=120) as client:
            response = client.embed(model=spec.model, input=[x['text'] for x in batch],
                                    truncate=False, options={'num_ctx': 8192})
        return validate_vectors(response.embeddings, rows=len(batch), dimensions=spec.dimensions,
                                dtype=spec.dtype, normalization=spec.normalization)

    # Warm both instances outside timing, including model load.
    for url in urls:
        request(url, batches[0])
    try:
        names = list(arms)
        for repetition in range(3):
            for name in names[repetition:] + names[:repetition]:
                endpoints = arms[name]
                start = perf_counter()
                with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
                    futures = [pool.submit(request, endpoints[i % len(endpoints)], batch)
                               for i, batch in enumerate(batches)]
                    matrix = np.concatenate([future.result() for future in futures])
                seconds = perf_counter() - start
                if reference is None:
                    reference = matrix.copy()
                result = {'arm': name, 'repetition': repetition, 'seconds': seconds,
                          'inputs_per_second': len(items) / seconds,
                          'exactly_equal_to_serial': bool(np.array_equal(matrix, reference)),
                          'max_abs_difference_from_serial': float(np.max(np.abs(matrix - reference))),
                          'exactly_equal_to_existing_cache': bool(np.array_equal(matrix, expected)),
                          'max_abs_difference_from_existing_cache': float(np.max(np.abs(matrix - expected))),
                          'vector_sha256': hashlib.sha256(matrix.astype('<f8').tobytes()).hexdigest()}
                output['rounds'].append(result)
                write_json(args.output / 'audit.json', output)
                print(json.dumps(result), flush=True)
        output['summary'] = {
            name: {'mean_inputs_per_second': statistics.mean(r['inputs_per_second'] for r in output['rounds'] if r['arm'] == name),
                   'all_exact_to_serial': all(r['exactly_equal_to_serial'] for r in output['rounds'] if r['arm'] == name),
                   'all_exact_to_existing_cache': all(r['exactly_equal_to_existing_cache'] for r in output['rounds'] if r['arm'] == name)}
            for name in names
        }
        output['status'] = 'completed'
    except BaseException as error:
        output.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        write_json(args.output / 'audit.json', output)


if __name__ == '__main__':
    main()

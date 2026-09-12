"""Prepare resumable embedding inputs using the unchanged public document/chunk APIs.

CPU workers operate on separate documents. A deterministic merge restores the
original source/chunk order and must match the prior full serial input digest
before publishing this operational plan. No models, retrieval or index writes.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache, partial
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
from time import perf_counter

from arkb.evaluation.external import digest, write_json
from arkb.knowledge.chunking import chunk_notes
from arkb.knowledge.documents import scan_notes
from arkb.knowledge.embeddings import (
    count_tokens, load_tokenizer, prepare_document, tokenizer_fingerprint,
    validate_input_tokens,
)

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = None


def initialize():
    global TOKENIZER
    TOKENIZER = load_tokenizer(cache_dir=ROOT / '.uv-cache/tokenizers', local_files_only=True)


def document_plan(note, output):
    path = Path(output) / (note.source + '.json.gz')
    receipt = path.with_suffix(path.suffix + '.sha256')
    if receipt.exists():
        if not path.exists() or digest(path) != receipt.read_text().strip():
            raise ValueError('Prepared source checksum mismatch: ' + note.source)
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            saved = json.load(stream)
        if saved['source'] != note.source or saved['document_revision'] != note.document_revision:
            raise ValueError('Prepared source changed: ' + note.source)
        return {'source': note.source, 'file': path.name, 'sha256': digest(path), 'chunks': len(saved['inputs']),
                'document_revision': note.document_revision, 'reused': True}
    counter = lru_cache(maxsize=8192)(partial(count_tokens, tokenizer=TOKENIZER))
    chunks = chunk_notes([note], count_tokens=counter, chunk_size=512, chunk_overlap=64)
    inputs = []
    for chunk in chunks:
        text = prepare_document(chunk)
        inputs.append([text, validate_input_tokens(text, tokenizer=TOKENIZER, max_tokens=8192)])
    saved = {'source': note.source, 'document_revision': note.document_revision, 'inputs': inputs,
             'chunks_sha256': hashlib.sha256(json.dumps([asdict(c) for c in chunks], sort_keys=True,
                                                       ensure_ascii=False).encode()).hexdigest()}
    temporary = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as stream:
        json.dump(saved, stream, ensure_ascii=False)
    os.replace(temporary, path)
    checksum = digest(path)
    temporary_receipt = receipt.with_suffix(receipt.suffix + '.tmp')
    temporary_receipt.write_text(checksum + '\n')
    os.replace(temporary_receipt, receipt)
    return {'source': note.source, 'file': path.name, 'sha256': checksum, 'chunks': len(inputs),
            'document_revision': note.document_revision, 'reused': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--expected-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError('Expected 1–12 preparation workers.')
    expected = json.loads(args.expected_checkpoint.read_text())
    initialize()
    identity = {'dataset_manifest_sha256': digest(args.dataset / 'manifest.json'),
                'expected_checkpoint_sha256': digest(args.expected_checkpoint),
                'tokenizer': tokenizer_fingerprint(TOKENIZER), 'chunk_size': 512, 'chunk_overlap': 64,
                'source_files': {name: digest(ROOT / name) for name in (
                    'src/arkb/knowledge/chunking.py', 'src/arkb/knowledge/documents.py',
                    'src/arkb/knowledge/embeddings.py', 'src/arkb/knowledge/models.py')},
                'script_sha256': digest(__file__)}
    if (identity['tokenizer'] != expected['tokenizer']
            or identity['dataset_manifest_sha256'] != expected['dataset_manifest_sha256']
            or expected['chunk_size'] != 512 or expected['chunk_overlap'] != 64
            or expected['embedding_spec']['document_template'] != 'title-body-v1'):
        raise ValueError('Plan identity differs from original serial preparation.')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'identity.json').exists():
        if json.loads((args.output / 'identity.json').read_text()) != identity:
            raise ValueError('Prepared plan identity changed; use a new directory.')
    else:
        write_json(args.output / 'identity.json', identity)
        shutil.copyfile(__file__, args.output / 'prepare_phase_c_inputs.py')
        shutil.copyfile(args.expected_checkpoint, args.output / 'original-serial-checkpoint.json')
    if (args.output / 'manifest.json').exists():
        raise FileExistsError('Completed preparation plans are immutable.')
    documents = args.output / 'documents'
    documents.mkdir(exist_ok=True)
    status = {'status': 'scanning', 'started_at': datetime.now(timezone.utc).isoformat(), 'workers': args.workers}
    write_json(args.output / 'status.json', status)
    started = perf_counter()
    try:
        notes = scan_notes(args.dataset / 'corpus')
        if len(notes) != expected['documents']:
            raise ValueError('Corpus count differs from original serial preparation.')
        status.update(status='preparing_documents', total_documents=len(notes), completed_documents=0)
        results = {}
        iterator = iter(notes)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize) as pool:
            pending = {}
            def submit():
                note = next(iterator, None)
                if note is not None:
                    pending[pool.submit(document_plan, note, documents)] = note.source
            for _ in range(args.workers * 2):
                submit()
            while pending:
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    source = pending.pop(future)
                    results[source] = future.result()
                    submit()
                if len(results) // 100 != status['completed_documents'] // 100:
                    status.update(completed_documents=len(results), elapsed_seconds=perf_counter() - started)
                    write_json(args.output / 'status.json', status)
                    print('prepared', len(results), '/', len(notes), flush=True)
        order = [note.source for note in notes]
        del notes
        status.update(status='merging_and_verifying', completed_documents=len(results))
        write_json(args.output / 'status.json', status)
        full_hash, unique_hash = hashlib.sha256(), hashlib.sha256()
        seen = set()
        chunk_count = 0
        temporary = args.output / 'inputs.jsonl.tmp'
        with temporary.open('w', encoding='utf-8', newline='\n') as target, (args.output / 'document-plans.jsonl').open('w') as index:
            for source in order:
                result = results[source]
                path = documents / result['file']
                if digest(path) != result['sha256']:
                    raise ValueError('Document plan changed during preparation.')
                index.write(json.dumps(result) + '\n')
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    document = json.load(stream)
                for text, tokens in document['inputs']:
                    full_hash.update((json.dumps(text, ensure_ascii=False) + '\n').encode())
                    chunk_count += 1
                    key = hashlib.sha256(text.encode()).digest()
                    if key not in seen:
                        seen.add(key)
                        line = json.dumps([text, tokens], ensure_ascii=False) + '\n'
                        target.write(line)
                        unique_hash.update(line.encode())
        actual = {'documents': len(order), 'chunks': chunk_count, 'unique_inputs': len(seen),
                  'input_sha256': full_hash.hexdigest()}
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError('Parallel preparation differs from the original full serial input plan.')
        os.replace(temporary, args.output / 'inputs.jsonl')
        manifest = {**actual, **identity, 'schema': 'arkb-embedding-input-plan-v1',
                    'inputs_file_sha256': unique_hash.hexdigest(),
                    'document_plans_sha256': digest(args.output / 'document-plans.jsonl'),
                    'embedding_spec': expected['embedding_spec'], 'status': 'completed',
                    'elapsed_seconds': perf_counter() - started,
                    'reused_document_plans': sum(result['reused'] for result in results.values())}
        write_json(args.output / 'manifest.json', manifest)
        status.update(status='completed', elapsed_seconds=manifest['elapsed_seconds'])
        print('Full original serial input digest matched.', flush=True)
    except BaseException as error:
        status.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        write_json(args.output / 'status.json', status)


if __name__ == '__main__':
    main()

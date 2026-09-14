"""Recover exact chunk metadata from checksum-anchored preparation documents.

Prepared embedding strings preserve chunk bodies but not their positions. Locate
candidate source slices, recreate metadata with the unchanged Markdown section
builder, and accept ONLY an exact match to the previously saved chunk hash.
Ambiguous repeated text falls back to the unchanged tokenizer/chunker. This is
an operational cache migration, not a new chunking algorithm. No index/model I/O.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache, partial
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from time import perf_counter

from arkb.evaluation.external import digest, write_json
from arkb.knowledge.chunking import _make_chunk, _sections, chunk_notes
from arkb.knowledge.documents import _load_note, note_paths
from arkb.knowledge.embeddings import count_tokens, load_tokenizer, prepare_document, tokenizer_fingerprint

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = None


class AmbiguousSlices(ValueError):
    pass


def chunk_digest(chunks):
    return hashlib.sha256(json.dumps([asdict(c) for c in chunks], sort_keys=True,
                                    ensure_ascii=False).encode()).hexdigest()


def recover(note, saved):
    """A candidate is never trusted without the original complete metadata hash."""
    if note.source != saved['source'] or note.document_revision != saved['document_revision']:
        raise ValueError('Prepared document identity changed.')
    sections = list(_sections(note))
    section_index = 0
    previous_start, previous_end = -1, 0
    occurrences = defaultdict(int)
    chunks = []
    prefix = note.title + '\n\n'
    for index, (text, tokens) in enumerate(saved['inputs']):
        if not text.startswith(prefix) or type(tokens) is not int or not 0 < tokens <= 8192:
            raise ValueError('Invalid prepared input template or token count.')
        body = text[len(prefix):]
        start = note.content.find(body, previous_start + 1)
        # Each production chunk extends coverage. A repeated substring inside
        # the previous chunk is not a new chunk, even when it matches its body.
        while start >= 0 and index and start + len(body) <= previous_end:
            start = note.content.find(body, start + 1)
        if start < 0 or start > previous_end:
            raise AmbiguousSlices('Cannot locate a contiguous ordered source slice.')
        end = start + len(body)
        while section_index + 1 < len(sections) and start >= sections[section_index].blocks[-1].end:
            section_index += 1
        section = sections[section_index]
        if end > section.blocks[-1].end:
            raise AmbiguousSlices('Candidate slice crosses a Markdown section.')
        key = (section_index, body)
        chunks.append(_make_chunk(note, section, start, end, index, occurrences[key]))
        occurrences[key] += 1
        previous_start, previous_end = start, end
    if previous_end != len(note.content) or chunk_digest(chunks) != saved['chunks_sha256']:
        raise AmbiguousSlices('Candidate metadata does not match the original chunk hash.')
    return chunks


def verify_inputs(chunks, saved):
    if len(chunks) != len(saved['inputs']) or chunk_digest(chunks) != saved['chunks_sha256']:
        raise ValueError('Recovered chunk metadata changed.')
    for chunk, (text, tokens) in zip(chunks, saved['inputs'], strict=True):
        if prepare_document(chunk) != text or type(tokens) is not int or not 0 < tokens <= 8192:
            raise ValueError('Recovered embedding input changed.')


def load_original(plan, entry):
    source = entry['source']
    if Path(source).name != source or entry['file'] != source + '.json.gz':
        raise ValueError('Unexpected prepared document path.')
    path = plan / 'documents' / entry['file']
    if digest(path) != entry['sha256']:
        raise ValueError('Original document plan checksum changed.')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        saved = json.load(stream)
    if (saved['source'] != source or saved['document_revision'] != entry['document_revision']
            or len(saved['inputs']) != entry['chunks']):
        raise ValueError('Original document plan metadata changed.')
    return saved


def worker(task):
    global TOKENIZER
    corpus, plan, output, entry = task
    source = entry['source']
    target = output / 'documents' / entry['file']
    receipt = target.with_suffix(target.suffix + '.sha256')
    started = perf_counter()
    saved = load_original(plan, entry)
    note = _load_note(corpus / source)
    if note.document_revision != entry['document_revision']:
        raise ValueError('Current corpus document changed: ' + source)
    method = 'recovered_slices'
    try:
        chunks = recover(note, saved)
    except AmbiguousSlices:
        method = 'original_chunker_fallback'
        if TOKENIZER is None:
            TOKENIZER = load_tokenizer(cache_dir=ROOT / '.uv-cache/tokenizers', local_files_only=True)
        counter = lru_cache(maxsize=8192)(partial(count_tokens, tokenizer=TOKENIZER))
        chunks = chunk_notes([note], count_tokens=counter, chunk_size=512, chunk_overlap=64)
    verify_inputs(chunks, saved)
    metadata = [{key: value for key, value in asdict(chunk).items()
                 if key not in ('content', 'title', 'source')} for chunk in chunks]
    document = {'source': source, 'document_revision': note.document_revision,
                'original_document_plan_sha256': entry['sha256'], 'chunks_sha256': saved['chunks_sha256'],
                'chunks': metadata, 'tokens': [row[1] for row in saved['inputs']], 'method': method}
    # Completed receipts make interrupted staging resumable, without trusting a
    # merely present file. Never overwrite a differing committed result.
    if receipt.exists():
        if digest(target) != receipt.read_text().strip():
            raise ValueError('Recovered document checksum changed.')
        with gzip.open(target, 'rt') as stream:
            prior = json.load(stream)
        if prior != document:
            raise ValueError('Committed recovered document differs.')
    else:
        temporary = target.with_suffix(target.suffix + '.tmp')
        with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as stream:
            json.dump(document, stream, ensure_ascii=False)
        os.replace(temporary, target)
        temporary_receipt = receipt.with_suffix(receipt.suffix + '.tmp')
        temporary_receipt.write_text(digest(target) + '\n')
        os.replace(temporary_receipt, receipt)
    return {'source': source, 'file': entry['file'], 'sha256': digest(target), 'chunks': len(chunks),
            'document_revision': note.document_revision, 'chunks_sha256': saved['chunks_sha256'],
            'method': method, 'seconds': perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError('Expected 1–12 workers.')
    manifest = json.loads((args.plan / 'manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['chunk_size'] != 512 or manifest['chunk_overlap'] != 64:
        raise ValueError('Unexpected preparation plan.')
    if digest(args.dataset / 'manifest.json') != manifest['dataset_manifest_sha256']:
        raise ValueError('Dataset identity changed.')
    for name, checksum in manifest['source_files'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError('Production source changed: ' + name)
    tokenizer = load_tokenizer(cache_dir=ROOT / '.uv-cache/tokenizers', local_files_only=True)
    if tokenizer_fingerprint(tokenizer) != manifest['tokenizer']:
        raise ValueError('Tokenizer changed.')
    del tokenizer
    index = args.plan / 'document-plans.jsonl'
    if digest(index) != manifest['document_plans_sha256']:
        raise ValueError('Original document inventory changed.')
    entries = [json.loads(line) for line in index.open()]
    sources = [row['source'] for row in entries]
    if len(sources) != manifest['documents'] or sources != [p.name for p in note_paths(args.dataset / 'corpus')]:
        raise ValueError('Corpus scope/order changed.')
    if args.sample:
        sample = set(json.loads(args.sample.read_text())['sample_sources'])
        entries = [row for row in entries if row['source'] in sample]
        if len(entries) != len(sample):
            raise ValueError('Sample source missing.')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'manifest.json').exists():
        raise FileExistsError('Completed recovered plans are immutable.')
    identity = {'original_plan_sha256': digest(args.plan / 'manifest.json'),
                'script_sha256': digest(__file__), 'source_files': manifest['source_files'],
                'sample_sha256': digest(args.sample) if args.sample else None}
    identity_path = args.output / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Staging identity changed.')
    write_json(identity_path, identity)
    (args.output / 'documents').mkdir(exist_ok=True)
    started = perf_counter()
    status = {'status': 'recovering_chunks', 'started_at': datetime.now(timezone.utc).isoformat(),
              'total_documents': len(entries), 'completed_documents': 0, 'workers': args.workers,
              'original_chunker_fallback_documents': 0}
    write_json(args.output / 'status.json', status)
    results = []
    try:
        tasks = ((args.dataset / 'corpus', args.plan, args.output, entry) for entry in entries)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            for result in pool.map(worker, tasks, chunksize=8):
                results.append(result)
                status['original_chunker_fallback_documents'] += result['method'] == 'original_chunker_fallback'
                if len(results) % 500 == 0:
                    status.update(completed_documents=len(results), elapsed_seconds=perf_counter() - started)
                    write_json(args.output / 'status.json', status)
                    print('recovered', len(results), '/', len(entries), 'fallback', status['original_chunker_fallback_documents'], flush=True)
        if [r['source'] for r in results] != [r['source'] for r in entries]:
            raise ValueError('Recovered source order changed.')
        if not args.sample and sum(row['chunks'] for row in results) != manifest['chunks']:
            raise ValueError('Recovered chunk count changed.')
        with (args.output / 'document-plans.jsonl').open('x') as stream:
            for result in results:
                stream.write(json.dumps(result) + '\n')
        status.update(status='completed', completed_documents=len(results), elapsed_seconds=perf_counter() - started)
        write_json(args.output / 'manifest.json', {
            **identity, **status, 'schema': 'arkb-recovered-chunks-v1',
            'scope': 'sample' if args.sample else 'full', 'documents': len(results),
            'chunks': sum(row['chunks'] for row in results), 'all_document_chunk_hashes_match': True,
            'all_embedding_inputs_match': True, 'model_calls': 0, 'index_writes': 0,
            'original_plan': str(args.plan.resolve()), 'dataset': str(args.dataset.resolve()),
            'dataset_manifest_sha256': manifest['dataset_manifest_sha256'], 'tokenizer': manifest['tokenizer'],
            'document_plans_sha256': digest(args.output / 'document-plans.jsonl')})
    except BaseException as error:
        status.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        write_json(args.output / 'status.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()

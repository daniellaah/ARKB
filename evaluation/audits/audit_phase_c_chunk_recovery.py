"""Compare prepared builds to the original builder and exercise corruption guards."""
import argparse
from dataclasses import asdict, replace
from functools import partial
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

import numpy as np
from arkb.evaluation.external import digest, write_json
from arkb.knowledge import indexing
from arkb.knowledge.chunking import chunk_notes
from arkb.knowledge.documents import _load_note
from arkb.knowledge.embeddings import count_tokens, load_tokenizer, prepare_document, validate_input_tokens
from arkb.knowledge.models import EmbeddingSpec, QdrantConfig, Note
from arkb.knowledge.sqlite import SQLiteStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from recover_phase_c_chunks import recover, chunk_digest, load_original, verify_inputs
from use_phase_c_chunks import PreparedChunks


class Remote:
    def __init__(self):
        self.calls = []

    def record(self, name, records, vectors):
        self.calls.append({'method': name, 'records_sha256': hashlib.sha256(json.dumps(
            [asdict(record) for record in records], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            'vectors_sha256': hashlib.sha256(vectors.tobytes()).hexdigest(), 'rows': len(records)})

    def upsert(self, records, vectors):
        self.record('upsert', records, vectors)

    def verify_snapshot(self, records, vectors):
        self.record('verify_snapshot', records, vectors)

    def wait_ready(self, *, expected_count):
        self.calls.append({'method': 'wait_ready', 'expected_count': expected_count})
        return {'fixture_count': expected_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.plan / 'manifest.json').read_text())
    corpus = Path(manifest['dataset']) / 'corpus'
    entries = [json.loads(line) for line in (args.plan / 'document-plans.jsonl').open()]
    notes = [_load_note(corpus / row['source']) for row in entries]
    tokenizer = load_tokenizer(cache_dir=Path('.uv-cache/tokenizers'), local_files_only=True)
    spec = EmbeddingSpec(model='qwen3-embedding:0.6b', model_revision='fixture-only', dimensions=4,
                         document_template='title-body-v1')
    checks = []
    def rejects(name, fn):
        try:
            fn()
        except (ValueError, FileNotFoundError):
            checks.append({'check': name, 'status': 'passed'})
        else:
            raise AssertionError('Expected rejection: ' + name)

    original_plan = Path(manifest['original_plan'])
    source_entries = {row['source']: row for line in (original_plan / 'document-plans.jsonl').open()
                      if (row := json.loads(line))['source'] in {n.source for n in notes}}
    texts = {}
    for note in notes:
        saved = load_original(original_plan, source_entries[note.source])
        for text, tokens in saved['inputs']:
            texts[text] = tokens
    remote_results, build_results, timings = [], [], {}
    original_functions = (indexing.chunk_notes, indexing.validate_input_tokens, Note.document_revision)
    with tempfile.TemporaryDirectory(prefix='arkb-chunk-build-audit-') as directory:
        for mode in ('original', 'prepared'):
            remote = Remote()
            with SQLiteStorage(Path(directory) / (mode + '.sqlite')) as storage:
                storage.put_embeddings(spec, list(texts), np.full((len(texts), 4), 0.5))
                arguments = dict(spec=spec, vault_id='fixture', client=object(), tokenizer=tokenizer,
                                 max_input_tokens=8192, chunk_size=512, chunk_overlap=64,
                                 index_version='chunk-recovery-audit-v1', qdrant_config=QdrantConfig(),
                                 qdrant_client=object())
                started = perf_counter()
                with patch.object(indexing, '_qdrant_index', return_value=remote):
                    if mode == 'prepared':
                        cache = PreparedChunks(args.plan, manifest_sha256=digest(args.plan / 'manifest.json'), allow_sample=True)
                        with cache.bind():
                            report = indexing.build_index(storage, notes, **arguments)
                    else:
                        report = indexing.build_index(storage, notes, **arguments)
                timings[mode] = perf_counter() - started
                result = asdict(report)
                result.pop('build_seconds')
                build_results.append(result)
                remote_results.append(remote.calls)
                assert report.embedded_inputs == 0 and report.manifest.status == 'ready'
            assert (indexing.chunk_notes, indexing.validate_input_tokens, Note.document_revision) == original_functions
    assert build_results[0] == build_results[1] and remote_results[0] == remote_results[1]
    checks.append({'check': 'original/prepared READY manifest, cache counts, exact ordered records/vectors and remote calls equal',
                   'status': 'passed', 'documents': len(notes), 'chunks': manifest['chunks']})
    checks.append({'check': 'all original bindings restored after successful build', 'status': 'passed'})

    for content in ('', '# Heading\n\nText.\n\n## Nested\n\n- a\n- b',
                    'repeat ' * 3000, '```\n# opaque\n' + 'x' * 2000 + '\n```',
                    'abc' * 1600 + '\n\nabc' * 800):
        note = Note(title='Fixture', content=content, source='fixture.md')
        chunks = chunk_notes([note], count_tokens=partial(count_tokens, tokenizer=tokenizer))
        saved = {'source': note.source, 'document_revision': note.document_revision,
                 'chunks_sha256': chunk_digest(chunks), 'inputs': [[prepare_document(c), validate_input_tokens(
                     prepare_document(c), tokenizer=tokenizer, max_tokens=8192)] for c in chunks]}
        try:
            result = recover(note, saved)
        except ValueError:
            result = chunk_notes([note], count_tokens=partial(count_tokens, tokenizer=tokenizer))
        verify_inputs(result, saved)
        assert [asdict(c) for c in result] == [asdict(c) for c in chunks]
    checks.append({'check': 'empty body, Markdown sections, fenced code, repeated/overlapping text preserve exact metadata', 'status': 'passed'})
    note = notes[0]
    saved = load_original(original_plan, source_entries[note.source])
    rejects('changed source revision', lambda: recover(replace(note, content=note.content + ' changed'), saved))
    rejects('wrong original metadata digest', lambda: recover(note, {**saved, 'chunks_sha256': '0' * 64}))
    rejects('wrong recovered plan manifest', lambda: PreparedChunks(args.plan, manifest_sha256='0' * 64, allow_sample=True))
    rejects('sample cannot be used for full validation', lambda: PreparedChunks(args.plan, manifest_sha256=digest(args.plan / 'manifest.json')))
    cache = PreparedChunks(args.plan, manifest_sha256=digest(args.plan / 'manifest.json'), allow_sample=True)
    rejects('different source order', lambda: cache.chunk_notes(list(reversed(notes)), count_tokens=len))
    rejects('different chunk budget', lambda: cache.chunk_notes(notes, count_tokens=len, chunk_size=256))
    with cache.bind():
        chunks = indexing.chunk_notes(notes, count_tokens=len)
        text = prepare_document(chunks[0])
        kwargs = dict(tokenizer=tokenizer, max_tokens=8192, source=f'{chunks[0].source}, chunk 0')
        rejects('changed input text', lambda: indexing.validate_input_tokens(text + ' changed', **kwargs))
        rejects('input budget smaller than prepared count', lambda: indexing.validate_input_tokens(text, **{**kwargs, 'max_tokens': 1}))
        for chunk in chunks:
            indexing.validate_input_tokens(prepare_document(chunk), tokenizer=tokenizer, max_tokens=8192,
                                            source=f'{chunk.source}, chunk {chunk.chunk_index}')
    try:
        with PreparedChunks(args.plan, manifest_sha256=digest(args.plan / 'manifest.json'), allow_sample=True).bind():
            raise RuntimeError('fixture interruption')
    except RuntimeError:
        pass
    assert (indexing.chunk_notes, indexing.validate_input_tokens, Note.document_revision) == original_functions
    checks.append({'check': 'all original bindings restored on exception', 'status': 'passed'})
    write_json(args.output, {'status': 'passed', 'checks': checks, 'sample_plan_sha256': digest(args.plan / 'manifest.json'),
                            'builder_seconds': timings, 'remote_calls': remote_results[0],
                            'model_calls': 0, 'real_qdrant_calls': 0, 'production_writes': 0,
                            'script_sha256': digest(__file__), 'adapter_sha256': digest(Path(sys.path[0]) / 'use_phase_c_chunks.py'),
                            'recovery_sha256': digest(Path(sys.path[0]) / 'recover_phase_c_chunks.py')})
    print(len(checks), 'checks passed; original/prepared exact build equivalence verified.', flush=True)


if __name__ == '__main__':
    main()

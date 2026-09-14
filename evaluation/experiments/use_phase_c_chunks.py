"""Bind verified, prepared chunk inputs only during a Phase C index build.

The original index builder, storage validation, vector transport and READY
publication remain in use. Only its pure chunk/input preparation calls are
served from a fully checked artifact; the bindings are restored before retrieval.
"""
import argparse
from contextlib import ExitStack, contextmanager
from functools import wraps
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sys
from time import perf_counter
from unittest.mock import patch

from arkb.evaluation.external import digest, write_json
from arkb.knowledge import indexing
from arkb.knowledge.embeddings import prepare_document, tokenizer_fingerprint
from arkb.knowledge.models import Chunk, Note
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.knowledge.qdrant import QdrantIndex
from recover_phase_c_chunks import load_original, verify_inputs


class PreparedChunks:
    def __init__(self, plan, *, manifest_sha256, progress=None, allow_sample=False):
        self.plan = Path(plan)
        if digest(self.plan / 'manifest.json') != manifest_sha256:
            raise ValueError('Recovered plan identity changed.')
        self.manifest = json.loads((self.plan / 'manifest.json').read_text())
        if (self.manifest['status'] != 'completed' or not self.manifest['all_document_chunk_hashes_match']
                or not self.manifest['all_embedding_inputs_match']
                or (self.manifest['scope'] != 'full' and not allow_sample)):
            raise ValueError('Recovered chunk plan is not complete.')
        original = Path(self.manifest['original_plan'])
        if digest(original / 'manifest.json') != self.manifest['original_plan_sha256']:
            raise ValueError('Original plan identity changed.')
        self.original = original
        self.original_manifest = json.loads((original / 'manifest.json').read_text())
        if digest(original / 'document-plans.jsonl') != self.original_manifest['document_plans_sha256']:
            raise ValueError('Original plan inventory changed.')
        if digest(self.plan / 'document-plans.jsonl') != self.manifest['document_plans_sha256']:
            raise ValueError('Recovered plan inventory changed.')
        self.entries = [json.loads(line) for line in (self.plan / 'document-plans.jsonl').open()]
        selected = {row['source'] for row in self.entries}
        self.original_entries = {row['source']: row for line in (original / 'document-plans.jsonl').open()
                                 if (row := json.loads(line))['source'] in selected}
        if len(self.entries) != self.manifest['documents'] or len(selected) != len(self.entries):
            raise ValueError('Recovered document inventory is inconsistent.')
        self.chunks = None
        self.tokens = []
        self.cursor = 0
        self.progress_path = progress
        self.started = perf_counter()
        self.tokenizer = None
        self.revisions = {}
        self.original_revision = Note.document_revision.fget

    def document_revision(self, note):
        saved = self.revisions.get(id(note))
        if saved is not None and saved[0] is note:
            return saved[1]
        return self.original_revision(note)

    def progress(self, stage, **extra):
        if self.progress_path:
            write_json(self.progress_path, {'stage': stage, 'elapsed_seconds': perf_counter() - self.started, **extra})
        print('prepared index:', stage, extra, flush=True)

    def chunk_notes(self, notes, *, count_tokens, chunk_size=512, chunk_overlap=64):
        if self.chunks is not None or (chunk_size, chunk_overlap) != (512, 64):
            raise ValueError('Prepared chunk cache has a different scope or was invoked twice.')
        if [note.source for note in notes] != [row['source'] for row in self.entries]:
            raise ValueError('Index corpus order/scope differs from recovered plan.')
        chunks = []
        input_hash = hashlib.sha256()
        for index, (note, entry) in enumerate(zip(notes, self.entries, strict=True)):
            if Path(entry['file']).name != entry['file'] or entry['file'] != note.source + '.json.gz':
                raise ValueError('Invalid recovered document path.')
            path = self.plan / 'documents' / entry['file']
            if digest(path) != entry['sha256']:
                raise ValueError('Recovered document checksum changed.')
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                saved = json.load(stream)
            original_entry = self.original_entries[note.source]
            original = load_original(self.original, original_entry)
            revision = self.original_revision(note)
            if (revision != entry['document_revision']
                    or saved['document_revision'] != revision
                    or saved['original_document_plan_sha256'] != original_entry['sha256']
                    or saved['chunks_sha256'] != original['chunks_sha256']
                    or saved['tokens'] != [row[1] for row in original['inputs']]):
                raise ValueError('Prepared document identity or token counts changed.')
            self.revisions[id(note)] = (note, revision)
            restored = []
            for metadata in saved['chunks']:
                start, end = metadata['start_char'], metadata['end_char']
                if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(note.content):
                    raise ValueError('Invalid prepared source range.')
                restored.append(Chunk(content=note.content[start:end], title=note.title,
                                      source=note.source, **metadata))
            verify_inputs(restored, original)
            for text, _ in original['inputs']:
                input_hash.update((json.dumps(text, ensure_ascii=False) + '\n').encode())
            chunks.extend(restored)
            self.tokens.extend(saved['tokens'])
            if (index + 1) % 1000 == 0:
                self.progress('loading_verified_chunks', documents=index + 1, total_documents=len(notes), chunks=len(chunks))
        if len(chunks) != self.manifest['chunks']:
            raise ValueError('Recovered chunk count changed.')
        if self.manifest['scope'] == 'full' and input_hash.hexdigest() != self.original_manifest['input_sha256']:
            raise ValueError('Complete ordered input digest changed.')
        self.chunks = chunks
        self.progress('chunks_loaded', documents=len(notes), chunks=len(chunks), input_sha256=input_hash.hexdigest())
        return chunks

    def validate_input_tokens(self, text, *, tokenizer, max_tokens, source='embedding input'):
        if self.chunks is None or self.cursor >= len(self.chunks):
            raise ValueError('Unexpected prepared input validation call.')
        if tokenizer is not self.tokenizer:
            if tokenizer_fingerprint(tokenizer) != self.manifest['tokenizer']:
                raise ValueError('Index tokenizer differs from prepared inputs.')
            self.tokenizer = tokenizer
        if (tokenizer.padding is not None or tokenizer.truncation is not None
                or type(max_tokens) is not int or max_tokens <= 0):
            raise ValueError('Invalid tokenizer input validation settings.')
        chunk = self.chunks[self.cursor]
        if source != f'{chunk.source}, chunk {chunk.chunk_index}' or text != prepare_document(chunk):
            raise ValueError('Index input text or ordering differs from prepared input.')
        count = self.tokens[self.cursor]
        if type(count) is not int or not 0 < count <= max_tokens:
            raise ValueError('Prepared input exceeds the active token budget.')
        self.cursor += 1
        return count

    @contextmanager
    def bind(self):
        with ExitStack() as bindings:
            bindings.enter_context(patch.object(indexing, 'chunk_notes', self.chunk_notes))
            bindings.enter_context(patch.object(indexing, 'validate_input_tokens', self.validate_input_tokens))
            bindings.enter_context(patch.object(Note, 'document_revision', property(self.document_revision)))
            def observe(original, label):
                @wraps(original)
                def call(*args, **kwargs):
                    self.progress(label, state='started')
                    result = original(*args, **kwargs)
                    self.progress(label, state='returned')
                    return result
                return call
            for cls, names in ((SQLiteStorage, ('create_build', 'add_chunks', 'load_snapshot', 'publish')),
                               (QdrantIndex, ('upsert', 'verify_snapshot', 'wait_ready'))):
                for name in names:
                    bindings.enter_context(patch.object(cls, name, observe(getattr(cls, name), cls.__name__ + '.' + name)))
            yield
            if self.chunks is None or self.cursor != len(self.chunks):
                raise ValueError('Original builder did not consume the complete prepared corpus.')
        self.progress('index_build_returned', validated_inputs=self.cursor)
        self.chunks = None
        self.tokens.clear()
        self.revisions.clear()
        self.original_entries.clear()
        self.entries.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--qdrant-url', default='http://127.0.0.1:6340')
    args = parser.parse_args()
    from arkb.runtime import Runtime
    from arkb.knowledge.sqlite import SQLiteStorage
    protocol = json.loads((args.run / 'protocol.json').read_text())
    for name, checksum in protocol['source_files'].items():
        if digest(args.run / 'measured-source' / name) != checksum:
            raise ValueError('Measured source changed.')
    if Path(indexing.__file__).resolve() != (args.run / 'measured-source/src/arkb/knowledge/indexing.py').resolve():
        raise ValueError('Expected the frozen measured-source index builder.')
    for artifact in ('legs.jsonl.gz', 'summary.json', 'rows.jsonl.gz'):
        if (args.run / artifact).exists():
            raise FileExistsError('Do not repeat retrieval: ' + artifact)
    with SQLiteStorage(args.run / 'index.sqlite', read_only=True) as storage:
        if storage.list_builds('browsecomp-plus'):
            raise ValueError('Candidate build already exists; inspect rather than repeat.')
    cache = PreparedChunks(args.plan, manifest_sha256=args.plan_sha256,
                           progress=args.run / 'prepared-index-progress.json')
    if (cache.manifest['dataset_manifest_sha256'] != protocol['dataset_manifest_sha256']
            or args.qdrant_url != protocol['qdrant_url']):
        raise ValueError('Validation protocol identity changed.')
    original_index = Runtime.index
    calls = 0

    def prepared_index(runtime, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 1 or kwargs['vault_id'] != 'browsecomp-plus' or kwargs['chunking'] != 'recursive':
            raise ValueError('Unexpected validation index invocation.')
        with cache.bind():
            return original_index(runtime, **kwargs)

    sys.path.insert(0, str(args.run))
    sys.argv = [str(args.run / 'validate_phase_c.py'), '--execute', '--dataset', cache.manifest['dataset'],
                '--output', str(args.run), '--qdrant-url', args.qdrant_url]
    with patch.object(Runtime, 'index', prepared_index):
        runpy.run_path(str(args.run / 'validate_phase_c.py'), run_name='__main__')


if __name__ == '__main__':
    main()

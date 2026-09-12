"""Operational input-plan equivalence and interruption guards; no model calls."""
import argparse
from dataclasses import asdict, replace
from functools import partial
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from arkb.evaluation.external import digest, write_json
from arkb.knowledge.chunking import chunk_notes
from arkb.knowledge.documents import scan_notes
from arkb.knowledge.embeddings import count_tokens, prepare_document, validate_input_tokens
from arkb.knowledge.models import EmbeddingSpec

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'evaluation/experiments'))
import prepare_phase_c_inputs as producer
from cache_phase_c_embeddings import load_prepared_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []

    def rejects(name, operation):
        try:
            operation()
        except ValueError:
            results.append({'check': name, 'status': 'passed'})
        else:
            raise AssertionError('Expected guard failure: ' + name)

    with tempfile.TemporaryDirectory(prefix='arkb-input-plan-audit-') as folder:
        work = Path(folder)
        corpus = work / 'corpus'
        corpus.mkdir()
        for source in json.loads(args.sample.read_text())['sample_sources']:
            (corpus / source).symlink_to(args.corpus / source)
        # An additional equal document checks global first-occurrence deduplication.
        (corpus / 'zz-duplicate.md').symlink_to(next(iter(sorted(corpus.iterdir()))))
        write_json(work / 'manifest.json', {'scope': 'temporary corpus-only operational audit'})
        producer.initialize()
        tokenizer = producer.TOKENIZER
        notes = scan_notes(corpus)
        chunks = chunk_notes(notes, count_tokens=partial(count_tokens, tokenizer=tokenizer),
                             chunk_size=512, chunk_overlap=64)
        full_hash, unique = hashlib.sha256(), {}
        for chunk in chunks:
            text = prepare_document(chunk)
            full_hash.update((json.dumps(text, ensure_ascii=False) + '\n').encode())
            if text not in unique:
                unique[text] = validate_input_tokens(text, tokenizer=tokenizer, max_tokens=8192)
        expected = json.loads(args.checkpoint.read_text())
        expected.update(dataset_manifest_sha256=digest(work / 'manifest.json'), documents=len(notes),
                        chunks=len(chunks), unique_inputs=len(unique), input_sha256=full_hash.hexdigest())
        write_json(work / 'expected.json', expected)
        output = work / 'plan'
        command = [sys.executable, '-u', str(ROOT / 'evaluation/experiments/prepare_phase_c_inputs.py'),
                   '--dataset', str(work), '--expected-checkpoint', str(work / 'expected.json'),
                   '--output', str(output), '--workers', '8']
        env = {**os.environ, 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        subprocess.run(command, check=True, env=env, cwd=ROOT)
        spec = EmbeddingSpec(**expected['embedding_spec'])
        load = lambda: load_prepared_inputs(output, dataset=work, tokenizer=tokenizer, spec=spec)
        manifest, actual = load()
        assert list(actual.items()) == list(unique.items())
        for note in notes:
            with gzip.open(output / 'documents' / (note.source + '.json.gz'), 'rt') as stream:
                shard = json.load(stream)
            baseline = [asdict(c) for c in chunks if c.source == note.source]
            assert shard['chunks_sha256'] == hashlib.sha256(json.dumps(baseline, sort_keys=True,
                                                                        ensure_ascii=False).encode()).hexdigest()
        results.append({'check': 'parallel chunk metadata, ordered inputs, token counts and deduplication equal serial',
                        'status': 'passed', 'documents': len(notes), 'chunks': len(chunks), 'unique_inputs': len(unique)})
        repeated = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True)
        assert repeated.returncode != 0 and 'Completed preparation plans are immutable' in repeated.stderr
        results.append({'check': 'completed plan cannot be overwritten', 'status': 'passed'})

        # A resumed document is checksum-verified, including persisted token counts.
        note = notes[0]
        reused = producer.document_plan(note, output / 'documents')
        assert reused['reused']
        results.append({'check': 'completed document shard reused without recomputation', 'status': 'passed'})
        rejects('source revision changed', lambda: producer.document_plan(
            replace(note, content=note.content + ' changed'), output / 'documents'))
        path = output / 'documents' / reused['file']
        original_shard = path.read_bytes()
        with gzip.open(path, 'rt') as stream:
            changed = json.load(stream)
        changed['inputs'][0][1] += 1
        with gzip.open(path, 'wt') as stream:
            json.dump(changed, stream)
        rejects('token count corruption rejected on resume', lambda: producer.document_plan(note, output / 'documents'))
        path.write_bytes(original_shard)

        inputs = output / 'inputs.jsonl'
        original_inputs = inputs.read_bytes()
        with inputs.open('ab') as stream:
            stream.write(b'["new input", 2]\n')
        rejects('changed prepared input rejected before cache write', load)
        inputs.write_bytes(original_inputs)
        changed_manifest = {**manifest, 'tokenizer': '0' * 64}
        write_json(output / 'manifest.json', changed_manifest)
        rejects('tokenizer identity drift rejected', load)
        write_json(output / 'manifest.json', manifest)

        # Uncommitted temporary/shard writes are recomputed, never trusted on resume.
        receipt = path.with_suffix(path.suffix + '.sha256')
        receipt.unlink()
        path.write_bytes(b'interrupted partial file')
        recovered = producer.document_plan(note, output / 'documents')
        assert not recovered['reused']
        with gzip.open(path, 'rt') as stream:
            recovered_inputs = json.load(stream)['inputs']
        assert recovered_inputs == [[prepare_document(c), unique[prepare_document(c)]]
                                    for c in chunks if c.source == note.source]
        results.append({'check': 'uncommitted document recomputed after interruption', 'status': 'passed'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {'status': 'passed', 'checks': results, 'model_calls': 0, 'retrieval_calls': 0,
                            'production_cache_writes': 0, 'sample_sha256': digest(args.sample),
                            'original_checkpoint_sha256': digest(args.checkpoint),
                            'scripts_sha256': {p.name: digest(p) for p in (Path(__file__), Path(producer.__file__),
                                               ROOT / 'evaluation/experiments/cache_phase_c_embeddings.py')}})
    print(len(results), 'operational checks passed; no models, retrieval or production cache writes.')


if __name__ == '__main__':
    main()

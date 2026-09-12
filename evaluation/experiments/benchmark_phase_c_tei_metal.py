"""Isolated TEI Metal feasibility probe; different precision, never a cache writer.

Compare original Q8_0 Ollama with official HF weights cast to float16 by TEI.
This is an operational backend investigation, not a Phase C retrieval arm or
permission to combine vectors from the two configurations in one index.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import plistlib
import shutil
import signal
import socket
import statistics
import subprocess
import threading
from time import perf_counter, sleep

import httpx
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
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise FileExistsError('Preserve the existing probe; use a fresh output directory.')
    items = json.loads((ROOT / 'evaluation/results/phase-c-v1/parallel-probe/inputs.json').read_text())
    assert len(items) == 256
    write_json(out / 'inputs.json', items)
    original_tokenizer = load_tokenizer(cache_dir=ROOT / '.uv-cache/tokenizers', local_files_only=True)
    hf_tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    hf_tokenizer.no_padding()
    hf_tokenizer.no_truncation()
    assert all(original_tokenizer.encode(item['text']).ids == hf_tokenizer.encode(item['text']).ids for item in items)
    model = out / 'model'
    model.mkdir()
    for path in args.model.iterdir():
        (model / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    # Configure the same 8192-token runtime acceptance limit without modifying
    # cached weights/configuration. All probe inputs contain only 400–512 tokens.
    if (model / 'sentence_bert_config.json').exists():
        raise ValueError('Unexpected pre-existing runtime length configuration.')
    write_json(model / 'sentence_bert_config.json', {'max_seq_length': 8192})
    database = Path('/Volumes/ARKBPhaseC/validation/fiqa-r2/index.sqlite')
    source_hash = digest(database)
    with SQLiteStorage(database, read_only=True) as storage:
        spec = storage.active_manifest('fiqa').embedding_spec
        expected = np.asarray([storage.get_embedding(spec, item['text']) for item in items])
    groups, current, tokens = [], [], 0
    for item in items:
        if current and (len(current) == 32 or tokens + item['tokens'] > 8192):
            groups.append(current)
            current, tokens = [], 0
        current.append(item)
        tokens += item['tokens']
    if current:
        groups.append(current)
    port = 11439
    for number in (port, 19039):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', number))
    command = [str(args.binary.resolve()), '--model-id', str(model), '--hostname', '127.0.0.1',
               '--port', str(port), '--prometheus-port', '19039', '--dtype', 'float16',
               '--pooling', 'last-token', '--max-batch-tokens', '8192', '--max-client-batch-size', '32',
               '--max-batch-requests', '32', '--tokenization-workers', '2', '--auto-truncate', 'false']
    protocol = {'scope': __doc__, 'justification': 'User requested GPU embedding acceleration; isolate backend throughput and precision differences without touching frozen Phase C inputs, caches or retrieval.',
                'source_sha256': digest(__file__), 'binary_sha256': digest(args.binary),
                'hf_model_revision': args.model.name, 'hf_weights_sha256': digest(args.model / 'model.safetensors'),
                'hf_config_sha256': digest(args.model / 'config.json'),
                'runtime_length_override': {'sentence_bert_config.json': {'max_seq_length': 8192}},
                'ollama_embedding_spec': asdict(spec), 'tei_dtype': 'float16', 'command': command,
                'input_sha256': digest(out / 'inputs.json'), 'input_count': len(items),
                'sample_token_ids_equal': True, 'request_groups': len(groups), 'repetitions': 3,
                'arms': ['ollama_q8', 'tei_f16_single', 'tei_f16_batch'],
                'no_production_cache_writes': True, 'no_retrieval': True,
                'resource_note': 'Eight-process BCP CPU preparation and prior serial preparation continue on this host; no claim of isolated maximum GPU throughput.'}
    write_json(out / 'protocol.json', protocol)
    meta = {'status': 'starting', 'rounds': [], 'protocol_sha256': digest(out / 'protocol.json')}
    write_json(out / 'audit.json', meta)
    proc, gpu_thread = None, None
    done, gpu_samples = threading.Event(), []
    phase = 'loading'
    start_clock = perf_counter()

    def observe_gpu():
        while not done.is_set():
            result = subprocess.run(['/usr/sbin/ioreg', '-a', '-r', '-c', 'AGXAccelerator', '-d', '1'],
                                    capture_output=True, check=True)
            for entry in plistlib.loads(result.stdout):
                values = entry.get('PerformanceStatistics', {})
                if 'Device Utilization %' in values:
                    gpu_samples.append({'seconds': perf_counter() - start_clock, 'phase': phase,
                                        'device_utilization_percent': values['Device Utilization %']})
            done.wait(0.5)

    try:
        with (out / 'server.log').open('x') as log:
            env = {key: os.environ[key] for key in ('PATH', 'HOME', 'TMPDIR') if key in os.environ}
            env.update(HF_HOME=str(out / 'isolated-hf-home'), HF_HUB_OFFLINE='1',
                       HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', RUST_LOG='info')
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=180) as tei, Client(host='http://127.0.0.1:11434', timeout=180) as ollama:
                if resolve_embedding_spec(ollama, spec.model, context_length=8192) != spec:
                    raise ValueError('Original Ollama model identity drift.')
                for _ in range(600):
                    if proc.poll() is not None:
                        raise RuntimeError('TEI exited; inspect preserved server.log.')
                    try:
                        if tei.get('/health').status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    sleep(0.2)
                else:
                    raise TimeoutError('TEI did not become ready.')
                response = tei.get('/info')
                response.raise_for_status()
                info = response.json()
                assert info['max_input_length'] == 8192 and info['auto_truncate'] is False
                write_json(out / 'server-info.json', info)
                gpu_thread = threading.Thread(target=observe_gpu, daemon=True)
                gpu_thread.start()

                def execute(arm):
                    batches = [[item] for item in items] if arm == 'tei_f16_single' else groups
                    vectors = []
                    for batch in batches:
                        texts = [item['text'] for item in batch]
                        if arm == 'ollama_q8':
                            raw = ollama.embed(model=spec.model, input=texts, truncate=False,
                                               options={'num_ctx': 8192}).embeddings
                        else:
                            response = tei.post('/embed', json={'inputs': texts, 'truncate': False, 'normalize': True})
                            response.raise_for_status()
                            raw = response.json()
                        vectors.append(validate_vectors(raw, rows=len(batch), dimensions=1024,
                                                        dtype='float64', normalization='l2'))
                    return np.concatenate(vectors)

                references = {}
                for arm in protocol['arms']:
                    phase = 'warmup_' + arm
                    references[arm] = execute(arm)
                    np.save(out / (arm + '-warmup.npy'), references[arm])
                    print(arm, 'full sample warmed', flush=True)
                meta['status'] = 'timing'
                for repetition in range(3):
                    arms = protocol['arms'][repetition:] + protocol['arms'][:repetition]
                    for arm in arms:
                        phase = arm
                        start = perf_counter()
                        matrix = execute(arm)
                        seconds = perf_counter() - start
                        def compare(reference):
                            cosine = np.sum(matrix * reference, axis=1) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(reference, axis=1))
                            return {'exact': bool(np.array_equal(matrix, reference)),
                                    'max_abs_difference': float(np.max(np.abs(matrix - reference))),
                                    'min_cosine': float(cosine.min()), 'mean_cosine': float(cosine.mean())}
                        row = {'arm': arm, 'repetition': repetition, 'seconds': seconds,
                               'inputs_per_second': len(items) / seconds, 'vs_original_cache': compare(expected),
                               'vs_own_warmup': compare(references[arm]),
                               'vs_tei_single': compare(references['tei_f16_single'])}
                        meta['rounds'].append(row)
                        np.save(out / f'{arm}-{repetition}.npy', matrix)
                        write_json(out / 'audit.json', meta)
                        print(json.dumps(row), flush=True)
                        phase = 'between_rounds'
                meta['summary'] = {arm: {'mean_inputs_per_second': statistics.mean(
                    row['inputs_per_second'] for row in meta['rounds'] if row['arm'] == arm)} for arm in protocol['arms']}
                meta['status'] = 'completed'
    except BaseException as error:
        meta.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        done.set()
        if gpu_thread:
            gpu_thread.join(timeout=5)
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        meta['temporary_server_stopped'] = proc is None or proc.poll() is not None
        meta['source_sqlite_unchanged'] = digest(database) == source_hash
        write_json(out / 'gpu-samples.json', gpu_samples)
        write_json(out / 'audit.json', meta)


if __name__ == '__main__':
    main()

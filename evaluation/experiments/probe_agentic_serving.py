"""Frozen request-level concurrency screen; no tool execution or answer scoring."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
from time import perf_counter

from arkb.evaluation.external import digest, write_json
from arkb.agent.tools import FINAL_SCHEMA
from evaluation.agentic_tools.contract import MODEL, OPTIONS, THINK
from evaluation.agentic_tools.prepare import utc
from evaluation.agentic_tools.records import read_records
from evaluation.agentic_tools.runner import verify_files, gpu_competitors
from evaluation.agentic_tools.stopping import StopController
from evaluation.agentic_tools.transport import LocalChatClient


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def workload(rows):
    first = {}
    for row in rows:
        s = row['schedule']
        first.setdefault(s['dataset'], s['id'])
    selected = []
    for row in rows:
        s = row['schedule']
        if s['id'] != first[s['dataset']] or s['arm'] not in ('F-S', 'A-All'):
            continue
        models = row['result']['observation']['models']
        if not models:
            raise ValueError('Selected trajectory has no actual provider request.')
        request = deepcopy(models[0]['request'])
        selected.append({'id': row['key'], 'dataset': s['dataset'], 'question_id': s['id'],
                         'variant': s['variant'], 'arm': s['arm'], 'request': request,
                         'request_sha256': hash_value(request)})
    if len(selected) != 10:
        raise ValueError('Expected ten outcome-independent requests including both MuSiQue variants.')
    return selected


def blocks(items):
    result = []
    for repeat, levels in enumerate(((1, 2, 4), (2, 4, 1), (4, 1, 2))):
        ids = sorted((x['id'] for x in items), key=lambda i: hash_value(['serving-v1', 20260914, repeat, i]))
        for concurrency in levels:
            result.append({'index': len(result), 'repeat': repeat, 'concurrency': concurrency, 'request_ids': ids})
    return result


def normalized_message(value):
    message = deepcopy(value.get('message') or {})
    content = message.get('content')
    if isinstance(content, str):
        try:
            message['content'] = json.loads(content)
        except (ValueError, TypeError):
            pass
    return hash_value(message)


def prepare(pilot, out, root):
    protocol, rows = read_records(pilot)
    verify_files(pilot, protocol)
    readiness = json.loads((pilot / 'tool-readiness.json').read_text())
    if len(rows) != 98 or readiness['status'] != 'passed' or not readiness['scope_fixtures_passed']:
        raise ValueError('Complete v3 tool readiness is required.')
    out.mkdir(parents=True, exist_ok=False)
    items = workload(rows)
    write_json(out / 'requests.json', items)
    write_json(out / 'blocks.json', blocks(items))
    shutil.copyfile(__file__, out / 'probe.py')
    shutil.copyfile(root / 'docs/agentic-serving-probe-v1.md', out / 'plan.md')
    shutil.copyfile(root / 'evaluation/agentic-tools/serving-v1/tests.xml', out / 'tests.xml')
    write_json(out / 'protocol.json', {
        'schema': 'arkb-agentic-serving-probe-v1', 'created_at': utc(),
        'parent': str(pilot), 'parent_protocol_sha256': digest(pilot / 'protocol.json'),
        'parent_readiness_sha256': digest(pilot / 'tool-readiness.json'),
        'parent_source': protocol['measured_source'], 'model': protocol['models']['chat'],
        'ollama_version': protocol['ollama_version'],
        'inference_lock': protocol['inference_lock'],
        'files': {n: digest(out / n) for n in ['requests.json', 'blocks.json', 'probe.py', 'plan.md', 'tests.xml']},
        'requests': 90, 'warmups': 9, 'concurrency_levels': [1, 2, 4], 'repetitions': 3,
        'options': OPTIONS, 'think': THINK, 'truncate': False, 'shift': False,
        'scope': 'request serving and output repeatability only; not full Agent throughput or answer quality',
        'screen': {'minimum_median_throughput_ratio': 1.10, 'operational_errors': 0, 'additional_length_outputs': 0},
        'core_dispatch_authorized': False})


def ollama_process_memory(lines):
    parsed = [line.strip().split(None, 3) for line in lines]
    parsed = [p for p in parsed if len(p) == 4]
    roots = {int(p[0]) for p in parsed if p[3].startswith('/Applications/Ollama.app/Contents/Resources/ollama serve')}
    selected = set(roots)
    while True:
        descendants = {int(p[0]) for p in parsed if int(p[1]) in selected}
        if descendants <= selected:
            break
        selected.update(descendants)
    return [{'pid': int(p[0]), 'rss_kib': int(p[2])} for p in parsed if int(p[0]) in selected]


def memory_sample():
    processes = ollama_process_memory(subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss=,command='], text=True).splitlines())
    return {'at': utc(), 'processes': processes, 'total_rss_kib': sum(p['rss_kib'] for p in processes)}


def one_call(item, out, call_id, identity, submitted, cancel):
    if cancel.is_set():
        return {'id': call_id, 'request_id': item['id'], 'status': 'unstarted'}
    start = perf_counter()
    directory = out / 'calls' / call_id
    directory.mkdir()
    events, response, error = [], None, None
    try:
        client = LocalChatClient(identity, journal=events.append)
        try:
            response = client.chat(**item['request']).model_dump(exclude_none=True)
            if response.get('done') is not True:
                raise ValueError('Nonstreaming provider response was incomplete.')
            if (response.get('message') or {}).get('thinking') or response.get('eval_count', OPTIONS['num_predict'] + 1) > OPTIONS['num_predict']:
                raise ValueError('Provider violated registered thinking or output budget.')
            if item['id'] == 'synthetic-ready' and json.loads(response['message']['content']) != {'answer': 'ready', 'status': 'answered', 'evidence_refs': []}:
                raise ValueError('Synthetic ready control did not pass.')
        finally:
            client.close()
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        cancel.set()
    elapsed = perf_counter() - start
    write_json(directory / 'provider.json', {'request': item['request'], 'journal': events, 'response': response, 'error': error})
    row = {'id': call_id, 'request_id': item['id'], 'status': 'error' if error else 'completed',
           'client_seconds': elapsed, 'client_admission_wait_seconds': start - submitted,
           'error': error, 'message_sha256': normalized_message(response) if response else None,
           'done_reason': response.get('done_reason') if response else None,
           'usage': {k: response.get(k) for k in ('total_duration', 'load_duration', 'prompt_eval_duration',
                     'eval_duration', 'prompt_eval_count', 'eval_count')} if response else {},
           'provider_sha256': digest(directory / 'provider.json')}
    write_json(directory / 'result.json', row)
    write_json(directory / 'complete.json', {'result_sha256': digest(directory / 'result.json'),
                                           'provider_sha256': row['provider_sha256']})
    return row


def loaded_context(client, identity):
    data = client.http.get('/api/ps').raise_for_status().json()
    model = next((m for m in data['models'] if m['name'] == MODEL), None)
    if not model or model['digest'] != identity['digest'] or model['context_length'] != OPTIONS['num_ctx']:
        raise ValueError('Effective model identity or context differs from registration.')
    return data


def summarize(batch_rows):
    groups = defaultdict(list)
    for block in batch_rows:
        groups[block['concurrency']].append(block)
    baseline = statistics.median(b['throughput_per_second'] for b in groups[1])
    baseline_length = sum(r.get('done_reason') == 'length' for b in groups[1] for r in b['calls'])
    summaries = []
    for level, batches in sorted(groups.items()):
        rows = [r for b in batches for r in b['calls']]
        completed = [r for r in rows if r['status'] == 'completed']
        times = sorted(r['client_seconds'] for r in rows if r['status'] != 'unstarted')
        throughput = statistics.median(b['throughput_per_second'] for b in batches)
        identities = defaultdict(set)
        for r in completed:
            identities[r['request_id']].add(r['message_sha256'])
        errors = sum(r['status'] == 'error' for r in rows)
        length = sum(r.get('done_reason') == 'length' for r in rows)
        unstarted = sum(r['status'] == 'unstarted' for r in rows)
        summaries.append({'concurrency': level, 'blocks': len(batches), 'completed': len(completed),
                          'operational_errors': errors, 'unstarted': unstarted, 'length_outputs': length,
                          'median_throughput_per_second': throughput, 'throughput_vs_one': throughput / baseline,
                          'client_seconds_p95': times[math.ceil(len(times) * .95) - 1] if times else None,
                          'identical_output_requests': sum(len(v) == 1 for v in identities.values()),
                          'distinct_requests': len(identities),
                          'eligible_for_agent_confirmation': len(batches) == 3 and not errors and not unstarted
                          and throughput / baseline >= 1.10 and length <= baseline_length})
    return summaries


def run(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    if Path(__file__).resolve() != (out / 'probe.py').resolve():
        raise ValueError('Execute the frozen probe.py.')
    for name, checksum in protocol['files'].items():
        if digest(out / name) != checksum:
            raise ValueError('Probe input changed: ' + name)
    pilot = Path(protocol['parent'])
    if digest(pilot / 'protocol.json') != protocol['parent_protocol_sha256'] or digest(pilot / 'tool-readiness.json') != protocol['parent_readiness_sha256']:
        raise ValueError('Parent identity changed.')
    parent = json.loads((pilot / 'protocol.json').read_text())
    verify_files(pilot, parent)
    if (out / 'status.json').exists():
        raise ValueError('This bounded probe does not automatically retry or resume attempts.')
    with Path(protocol['inference_lock']).open('a') as lock, StopController(out / 'stop-request.json') as stop:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (out / 'calls').mkdir()
        items = {r['id']: r for r in json.loads((out / 'requests.json').read_text())}
        batches = json.loads((out / 'blocks.json').read_text())
        records, memory = [], []
        meta = {'status': 'running', 'pid': os.getpid(), 'started_at': utc(), 'completed': 0, 'total': 90,
                'protocol_sha256': digest(out / 'protocol.json')}
        def status(**values):
            meta.update(values, updated_at=utc())
            write_json(out / 'status.json', meta)
            print(json.dumps(meta), flush=True)
        client = LocalChatClient(protocol['model'])
        awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        try:
            status()
            for batch in batches:
                competitors = gpu_competitors()
                if competitors:
                    stop.request('competing_gpu_work')
                if stop.requested():
                    status(status='paused', administrative_stop=stop.persist())
                    return
                client.verify_identity()
                if client.http.get('/api/version').raise_for_status().json() != protocol['ollama_version']:
                    raise ValueError('Provider software version changed.')
                warmup = {'id': 'synthetic-ready', 'request': {'model': MODEL, 'messages': [{'role': 'user',
                    'content': 'Synthetic configuration check: return answer=ready, status=answered, evidence_refs=[].'}],
                    'stream': False, 'think': THINK, 'truncate': False, 'shift': False, 'options': OPTIONS, 'format': FINAL_SCHEMA}}
                warm = one_call(warmup, out, 'warmup-' + str(batch['index']), protocol['model'], perf_counter(), threading.Event())
                if warm['status'] != 'completed':
                    raise ValueError('Synthetic warmup failed.')
                before = loaded_context(client, protocol['model'])
                cancel = threading.Event()
                start = perf_counter()
                call_rows, contamination = [], []
                status(block=batch['index'], concurrency=batch['concurrency'], repetition=batch['repeat'])
                with ThreadPoolExecutor(max_workers=batch['concurrency']) as pool:
                    futures = {pool.submit(one_call, items[item_id], out, f"b{batch['index']}-{index}",
                               protocol['model'], perf_counter(), cancel) for index, item_id in enumerate(batch['request_ids'])}
                    last_sample = 0
                    while futures:
                        done, futures = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                        for future in done:
                            call_rows.append(future.result())
                        if perf_counter() - last_sample >= 5:
                            sample = memory_sample()
                            memory.append({'block': batch['index'], **sample})
                            contamination.extend(gpu_competitors())
                            last_sample = perf_counter()
                        if stop.requested() or contamination:
                            cancel.set()
                wall = perf_counter() - start
                record = {**batch, 'wall_seconds': wall, 'calls': sorted(call_rows, key=lambda r: r['id']),
                          'throughput_per_second': sum(r['status'] == 'completed' for r in call_rows) / wall,
                          'loaded_before': before, 'loaded_after': loaded_context(client, protocol['model']),
                          'competing_processes': sorted(set(contamination))}
                records.append(record)
                write_json(out / 'blocks-results.json', records)
                write_json(out / 'memory-samples.json', memory)
                status(completed=sum(r['status'] != 'unstarted' for b in records for r in b['calls']))
                client.verify_identity()
                if contamination or any(r['status'] != 'completed' for r in call_rows):
                    stop.request('operational_failure_or_gpu_competition')
                if stop.requested():
                    status(status='paused', administrative_stop=stop.persist())
                    return
            verify_files(pilot, parent)
            for name, checksum in protocol['files'].items():
                if digest(out / name) != checksum:
                    raise ValueError('Probe source or inputs changed during execution.')
            write_json(out / 'summary.json', {'status': 'completed', 'completed_at': utc(), 'groups': summarize(records),
                       'protocol_sha256': meta['protocol_sha256'], 'peak_sampled_ollama_rss_kib': max(s['total_rss_kib'] for s in memory),
                       'server_queue_seconds': None, 'server_queue_qualification': 'Unavailable; client admission wait and provider durations are separate fields.',
                       'quality_scores_used': False, 'full_agent_confirmation_required': True})
            status(status='completed')
        except BaseException as exc:
            status(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            client.close()
            awake.terminate()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--pilot', type=Path)
    p.add_argument('--root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.prepare:
        prepare(a.pilot.resolve(), a.output.resolve(), a.root.resolve())
    else:
        run(a.output.resolve())


if __name__ == '__main__':
    main()

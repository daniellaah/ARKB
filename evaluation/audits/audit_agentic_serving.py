"""Verify the complete serving screen and reproduce its descriptive results."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path

from ollama import ChatResponse
from arkb.evaluation.external import digest, write_json
from evaluation.agentic_tools.prepare import utc
from evaluation.agentic_tools.runner import verify_files


def audit(out, destination):
    protocol = json.loads((out / 'protocol.json').read_text())
    for name, checksum in protocol['files'].items():
        if digest(out / name) != checksum:
            raise ValueError('Frozen screen input changed: ' + name)
    parent = Path(protocol['parent'])
    if digest(parent / 'protocol.json') != protocol['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed.')
    verify_files(parent, json.loads((parent / 'protocol.json').read_text()))
    spec = importlib.util.spec_from_file_location('frozen_serving_probe', out / 'probe.py')
    measured = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measured)
    status = json.loads((out / 'status.json').read_text())
    if status['status'] != 'completed' or status['completed'] != 90:
        raise ValueError('The screen must complete before final analysis.')
    items = {r['id']: r for r in json.loads((out / 'requests.json').read_text())}
    schedule = json.loads((out / 'blocks.json').read_text())
    results = json.loads((out / 'blocks-results.json').read_text())
    if len(results) != 9 or len(items) != 10:
        raise ValueError('Workload or block count differs from registration.')
    expected_dirs, files, calls, fingerprints = set(), {}, [], defaultdict(dict)

    def checked(call_id, request=None):
        path = out / 'calls' / call_id
        expected_dirs.add(call_id)
        manifest = json.loads((path / 'complete.json').read_text())
        for name, field in [('result.json', 'result_sha256'), ('provider.json', 'provider_sha256')]:
            if digest(path / name) != manifest[field]:
                raise ValueError('Immutable provider record changed: ' + call_id)
        files[call_id] = manifest
        row = json.loads((path / 'result.json').read_text())
        provider = json.loads((path / 'provider.json').read_text())
        if row['status'] != 'completed' or provider['error'] is not None:
            raise ValueError('Unexpected operational failure in completed screen.')
        if request is not None and provider['request'] != request:
            raise ValueError('Replayed request differs from frozen workload.')
        journal = provider['journal']
        if len(journal) != 2 or journal[0] != {'event': 'provider_request', 'request': provider['request']} or journal[1]['event'] != 'provider_response' or journal[1]['http_status'] != 200:
            raise ValueError('Request/response journal contract mismatch.')
        raw = ChatResponse.model_validate(json.loads(journal[1]['body'])).model_dump(exclude_none=True)
        if raw != provider['response'] or measured.normalized_message(raw) != row['message_sha256']:
            raise ValueError('Saved model output or stability fingerprint changed.')
        if raw.get('done') is not True or raw['message'].get('thinking'):
            raise ValueError('Completion/thinking contract violation.')
        req = provider['request']
        for key, expected in [('options', protocol['options']), ('think', False), ('truncate', False), ('shift', False)]:
            if req[key] != expected:
                raise ValueError('Actual request escaped registered settings.')
        if row['client_seconds'] < 0 or row['client_admission_wait_seconds'] < 0:
            raise ValueError('Invalid measured duration.')
        return row

    for planned, result in zip(schedule, results, strict=True):
        if any(result[key] != value for key, value in planned.items()) or result['competing_processes']:
            raise ValueError('Block identity or isolation differs from registration.')
        warm = checked('warmup-' + str(planned['index']))
        warm_raw = json.loads((out / 'calls' / warm['id'] / 'provider.json').read_text())['response']
        if json.loads(warm_raw['message']['content']) != {'answer': 'ready', 'status': 'answered', 'evidence_refs': []}:
            raise ValueError('Warmup did not pass.')
        by_id = {row['id']: row for row in result['calls']}
        if len(by_id) != 10:
            raise ValueError('Block does not contain every request exactly once.')
        for index, item_id in enumerate(planned['request_ids']):
            row = checked(f"b{planned['index']}-{index}", items[item_id]['request'])
            if row != by_id[row['id']] or row['request_id'] != item_id:
                raise ValueError('Block results differ from immutable call records.')
            fingerprints[item_id][planned['concurrency'], planned['repeat']] = row['message_sha256']
            calls.append(row)
        for key in ('loaded_before', 'loaded_after'):
            active = next(m for m in result[key]['models'] if m['name'] == protocol['model']['name'])
            if active['digest'] != protocol['model']['digest'] or active['context_length'] != protocol['options']['num_ctx']:
                raise ValueError('Effective model/context changed.')
        if result['throughput_per_second'] != 10 / result['wall_seconds']:
            raise ValueError('Block throughput does not match completed calls.')
    if {p.name for p in (out / 'calls').iterdir() if p.is_dir()} != expected_dirs:
        raise ValueError('Unexpected or missing call directory.')
    summary = json.loads((out / 'summary.json').read_text())
    groups = measured.summarize(results)
    if groups != summary['groups']:
        raise ValueError('Reported concurrency summary differs from independent replay.')
    stability = [{'request_id': item_id, 'dataset': items[item_id]['dataset'], 'arm': items[item_id]['arm'],
                  'variant': items[item_id]['variant'], 'distinct_messages_all_levels': len(set(values.values())),
                  'same_as_paired_concurrency_one': {str(c): sum(values[c, r] == values[1, r] for r in range(3)) for c in (2, 4)}}
                 for item_id, values in fingerprints.items()]
    report = {'status': 'passed', 'checked_at': utc(), 'audit_source_sha256': digest(__file__),
              'protocol_sha256': digest(out / 'protocol.json'), 'measured_calls': len(calls), 'warmups': 9,
              'groups': groups, 'request_stability': stability, 'call_checksums': files,
              'data_files': {n: digest(out / n) for n in ('blocks-results.json', 'memory-samples.json', 'summary.json', 'status.json')},
              'quality_scores_used': False, 'full_agent_confirmation_required': True,
              'interpretation': 'Three development repetitions per concurrency, current service/cache behavior. No exact server queue measurement and no full-Agent speedup or quality claim.'}
    write_json(destination, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.output, a.report)
    print(json.dumps({k: result[k] for k in ('status', 'measured_calls', 'groups')}))


if __name__ == '__main__':
    main()

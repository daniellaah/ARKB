"""Audit recorded execution costs without reading answers or assigning quality scores."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_run(root, phase):
    protocol_hash = sha(root / 'protocol.json')
    schedule = json.loads((root / f'{phase}-schedule.json').read_text())
    groups = defaultdict(list)
    manifest, incomplete = [], []
    seen = set()
    for directory in sorted((root / f'{phase}-attempts').iterdir()):
        if not directory.is_dir():
            continue
        attempt = json.loads((directory / 'attempt.json').read_text())
        row = attempt['schedule']
        key = hashlib.sha256(json.dumps([
            protocol_hash, row['dataset'], row['id'], row['variant'],
            row['arm'], row['repetition']], ensure_ascii=False,
            separators=(',', ':')).encode()).hexdigest()
        assert row in schedule and attempt['protocol_sha256'] == protocol_hash
        assert attempt['key'] == directory.name == key
        identity = tuple(row[k] for k in ('dataset', 'id', 'variant', 'arm', 'repetition'))
        assert identity not in seen
        seen.add(identity)
        files = {p.name: sha(p) for p in directory.iterdir() if p.is_file()}
        if not (directory / 'complete.json').exists():
            incomplete.append({'key': directory.name, 'files_sha256': files})
            continue
        record = json.loads((directory / 'result.json').read_text())
        complete = json.loads((directory / 'complete.json').read_text())
        assert complete['result_sha256'] == files['result.json']
        assert complete['provider_sha256'] == files['provider.jsonl']
        assert record['schedule'] == row and record['protocol_sha256'] == protocol_hash
        observation = (record.get('result') or {}).get('observation') or {}
        models = observation.get('models') or []
        tools = [t for t in observation.get('tools', []) if t.get('executed')]
        values = Counter(total_seconds=record['elapsed_ms'] / 1000,
                         model_seconds=sum(t.get('elapsed_ms') or 0 for t in models) / 1000,
                         tool_seconds=sum(t.get('elapsed_ms') or 0 for t in tools) / 1000,
                         model_calls=len(models), tool_calls=len(tools))
        for tool in tools:
            name = tool['name']
            values[name + '_seconds'] += (tool.get('elapsed_ms') or 0) / 1000
            values[name + '_calls'] += 1
            error = tool.get('error') or {}
            if isinstance(error, dict) and error.get('code') == 'exact_timeout':
                values['match_timeout_calls'] += 1
        groups[row['dataset'], row['arm']].append(values)
        manifest.append({'key': directory.name, 'files_sha256': files})
    aggregates = []
    for (dataset, arm), rows in sorted(groups.items()):
        sums = Counter()
        for row in rows:
            sums.update(row)
        aggregates.append({'dataset': dataset, 'arm': arm, 'completed': len(rows),
                           'sums': dict(sums),
                           'means': {k: v / len(rows) for k, v in sums.items()}})
    return {'path': str(root), 'protocol_sha256': protocol_hash,
            'scheduled': len(schedule), 'completed': len(manifest),
            'incomplete': incomplete, 'unstarted': len(schedule) - len(seen),
            'groups': aggregates, 'attempt_files': manifest}, schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pilot, _ = inspect_run(args.root / 'pilot-v2', 'pilot')
    core, schedule = inspect_run(args.root / 'core-v1', 'core')
    assert pilot['completed'] == 98 and not pilot['incomplete']
    multiplicities = Counter((r['dataset'], r['arm']) for r in schedule)
    projected = Counter()
    for group in pilot['groups']:
        count = multiplicities[group['dataset'], group['arm']]
        for key, value in group['means'].items():
            projected[key] += value * count
    root = args.root / 'core-v1'
    source_manifest = json.loads((root / 'measured-source-manifest.json').read_text())
    assert all(sha(root / 'measured-source' / name) == value
               for name, value in source_manifest.items())
    result = {'schema': 'arkb-agentic-execution-cost-audit-v1',
              'checked_at': datetime.now(timezone.utc).isoformat(),
              'audit_source_sha256': sha(Path(__file__)),
              'pilot': pilot, 'core': core,
              'projected_core_hours': {k.removesuffix('_seconds'): v / 3600
                                       for k, v in projected.items() if k.endswith('_seconds')},
              'core_source_files_verified': len(source_manifest),
              'interpretation': [
                  'Projected hours extrapolate small pilot group means, not a completion guarantee.',
                  'Tool times include CPU, filesystem, provider and service work; no GPU utilization is inferred.',
                  'Match timeout is a tool-call outcome, even when the final trajectory is schema-valid.',
                  'Core was stopped administratively; incomplete and unstarted work are not model failures.',
                  'No answer correctness or strategy-superiority analysis was performed.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'pilot_completed': pilot['completed'],
                      'core_completed': core['completed'], 'core_incomplete': len(core['incomplete']),
                      'core_unstarted': core['unstarted'],
                      'projected_core_hours': result['projected_core_hours']}))


if __name__ == '__main__':
    main()

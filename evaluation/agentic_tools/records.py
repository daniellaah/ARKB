"""Read immutable scheduled attempts; do not discover a cohort from outcomes."""
import json
from pathlib import Path

from arkb.evaluation.external import digest
from .selection import attempt_key


def read_records(output, *, require_complete=True):
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    phase = protocol['phase']
    schedule = json.loads((output / f'{phase}-schedule.json').read_text())
    protocol_hash = digest(output / 'protocol.json')
    rows = []
    for planned in schedule:
        key = attempt_key(protocol_hash, planned)
        directory = output / f'{phase}-attempts' / key
        if not (directory / 'complete.json').exists():
            if require_complete:
                raise ValueError('A scheduled attempt is incomplete: ' + key)
            continue
        manifest = json.loads((directory / 'complete.json').read_text())
        for name, field in [('result.json', 'result_sha256'), ('provider.jsonl', 'provider_sha256')]:
            if digest(directory / name) != manifest[field]:
                raise ValueError('Attempt checksum changed: ' + key)
        row = json.loads((directory / 'result.json').read_text())
        if row['key'] != key or row['schedule'] != planned or row['protocol_sha256'] != protocol_hash:
            raise ValueError('Scheduled attempt identity changed.')
        report = (row.get('result') or {}).get('observation')
        if report:
            requests = [event['request'] for line in (directory / 'provider.jsonl').read_text().splitlines()
                        if (event := json.loads(line)).get('event') == 'provider_request']
            if requests != [m['request'] for m in report['models']]:
                raise ValueError('Observed requests disagree with the actual provider journal.')
        rows.append(row)
    return protocol, rows

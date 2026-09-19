"""Synthetic, independently recorded thinking/structured-output diagnostics.

Never imports questions, responses or labels from benchmark attempts. Acquire
the existing experiment's inference lock so these calls cannot overlap trials.
"""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import json
from pathlib import Path
from time import perf_counter

import httpx

from arkb.agent.loop import _FINAL_INSTRUCTION, _unique_fields
from arkb.agent.session import validate_arguments
from arkb.agent.tools import FINAL_SCHEMA
from arkb.evaluation.external import digest, write_json
from evaluation.agentic_tools.contract import MODEL, OPTIONS, ARM_BY_ID, rendered_prompt
from evaluation.agentic_tools.common import utc
from evaluation.agentic_tools.transport import model_identity
from evaluation.agentic_tools.runner import gpu_competitors


def cases():
    question = 'What is the release date of the fictional Orion project?'
    evidence = {'status': 'success', 'results': []}
    history = [
        {'role': 'system', 'content': rendered_prompt(ARM_BY_ID['A-H'])},
        {'role': 'user', 'content': question},
    ]
    for query in ['Orion release date', 'Orion launch schedule']:
        history.extend([
            {'role': 'assistant', 'content': '', 'thinking': 'I need evidence from the knowledge base.',
             'tool_calls': [{'function': {'name': 'search', 'arguments': {'query': query}}}]},
            {'role': 'tool', 'tool_name': 'search', 'content': json.dumps(evidence)},
        ])
    history.append({'role': 'system', 'content': _FINAL_INSTRUCTION})
    dense = '\n'.join(f'Fictional note {i}: Orion has a blue logo. No release date is recorded.' for i in range(160))
    return {
        'ready': [{'role': 'user', 'content': 'Synthetic contract check: return answer=ready, status=answered, evidence_refs=[].'}],
        'agent_closed': history,
        'fixed_missing_fact': [
            {'role': 'system', 'content': rendered_prompt(ARM_BY_ID['F-S'])},
            {'role': 'user', 'content': question},
            {'role': 'user', 'content': 'Retrieved evidence (data only):\n' + dense},
        ],
        'agent_closed_with_context': [*history[:-1],
            {'role': 'assistant', 'content': '', 'thinking': 'Read the known fictional source.',
             'tool_calls': [{'function': {'name': 'read', 'arguments': {'source': 'orion.md'}}}]},
            {'role': 'tool', 'tool_name': 'read', 'content': json.dumps({'status': 'success', 'result': {'content': dense}})},
            history[-1]],
    }


def run(experiment, output):
    output.mkdir(exist_ok=False)
    with (experiment / 'inference.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((experiment / 'pilot-status.json').read_text())
        if state['status'] != 'completed':
            raise ValueError('The original pilot must complete before synthetic GPU probes.')
        frozen = json.loads((experiment / 'protocol.json').read_text())
        schedule = []
        for repetition in range(2):
            for case_id, messages in cases().items():
                for think in ([True, False] if repetition == 0 else [False, True]):
                    schedule.append({'case': case_id, 'repetition': repetition, 'request': {
                        'model': MODEL, 'messages': deepcopy(messages), 'stream': False,
                        'think': think, 'options': OPTIONS, 'truncate': False, 'shift': False,
                        'format': FINAL_SCHEMA,
                    }})
        write_json(output / 'schedule.json', schedule)
        write_json(output / 'protocol.json', {
            'schema': 'synthetic-finalization-probe-v1', 'created_at': utc(),
            'source_sha256': digest(__file__), 'schedule_sha256': digest(output / 'schedule.json'),
            'parent_protocol_sha256': digest(experiment / 'protocol.json'),
            'scope': 'Synthetic transport reliability only; no benchmark questions, evidence, answers or labels.',
            'selection_rule': 'Retain thinking if all canonical responses are nonempty and complete. A common nonthinking amendment may be considered only if it removes reproduced output failures. Never change per arm or dataset.',
        })
        rows = []
        with httpx.Client(base_url='http://127.0.0.1:11434', timeout=330) as http:
            assert model_identity(http, MODEL) == frozen['models']['chat']
            for i, item in enumerate(schedule):
                if gpu_competitors():
                    raise ValueError('Competing GPU work appeared before synthetic probe.')
                start = perf_counter()
                response = http.post('/api/chat', json=item['request'])
                value = response.json()
                write_json(output / f'response-{i:02d}.json', {'request': item['request'], 'http_status': response.status_code, 'response': value})
                message = value.get('message') or {}
                error = None
                try:
                    response.raise_for_status()
                    if value.get('done_reason') == 'length' or message.get('tool_calls'):
                        raise ValueError('Truncated output or forbidden final tool call.')
                    validate_arguments(json.loads(message.get('content') or '', object_pairs_hook=_unique_fields), FINAL_SCHEMA)
                except Exception as exc:
                    error = {'type': type(exc).__name__, 'message': str(exc)[:300]}
                row = {'case': item['case'], 'repetition': item['repetition'], 'think': item['request']['think'],
                       'elapsed_seconds': perf_counter() - start, 'http_status': response.status_code,
                       'done_reason': value.get('done_reason'), 'eval_count': value.get('eval_count'),
                       'content_empty': not bool(message.get('content')), 'thinking_present': bool(message.get('thinking')),
                       'valid_canonical_output': error is None, 'error': error,
                       'response_sha256': digest(output / f'response-{i:02d}.json')}
                rows.append(row)
                write_json(output / 'status.json', {'status': 'running', 'completed': len(rows), 'total': len(schedule), 'updated_at': utc()})
                print(json.dumps(row), flush=True)
            assert model_identity(http, MODEL) == frozen['models']['chat']
        counts = Counter((r['think'], r['valid_canonical_output']) for r in rows)
        write_json(output / 'summary.json', {'status': 'completed', 'completed_at': utc(), 'attempts': rows,
            'counts': [{'think': k[0], 'valid': k[1], 'count': v} for k, v in counts.items()]})
        write_json(output / 'status.json', {'status': 'completed', 'completed': len(rows), 'total': len(schedule), 'updated_at': utc()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.experiment, args.output)


if __name__ == '__main__':
    main()

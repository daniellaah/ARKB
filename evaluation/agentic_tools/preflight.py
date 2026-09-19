"""Synthetic provider checks before benchmark trials; no benchmark text involved."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import httpx
from arkb.evaluation.external import write_json
from arkb.agent.tools import FINAL_SCHEMA
from .contract import MODEL, OPTIONS, THINK
from .transport import model_identity
from .prepare import utc


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    with httpx.Client(base_url='http://127.0.0.1:11434', timeout=180) as http:
        identity = model_identity(http, MODEL)
        report = {'status': 'running', 'started_at': utc(), 'chat_model': identity, 'probes': []}
        dest = a.output / 'provider-preflight.json'
        write_json(dest, report)
        overflow = {'model': MODEL, 'messages': [{'role': 'user', 'content': ' alpha' * 5000}],
                    'stream': False, 'think': THINK, 'truncate': False, 'shift': False,
                    'options': {**OPTIONS, 'num_ctx': 1024, 'num_predict': 1}}
        response = http.post('/api/chat', json=overflow)
        report['probes'].append({'kind': 'oversized_input', 'input': '5000 repetitions of synthetic alpha',
                                 'options': overflow['options'], 'http_status': response.status_code,
                                 'response': response.text})
        write_json(dest, report)
        if response.status_code < 400 or not any(w in response.text.lower() for w in ('context', 'length', 'tokens')):
            raise ValueError('Provider did not explicitly reject oversized non-truncating input.')
        request = {'model': MODEL, 'messages': [{'role': 'user', 'content':
                    'This is a synthetic configuration check, not a knowledge question. Return answer=ready, status=answered, evidence_refs=[].'}],
                   'stream': False, 'think': THINK, 'truncate': False, 'shift': False,
                   'options': OPTIONS, 'format': FINAL_SCHEMA}
        response = http.post('/api/chat', json=request)
        report['probes'].append({'kind': 'registered_context_and_format', 'request': request,
                                 'http_status': response.status_code, 'response': response.json()})
        write_json(dest, report)
        response.raise_for_status()
        value = response.json()
        if not THINK and value['message'].get('thinking'):
            raise ValueError('Provider returned thinking despite the nonthinking protocol.')
        final = json.loads(value['message']['content'])
        if final != {'answer': 'ready', 'status': 'answered', 'evidence_refs': []}:
            raise ValueError('Synthetic canonical response did not pass.')
        loaded = http.get('/api/ps').raise_for_status().json()['models']
        active = next(m for m in loaded if m['name'] == MODEL)
        if active['digest'] != identity['digest'] or active['context_length'] != OPTIONS['num_ctx']:
            raise ValueError('Effective model/context differs from requested settings.')
        if model_identity(http, MODEL) != identity:
            raise ValueError('Model changed during provider preflight.')
        report.update(status='passed', completed_at=utc(), loaded_model=active,
                      context_safety='truncate=false and shift=false, verified oversized input rejection',
                      requested_options=OPTIONS, requested_think=THINK,
                      output_limit_check='num_predict explicitly submitted and accepted; returned eval_count checked against it')
        if value.get('eval_count', OPTIONS['num_predict'] + 1) > OPTIONS['num_predict']:
            raise ValueError('Provider exceeded output token limit.')
        write_json(dest, report)
        print(json.dumps({'status': 'passed', 'context_length': active['context_length'],
                          'model_digest': identity['digest'], 'overflow_http_status': report['probes'][0]['http_status']}), flush=True)


if __name__ == '__main__':
    main()

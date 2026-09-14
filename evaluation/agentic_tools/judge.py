"""Local scoring and blinded calibration export, separate from Agent inference."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from time import perf_counter

import httpx

from arkb.evaluation.external import digest, write_json
from .labels import parse_judge
from .prepare import utc
from .records import read_records
from .runner import gpu_competitors
from .scoring import canonical, answer_eligible, musique_row
from .transport import model_identity


def calibration_rows(rows):
    selected = [r for r in rows if r['schedule']['dataset'] == 'browsecomp-plus' or
                (r['schedule']['dataset'] == 'musique' and r['schedule']['stratum'] == '2')]
    if len(selected) != 42:
        raise ValueError('Calibration requires all 42 registered responses.')
    return selected


def execute(source, output, *, calibration=False):
    import hashlib
    source, output = source.resolve(), output.resolve()
    protocol, rows = read_records(source)
    if calibration and protocol['phase'] != 'pilot':
        raise ValueError('Calibration uses pilot responses only.')
    rows = calibration_rows(rows) if calibration else [r for r in rows if r['schedule']['dataset'] == 'browsecomp-plus']
    output.mkdir(exist_ok=True)
    design = json.loads((source / 'judge-design.json').read_text())
    if digest(Path(__file__).with_name('labels.py')) != design['parser_sha256']:
        raise ValueError('Frozen judge parser changed.')
    if digest(source / 'scoring/browsecomp-answers.json') != design['labels_sha256']:
        raise ValueError('Scoring answer labels changed.')
    scenarios = json.loads((source / 'inference/scenarios.json').read_text())
    answers = json.loads((source / 'scoring/browsecomp-answers.json').read_text())['answers']
    musique = json.loads((source / 'scoring/musique.json').read_text())
    payloads, review, keys = [], [], {}
    for row in rows:
        final = canonical(row)
        case = scenarios[row['scenario_id']]
        ds = row['schedule']['dataset']
        blinded = hashlib.sha256(('blind-review-v1|' + row['key']).encode()).hexdigest()
        automatic = None
        eligible = answer_eligible(row)
        if ds == 'musique':
            gold = musique[row['scenario_id']]
            mapped = musique_row(row, gold)
            if not gold['answerable']:
                eligible = False
                automatic = mapped['answerability_correct']
            expected = gold['answer']
        else:
            expected = answers[row['schedule']['id']]
        if not eligible and automatic is None:
            automatic = 0.0
        text = final.answer if final and final.answer else '[No final answer was produced.]'
        request = {'model': design['model'], 'stream': False, 'think': design['think'],
                   'truncate': design['truncate'], 'shift': design['shift'], 'options': design['options'],
                   'messages': [{'role': 'user', 'content': design['rubric'].format(
                       question=case['query'], response=text, correct_answer=expected)}]} if eligible else None
        payloads.append({'id': blinded, 'request': request, 'automatic_score': automatic,
                         'score_basis': 'answer_judge' if eligible else
                         ('canonical_answerability' if ds == 'musique' and not gold['answerable'] else 'execution_error_or_nonanswer')})
        keys[blinded] = {'key': row['key'], 'schedule': row['schedule'], 'scenario_id': row['scenario_id']}
        review.append({'review_id': blinded, 'dataset': ds, 'question': case['query'],
                       'response': text, 'final_status': final.status if final else 'error',
                       'citations': final.citations if final else [], 'expected_answer': expected,
                       'gold_answerable': musique[row['scenario_id']]['answerable'] if ds == 'musique' else True,
                       'human_answer_correct': None, 'human_answerability_correct': None,
                       'reviewer': None, 'adjudication': None})
    # Hash ordering blinds arm order; no arm, repetition or attempt key appears in review rows.
    review.sort(key=lambda r: r['review_id'])
    payloads.sort(key=lambda r: r['id'])
    def frozen(name, value):
        target = output / name
        if target.exists():
            if json.loads(target.read_text()) != value:
                raise ValueError('Refusing to change frozen scoring input: ' + name)
        else:
            write_json(target, value)
    frozen('requests.json', payloads)
    frozen('blinded-review.json', review)
    frozen('private-review-key.json', keys)
    sources = {name: digest(Path(__file__).with_name(name)) for name in
               ('judge.py', 'labels.py', 'scoring.py', 'records.py')}
    frozen('protocol.json', {'schema': 'agentic-local-grading-v1', 'source_protocol_sha256': digest(source / 'protocol.json'),
                            'judge_design_sha256': digest(source / 'judge-design.json'),
                            'requests_sha256': digest(output / 'requests.json'), 'model': protocol['models']['judge'],
                            'source_sha256': sources, 'calibration': calibration,
                            'automatic_score_scope': 'MuSiQue answerability on unanswerable variants is canonical, not an LLM answer-quality label.',
                            'human_review_status': 'pending', 'quality_status': 'provisional'})
    status = {'status': 'running', 'pid': os.getpid(), 'started_at': utc(), 'completed': 0,
              'total': len(payloads), 'human_review': 'pending', 'quality_status': 'provisional'}
    def progress(**changes):
        status.update(changes, updated_at=utc())
        write_json(output / 'status.json', status)
    lock_path = Path(protocol.get('inference_lock', source / 'inference.lock'))
    with lock_path.open('a') as lock, httpx.Client(base_url='http://127.0.0.1:11434', timeout=330) as http:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if gpu_competitors():
                raise ValueError('Competing GPU workload prevents judge dispatch.')
            assert model_identity(http, design['model']) == protocol['models']['judge']
            progress(stage='synthetic_controls')
            results = []
            def call(identifier, request):
                for attempt in range(1 + design['max_unchanged_retries']):
                    prefix = output / f'{identifier}-{attempt}'
                    req_path = prefix.with_suffix('.request.json')
                    response_path = prefix.with_suffix('.response.json')
                    if req_path.exists() and not response_path.exists():
                        raise ValueError('An interrupted judge request needs explicit accounting: ' + identifier)
                    if response_path.exists():
                        saved = json.loads(response_path.read_text())
                        if json.loads(req_path.read_text()) != request:
                            raise ValueError('Judge retry request changed.')
                    else:
                        if gpu_competitors():
                            raise ValueError('Competing GPU workload appeared.')
                        write_json(req_path, request)
                        start = perf_counter()
                        try:
                            response = http.post('/api/chat', json=request)
                            saved = {'http_status': response.status_code, 'body': response.json(),
                                     'elapsed_ms': (perf_counter() - start) * 1000}
                        except Exception as exc:
                            saved = {'http_status': None, 'body': {}, 'elapsed_ms': (perf_counter() - start) * 1000,
                                     'transport_error': {'type': type(exc).__name__, 'message': str(exc)}}
                        write_json(response_path, saved)
                    body = saved['body']
                    parsed = parse_judge((body.get('message') or {}).get('content') or '')
                    good = (saved['http_status'] == 200 and body.get('done_reason') != 'length'
                            and not (body.get('message') or {}).get('thinking') and not parsed['parse_error'])
                    if good:
                        return {'correct': parsed['correct'], 'parse_error': False, 'attempts': attempt + 1,
                                'response_files': [f'{identifier}-{i}.response.json' for i in range(attempt + 1)]}
                return {'correct': None, 'parse_error': True, 'attempts': 1 + design['max_unchanged_retries'],
                        'response_files': [f'{identifier}-{i}.response.json' for i in range(1 + design['max_unchanged_retries'])]}
            controls = []
            for index, (candidate, expected) in enumerate([('seven', True), ('eight', False)]):
                request = {'model': design['model'], 'stream': False, 'think': False, 'truncate': False, 'shift': False,
                           'options': design['options'], 'messages': [{'role': 'user', 'content': design['rubric'].format(
                               question='What is three plus four?', response=candidate, correct_answer='seven')}]}
                result = call('control-' + str(index), request)
                controls.append({'expected': expected, **result})
            write_json(output / 'controls.json', controls)
            if any(r['correct'] is not r['expected'] for r in controls):
                raise ValueError('Judge failed a synthetic positive/negative control.')
            loaded = http.get('/api/ps').raise_for_status().json()
            active = next(m for m in loaded['models'] if m['name'] == design['model'])
            if active['digest'] != protocol['models']['judge']['digest'] or active['context_length'] != design['options']['num_ctx']:
                raise ValueError('Effective grader model/context differs.')
            write_json(output / 'loaded-judge.json', loaded)
            for payload in payloads:
                progress(stage='grading', current=payload['id'])
                result = call(payload['id'], payload['request']) if payload['request'] else {
                    'correct': bool(payload['automatic_score']), 'parse_error': False, 'attempts': 0, 'response_files': []}
                results.append({'id': payload['id'], 'basis': payload['score_basis'], **result,
                                'score': float(result['correct']) if result['correct'] is not None else None,
                                'review_status': 'pending', 'quality_status': 'provisional'})
                write_json(output / 'scores.json', results)
                progress(completed=len(results))
            assert model_identity(http, design['model']) == protocol['models']['judge']
            for name, expected in sources.items():
                if digest(Path(__file__).with_name(name)) != expected:
                    raise ValueError('Grading implementation changed during execution.')
            write_json(output / 'summary.json', {'status': 'completed', 'responses': len(results),
                'judge_requests': sum(r['attempts'] for r in results), 'pending_scores': sum(r['score'] is None for r in results),
                'human_review': 'pending', 'quality_status': 'provisional', 'agreement_with_humans': None})
            progress(status='completed', stage='grading_complete_human_review_pending')
        except BaseException as exc:
            progress(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            # Release only this grader's VRAM before the separately locked core worker starts.
            unload = http.post('/api/generate', json={'model': design['model'], 'keep_alive': 0})
            write_json(output / 'unload.json', {'http_status': unload.status_code, 'response': unload.text})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', action='store_true')
    args = p.parse_args()
    execute(args.source, args.output, calibration=args.calibration)


if __name__ == '__main__':
    main()

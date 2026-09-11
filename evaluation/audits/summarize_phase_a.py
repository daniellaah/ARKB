"""Offline reliability report with independently checked evidence and usage totals."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from arkb.evaluation.external_agent import coverage, evidence_hits, evidence_packet
from arkb.evaluation.multihop import musique_metrics
from arkb.evaluation.reliability import summarize_reliability
from arkb.knowledge.documents import DocumentAccess
from arkb.knowledge.embeddings import count_tokens
from tokenizers import Tokenizer


def inspect_arm(root, arm, cases):
    rows=[json.loads(line) for line in (root/arm/'rows.jsonl').read_text().splitlines()]
    if len(rows)!=len(cases) or [r['key'] for r in rows]!=[c['key'] for c in cases]:
        raise ValueError('Incomplete or reordered matrix.')
    tokenizer=Tokenizer.from_file(str(root/arm/'reference-tokenizer.json'))
    count=lambda value:count_tokens(value,tokenizer=tokenizer)
    diagnostics=[]; spans=0; model_requests=0; references=0
    for row,case in zip(rows,cases):
        result=row['result']
        if result is None:
            if row['error'] is None:raise ValueError('Missing run without failure.')
            continue
        report=result['observation']; canonical=bool(result.get('final'))
        access=DocumentAccess(Path(case['directory']),vault_id=case['vault'])
        cache={}; returned=delivered=0; unique={}; delivered_refs=set()
        for event in report['tools']:
            hits=evidence_hits(event['raw_result'],event['name'],report.get('evidence_references'))
            if event['returned_evidence_tokens'] is not None:
                tokens=sum(count(h['content']) for h in hits)
                if tokens!=event['returned_evidence_tokens']:raise ValueError('Returned evidence accounting differs.')
                returned+=tokens
                if event['delivered_to_conversation']:
                    if event['delivered_evidence_tokens']!=tokens:raise ValueError('Delivered evidence accounting differs.')
                    delivered+=tokens
                    for h in hits:
                        identity=json.dumps({k:h.get(k) for k in ('source','document_revision','start_char','end_char','content')},sort_keys=True,ensure_ascii=False)
                        unique[identity]=count(h['content'])
                        if canonical:delivered_refs.add(h['ref'])
            for hit in hits:
                if hit['source'] not in cache:
                    cache[hit['source']]=access.read(source=hit['source'])
                current=cache[hit['source']]
                if (hit['document_id']!=current.document_id or hit['document_revision']!=current.document_revision
                        or hit['content']!=current.content[hit['start_char']:hit['end_char']]):
                    raise ValueError('Evidence identity or body coordinates differ.')
                spans+=1
        if report['evidence']!={'returned_tokens':returned,'delivered_tokens':delivered,'unique_exact_excerpt_tokens':sum(unique.values())}:
            raise ValueError('Evidence totals differ.')
        if canonical:
            final=result['final']
            if final!=report['final'] or result['response']!=final['answer']:raise ValueError('Canonical outcome differs.')
            for citation in final['citations']:
                ref=citation['ref']; references+=1
                if ref not in delivered_refs or {k:v for k,v in citation.items() if k!='ref'}!=report['evidence_references'][ref]:
                    raise ValueError('Final citation was not delivered or differs from its binding.')
            if result['stop_reason']=='final':
                message=report['models'][-1]['response']['message']
                proposal=(next(c['function']['arguments'] for c in message['tool_calls'] if c['function']['name']=='finish')
                          if message.get('tool_calls') else json.loads(message['content']))
                if proposal['answer']!=final['answer'] or proposal['status']!=final['status']:
                    raise ValueError('Final answer differs from model proposal.')
        if len(report['models'])!=result['state']['turn']:raise ValueError('Turn count differs.')
        for model in report['models']:
            request=model['request']; model_requests+=1
            if count(json.dumps(request,ensure_ascii=False,sort_keys=True))!=model['request_json_reference_tokens']:
                raise ValueError('Model request reference tokens differ.')
            if request['messages']!=result['state']['messages'][:len(request['messages'])]:
                raise ValueError('Model request differs from retained conversation.')
        for name in ('prompt_eval_count','eval_count'):
            values=[m['usage'].get(name) if m['usage'] else None for m in report['models']]
            known=[v for v in values if type(v)is int and v>=0]
            expected={'total':sum(known) if len(known)==len(values) else None,
                      'known_total':sum(known),'defined_requests':len(known),'requests':len(values)}
            if expected!=report['usage'][name]:raise ValueError('Provider usage totals differ.')
        packet=evidence_packet(result,row['source_map'])
        # JSON object keys are strings; normalize scoring identities on both sides.
        score_packet={k:[str(v) for v in packet[k]] for k in ('returned','delivered','submitted')}
        score=coverage({str(k):v for k,v in row['qrels'].items()},row['aspects'],score_packet)
        diagnostics.append({'key':row['key'],'coverage':score,'stop_reason':row['stop_reason'],
            'termination_reason':(result.get('final') or {}).get('termination_reason'),
            'answer_status':(result.get('final') or {}).get('status'),
            'error':row['error'],'elapsed_ms':row['elapsed_ms'],
            'tool_statuses':dict(Counter(e['status'] for e in report['tools']))})
    mu=[r for r in rows if r['track']=='musique']
    gold=json.loads((root/'musique-gold.json').read_text())
    predictions=[{'id':r['id'],**r['prediction']} for r in mu]
    averaged={phase:{metric:mean(values) if values else None for metric in
        ('positive_document_recall','weighted_aspect_coverage')
        for values in [[d['coverage'][phase][metric] for d in diagnostics if d['coverage'][phase][metric] is not None]]}
        for phase in ('returned','delivered','submitted')}
    return {'reliability':summarize_reliability(rows),'musique_metrics':musique_metrics(gold,predictions),
        'evidence_coverage_macro':averaged,
        'coverage_defined_cases':{phase:{metric:sum(d['coverage'][phase][metric] is not None for d in diagnostics)
            for metric in ('positive_document_recall','weighted_aspect_coverage')} for phase in ('returned','delivered','submitted')},
        'cases':diagnostics,
        'audit':{'runs':len(rows),'source_spans_verified':spans,'model_requests_verified':model_requests,'final_references_verified':references},
        'answer_quality_bright':'No independent semantic judgments; coverage is not answer correctness.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();root=args.run.resolve()
    cases=json.loads((root/'cases.json').read_text())
    result={'schema':'arkb-phase-a-comparison-v1','arms':{arm:inspect_arm(root,arm,cases) for arm in ('old','new')},
        'release_eligible':False,'scope':'Eight fixed cases per arm; protocol reliability only, no score tuning or benchmark claim.'}
    for arm in ('old','new'):
        metadata=json.loads((root/arm/'experiment.json').read_text())
        if metadata['status']!='completed':raise ValueError('Incomplete run.')
        prior=root/arm/'attempt-1-experiment.json'
        if prior.exists():
            first=json.loads(prior.read_text())
            if any(metadata[k]!=first[k] for k in ('environment','models_before','model_definition','template_sha256')):
                raise ValueError('Resume changed the measured environment.')
        if arm=='old':before=metadata
        elif any(metadata[k]!=before[k] for k in ('environment','models_before','models_after','model_definition','template_sha256')):
            raise ValueError('Measured service/model environment differs between arms.')
    (root/'summary.json').write_text(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({arm:result['arms'][arm]['reliability'] for arm in ('old','new')},indent=2))


if __name__=='__main__':main()

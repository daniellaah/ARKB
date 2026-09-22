"""Diagnostic evidence coverage for public Agent runs; never answer-quality gold."""
from copy import deepcopy


def evidence_hits(raw, name, references=None):
    """Read either archived v1 evidence or v2 evidence with its diagnostic map."""
    if raw is None or raw.get('status', 'success') != 'success':
        return []
    hits = [raw.get('result')] if name == 'read' else raw.get('results', [])
    result = []
    for hit in hits:
        if not isinstance(hit, dict) or not isinstance(hit.get('content'), str):
            raise ValueError('Malformed retained evidence.')
        if 'ref' in hit:
            if references is None or hit['ref'] not in references:
                raise ValueError('Missing evidence reference mapping.')
            internal = references[hit['ref']]
            if any(hit[k] != internal[k] for k in ('source', 'title', 'content')):
                raise ValueError('Evidence reference mapping disagrees with observation.')
            hit = {**deepcopy(internal), **hit}
        result.append(hit)
    return result


def evidence_packet(result, source_map):
    """Separate returned, delivered and model-submitted nonempty body evidence.

    Submitted means present in a subsequent attempted provider request. A
    provider failure may leave actual processing unconfirmed; it is not a
    measure of comprehension or successful use in a final answer.
    """
    observation=result.get('observation',{}) if result else {}
    groups={key:[] for key in ('returned','delivered','submitted')}
    excerpts=[]
    for event in observation.get('tools',[]):
        delivered=event.get('delivered_to_conversation',False)
        submitted=event.get('submitted_to_model',False)
        expected=delivered and any(m['turn']>event['turn'] for m in observation.get('models',[]))
        if submitted!=expected:raise ValueError('Evidence submission flags disagree with model requests.')
        raw=event.get('raw_result')
        if raw is None:continue
        hits=evidence_hits(raw, event['name'], observation.get('evidence_references'))
        for hit in hits:
            if not isinstance(hit,dict) or not isinstance(hit.get('content'),str):
                raise ValueError('Malformed retained evidence.')
            if hit['source'] not in source_map:raise ValueError('Evidence outside the registered corpus.')
            if not hit['content'].strip():continue
            document=source_map[hit['source']]
            for key,include in (('returned',True),('delivered',delivered),('submitted',submitted)):
                if include and document not in groups[key]:groups[key].append(document)
            if submitted:
                excerpts.append({'tool_index':event['index'],'turn':event['turn'],
                                 'corpus_id':document,'evidence':deepcopy(hit)})
    return {**groups,'submitted_excerpts':excerpts}


def coverage(qrels, aspects, evidence):
    positives={d for d,r in qrels.items() if r>0}
    total=sum(a['weight'] for a in aspects)
    result={}
    for phase in ('returned','delivered','submitted'):
        seen=set(evidence[phase])
        result[phase]={'unique_documents':len(seen),
            'positive_document_recall':len(positives & seen)/len(positives) if positives else None,
            'weighted_aspect_coverage':sum(a['weight'] for a in aspects if seen.intersection(a['supporting_docs']))/total if total else None}
    return result


def operation_cost(row):
    report=(row.get('result') or {}).get('observation',{})
    events=report.get('tools',[])
    return {'elapsed_ms':row['elapsed_ms'],'model_requests':len(report.get('models',[])),
        'requested_tools':len(events),'executed_tools':sum(e['executed'] for e in events),
        'budget_stop_reason':report.get('budget_stop_reason'),
        'evidence':report.get('evidence'), 'provider_usage':report.get('usage'),
        'observed':bool(report)}

"""Preregistered source-fusion experiments over existing retrieval contracts.

No queries, labels, model calls or benchmark identities influence the policy.
Not connected to production until the development/validation adoption gates pass.
"""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
import math

from arkb.retrieval.models import SearchResponse, validate_request


def fuse_sources(legs: Mapping[str, SearchResponse], *, policy: str,
                 top_k: int = 100, filters=None) -> SearchResponse:
    """Fuse pinned ranked chunks to unique sources with explicit score attribution.

    k=60 is fixed. Ties follow first occurrence in sorted leg-name/input order.
    Filters are applied before aggregation; normal callers filter upstream too.
    """
    if policy not in ('C1','C2'):
        raise ValueError('Unknown source fusion policy.')
    if not legs or any(not isinstance(n,str) or not n.strip() for n in legs):
        raise ValueError('Source fusion requires named retrieval legs.')
    first=legs[sorted(legs)[0]]
    filters=validate_request(first.query,top_k,filters)
    if first.index_id is None:
        raise ValueError('Source fusion requires a pinned snapshot.')
    source_order={}; originals={}; sources={}; addresses={}; evidence={}; leg_votes=defaultdict(list)
    for name in sorted(legs):
        response=legs[name]
        if (response.query,response.index_id)!=(first.query,first.index_id):
            raise ValueError('Source fusion requires the same query and snapshot.')
        grouped={}; seen=set()
        for rank,h in enumerate(response.results,1):
            if h.metadata.get('index_version')!=first.index_id:
                raise ValueError('Contributing chunk does not belong to the pinned snapshot.')
            description=(h.source,h.metadata.get('document_revision'),h.metadata.get('vault_id'))
            if sources.setdefault(h.source_id,description)!=description:
                raise ValueError('Conflicting source revision or address.')
            if addresses.setdefault(h.source,h.source_id)!=h.source_id:
                raise ValueError('Source address has conflicting identities.')
            signature=(h.source_id,h.source,h.content,h.start_char,h.end_char,
                       {k:h.metadata.get(k) for k in ('title','section_id','section_start_char',
                        'section_end_char','heading_path','occurrence','chunk_index')})
            if evidence.setdefault(h.identity,signature)!=signature:
                raise ValueError('Conflicting chunk evidence.')
            if 'source' in filters and h.source!=filters['source']:
                continue
            if h.identity in seen:
                continue
            seen.add(h.identity); originals.setdefault(h.identity,h)
            source_order.setdefault(h.source_id,len(source_order))
            chunks=grouped.setdefault(h.source_id,[])
            if len(chunks)<(1 if policy=='C1' else 2):
                chunks.append((rank,h))
        aggregates={source:math.fsum(1/(60+rank) for rank,h in chunks) for source,chunks in grouped.items()}
        order=sorted(grouped,key=lambda source:-aggregates[source])
        for source_rank,source in enumerate(order,1):
            chunks=grouped[source]
            vote=1/(60+source_rank)
            leg_votes[source].append({'leg':name,'source_rank':source_rank,'vote':vote,
                'aggregation_score':aggregates[source], 'chunks':[{
                    'chunk_id':h.chunk_id,'identity':list(h.identity),'chunk_rank':rank,
                    'raw_score':h.score,'score_type':h.score_type,
                    'contribution':vote if len(chunks)==1 else vote*(1/(60+rank))/aggregates[source]} for rank,h in chunks]})
    scores={source:math.fsum(v['vote'] for v in votes) for source,votes in leg_votes.items()}
    ranked=sorted(leg_votes,key=lambda source:(-scores[source],source_order[source])); output=[]
    for source_rank,source in enumerate(ranked[:top_k],1):
        contributions=defaultdict(list)
        for vote in leg_votes[source]:
            for chunk in vote['chunks']:
                contributions[tuple(chunk['identity'])].append(chunk['contribution'])
        representative=max(contributions,key=lambda key:math.fsum(contributions[key]))
        original=originals[representative]
        output.append(replace(original,method='hybrid',score=scores[source],score_type='source_rrf',
            metadata={**original.metadata,'source_fusion':{'policy':policy,'rrf_k':60,
                'source_rank':source_rank,'representative':list(representative),
                'representative_policy':'R0-proportional-vote','legs':leg_votes[source]}}))
    return SearchResponse(query=first.query,method='hybrid',results=tuple(output),index_id=first.index_id)

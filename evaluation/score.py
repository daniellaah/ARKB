"""Offline scoring of saved run records; labels are read here and nowhere else.

Every function takes the record that run.py wrote (scenario, result with the
observation trace and canonical final, error) and the labels for that
scenario. Nothing here influences inference.
"""
from collections import Counter, defaultdict
import re
import string

import numpy as np

from arkb.agent.state import AgentFinal

from .common import ROOT, read_json
from .v2 import evidence_scores, load_dataset

PRIMARY = {'v2': 'evidence_coverage_delivered', 'exact-v2': 'completeness_cited', 'exact-nfcorpus': 'completeness_cited',
           'recall-nfcorpus': 'positive_recall_delivered', 'recall-fiqa': 'positive_recall_delivered',
           'musique': 'answer_f1_first_line', 'long-browsecomp': 'positive_recall_delivered'}
METRICS = ('evidence_coverage_delivered', 'critical_covered_delivered', 'source_recall_cited',
           'completeness_cited', 'spurious_cited', 'complete_and_exact_cited',
           'positive_recall_delivered', 'gold_recall_delivered',
           'answerability_correct', 'answer_f1_first_line', 'answer_em_first_line', 'support_f1')


# --- records ---------------------------------------------------------------

def canonical(row):
    value = (row.get('result') or {}).get('final')
    return AgentFinal(**value) if value else None


def cited_sources(row):
    final = (row.get('result') or {}).get('final') or {}
    return sorted({c['source'] for c in final.get('citations') or [] if isinstance(c, dict) and c.get('source')})


def executed_tools(row):
    report = (row.get('result') or {}).get('observation') or {}
    return [t for t in report.get('tools') or [] if t.get('executed')]


def _hits(event):
    raw = event.get('raw_result') or {}
    hits = [raw['result']] if 'result' in raw else raw.get('results', [])
    return [h for h in hits if isinstance(h, dict)]


def delivered_observations(report):
    """Evidence that entered the conversation, as bound references; the shape the v2 span scorer accepts."""
    refs = report.get('evidence_references') or {}
    observations = []
    for event in report.get('tools') or []:
        if not event.get('delivered_to_conversation'):
            continue
        hits = _hits(event)
        allowed = set(event.get('delivered_refs', [h.get('ref') for h in hits]))
        for hit in hits:
            bound = refs.get(hit.get('ref'))
            if bound and hit['ref'] in allowed:
                observations.append({k: bound.get(k) for k in ('source', 'document_revision', 'start_char', 'end_char', 'content')})
    return observations


def delivered_sources(report):
    """Distinct sources delivered to the conversation; blank bodies do not count."""
    refs = report.get('evidence_references') or {}
    sources = set()
    for event in report.get('tools') or []:
        if not event.get('delivered_to_conversation'):
            continue
        hits = _hits(event)
        allowed = set(event.get('delivered_refs', [h['ref'] for h in hits]))
        for hit in hits:
            bound = refs.get(hit['ref'])
            if bound and hit['ref'] in allowed and bound['content'].strip():
                sources.add(bound['source'])
    return sorted(sources)


def match_limit_hits(row):
    hits = 0
    for event in executed_tools(row):
        if event['name'] == 'match' and event.get('status') == 'success':
            limit = (event.get('arguments') or {}).get('limit', 5)
            if len((event.get('raw_result') or {}).get('results', [])) >= limit:
                hits += 1
    return hits


def costs(row):
    report = (row.get('result') or {}).get('observation') or {}
    tools = executed_tools(row)
    usage = report.get('usage') or {}
    return {'elapsed_ms': row['elapsed_ms'], 'model_requests': len(report.get('models') or []),
            'tool_calls': len(tools), 'tools_executed': dict(Counter(t['name'] for t in tools)),
            'search_modes': [(t['arguments'] or {}).get('mode') or 'default' for t in tools if t['name'] == 'search'],
            'responses_cut': sum(1 for m in report.get('models') or [] if ((m.get('response') or {}).get('done_reason') == 'length')),
            'prompt_tokens': (usage.get('prompt_eval_count') or {}).get('known_total'),
            'eval_tokens': (usage.get('eval_count') or {}).get('known_total'),
            'delivered_evidence_tokens': (report.get('evidence') or {}).get('delivered_tokens'),
            'budget_stop_reason': report.get('budget_stop_reason')}


# --- MuSiQue ---------------------------------------------------------------

def normalize_answer(text):
    no_punctuation = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', no_punctuation).split())


def answer_f1(prediction, aliases):
    p = Counter(normalize_answer(prediction).split())
    values = []
    for alias in aliases:
        g = Counter(normalize_answer(alias).split())
        denominator = sum(p.values()) + sum(g.values())
        values.append(2 * sum((p & g).values()) / denominator if denominator else 1.0)
    return max(values)


def support_f1(predicted, gold):
    p, g = set(predicted), set(gold)
    return 2 * len(p & g) / (len(p) + len(g)) if p or g else 1.0


def first_line(text):
    """The short answer the product asks for: the first nonblank line."""
    return next((line.strip() for line in (text or '').splitlines() if line.strip()), '')


def musique_scores(row, gold):
    """Canonical final against gold; an error final scores zero, an unanswerable pair has no answer scores."""
    final = canonical(row)
    valid = final is not None and final.status != 'error'
    if valid:
        sources = list(dict.fromkeys(c['source'] for c in final.citations))
        if any(s not in gold['source_map'] for s in sources):
            raise ValueError('Citation is outside the prepared context.')
        answerable, support = final.status == 'answered', [gold['source_map'][s] for s in sources]
    else:
        answerable, support = None, []
    aliases = [gold['answer'], *gold['answer_aliases']]
    line = first_line(final.answer) if valid else ''
    scored = {'answerability_correct': float(answerable is gold['answerable']), 'predicted_answerable': answerable,
              'answer_f1_first_line': None, 'answer_em_first_line': None, 'support_f1': None}
    if gold['answerable']:
        scored['answer_f1_first_line'] = answer_f1(line, aliases) if valid else 0.0
        scored['answer_em_first_line'] = float(any(normalize_answer(line) == normalize_answer(a) for a in aliases)) if valid else 0.0
        scored['support_f1'] = support_f1(support, gold['support_idxs']) if valid else 0.0
    return scored


# --- per-scenario scoring --------------------------------------------------

def load_resources(devset):
    """Datasets and source maps the scorers need; source maps for absent optional slices are simply missing."""
    devset = devset if devset.is_absolute() else ROOT / devset
    resources = {'v2': load_dataset(devset / 'v2-pilot', allow_provisional=True), 'source_maps': {}}
    for path in sorted((devset / 'scoring').glob('*.json')) if (devset / 'scoring').exists() else []:
        resources['source_maps'][path.stem] = read_json(path)['source_map']
    return resources


def score_row(row, scenario, labels, resources):
    slice_name = scenario['slice']
    report = (row.get('result') or {}).get('observation') or {}
    final = canonical(row)
    status = final.status if final else 'error'
    delivered = delivered_sources(report)
    cited = cited_sources(row)
    tools = executed_tools(row)
    scored = {'final_status': status, 'stop_reason': (row.get('result') or {}).get('stop_reason', 'error'),
              'error': row.get('error'), 'costs': costs(row), 'tool_names': [t['name'] for t in tools],
              'delivered_sources': delivered, 'cited_sources': cited}
    if slice_name == 'v2':
        dataset = resources['v2']
        case = dataset.case(scenario['id'])
        expected = dataset.expected_sources(case['id'])
        e = evidence_scores(case, delivered_observations(report), dataset)
        scored['evidence_coverage_delivered'] = e['evidence_coverage']
        scored['critical_covered_delivered'] = e['all_critical_facets_covered']
        scored['expected_sources'] = sorted(expected)
        scored['source_recall_cited'] = len(expected & set(cited)) / len(expected) if expected else None
        task = scenario['task_type']
        scored['behavior'] = {
            'no_retrieval_respected': not tools if task == 'no_retrieval' else None,
            'read_only_respected': all(t['name'] in ('read', 'finish') for t in tools) if task == 'direct_read' else None,
            'abstained': status == 'insufficient_evidence' if task == 'evidence_gap' else None,
            'answered': status in ('answered', 'partial')}
    elif slice_name.startswith('exact-'):
        expected = set(labels['expected_sources'])
        scored['completeness_cited'] = len(expected & set(cited)) / len(expected) if expected else None
        scored['completeness_delivered'] = len(expected & set(delivered)) / len(expected) if expected else None
        scored['spurious_cited'] = len(set(cited) - expected)
        scored['complete_and_exact_cited'] = set(cited) == expected
        scored['match_calls'] = sum(t['name'] == 'match' for t in tools)
        scored['match_limit_hits'] = match_limit_hits(row)
    elif slice_name.startswith('recall-') or slice_name == 'long-browsecomp':
        source_map = resources['source_maps'][scenario['scope']]
        if any(s not in source_map for s in delivered):
            raise ValueError('Delivered source is outside the frozen corpus mapping.')
        observed = {str(source_map[s]) for s in delivered}
        for name in ('qrels', 'gold_qrels'):
            if name in labels:
                positive = {str(k) for k, v in labels[name].items() if v > 0}
                key = 'positive_recall_delivered' if name == 'qrels' else 'gold_recall_delivered'
                scored[key] = len(observed & positive) / len(positive) if positive else None
        scored['answered'] = status in ('answered', 'partial')
    elif slice_name == 'musique':
        scored.update(musique_scores(row, labels['gold']))
    return scored


# --- summaries -------------------------------------------------------------

def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(records):
    by_slice = defaultdict(list)
    for r in records:
        by_slice[r['scenario']['slice']].append(r)
    summary = {}
    for name, rows in sorted(by_slice.items()):
        s = [r['scores'] for r in rows]
        elapsed = [x['costs']['elapsed_ms'] / 1000 for x in s]
        block = {'scenarios': len(rows), 'errors': sum(x['final_status'] == 'error' for x in s),
                 'final_statuses': dict(Counter(x['final_status'] for x in s)),
                 'stop_reasons': dict(Counter(x['stop_reason'] for x in s)),
                 'elapsed_seconds_mean': mean(elapsed), 'elapsed_seconds_p95': float(np.quantile(elapsed, .95)),
                 'model_requests_mean': mean([x['costs']['model_requests'] for x in s]),
                 'tool_calls_mean': mean([x['costs']['tool_calls'] for x in s]),
                 'tools_executed': dict(sum((Counter(x['costs']['tools_executed']) for x in s), Counter())),
                 'responses_cut': sum(x['costs']['responses_cut'] for x in s),
                 'prompt_tokens_mean': mean([x['costs']['prompt_tokens'] for x in s]),
                 'eval_tokens_mean': mean([x['costs']['eval_tokens'] for x in s]),
                 'delivered_evidence_tokens_mean': mean([x['costs']['delivered_evidence_tokens'] for x in s])}
        for metric in METRICS:
            if any(metric in x for x in s):
                block[metric] = mean([x.get(metric) for x in s])
        if name == 'v2':
            by_task = defaultdict(list)
            for r in rows:
                by_task[r['scenario']['task_type']].append(r['scores'])
            block['by_task_type'] = {task: {'scenarios': len(v), 'evidence_coverage_delivered': mean([x['evidence_coverage_delivered'] for x in v]),
                                            **{k: mean([x['behavior'][k] for x in v]) for k in ('answered', 'no_retrieval_respected', 'read_only_respected', 'abstained')}}
                                     for task, v in sorted(by_task.items())}
        if name.startswith('exact-'):
            block['match_limit_hits'] = sum(x['match_limit_hits'] for x in s)
        summary[name] = block
    return summary


def markdown(summary, meta):
    lines = [f"# Run `{meta['label']}`", '',
             f"model={meta['model']} think={meta['think']} git={meta['git_head'][:10]}{'+dirty' if meta['dirty'] else ''} "
             f"scenarios={meta['scenarios']} wall={meta.get('wall_seconds', 0):.0f}s", '',
             '| slice | n | primary | value | errors | cut | elapsed s | requests | tool calls |',
             '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, block in summary.items():
        primary = PRIMARY.get(name, '')
        value = block.get(primary)
        lines.append(f"| {name} | {block['scenarios']} | {primary} | {value if value is None else f'{value:.3f}'} | {block['errors']} | "
                     f"{block['responses_cut']} | {block['elapsed_seconds_mean']:.1f} | {block['model_requests_mean']:.2f} | {block['tool_calls_mean']:.2f} |")
    for name, block in summary.items():
        lines += ['', f'## {name}', '']
        for key, value in block.items():
            if key == 'by_task_type':
                lines.append('- by task type:')
                for task, inner in value.items():
                    lines.append(f'  - {task}: ' + ', '.join(f"{k}={v if not isinstance(v, float) else round(v, 3)}" for k, v in inner.items()))
            elif isinstance(value, float):
                lines.append(f'- {key}: {value:.3f}')
            else:
                lines.append(f'- {key}: {value}')
    return '\n'.join(lines) + '\n'

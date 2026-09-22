"""The v2 pilot dataset: validated evidence spans and the span-coverage scorer.

Coordinates reference Note.content, not raw Markdown. Provisional annotations
load only by explicit opt-in and never imply independent human review.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

from arkb.knowledge.documents import load_notes

from .common import digest_text, read_json, read_jsonl

SCHEMA = 'arkb-eval-v2'
TASKS = {'semantic_discovery', 'exploratory_retrieval', 'knowledge_qa', 'multi_hop_qa',
         'exact_lookup', 'direct_read', 'evidence_gap', 'no_retrieval'}
ANSWERABILITY = {'answerable', 'partial', 'unanswerable', 'needs_clarification', 'not_applicable'}


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _fields(row, required, optional=()):
    if not isinstance(row, dict) or set(row) - set(required) - set(optional) or set(required) - set(row):
        raise ValueError(f'Invalid fields; expected {sorted(required)}, optional {sorted(optional)}.')


def _strings(values, *, nonempty=False):
    return (isinstance(values, list) and (bool(values) or not nonempty)
            and all(_text(v) for v in values) and len(values) == len(set(values)))


def _status(row):
    if row['annotation_status'] not in ('provisional', 'reviewed') or not _strings(row['reviewers']):
        raise ValueError('Invalid annotation status/reviewers.')
    if row['annotation_status'] == 'reviewed' and not row['reviewers']:
        raise ValueError('Reviewed labels require identified reviewers.')


@dataclass(frozen=True)
class Dataset:
    manifest: dict
    cases: tuple[dict, ...]
    evidence: dict[str, dict]
    qrels: tuple[dict, ...]
    bodies: dict[str, str]

    @property
    def provisional(self):
        return any(row['annotation_status'] != 'reviewed' for row in (*self.cases, *self.evidence.values(), *self.qrels))

    def case(self, case_id):
        return next(c for c in self.cases if c['id'] == case_id)

    def expected_sources(self, case_id, *, grade=2):
        return {r['source'] for r in self.qrels if r['query_id'] == case_id and r['grade'] >= grade}


def load_dataset(directory, *, allow_provisional=False):
    """Load and validate the dataset directory; the corpus is its `corpus/` subdirectory."""
    directory = Path(directory)
    notes_dir = directory / 'corpus'
    manifest = read_json(directory / 'manifest.json')
    _fields(manifest, {'schema_version', 'corpus_id', 'files', 'corpus', 'description'})
    if manifest['schema_version'] != SCHEMA or not _text(manifest['corpus_id']):
        raise ValueError('Unsupported dataset schema/corpus.')
    if set(manifest['files']) != {'queries.jsonl', 'evidence.jsonl', 'qrels.jsonl'}:
        raise ValueError('Manifest must hash all three data files.')
    for name, expected in manifest['files'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Dataset checksum mismatch: {name}.')
    notes = {note.source: note for note in load_notes(notes_dir)}
    recorded = set()
    for doc in manifest['corpus']:
        _fields(doc, {'source', 'document_revision', 'body_sha256', 'origin'})
        source = doc['source']
        if not _text(source) or source in recorded or source not in notes:
            raise ValueError('Invalid, duplicate, or external corpus source.')
        recorded.add(source)
        note = notes[source]
        if note.document_revision != doc['document_revision'] or digest_text(note.content) != doc['body_sha256']:
            raise ValueError(f'Corpus revision mismatch: {source}.')
    if recorded != set(notes):
        raise ValueError('Corpus file set differs from manifest.')
    evidence = {}
    for span in read_jsonl(directory / 'evidence.jsonl'):
        _fields(span, {'id', 'source', 'document_revision', 'start_char', 'end_char', 'quote', 'quote_sha256',
                       'annotation_status', 'reviewers'})
        _status(span)
        if not _text(span['id']) or span['id'] in evidence or span['source'] not in notes:
            raise ValueError('Invalid/duplicate evidence ID or source.')
        note = notes[span['source']]
        start, end = span['start_char'], span['end_char']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(note.content):
            raise ValueError('Invalid evidence body coordinates.')
        if (span['document_revision'] != note.document_revision or note.content[start:end] != span['quote']
                or digest_text(span['quote']) != span['quote_sha256']):
            raise ValueError('Evidence revision/quote mismatch.')
        evidence[span['id']] = span
    cases, ids = [], set()
    for case in read_jsonl(directory / 'queries.jsonl'):
        _fields(case, {'id', 'intent_family_id', 'task_type', 'query', 'query_language', 'split', 'answerability',
                       'query_origin', 'annotation_status', 'reviewers', 'evidence_requirements', 'required_facts',
                       'expected_behavior'}, {'exact_pattern', 'read_source'})
        _status(case)
        if case['id'] in ids or case['task_type'] not in TASKS or case['answerability'] not in ANSWERABILITY:
            raise ValueError('Invalid/duplicate case identity, task or answerability.')
        if not _strings(case['required_facts']) or not isinstance(case['evidence_requirements'], list):
            raise ValueError('Invalid required facts or evidence requirements.')
        facet_ids = set()
        for facet in case['evidence_requirements']:
            _fields(facet, {'id', 'weight', 'critical', 'alternatives'})
            if (not _text(facet['id']) or facet['id'] in facet_ids or type(facet['critical']) is not bool
                    or type(facet['weight']) not in (int, float) or not math.isfinite(facet['weight']) or facet['weight'] <= 0):
                raise ValueError('Invalid facet identity/weight/critical flag.')
            facet_ids.add(facet['id'])
            if not facet['alternatives'] or any(not _strings(a, nonempty=True) or set(a) - evidence.keys()
                                               for a in facet['alternatives']):
                raise ValueError('Invalid/unknown evidence in alternative.')
        if case['task_type'] == 'no_retrieval' and case['evidence_requirements']:
            raise ValueError('no_retrieval must not require knowledge evidence.')
        ids.add(case['id'])
        cases.append(case)
    if not cases:
        raise ValueError('Dataset must not be empty.')
    qrels, pairs = [], set()
    for rel in read_jsonl(directory / 'qrels.jsonl'):
        _fields(rel, {'query_id', 'source', 'grade', 'annotation_status', 'reviewers'})
        _status(rel)
        key = (rel['query_id'], rel['source'])
        if key in pairs or key[0] not in ids or key[1] not in notes or type(rel['grade']) is not int or not 0 <= rel['grade'] <= 3:
            raise ValueError('Invalid/duplicate relevance judgment.')
        pairs.add(key)
        qrels.append(rel)
    dataset = Dataset(manifest, tuple(cases), evidence, tuple(qrels), {s: n.content for s, n in notes.items()})
    if dataset.provisional and not allow_provisional:
        raise ValueError('Provisional dataset requires explicit allow_provisional=True; not a release gold set.')
    return dataset


def observation_intervals(observations, dataset):
    """Validate observed verbatim bodies/ranges and keep per-source intervals; anything else is rejected."""
    intervals, rejected = {}, 0
    revisions = {s['source']: s['document_revision'] for s in dataset.manifest['corpus']}
    for hit in observations:
        if not isinstance(hit, dict):
            rejected += 1
            continue
        source, start, end = hit.get('source'), hit.get('start_char'), hit.get('end_char')
        # A whole-body read may come without a range; derive it only on exact equality.
        if isinstance(source, str) and source in dataset.bodies and start is None and end is None \
                and hit.get('content') == dataset.bodies[source]:
            start, end = 0, len(dataset.bodies[source])
        if (not isinstance(source, str) or source not in dataset.bodies or hit.get('document_revision') != revisions[source]
                or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(dataset.bodies[source])
                or hit.get('content') != dataset.bodies[source][start:end]):
            rejected += 1
            continue
        intervals.setdefault(source, []).append((start, end))
    return intervals, rejected


def evidence_scores(case, observations, dataset):
    """Weighted facet coverage from exact observed bodies at the labeled revision."""
    intervals, rejected = observation_intervals(observations, dataset)
    fractions = {}
    for facet in case['evidence_requirements']:
        for alternative in facet['alternatives']:
            for evidence_id in alternative:
                span = dataset.evidence[evidence_id]
                start, end = span['start_char'], span['end_char']
                cursor, covered = start, 0
                for left, right in sorted(intervals.get(span['source'], [])):
                    left, right = max(start, left), min(end, right)
                    if left < right:
                        covered += max(0, right - max(cursor, left))
                        cursor = max(cursor, right)
                fractions[evidence_id] = covered / (end - start)
    satisfied = {f['id']: any(all(fractions[e] == 1 for e in alt) for alt in f['alternatives'])
                 for f in case['evidence_requirements']}
    weight = sum(f['weight'] for f in case['evidence_requirements'])
    critical = [satisfied[f['id']] for f in case['evidence_requirements'] if f['critical']]
    return {'evidence_coverage': sum(f['weight'] * satisfied[f['id']] for f in case['evidence_requirements']) / weight if weight else None,
            'all_critical_facets_covered': all(critical) if critical else None,
            'facet_satisfied': satisfied, 'span_coverage': fractions, 'rejected_observations': rejected}

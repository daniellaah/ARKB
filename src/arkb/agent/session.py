"""One run's validated tool boundary and opaque evidence registry.

Capabilities retain their low-level Python API. Only this boundary accepts
model arguments; internal IDs and body coordinates are never model selectors.
"""
from copy import deepcopy
import json
from uuid import uuid4

from arkb.retrieval.exact import ExactPatternError, ExactTimeout, ExactCancelled
from arkb.knowledge.documents import DocumentNotFound
from arkb.knowledge.models import is_canonical_source
from arkb.agent.tools import DEFAULT_LIST_LIMIT, DEFAULT_SEARCH_LIMIT


class ToolInputError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def error_result(code, message, *, fatal=False):
    return {'status': 'fatal_error' if fatal else 'recoverable_error',
            'error': {'code': code, 'message': message}}


def validate_arguments(arguments, schema):
    """Validate the deliberately small tool-schema vocabulary without coercion."""
    if not isinstance(arguments, dict):
        raise ToolInputError('invalid_arguments', 'Arguments must be a JSON object.')
    properties = schema['properties']
    if set(arguments) - set(properties) or set(schema.get('required', [])) - set(arguments):
        raise ToolInputError('invalid_arguments', 'Use only the defined fields and supply all required fields.')
    for key, value in arguments.items():
        spec = properties[key]
        types = spec['type'] if isinstance(spec['type'], list) else [spec['type']]
        actual = {str: 'string', int: 'integer', bool: 'boolean', list: 'array', type(None): 'null'}.get(type(value))
        if actual not in types or ('enum' in spec and value not in spec['enum']):
            raise ToolInputError('invalid_arguments', f'Invalid {key}; follow its advertised type and allowed values.')
        if isinstance(value, str) and (not value.strip() or '\x00' in value):
            raise ToolInputError('invalid_arguments', f'{key} must be nonblank text without NUL characters.')
        if type(value) is int and value < spec.get('minimum', value):
            raise ToolInputError('invalid_arguments', f'{key} is below its minimum.')
        if actual == 'array' and any(not isinstance(v, str) or not v.strip() for v in value):
            raise ToolInputError('invalid_arguments', f'{key} must contain reference strings.')
    source = arguments.get('source')
    if source is not None and not is_canonical_source(source):
        raise ToolInputError('invalid_arguments', 'source must be a vault-relative path such as '
                             'area/note.md, with no leading slash and no ".." segment.')


SECTION_WINDOW_CHARS = 3000


def section_bounds(bound, length):
    """Bounded expansion around bound evidence: its heading section, else a window."""
    start, end = bound.get('section_start_char'), bound.get('section_end_char')
    if type(start) is int and type(end) is int and 0 <= start < end <= length:
        return start, end
    span_start, span_end = bound.get('start_char'), bound.get('end_char')
    if type(span_start) is not int or type(span_end) is not int:
        return 0, length
    return max(0, span_start - SECTION_WINDOW_CHARS), min(length, span_end + SECTION_WINDOW_CHARS)


class ToolSession:
    def __init__(self, tools):
        self.tools = tools
        self.references = {}
        self.resolutions = []
        self._keys = {}
        self._delivered = set()
        self.retrieval_attempted = False
        self._prefix = 'ev_' + uuid4().hex + '_'
        self.definitions = tools.tool_definitions()
        self._schemas = {d['name']: d['parameters'] for d in self.definitions}

    def _present(self, evidence, index_id=None):
        internal = deepcopy(evidence)
        if index_id is not None:
            internal['index_version'] = index_id
        key = json.dumps(internal, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if key not in self._keys:
            ref = self._prefix + str(len(self.references) + 1)
            self._keys[key] = ref
            self.references[ref] = internal
        return {'ref': self._keys[key], **{k: evidence[k] for k in ('source', 'title', 'content')}}

    def deliver(self, result):
        hits = [result['result']] if 'result' in result else result.get('results', [])
        self._delivered.update(hit['ref'] for hit in hits)

    def _resolve(self, ref):
        record = {'ref': ref, 'status': 'requested'}
        self.resolutions.append(record)
        try:
            if ref not in self.references:
                raise ToolInputError('invalid_reference', self._unknown_reference(ref))
            if ref not in self._delivered:
                raise ToolInputError('undelivered_reference', 'This evidence was not delivered within the budget.')
            bound = self.references[ref]
            record['evidence'] = deepcopy(bound)
            try:
                current = self.tools.read(bound['document_id'], source=bound['source'])['result']
            except DocumentNotFound as error:
                raise ToolInputError('source_unavailable', 'Source was deleted, renamed or is outside the live scope. Search again.') from error
            if not bound['document_revision'] or current['document_revision'] != bound['document_revision']:
                raise ToolInputError('stale_reference', 'Source revision changed. Search again or read the known source filename.')
            start, end = bound['start_char'], bound['end_char']
            if start is not None and current['content'][start:end] != bound['content']:
                raise RuntimeError('Evidence coordinates disagree with their bound document revision.')
            record['status'] = 'success'
            return bound, current
        except ToolInputError as error:
            record.update(status='recoverable_error', error={'code': error.code, 'message': str(error)})
            raise

    def _unknown_reference(self, ref):
        """Say what was wrong, not only that something was.

        Six of the seven citation failures measured over the real vault were
        one mistake: the model put the note's source path where a reference
        belongs. The old message said the reference did not exist, which is
        true and unusable -- one run proposed the same rejected finish five
        times. When the string names a source whose evidence was delivered,
        the refs for it are named here, because they are already in the
        conversation and repeating them is what lets the next turn recover.

        Neither message carries a specimen reference. One that did was copied
        verbatim into a final answer by 38 of 180 runs of a smaller model,
        which called no tool at all.
        """
        delivered = [r for r in self._delivered
                     if self.references.get(r, {}).get('source') == ref]
        if delivered:
            shown = ', '.join(sorted(delivered)[:5])
            return (f'{ref} is a source path, not a reference. Evidence from that note was delivered as: '
                    f'{shown}. Cite those.')
        return ('No such reference in this run. Cite a ref returned beside a piece of evidence by a tool '
                'in this run; a source path, a title, or an invented identifier is not a reference.')

    def _read(self, *, ref=None, source=None, expand='section'):
        if (ref is None) == (source is None):
            raise ToolInputError('invalid_arguments', 'Supply exactly one of ref or source.')
        if ref is not None:
            bound, current = self._resolve(ref)
            if expand == 'snippet':
                return self._present(bound)
            if expand == 'document':
                # A document expansion is taken from the single revision-checked read.
                return self._present(current)
            start, end = section_bounds(bound, len(current['content']))
            if (start, end) == (0, len(current['content'])):
                return self._present(current)
            # The bounded section is a verbatim slice of the same revision-checked read.
            return self._present({**current, 'content': current['content'][start:end],
                                  'start_char': start, 'end_char': end, 'chunk_id': None,
                                  'section_id': bound.get('section_id')})
        if expand not in ('document', 'section'):
            raise ToolInputError('invalid_arguments', 'Reading a known source returns the whole document; use snippet only with a ref.')
        try:
            current = self.tools.read(source=source)['result']
        except DocumentNotFound as error:
            raise ToolInputError('source_unavailable', 'Source does not exist in this live knowledge base.') from error
        return self._present(current)

    def corpus(self, sources):
        """Present whole notes as evidence without a model-selected tool call.

        The loop's small-scope bypass hands the scope over instead of searching
        it. Each note is presented exactly as read(source=...) would present it,
        so its reference binds the same source and revision, resolves the same
        way, and is accepted by finish. A note that disappeared between the
        scan and the read is left out rather than failing the delivery.
        """
        self.retrieval_attempted = True
        results = []
        for source in sources:
            try:
                results.append(self._read(source=source, expand='document'))
            except ToolInputError:
                continue
        return {'status': 'success', 'results': results}

    def _finish(self, arguments):
        refs = arguments['evidence_refs']
        if len(set(refs)) != len(refs):
            raise ToolInputError('invalid_citation', 'Cite each evidence reference at most once.')
        if arguments['status'] in ('answered', 'partial') and self.retrieval_attempted and not refs:
            if not self.references:
                raise ToolInputError('invalid_citation', 'Nothing citable has been returned yet: listings are not evidence. '
                                     'Read or match the notes the answer relies on, then cite their refs.')
            raise ToolInputError('invalid_citation', 'Cite returned evidence supporting this answer, or report insufficient_evidence.')
        citations = []
        for ref in refs:
            bound, _ = self._resolve(ref)
            citations.append({'ref': ref, **deepcopy(bound)})
        return {**arguments, 'citations': citations}

    def invoke(self, name, arguments, *, exact_timeout=30):
        """Expected mistakes return observations; unexpected backend errors propagate."""
        try:
            if name in ('match', 'search', 'read', 'list', 'links'):
                self.retrieval_attempted = True
            if name not in self._schemas:
                raise ToolInputError('unknown_tool', 'Use one of the advertised tools, or finish.')
            validate_arguments(arguments, self._schemas[name])
            if name == 'list':
                return {'status': 'success', **self.tools.list(**{'limit': DEFAULT_LIST_LIMIT, **arguments})}
            if name == 'links':
                # Navigation, like list: it returns no evidence references, so
                # nothing here is citable and nothing is charged for it.
                try:
                    return {'status': 'success', **self.tools.links(**arguments)}
                except DocumentNotFound as error:
                    raise ToolInputError('source_unavailable', 'No note has this source path in the live '
                                         'knowledge base; list or search for its current path.') from error
            if name == 'read':
                return {'status': 'success', 'result': self._read(**arguments)}
            if name == 'finish':
                return {'status': 'success', 'final': self._finish(arguments)}
            if name == 'match':
                raw = self.tools.match(**arguments, timeout=exact_timeout)
            else:
                mode = arguments.get('mode') or self.tools._mode
                limit = arguments.get('limit', DEFAULT_SEARCH_LIMIT)
                engine = self.tools._engine
                if mode == 'hybrid' and limit > engine.candidate_k:
                    raise ToolInputError('invalid_arguments', f'Hybrid limit must be <= {engine.candidate_k}.')
                if self.tools._rerank and limit > engine.rerank_candidates:
                    raise ToolInputError('invalid_arguments', f'Reranked limit must be <= {engine.rerank_candidates}.')
                raw = self.tools.search(**arguments)
            observation = {'status': 'success', 'query': raw['query'],
                           'results': [self._present(hit, raw.get('index_id')) for hit in raw['results']]}
            if 'truncated' in raw:
                observation['truncated'] = raw['truncated']
            return observation
        except ToolInputError as error:
            return error_result(error.code, str(error))
        except ExactPatternError as error:
            return error_result('invalid_pattern', str(error))
        except ExactTimeout as error:
            return error_result('exact_timeout', str(error))
        except ExactCancelled as error:
            return error_result('exact_cancelled', str(error))

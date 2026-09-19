"""Low-level capability adapters and the schemas advertised by ToolSession."""

from copy import deepcopy
from typing import TypedDict

from arkb.knowledge.documents import DocumentAccess
from arkb.knowledge.models import ConfigValue
from arkb.retrieval.engine import RetrievalEngine
from arkb.retrieval.exact import ExactRetriever
from arkb.retrieval.models import SearchResponse, SearchResult, validate_request


class Evidence(TypedDict):
    document_id: str
    source: str
    title: str | None
    content: str
    document_revision: str | None
    chunk_id: str | None
    section_id: str | None
    start_char: int | None
    end_char: int | None
    section_start_char: int | None
    section_end_char: int | None


class QueryResult(TypedDict):
    query: str
    results: list[Evidence]


class ReadResult(TypedDict):
    result: Evidence


def _evidence(result: SearchResult) -> Evidence:
    def text(key: str) -> str | None:
        value = result.metadata.get(key)
        return value if isinstance(value, str) else None

    def offset(key: str) -> int | None:
        value = result.metadata.get(key)
        return value if type(value) is int else None

    return Evidence(document_id=result.source_id, source=result.source, title=text('title'),
                    content=result.content, document_revision=text('document_revision'),
                    chunk_id=result.chunk_id, section_id=text('section_id'),
                    start_char=result.start_char, end_char=result.end_char,
                    section_start_char=offset('section_start_char'), section_end_char=offset('section_end_char'))


def _query_result(response: SearchResponse) -> QueryResult:
    return {**QueryResult(query=response.query, results=[_evidence(hit) for hit in response.results]),
            **({'index_id': response.index_id} if response.index_id is not None else {}),
            **({'truncated': response.truncated} if response.truncated is not None else {})}


def tool_definitions(modes: tuple[str, ...], *, default_mode: str) -> tuple[dict[str, ConfigValue], ...]:
    """Render the effective tool schemas without constructing capability objects."""
    definitions = deepcopy(TOOL_DEFINITIONS)
    search = next(definition for definition in definitions if definition['name'] == 'search')
    search['parameters']['properties']['mode']['enum'] = [*modes, None]
    meanings = {'bm25': 'keyword ranking', 'semantic': 'meaning', 'hybrid': 'keywords and meaning'}
    strategies = ', '.join(f'{mode} ({meanings[mode]})' for mode in modes)
    search['parameters']['properties']['mode']['description'] = (
        f'Available strategies: {strategies or "none"}; omitted/null uses {default_mode}.')
    return definitions


DEFAULT_MATCH_LIMIT = 5
UNIQUE_SOURCES_LIMIT = 50
DEFAULT_SEARCH_LIMIT = 10


class AgentTools:
    """Adapt injected capabilities without owning resources or making retrieval decisions.

    mode is the default strategy; an agent may override it on each search.
    Reranking remains an application composition setting.
    The existing engine remains responsible for all retrieval strategy execution.
    """

    def __init__(self, *, documents: DocumentAccess, exact: ExactRetriever,
                 engine: RetrievalEngine, mode: str = 'semantic', rerank: bool = False):
        if mode not in ('semantic', 'bm25', 'lexical', 'hybrid'):
            raise ValueError('Unknown configured retrieval mode.')
        if type(rerank) is not bool:
            raise ValueError('rerank must be a boolean.')
        self._documents = documents
        self._exact = exact
        self._engine = engine
        self._mode = mode
        self._rerank = rerank

    def tool_definitions(self) -> tuple[dict[str, ConfigValue], ...]:
        """Describe only search modes supported by the composed engine.

        This advertises capabilities, without choosing a strategy for the agent.
        The shared catalog remains unchanged for other compositions.
        """
        modes = []
        if self._engine.bm25 is not None:
            modes.append('bm25')
        if self._engine.semantic is not None:
            modes.append('semantic')
        if len(modes) == 2:
            modes.append('hybrid')
        return tool_definitions(tuple(modes), default_mode=self._mode)

    def match(self, query: str, *, target: str = 'content', regex: bool = False,
              case_sensitive: bool = True, source: str | None = None, limit: int | None = None,
              timeout: float = 30, unique_sources: bool = False) -> QueryResult:
        """Use when you know an exact word, phrase, symbol, filename, or text pattern.

        unique_sources lists each matching note once; the result's truncated
        flag says whether more matches existed beyond limit. An omitted limit
        means five occurrences, or UNIQUE_SOURCES_LIMIT notes when unique_sources
        is set, because one hit per note costs only a few evidence tokens.
        """
        if limit is None:
            limit = UNIQUE_SOURCES_LIMIT if unique_sources else DEFAULT_MATCH_LIMIT
        return _query_result(self._exact.search(
            query, target=target, regex=regex, case_sensitive=case_sensitive,
            filters={'source': source} if source is not None else None, top_k=limit, timeout=timeout,
            unique_sources=unique_sources,
        ))

    def search(self, query: str, *, source: str | None = None, limit: int = DEFAULT_SEARCH_LIMIT,
               mode: str | None = None) -> QueryResult:
        """Use to discover relevant knowledge about a question, topic, or concept
        when you do not know the document's exact wording.

        The default depth is ten chunks: on the development tracks one
        five-chunk call bounded document recall at the engine's recall@5, while
        ten results reach recall@10 at a modest evidence cost. Oversized results
        are delivered as a fitting prefix by the observer, never silently cut.
        """
        filters = validate_request(query, limit, {'source': source} if source is not None else None)
        if mode is not None and mode not in ('bm25', 'semantic', 'hybrid'):
            raise ValueError('mode must be bm25, semantic or hybrid.')
        return _query_result(self._engine.search(query, mode=self._mode if mode is None else mode,
                                                rerank=self._rerank,
                                                filters=filters, top_k=limit))

    def read(self, document_id: str | None = None, *, source: str | None = None,
             section_id: str | None = None,
             start_char: int | None = None, end_char: int | None = None) -> ReadResult:
        """Read by returned document ID or known source filename. Both selectors
        must agree when supplied together. Optionally select a section or range.
        """
        document = self._documents.read(document_id, source=source, section_id=section_id,
                                         start_char=start_char, end_char=end_char)
        return ReadResult(result=Evidence(
            document_id=document.document_id, source=document.source, title=document.title,
            content=document.content, document_revision=document.document_revision,
            chunk_id=None, section_id=document.section_id,
            start_char=document.start_char, end_char=document.end_char,
            section_start_char=None, section_end_char=None))


# Plain JSON schemas, independent of any model provider, framework, or dispatcher.
TOOL_DEFINITIONS: tuple[dict[str, ConfigValue], ...] = (
    {
        'name': 'match',
        'description': 'Use only when you know an exact word, phrase, symbol, filename, or text '
                       'pattern. Returns literal occurrences by default; enable regex for '
                       'patterns. Use target=source for filenames instead of document bodies. '
                       'unique_sources=true lists each matching note once (up to 50 unless limit '
                       'is given). truncated=true in the response means more matches exist beyond '
                       'limit: repeat the same call with a higher limit before claiming a complete list.',
        'parameters': {
            'type': 'object', 'required': ['query'], 'additionalProperties': False,
            'properties': {
                'query': {'type': 'string', 'minLength': 1},
                'target': {'type': 'string', 'enum': ['content', 'source'], 'default': 'content'},
                'regex': {'type': 'boolean', 'default': False},
                'case_sensitive': {'type': 'boolean', 'default': True},
                'source': {'type': ['string', 'null'], 'minLength': 1,
                           'description': 'Restrict to this exact knowledge-relative source path.'},
                'limit': {'type': 'integer', 'minimum': 1,
                          'description': 'Maximum results; defaults to 5 occurrences, or 50 notes with unique_sources.'},
                'unique_sources': {'type': 'boolean', 'default': False,
                                   'description': 'At most one result per note; use it to enumerate matching notes completely.'},
            },
        },
    },
    {
        'name': 'search',
        'description': 'Use to discover relevant knowledge about a question, topic, or '
                       'concept when you do not know the exact wording. Results are ordered '
                       'by relevance. Choose an available strategy from the mode parameter, '
                       'or omit it for the default. If the results do not cover the question, '
                       'search again with different wording, another mode, or a larger limit. '
                       'Use read with a returned ref to expand context.',
        'parameters': {
            'type': 'object', 'required': ['query'], 'additionalProperties': False,
            'properties': {
                'query': {'type': 'string', 'minLength': 1},
                'mode': {'type': ['string', 'null'], 'enum': ['bm25', 'semantic', 'hybrid', None],
                         'description': 'Retrieval strategy for this call; omitted/null uses the configured default.'},
                'source': {'type': ['string', 'null'], 'minLength': 1,
                           'description': 'Restrict to this exact knowledge-relative source path.'},
                'limit': {'type': 'integer', 'minimum': 1, 'default': 10,
                          'description': 'Number of ranked chunks; use up to 20 for broad questions.'},
            },
        },
    },
    {
        'name': 'read',
        'description': 'Expand returned evidence using ref. Alternatively read a known source filename. '
                       'Supply exactly one selector. References are bound to their source revision; '
                       'if stale, search again or read the filename to get current text. expand=section '
                       '(default) returns the bounded section around the evidence; snippet returns only '
                       'the excerpt; document returns the whole note, which can be very long.',
        'parameters': {
            'type': 'object', 'oneOf': [{'required': ['ref']}, {'required': ['source']}],
            'additionalProperties': False,
            'properties': {
                'ref': {'type': 'string', 'minLength': 1},
                'source': {'type': 'string', 'minLength': 1,
                           'description': 'Known filename in the knowledge base, e.g. rag.md.'},
                'expand': {'type': 'string', 'enum': ['section', 'snippet', 'document'], 'default': 'section',
                           'description': 'section: heading section around the evidence (bounded); snippet: the excerpt only; document: the entire note.'},
            },
        },
    },
    {
        'name': 'finish',
        'description': 'Return your final answer with supporting evidence refs. Use insufficient_evidence '
                       'when the available evidence cannot answer the question; explain gaps in answer. '
                       'For a greeting or other request needing no retrieval, refs can be empty.',
        'parameters': {
            'type': 'object', 'required': ['answer', 'status', 'evidence_refs'],
            'additionalProperties': False,
            'properties': {
                'answer': {'type': 'string', 'minLength': 1},
                'status': {'type': 'string', 'enum': ['answered', 'partial', 'insufficient_evidence']},
                'evidence_refs': {'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True},
            },
        },
    },
)

FINAL_SCHEMA = deepcopy(TOOL_DEFINITIONS[-1]['parameters'])

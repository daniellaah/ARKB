"""Serve the read-only knowledge tools to an outer agent over MCP stdio.

Claude Code and the Claude desktop app are the agent in this mode; ARKB
contributes retrieval only and loads no model of its own. The boundary is
therefore stateless: the client is another agent whose requests need not belong
to one run, so every result carries its vault-relative source, the document
revision it was read from and body coordinates, instead of the agent loop's
run-scoped ev_* references. Nothing here writes, deletes or indexes.
"""

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from arkb.agent.session import ToolInputError, validate_arguments
from arkb.agent.tools import DEFAULT_LIST_LIMIT, DEFAULT_SEARCH_LIMIT
from arkb.config import DEFAULT_DB, DEFAULT_RETRIEVAL_MODE
from arkb.knowledge.chunking import _markdown_blocks
from arkb.knowledge.documents import DocumentNotFound
from arkb.retrieval.exact import ExactCancelled, ExactPatternError, ExactTimeout


SERVER_NAME = 'arkb'

INSTRUCTIONS = (
    'Read-only retrieval over the "{vault_id}" Markdown knowledge base. list browses notes, '
    'match finds literal text, search ranks chunks by relevance, read returns a note or one of '
    'its sections, and links -- when the index holds a link graph -- follows the notes one note '
    'links to or is linked from. Results identify evidence by its vault-relative source path '
    'and the document revision it was read from; quote and cite those paths. This server never '
    'writes to the knowledge base.'
)

# Only these leave the boundary: enough to quote and re-read a result, and
# nothing that names an internal identity a later request could not resolve.
EVIDENCE_FIELDS = ('source', 'title', 'content', 'document_revision', 'start_char', 'end_char')

# One sentence of the shared search description points at the agent loop's
# references; a test asserts no advertised schema mentions them.
REF_SENTENCE = 'Use read with a returned ref to expand context.'
SOURCE_SENTENCE = 'Use read with a returned source to expand context.'

READ_DESCRIPTION = (
    'Read a note by its vault-relative source path, exactly as list, match or search returned it. '
    'expand=document (default) returns the whole note, which can be long; expand=section with a '
    'start_char taken from a result returns the bounded Markdown section holding that offset. The '
    'response repeats the source and the document revision it was read from.'
)

READ_PARAMETERS = {
    'type': 'object', 'required': ['source'], 'additionalProperties': False,
    'properties': {
        'source': {'type': 'string', 'minLength': 1,
                   'description': 'Vault-relative source path, e.g. rag.md or 04-Areas/rag.md.'},
        'expand': {'type': 'string', 'enum': ['section', 'document'], 'default': 'document',
                   'description': 'document: the entire note; section: the heading section around start_char.'},
        'start_char': {'type': 'integer', 'minimum': 0,
                       'description': 'Body offset from a list, match or search result; anchors expand=section.'},
    },
}


def tool_schemas(definitions) -> tuple[dict, ...]:
    """Republish the agent's own tool schemas without finish or run-scoped refs.

    Names, descriptions and parameters come from TOOL_DEFINITIONS so one
    contract serves the internal loop and MCP clients alike. read is the single
    exception: its ref selector addresses evidence delivered earlier in one
    agent run and has no meaning for a client that is itself the agent.
    """
    schemas = []
    for definition in definitions:
        if definition['name'] == 'finish':
            continue
        definition = deepcopy(definition)
        if definition['name'] == 'read':
            definition['description'] = READ_DESCRIPTION
            definition['parameters'] = deepcopy(READ_PARAMETERS)
        else:
            definition['description'] = definition['description'].replace(REF_SENTENCE, SOURCE_SENTENCE)
        schemas.append(definition)
    return tuple(schemas)


def _result(evidence) -> dict:
    return {field: evidence[field] for field in EVIDENCE_FIELDS}


def _section_bounds(content: str, anchor: int) -> tuple[int, int]:
    """Bound the heading section holding anchor, as a section read would.

    This uses the block recognizer the indexer uses, so a section is its
    heading and direct body up to the next heading of any level; text before
    the first heading is the note's root section.
    """
    start, end = 0, len(content)
    for block in _markdown_blocks(content):
        if block.kind != 'heading':
            continue
        if block.start <= anchor:
            start = block.start
        else:
            end = block.start
            break
    return start, end


class ReadOnlyTools:
    """Validate one stateless request and project its result for an outer agent.

    The capabilities are the ones the agent loop uses; what changes is the
    contract around them. There is no evidence registry, no budget and no
    conversation, because each request is complete on its own.
    """

    def __init__(self, tools):
        self.tools = tools
        self.definitions = tool_schemas(tools.tool_definitions())
        self._schemas = {definition['name']: definition['parameters'] for definition in self.definitions}

    def _read(self, *, source: str, expand: str = 'document', start_char: int | None = None) -> dict:
        try:
            note = self.tools.read(source=source)['result']
        except DocumentNotFound as error:
            raise ToolInputError('source_unavailable', 'No note has this source path in the live '
                                 'knowledge base; list or search for its current path.') from error
        content = note['content']
        if start_char is not None and start_char > len(content):
            raise ToolInputError('invalid_arguments', 'start_char is past the end of this note; '
                                 'the note changed, so search again for current offsets.')
        if expand == 'section' and start_char is not None:
            start, end = _section_bounds(content, start_char)
            note = {**note, 'content': content[start:end], 'start_char': start, 'end_char': end}
        return {'result': _result(note)}

    def invoke(self, name: str, arguments: dict, *, exact_timeout: float = 30) -> dict:
        """Run one advertised tool; expected mistakes raise ToolInputError."""
        if name not in self._schemas:
            raise ToolInputError('unknown_tool', f'Use {", ".join(sorted(self._schemas))}.')
        validate_arguments(arguments, self._schemas[name])
        if name == 'list':
            return self.tools.list(**{'limit': DEFAULT_LIST_LIMIT, **arguments})
        if name == 'links':
            try:
                return self.tools.links(**arguments)
            except DocumentNotFound as error:
                raise ToolInputError('source_unavailable', 'No note has this source path in the live '
                                     'knowledge base; list or search for its current path.') from error
        if name == 'read':
            return self._read(**arguments)
        if name == 'match':
            raw = self.tools.match(**arguments, timeout=exact_timeout)
        else:
            engine = self.tools._engine
            if ((arguments.get('mode') or self.tools._mode) == 'hybrid'
                    and arguments.get('limit', DEFAULT_SEARCH_LIMIT) > engine.candidate_k):
                raise ToolInputError('invalid_arguments', f'Hybrid limit must be <= {engine.candidate_k}.')
            raw = self.tools.search(**arguments)
        return {'query': raw['query'], 'results': [_result(hit) for hit in raw['results']],
                **({'truncated': raw['truncated']} if 'truncated' in raw else {}),
                **({'index_version': raw['index_id']} if raw.get('index_id') is not None else {})}


def _error(code: str, message: str) -> types.CallToolResult:
    """Report an expected mistake as a tool error, so one bad call cannot end the session."""
    return types.CallToolResult(isError=True, content=[types.TextContent(type='text', text=f'{code}: {message}')])


def _version() -> str:
    try:
        return version('arkb')
    except PackageNotFoundError:  # pragma: no cover
        return '0'


def build_server(tools, *, vault_id: str) -> Server:
    """Bind prepared capabilities to an MCP server, transport independent."""
    vault = ReadOnlyTools(tools)
    server = Server(SERVER_NAME, version=_version(), instructions=INSTRUCTIONS.format(vault_id=vault_id))

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        # The hints state what the server guarantees: these calls only read,
        # and they read this knowledge base rather than the open world.
        return [types.Tool(name=schema['name'], description=schema['description'],
                           inputSchema=schema['parameters'],
                           annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                                             idempotentHint=True, openWorldHint=False))
                for schema in vault.definitions]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        # Retrieval is synchronous and a stdio server has one client, so calls
        # run in arrival order instead of on worker threads.
        try:
            return vault.invoke(name, arguments)
        except ToolInputError as error:
            return _error(error.code, str(error))
        except ExactPatternError as error:
            return _error('invalid_pattern', str(error))
        except ExactTimeout as error:
            return _error('exact_timeout', str(error))
        except ExactCancelled as error:
            return _error('exact_cancelled', str(error))
        except (ValueError, LookupError) as error:
            return _error('unavailable', str(error))

    return server


def serve(runtime, *, db: Path = DEFAULT_DB, vault_id: str = 'default',
          notes_dir: Path | None = None, mode: str = DEFAULT_RETRIEVAL_MODE) -> None:
    """Serve one snapshot over stdio until the client closes the connection."""
    with runtime.live_tools(db=db, vault_id=vault_id, notes_dir=notes_dir, mode=mode) as tools:
        server = build_server(tools, vault_id=vault_id)
        options = server.create_initialization_options()

        async def session():
            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream, options)

        anyio.run(session)

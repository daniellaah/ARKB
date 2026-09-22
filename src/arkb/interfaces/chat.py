"""A multi-turn session over the bounded agent loop: the conversation carries, the budget does not.

Each turn is one complete agent run with its own turn allowance and its own
budget, so a long session cannot spend a later turn's allowance on an earlier
one. What carries across turns is the conversation and one ToolSession.

Reusing the session is the deliberate choice for cross-turn references. The
`ev_` references are per-session identifiers bound to a source and a document
revision, and citing one re-reads the source and rejects a changed revision, so
a reference collected two turns ago is either still exact or refused. The
alternative — a fresh session per turn — would invalidate every earlier
reference and force the model to re-fetch evidence it already has, which costs
a tool call and a read budget per follow-up. `/reset` is where references do
expire: it starts a new session together with a new conversation.

Follow-ups need no query rewriting. The model resolves "it" and "that note"
from the history it is given; the work here is to give it a correct history and
to keep that history bounded.

The session logic is a function over an iterator of input lines, so a test can
drive it directly; the CLI adds input() and printing around it.
"""

from collections.abc import Iterable, Iterator
import json

DEFAULT_HISTORY_TOKENS = 24000
# Four characters per token: a transport-independent estimate, deliberately not the
# embedding tokenizer the observer uses to measure evidence. It only decides when to
# compact, so being approximate costs a slightly early or late compaction, nothing else.
CHARS_PER_TOKEN = 4
COMPACTED_NOTE = ('Earlier observation; its text was dropped to bound this conversation. '
                  'Read a source again if its exact wording matters.')
EXIT_HINT = 'Ask a question. Blank line quits; /reset clears the session; /trace toggles the trajectory.'


def estimate_tokens(messages: Iterable[dict]) -> int:
    """Approximate the conversation's size without loading a tokenizer."""
    return sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in messages) // CHARS_PER_TOKEN


def is_compacted(content: str) -> bool:
    """Whether an observation has already lost its bodies; compacting twice would lose its sources."""
    try:
        return bool(json.loads(content).get('compacted'))
    except ValueError:
        return False


def summarize_observation(content: str) -> str:
    """Replace an observation's bodies with the sources it delivered, still as valid JSON.

    The conversation stays well formed for every transport: each tool call keeps
    exactly one answer, and the answer keeps the source paths, so the model can
    ask for a note again by name instead of losing the fact that it saw it.
    """
    try:
        result = json.loads(content)
    except ValueError:
        return json.dumps({'status': 'success', 'compacted': True, 'sources': [], 'note': COMPACTED_NOTE},
                          ensure_ascii=False)
    hits = [result['result']] if 'result' in result else result.get('results') or []
    sources = list(dict.fromkeys(h['source'] for h in hits if isinstance(h, dict) and h.get('source')))
    return json.dumps({'status': result.get('status', 'success'), 'compacted': True,
                       'sources': sources, 'note': COMPACTED_NOTE}, ensure_ascii=False)


def carry_history(messages: Iterable[dict], *, limit: int = DEFAULT_HISTORY_TOKENS) -> list[dict]:
    """Prepare the conversation the next turn replays: no operator instructions, bounded size.

    Turn-scoped system messages (the closing instruction, the stall reminder)
    are dropped. They addressed the turn that ended, and the next run supplies
    its own instruction as message zero; keeping them would also rewrite the
    stable prefix that hosted prompt caching depends on.

    Above the limit the earliest tool observations lose their bodies first and
    keep a summary of the sources they delivered. limit=0 disables the bound,
    which lets the context grow until the provider refuses it.
    """
    kept = [dict(m) for m in messages if m['role'] != 'system']
    if not limit:
        return kept
    for index, message in enumerate(kept):
        if estimate_tokens(kept) <= limit:
            break
        if message['role'] == 'tool' and not is_compacted(message['content']):
            kept[index] = {**message, 'content': summarize_observation(message['content'])}
    return kept


def trace_lines(result) -> Iterator[str]:
    """The available trajectory: requested calls with their arguments, then why the run stopped."""
    trace = result.trace
    for step, call in enumerate(trace.tool_calls, 1):
        yield f'[{step}] {call.name}'
        for name, value in call.arguments.items():
            yield f'{name}: {json.dumps(value, ensure_ascii=False)}'
        yield ''
    yield f'[{len(trace.tool_calls) + 1}] {trace.stop_reason}'


def converse(inputs: Iterable[str], *, run, reset=None, usage=None, trace: bool = False,
             history_tokens: int = DEFAULT_HISTORY_TOKENS) -> Iterator[str]:
    """Drive one chat session over an iterator of input lines, yielding output lines.

    A blank line ends the session, /reset clears the conversation and whatever
    the caller keeps beside it, and /trace toggles the trajectory. `run(query,
    history)` performs one complete agent run and returns its AgentResult;
    `usage()` returns the accounting lines for the run that just finished.
    """
    history: list[dict] = []
    for line in inputs:
        text = line.strip()
        if not text:
            return
        if text == '/reset':
            history = []
            if reset is not None:
                reset()
            yield 'Session cleared; earlier evidence references no longer resolve.'
            continue
        if text == '/trace':
            trace = not trace
            yield f'Trace {"on" if trace else "off"}.'
            continue
        result = run(text, history)
        history = carry_history(result.state.messages, limit=history_tokens)
        if trace:
            yield from trace_lines(result)
        if result.response is not None:
            yield result.response
        else:
            reason = result.final.termination_reason if result.final else result.stop_reason
            yield f'No final response: {reason}.'
        if trace and usage is not None:
            yield from usage()

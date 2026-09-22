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

# One conversation bound with one meaning: the same estimate and the same
# compaction the agent loop applies inside a run, applied here between turns.
from arkb.agent.context import (CHARS_PER_TOKEN, COMPACTED_NOTE, DEFAULT_HISTORY_TOKENS,  # noqa: F401
                                compact, estimate_tokens, is_compacted, summarize_observation)

EXIT_HINT = 'Ask a question. Blank line quits; /reset clears the session; /trace toggles the trajectory.'


def carry_history(messages: Iterable[dict], *, limit: int = DEFAULT_HISTORY_TOKENS) -> list[dict]:
    """Prepare the conversation the next turn replays: no operator instructions, bounded size.

    Turn-scoped system messages (the closing instruction, the stall reminder,
    and the context the next run decides for itself: a vault map, a delivered
    small scope) are dropped. They addressed the turn that ended, and the next
    run supplies its own instruction as message zero; keeping them would also
    rewrite the stable prefix that hosted prompt caching depends on.

    Above the limit the earliest tool observations lose their bodies first and
    keep a summary of the sources they delivered. limit=0 disables the bound,
    which lets the context grow until the provider refuses it.
    """
    return compact([m for m in messages if m['role'] != 'system'], limit=limit)[0]


def trace_lines(result) -> Iterator[str]:
    """The available trajectory: what context the run was given, its calls, then why it stopped."""
    trace = result.trace
    decided = (getattr(result, 'observation', None) or {}).get('context')
    if decided:
        yield '[context] ' + json.dumps(decided, ensure_ascii=False, sort_keys=True)
        yield ''
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

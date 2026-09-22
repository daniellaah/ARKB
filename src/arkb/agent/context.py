"""What a run forgets when it grows long, and the setting that decides it.

Compaction drops the bodies of the oldest observations from the replayed
conversation. The session's reference table is not part of the conversation, so
a reference whose observation was compacted still resolves and is still
accepted by finish; only the text the model can re-read is gone.

It is off when its setting is zero, which returns the run to the unbounded
conversation the loop had before.
"""

from collections.abc import Iterable
from dataclasses import dataclass
import json

# Four characters per token: a transport-independent estimate, deliberately not
# the embedding tokenizer the observer uses to measure evidence. It only decides
# when to compact, so being approximate costs a slightly early or late decision,
# nothing else.
CHARS_PER_TOKEN = 4
# Bound the replayed conversation below the 32k context the evaluation and the
# local models run with, leaving the reserved finalization room for its own
# output. The same number bounds a chat session between turns: one conversation
# bound, one meaning.
DEFAULT_HISTORY_TOKENS = 24000

COMPACTED_NOTE = ('Earlier observation; its text was dropped to bound this conversation. '
                  'Read a source again if its exact wording matters.')


@dataclass(frozen=True, kw_only=True)
class ContextPolicy:
    """What one run does with its context; each setting is off at zero.

    history_tokens compacts the oldest observations once the replayed
    conversation is estimated above that size. On by default, because a bound
    that is never reached costs nothing and an unbounded conversation ends as a
    provider error rather than an answer.
    """

    history_tokens: int = DEFAULT_HISTORY_TOKENS

    def __post_init__(self):
        if type(self.history_tokens) is not int or self.history_tokens < 0:
            raise ValueError('history_tokens must be a nonnegative integer.')


# The unbounded conversation the loop had before.
OFF = ContextPolicy(history_tokens=0)


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


def compact(messages: Iterable[dict], *, limit: int, measure=estimate_tokens,
            protect: int = 0) -> tuple[list[dict], int]:
    """Drop the oldest observation bodies until the conversation fits limit.

    Only tool observations lose text, oldest first, and each keeps the sources
    it delivered. Assistant turns, and therefore the tool calls every transport
    pairs its results with, are never touched. measure sizes the conversation as
    it will actually be sent, which is not always the list given here. limit=0
    disables the bound, which lets the context grow until the provider refuses it.

    protect holds that many messages at the end back from compaction. Inside a
    run these are the observations the model has not read yet: dropping the
    result a tool just returned would leave the model with a trajectory and no
    evidence, so the conversation is allowed to exceed the limit instead. A
    caller compacting a conversation the model has already seen protects nothing.

    Returns the rewritten conversation and how many observations were compacted.
    """
    kept = [dict(message) for message in messages]
    if not limit:
        return kept, 0
    compacted = 0
    for index, message in enumerate(kept[:len(kept) - protect] if protect else kept):
        if measure(kept) <= limit:
            break
        if message['role'] == 'tool' and not is_compacted(message['content']):
            kept[index] = {**message, 'content': summarize_observation(message['content'])}
            compacted += 1
    return kept, compacted

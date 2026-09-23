"""What the answer may cite: one sentence before the choice, one check after it.

Two guards on the citations a final object carries. Each is independent and
each is switchable, because only measurement decides whether either earns its
cost on a given model.

- Discipline states the rule where the model chooses its citations: a citation
  belongs to a statement, not to a topic. The sentence goes into the finish
  tool's description and into the closing instruction, because a run reaches
  finish through either one.
- Verification is a check after the answer exists. Each cited note goes back to
  the model with the answer in one small request that asks a single question:
  does this note support a specific statement in the answer? A note that does
  not is dropped. This is not a second agent loop: it has no tools, no turns,
  and no memory from one citation to the next.

Verification can only remove a citation, never add one, and a check that fails
or returns something unreadable keeps the citation: a broken verifier must not
quietly strip an answer's evidence. Dropping every citation is reported to the
caller, which downgrades the status rather than returning an answered final
with nothing behind it.
"""

from copy import deepcopy
from dataclasses import dataclass
import json

DISCIPLINE = ('Cite a note only when a specific statement in the answer rests on it; '
              'a note being related to the subject is not a reason to cite it.')

VERIFY_INSTRUCTION = (
    'You check one citation. You are given an answer and a verbatim excerpt from one note that the '
    'answer cites. Decide whether the excerpt supports a specific statement the answer makes about '
    'that note. A statement that the note contains, mentions or is about something is supported when '
    'the excerpt shows it, however short the excerpt is. Shared subject matter alone is not support. '
    'Return only the object {"supported": true} or {"supported": false}.')

VERIFY_SCHEMA = {'type': 'object', 'required': ['supported'], 'additionalProperties': False,
                 'properties': {'supported': {'type': 'boolean'}}}

# A cheap check stays cheap. Cited evidence is a chunk, a heading section or a
# whole note, and a note that needs more than this to show its support is not
# something one boolean can settle; the excerpt is cut with the cut marked.
VERIFY_MAX_CHARS = 12000
CUT_NOTE = '\n[Note truncated for this check.]'


@dataclass(frozen=True, kw_only=True)
class CitationPolicy:
    """Whether the run states the citation rule, and whether it checks the result.

    Both are off, because both were measured on all 114 evaluation questions
    (2026-09-22, qwen3.5:9b and qwen3.5:27b, citation precision and recall):

    - discipline moved 27b by -0.001 precision and -0.002 recall, which is
      nothing, and bought 9b +0.011 precision for -0.028 recall. It costs about
      500 prompt tokens per question, every turn, for no measured gain.
    - verify on 27b rejected 4 of 431 citations: the model that chose a
      citation almost always confirms it, so self-verification buys -0.001
      precision for -0.016 recall at twice the model requests. On 9b it
      rejected 80 of 465, most of them correct, for -0.077 recall.

    The sound version of verify needs a checker that is not the author, which
    is a hosted model and a separate measurement, not a default.
    """

    discipline: bool = False
    verify: bool = False

    def __post_init__(self):
        if type(self.discipline) is not bool or type(self.verify) is not bool:
            raise ValueError('discipline and verify must be booleans.')


# Neither guard: the final object is exactly what the model proposed.
OFF = CitationPolicy()


def disciplined(definitions):
    """Repeat the rule in the finish description, where the model picks its citations."""
    return [{**definition, 'description': definition['description'] + ' ' + DISCIPLINE}
            if definition['name'] == 'finish' else definition for definition in definitions]


def final_instruction(instruction: str, policy: CitationPolicy) -> str:
    """The closing instruction, with the rule when the finalization is the only turn that sees it."""
    return f'{instruction} {DISCIPLINE}' if policy.discipline else instruction


def verification_request(model: str, *, answer: str, citation: dict, options=None) -> dict:
    """One self-contained request: the answer, one cited note, one boolean back.

    Thinking is off and the response is schema-constrained: the verdict is a
    judgement on text already in front of the model, not a task that improves
    with reasoning, and an unconstrained answer would have to be parsed out of
    prose.
    """
    content = citation.get('content') or ''
    if len(content) > VERIFY_MAX_CHARS:
        content = content[:VERIFY_MAX_CHARS] + CUT_NOTE
    title = f"# {citation['title']}\n" if citation.get('title') else ''
    question = (f'Answer:\n{answer}\n\nExcerpt from the cited note {citation.get("source")}:\n'
                f'{title}{content}\n\nDoes this excerpt support a specific statement in the answer?')
    return dict(model=model, messages=[{'role': 'system', 'content': VERIFY_INSTRUCTION},
                                       {'role': 'user', 'content': question}],
                stream=False, think=False, options={'temperature': 0, **(options or {})},
                format=deepcopy(VERIFY_SCHEMA))


def supported(response) -> bool | None:
    """The verdict, or None when the check did not return one and the citation stands."""
    if getattr(response, 'done_reason', None) == 'length':
        return None
    try:
        value = json.loads(getattr(getattr(response, 'message', None), 'content', None) or '')
    except ValueError:
        return None
    return value['supported'] if isinstance(value, dict) and type(value.get('supported')) is bool else None

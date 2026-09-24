"""Write one question per vault note, then throw away the ones retrieval cannot reach.

The evaluation corpus is the only instrument this project has, so training
data comes from the real vault instead. The labels come with it: a question
written from one note has that note as its expected source, which is why no
one has to annotate anything.

Two filters decide what survives, because a generated question fails in two
ways that a human would catch and a generator will not:

- It quotes the note. A question built from the note's own sentences trains
  keyword matching, not retrieval, so a long verbatim span shared with the
  note rejects it.
- It is unanswerable as asked. The question is put to the retrieval engine
  once; if its own note is not in the first twenty results, the question was
  written about something the note does not really say, and it goes.

    python -m training.make_questions --notes 120 --out training/data/questions-pilot.json
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import unicodedata

from arkb.agent.transports import OllamaClient
from arkb.config import RuntimeConfig
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]
VAULT = Path('/Users/daboluo/ObsidianVault/MyObsidian')
INDEX = ROOT / '.arkb/obsidian.sqlite'
VAULT_ID = 'obsidian'
QDRANT_URL = 'http://127.0.0.1:6340'
EXAMPLES = ROOT / 'evaluation/questions.json'

# Folders that hold no answerable prose: empty templates, vault plumbing,
# drawings, and an archive the live notes have already superseded. Daily
# journal notes are excluded for a different reason: they are the same habit
# checklist with a different date, so a question written from one is answered
# equally well by four hundred others and its single expected source is a
# wrong label. The first sample showed exactly that -- a question about a
# template line, whose own note ranked nineteenth.
SKIP_PREFIXES = ('90-Templates/', '00-ObsSys/', '99-Archive/', 'Excalidraw/', 'Attachments/', '01-Journal/')
MIN_PROSE_CHARS = 700
MAX_NOTE_CHARS = 6000
# A shared span is counted in units, where a CJK character is one unit and a
# run of letters or digits is one unit, because a character threshold means two
# different things in the two scripts this vault is written in. Ten units is a
# quotation in either: ten consecutive identical Chinese characters, or ten
# consecutive identical English words. The first version of this filter counted
# characters and rejected fifty of a hundred and forty notes, almost all of them
# English questions that shared a two-word technical term with their note.
MAX_SHARED_SPAN = 10
TYPES = ('knowledge_qa', 'semantic_discovery')

INSTRUCTION = """You write questions that a person would really ask their own notes.

You are given one note from a personal knowledge base. Write exactly one question
whose answer is in that note.

Rules:
- Write the question in the note's own main language.
- Do NOT reuse the note's wording. Describe the situation or the problem in your
  own words, the way somebody would who remembers the idea but not the sentence.
  Never copy a phrase of more than a few characters from the note.
- Do not mention the note, its title, its filename, or its author.
- The question must be answerable from this note alone, and specific enough that
  a different note would not answer it.
- One or two sentences. No preamble.
- type is semantic_discovery when the question asks to find material about a
  situation, and knowledge_qa when it asks a direct question about the content.
- reference is one short sentence, in Chinese, saying what the answer is. It is
  for a human reading results; it is not shown to the model being trained.

Return only a JSON object: {"type": ..., "question": ..., "reference": ...}"""

SCHEMA = {'type': 'object', 'properties': {'type': {'type': 'string', 'enum': list(TYPES)},
                                           'question': {'type': 'string'}, 'reference': {'type': 'string'}},
          'required': ['type', 'question', 'reference']}


def prose(text):
    """What is left of a note once frontmatter, code and markup are removed."""
    text = re.sub(r'\A---\n.*?\n---\n', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'!?\[\[[^\]]*\]\]|!?\[[^\]]*\]\([^)]*\)', '', text)
    text = re.sub(r'^[#>\-*\s]+$', '', text, flags=re.MULTILINE)
    return re.sub(r'\s+', '', text)


def candidates(vault, *, seed, wanted):
    """Notes with enough prose to answer a question, spread over the vault's folders."""
    notes = []
    for path in sorted(vault.rglob('*.md')):
        source = path.relative_to(vault).as_posix()
        if source.startswith('.') or source.startswith(SKIP_PREFIXES) or '/.' in source:
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            continue
        if len(prose(text)) < MIN_PROSE_CHARS:
            continue
        notes.append((source, text))
    random.Random(seed).shuffle(notes)
    # Round-robin over top-level folders so one large area cannot take the set.
    by_folder = {}
    for source, text in notes:
        by_folder.setdefault(source.split('/')[0], []).append((source, text))
    # The whole order is built before it is cut, so a larger --notes extends the
    # same sequence instead of choosing a different one; that is what makes
    # --resume continue a run rather than start a new sample.
    picked, folders = [], sorted(by_folder)
    while any(by_folder.values()):
        for folder in folders:
            if by_folder[folder]:
                picked.append(by_folder[folder].pop())
    return picked[:wanted]


_UNIT = re.compile(r'[\w\d]+|[^\s\w\d]', re.UNICODE)
_CJK = re.compile(r'[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]')


def units(text):
    """Words, single CJK characters and punctuation, lowercased."""
    out = []
    for token in _UNIT.findall(unicodedata.normalize('NFKC', text).lower()):
        out.extend(token) if _CJK.search(token) else out.append(token)
    return out


def shared_span(question, note):
    """Units in the longest run the question and the note have in common."""
    a, b = units(question), units(note)
    index = {}
    for position, unit in enumerate(b):
        index.setdefault(unit, []).append(position)
    best = 0
    for start in range(len(a)):
        for position in index.get(a[start], ()):
            length = 0
            while (start + length < len(a) and position + length < len(b)
                   and a[start + length] == b[position + length]):
                length += 1
            best = max(best, length)
    return best


def few_shot():
    questions = json.loads(EXAMPLES.read_text())
    chosen = [q for q in questions if q['type'] in TYPES][:2] + [q for q in questions if q['type'] == 'knowledge_qa'][:2]
    return '\n\n'.join(f"Example ({q['type']}):\n" + json.dumps(
        {'type': q['type'], 'question': q['question'], 'reference': q['reference']}, ensure_ascii=False)
        for q in chosen)


def ask(client, model, source, text, shots):
    body = text[:MAX_NOTE_CHARS]
    messages = [{'role': 'system', 'content': INSTRUCTION + '\n\n' + shots},
                {'role': 'user', 'content': f'Note path: {source}\n\nNote content:\n{body}'}]
    response = client.chat(model=model, messages=messages, stream=False, think=False,
                           format=SCHEMA, options={'temperature': 0.8, 'num_ctx': 16384, 'num_predict': 1024})
    return json.loads(response.message.content)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'training/data/questions-pilot.json')
    parser.add_argument('--notes', type=int, default=120, help='notes to try; the kept set is smaller')
    parser.add_argument('--keep', type=int, default=100, help='stop once this many questions survive')
    parser.add_argument('--model', default='qwen3.5:27b')
    parser.add_argument('--vault', type=Path, default=VAULT)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='keep the questions already written and extend them')
    parser.add_argument('--exclude', type=Path, action='append', default=[],
                        help='a question file whose notes are off limits; repeatable')
    args = parser.parse_args()

    shots = few_shot()
    notes = candidates(args.vault, seed=args.seed, wanted=args.notes)
    kept_already = json.loads(args.out.read_text()) if args.resume and args.out.exists() else []
    # Carried-over questions face the rule this run uses, not the one that let
    # them in, so the whole set answers to one filter.
    kept_already = [q for q in kept_already
                    if shared_span(q['question'], (args.vault / q['expected_sources'][0]).read_text(
                        encoding='utf-8')) <= MAX_SHARED_SPAN]
    already = {q['expected_sources'][0] for q in kept_already}
    # A held-out set is only held out if it is written from notes the trained
    # set never saw, so the sources of an excluded file are off limits here.
    for path in args.exclude:
        already |= {q['expected_sources'][0] for q in json.loads(path.read_text())}
    notes = [(source, text) for source, text in notes if source not in already]
    print(f'{len(notes)} candidate notes ({len(kept_already)} questions carried over)', flush=True)
    client = OllamaClient(args.model, options={'num_ctx': 16384}, think=False)
    kept, rejected = list(kept_already), Counter()
    with Runtime(RuntimeConfig(offline=True, tokenizer_cache=(ROOT / '.uv-cache/tokenizers').resolve(),
                               qdrant_url=QDRANT_URL)) as runtime:
        with SQLiteStorage(INDEX, read_only=True) as storage:
            manifest = storage.active_manifest(VAULT_ID)
            engine = runtime.retrieval_engine(storage, manifest, modes=('hybrid',))
            for index, (source, text) in enumerate(notes):
                if len(kept) >= args.keep:
                    break
                try:
                    written = ask(client, args.model, source, text, shots)
                except Exception as error:
                    rejected['generation_failed'] += 1
                    print(json.dumps({'source': source, 'rejected': 'generation_failed',
                                      'error': str(error)[:120]}, ensure_ascii=False), flush=True)
                    continue
                question = (written.get('question') or '').strip()
                span = shared_span(question, text)
                if not question or written.get('type') not in TYPES:
                    rejected['malformed'] += 1
                    continue
                if span > MAX_SHARED_SPAN:
                    rejected['copies_the_note'] += 1
                    print(json.dumps({'source': source, 'rejected': 'copies_the_note', 'span': span,
                                      'question': question}, ensure_ascii=False), flush=True)
                    continue
                hits = engine.search(question, mode='hybrid', top_k=args.top_k).results
                ranked = [hit.source for hit in hits]
                if source not in ranked:
                    rejected['not_retrievable'] += 1
                    print(json.dumps({'source': source, 'rejected': 'not_retrievable',
                                      'question': question}, ensure_ascii=False), flush=True)
                    continue
                kept.append({'id': None, 'type': written['type'], 'question': question,
                             'question_zh': question, 'expected_sources': [source],
                             'reference': (written.get('reference') or '').strip(),
                             'rank': ranked.index(source) + 1, 'shared_span': span})
                print(json.dumps({'done': index + 1, 'kept': len(kept), 'rank': kept[-1]['rank'],
                                  'source': source}, ensure_ascii=False), flush=True)
    client.close()
    for number, question in enumerate(kept, 1):
        question['id'] = f'vault_{number:03d}'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(kept, ensure_ascii=False, indent=1) + '\n')
    report = {'tried': len(kept) - len(kept_already) + sum(rejected.values()), 'carried_over': len(kept_already),
              'kept': len(kept),
              'rejected': dict(rejected), 'types': dict(Counter(q['type'] for q in kept)),
              'mean_rank': round(sum(q['rank'] for q in kept) / len(kept), 2) if kept else None,
              'model': args.model, 'out': str(args.out)}
    (args.out.parent / 'questions-pilot-report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

"""Scoring-side import of official BrowseComp answers; never imported by runner."""
import argparse
import ast
from pathlib import Path
import json
import re

from arkb.evaluation.browsecomp import CANARY, decrypt
from arkb.evaluation.external import digest, write_json

OFFICIAL_SCRIPT_SHA256 = '1a21233937c377ab6323c98ff9af67742756a57fbacab4ebf9bc30852eae530a'


def rubric(path):
    if digest(path) != OFFICIAL_SCRIPT_SHA256:
        raise ValueError('Official rubric source changed.')
    tree = ast.parse(Path(path).read_text())
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'GRADER_TEMPLATE' for t in node.targets)]
    value = assignments[0].value
    # Extract a reviewed constant, never import/execute the remote vLLM script.
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == 'strip' and isinstance(value.func.value, ast.Constant)):
        raise ValueError('Unexpected official rubric expression.')
    return ast.literal_eval(value.func.value).strip()


def parse_judge(text):
    # Anchor field names; 'incorrect:' must not be mistaken for 'correct:'.
    # Conflicting or repeated correctness fields remain pending, never guessed.
    matches = re.findall(r'^\s*(?:\*\*)?correct(?:\*\*)?\s*:\s*(?:\*\*)?\s*(yes|no)\b',
                         text, re.IGNORECASE | re.MULTILINE)
    if len(matches) != 1:
        return {'correct': None, 'parse_error': True}
    return {'correct': matches[0].lower() == 'yes', 'parse_error': False}


def import_labels(output, inputs):
    import pyarrow.parquet as pq
    manifest = json.loads((inputs / 'downloads.json').read_text())
    files = {k: v for k, v in manifest['files'].items()
             if k.startswith('browsecomp-plus/data/') and k.endswith('.parquet')}
    queries = json.loads(Path('/Volumes/ARKBPhaseC/data/browsecomp-plus/queries.json').read_text())
    answers = {}
    for name, metadata in sorted(files.items()):
        if digest(inputs / name) != metadata['sha256']:
            raise ValueError('Official encrypted answer source changed.')
        for batch in pq.ParquetFile(inputs / name).iter_batches(columns=['query_id', 'query', 'answer']):
            for row in batch.to_pylist():
                qid = str(row['query_id'])
                if qid in answers or decrypt(row['query'], CANARY) != queries[qid]:
                    raise ValueError('Official answer/query identity mismatch.')
                answers[qid] = decrypt(row['answer'], CANARY)
                if not answers[qid].strip():
                    raise ValueError('Missing official answer.')
    if set(answers) != set(queries) or len(answers) != 830:
        raise ValueError('Incomplete official answer set.')
    target = output / 'scoring/browsecomp-answers.json'
    payload = {'schema': 'scoring-only-browsecomp-answers-v1', 'answers': answers, 'source_files': files}
    if target.exists():
        if json.loads(target.read_text()) != payload:
            raise ValueError('Refusing to overwrite scoring labels.')
    else:
        write_json(target, payload)
    template = rubric(output / 'provenance/official-evaluate_run.py')
    protocol = {'schema': 'agentic-tools-judge-v1', 'status': 'awaiting_model_identity_and_calibration',
                'model': 'qwen3:32b-q8_0', 'runtime': 'Ollama 0.33.2', 'precision': 'Q8_0',
                'adaptation': 'ARKB adaptation: Ollama/GGUF Q8_0 instead of official vLLM weights; official rubric retained.',
                'options': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'num_predict': 4096,
                            'num_ctx': 32768, 'repeat_penalty': 1},
                'think': False, 'truncate': False, 'shift': False,
                'official_revision': '046949032b0328319cc9a02663a759ec601d9402',
                'official_script_sha256': OFFICIAL_SCRIPT_SHA256, 'rubric': template,
                'labels_sha256': digest(target), 'parser_sha256': digest(__file__),
                'parser_adaptation': 'Anchored single correct field; ambiguous/absent correctness is pending.',
                'max_unchanged_retries': 2, 'human_calibration': 'pending independent review',
                'invalid_or_nonanswer_score': 0, 'judge_failure_score': None}
    dest = output / 'judge-design.json'
    if dest.exists() and json.loads(dest.read_text()) != protocol:
        raise ValueError('Judge design changed; record an amendment.')
    if not dest.exists():
        write_json(dest, protocol)
    print(json.dumps({'answers_imported': 830, 'labels_sha256': digest(target), 'raw_answers_displayed': False}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inputs', type=Path, default=Path('/Volumes/闪迪1T/arkb-phase-c-v1/inputs'))
    a = p.parse_args()
    import_labels(a.output, a.inputs)


if __name__ == '__main__':
    main()

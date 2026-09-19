"""Create an immutable executable pilot protocol after preparation checks."""
import argparse
from pathlib import Path
import json
import shutil
import subprocess

import httpx
from arkb.evaluation.external import digest, write_json
from evaluation.agentic_tools.common import ROOT, PUBLIC, utc
from evaluation.agentic_tools.transport import model_identity
from evaluation.agentic_tools.dependencies import verify_dependencies
from evaluation.agentic_tools.contract import OPTIONS, THINK
from evaluation.agentic_tools.readiness import POLICY


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--public-dir', type=Path, default=PUBLIC)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--tool-readiness', action='store_true')
    a = parser.parse_args(argv)
    out = a.output.resolve()
    public = a.public_dir.resolve()
    public.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise ValueError('Protocol already exists; use that registered run or an explicit amendment.')
    if a.tool_readiness:
        if not a.parent or json.loads((out / 'readiness-policy.json').read_text()) != POLICY:
            raise ValueError('A repaired pilot requires its parent and the exact executable readiness policy.')
    preflight = json.loads((out / 'provider-preflight.json').read_text())
    if preflight['status'] != 'passed':
        raise ValueError('Provider preflight has not passed.')
    design = json.loads((out / 'design.json').read_text())
    if (design['options'] != OPTIONS or design['think'] is not THINK
            or preflight['requested_options'] != OPTIONS or preflight['requested_think'] is not THINK):
        raise ValueError('Design, tested provider settings and executable adapter disagree.')
    contexts = json.loads((out / 'context-indexes.json').read_text())
    if len(contexts) != 66:
        raise ValueError('Context preparation incomplete.')
    if json.loads((ROOT / 'evaluation/phase-c/v1/completion.json').read_text())['status'] != 'completed':
        raise ValueError('Phase C not complete.')
    with httpx.Client(base_url='http://127.0.0.1:11434', timeout=30) as http:
        chat = model_identity(http, 'qwen3.5:4b')
        embedding = model_identity(http, 'qwen3-embedding:0.6b')
        judge = model_identity(http, 'qwen3:32b-q8_0')
        version = http.get('/api/version').raise_for_status().json()
    if chat != preflight['chat_model'] or version != {'version': '0.33.2'}:
        raise ValueError('Runtime changed after provider verification.')
    dependencies = json.loads((out / 'dependencies.json').read_text())
    verify_dependencies(dependencies)
    source = out / 'measured-source'
    source.mkdir(exist_ok=False)
    shutil.copytree(ROOT / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT / 'evaluation/agentic_tools', source / 'evaluation/agentic_tools',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(ROOT / 'pyproject.toml', source / 'pyproject.toml')
    files = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*') if p.is_file()}
    write_json(out / 'measured-source-manifest.json', files)
    inputs = ['selection.json', 'design.json', 'inputs.json', 'context-indexes.json',
              'inference/scenarios.json', 'pilot-schedule.json', 'core-schedule.json',
              'provider-preflight.json', 'judge-design.json', 'measured-source-manifest.json', 'dependencies.json',
              'boundary-tests.xml',
              'scoring/browsecomp-answers.json', 'scoring/musique.json',
              'scoring/browsecomp-plus.json', 'scoring/fiqa.json', 'scoring/nfcorpus.json']
    if a.parent:
        inputs.extend(['amendment.md', 'parent-pilot-accounting.json'])
        inputs.extend(['readiness-policy.json', 'repair-verification.json', 'preparation-provenance.json', 'registration-inputs.json']
                     if a.tool_readiness else ['finalization-probe-summary.json'])
    protocol = {'schema': 'agentic-tools-v1-executable-pilot', 'created_at': utc(), 'phase': 'pilot',
                'project_root': str(ROOT), 'output': str(out), 'public_dir': str(public),
                'measured_source': str(source), 'models': {'chat': chat, 'embedding': embedding, 'judge': judge},
                'ollama_version': version, 'qdrant_url': 'http://127.0.0.1:6340',
                'tokenizer_cache': str(ROOT / '.uv-cache/tokenizers'),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'plan_sha256': digest(ROOT / 'docs/agentic-tool-selection-evaluation-plan.md'),
                'files': {name: digest(out / name) for name in inputs},
                'transport': {'kind': 'direct local HTTP, strict Ollama ChatResponse parsing',
                              'truncate': False, 'shift': False, 'timeout_seconds': 330},
                'primary_denominator': 'all attempted trajectories; errors and nonanswers zero; judge failures pending',
                'pilot_attempts': 98, 'core_attempts': 5460, 'core_dispatch_authorized': False,
                'core_gate': 'complete pilot, resource estimate, technical review and separate final core protocol',
                'human_review': 'pending independent review; scores provisional until resolved',
                'snapshot_policy': 'reuse Phase C read-only SQLite and Qdrant indexes; MuSiQue prepared once separately'}
    if a.parent:
        parent = a.parent.resolve()
        if json.loads((parent / 'pilot-status.json').read_text())['status'] != 'completed':
            raise ValueError('Parent pilot is not complete.')
        parent_protocol = json.loads((parent / 'protocol.json').read_text())
        protocol.update(parent_protocol_sha256=digest(parent / 'protocol.json'),
                        parent_output=str(parent), inference_lock=parent_protocol.get('inference_lock', str(parent / 'inference.lock')),
                        amendment_sha256=digest(out / 'amendment.md'))
    if a.tool_readiness:
        protocol.update(revision='tool-repair-pilot-v3', readiness_policy='readiness-policy.json',
                        core_schedule_scope='historical capacity projection only; replacement core design not selected or authorized',
                        core_gate='scope fixtures and all 98 pilot traces pass tool readiness; concurrency/repeatability study and justified new core protocol still required',
                        administrative_stop='SIGINT/SIGTERM or stop-request.json; finish current trajectory, persist stop, never automatically retry a failed gate')
    write_json(out / 'protocol.json', protocol)
    write_json(public / 'pilot-protocol.json', protocol)
    print(json.dumps({'protocol_sha256': digest(out / 'protocol.json'), 'source_files': len(files), 'pilot_attempts': 98}), flush=True)


if __name__ == '__main__':
    main()

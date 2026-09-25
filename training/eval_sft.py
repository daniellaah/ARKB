"""Run the unchanged evaluation against a student served by `mlx_lm.server`.

`evaluation/eval.py` picks its transport from the model name, and a LoRA-tuned
model has no name the product would recognise. Rather than teach the product
about a local endpoint, this replaces the factory the harness imported, for
this process only: every other thing the run does -- the questions, the
corpus, the budgets, the agent loop, `score()` -- is the harness's own. The
label and the recorded `chat_model` say which endpoint answered.

    python -m training.eval_sft --label sft-pilot-4b
"""

import argparse
import json
from pathlib import Path

from arkb.config import load_env_file

import evaluation.eval as harness
from training import chat_format
from training.transport import MLX_SERVER_URL, MlxServerClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label', required=True)
    parser.add_argument('--base-url', default=MLX_SERVER_URL)
    parser.add_argument('--model-path', default=chat_format.MODEL_PATH,
                        help='the served weights, which decide the chat template dialect')
    parser.add_argument('--limit', type=int, default=0, help='questions per type, for a quick check')
    parser.add_argument('--types', default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', default='', help="override the agent's default search strategy")
    args = parser.parse_args()

    load_env_file(ROOT / '.env')
    if args.mode:
        # The harness composes its tools without naming a strategy, so the
        # default is AgentTools'. Supplying one here measures a candidate
        # default on the development set before it becomes the product's.
        import arkb.runtime
        composed = arkb.runtime.Runtime.agent_tools
        arkb.runtime.Runtime.agent_tools = lambda self, **kw: composed(self, **{'mode': args.mode, **kw})
    options = harness.OPTIONS
    harness.make_client = lambda model, *, options, think, effort='high': MlxServerClient(
        model, base_url=args.base_url, max_tokens=options['num_predict'],
        temperature=options.get('temperature', 0), model_path=args.model_path)
    output = (args.output or harness.RESULTS / args.label).resolve()
    summary = harness.run(output, label=args.label, model=args.label, think=True, options=options,
                          limit=args.limit, resume=args.resume,
                          types=[t for t in args.types.split(',') if t] or None)
    print(harness.markdown(summary, json.loads((output / 'run.json').read_text())))


if __name__ == '__main__':
    main()

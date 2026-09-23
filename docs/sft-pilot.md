# SFT pilot: distil DeepSeek into qwen3.5:4b

Written 2026-09-23. A staged plan and one self-contained prompt for a fresh
session. The pilot is 100 questions; its purpose is to prove the pipeline
works end to end, not to move a score.

## Why 4B, and what behaviour is being taught

Measured on the 114-question development set, all on the current code:

| | 4b | 9b (t4) | 27b | deepseek-reasoner |
| --- | ---: | ---: | ---: | ---: |
| cited recall | **0.833** | 0.954 | 0.973 | 0.991 |
| cited precision | **0.960** | 0.958 | 0.918 | 0.795 |
| answered | **0.763** | 0.939 | — | 0.947 |
| tool calls / question | **5.17** | 3.95 | — | 6.5 |
| seconds / question | **16.5** | 21.9 | 55.5 | 8.8 |

4B does not cite badly — its precision is the highest of any model here. It
**fails to answer**: 76% answered against 94%, while making *more* tool calls
than 9B. It searches, does not conclude, and gives up. The teachable
behaviour is therefore narrow and concrete: keep going until the evidence is
there, then commit to an answer and cite it.

This is worth knowing before starting: 4B is only 25% faster than 9B per
question, because most of the wall clock is tool execution and the loop, not
token generation. The pilot's value is the technique and the pipeline, not a
fast local model. A vault-specific behaviour model is the plausible payoff
later; this pilot is what makes that possible.

## Cost and time, computed from measurements

A DeepSeek rollout on this system measured 19,690 prompt and 1,455 output
tokens over 3.9 requests per question (from `v2-deepseek-reasoner`, 114
questions). At the prices committed in `src/arkb/agent/transports.py`
(checked 2026-09-22: $0.30/M input, $1.20/M output, $0.006/M cache read):

| scale | uncached | 50% cache | 80% cache |
| --- | ---: | ---: | ---: |
| 100 questions × 4 samples (the pilot) | $3.06 | $1.90 | $1.21 |
| 1,000 × 4 (a real training set) | $30.61 | $19.04 | $12.09 |
| 2,000 × 4 | $61.23 | $38.07 | $24.18 |

Cache hit rates should be high: every rollout shares the same system prompt
and tool definitions, and the four samples of one question share the question
too. DeepSeek caches server-side with no request-side marking.

Wall clock is a function of concurrency, not of the 8.8 s/question measured
by the evaluation harness — that number is an artefact of its sequential
loop. At 16-way concurrency the pilot's 400 rollouts take a few minutes.

## Stages, in reverse order of the data flow

The training and serving path is validated **first**, with throwaway data,
because that is where an environment-level blocker would live. If a
fine-tuned model cannot be served back to the agent loop, no amount of data
matters, and finding that out costs nothing.

| Stage | What | Cost | Passes when |
| ---: | --- | --- | --- |
| 0 | Fine-tune and serve a throwaway LoRA; run one `arkb ask` through it | $0, ~1h | The model answers and calls one tool. Quality is irrelevant. |
| 1 | Generate 100 questions from the real vault, reverse-validate, read 10 by hand | $0 (local) or ~$0.2 | Questions read like a person's; `expected_sources` is right |
| 2 | Generate 400 rollouts concurrently with DeepSeek; filter by reward | $1–3 | Concurrency holds; a sane fraction survives the filter |
| 3 | Convert trajectories to SFT samples with loss masking | $0 | **A human has read rendered samples** and confirmed masking |
| 4 | LoRA, then evaluate on the 114 questions | $0, ~2h | The pipeline ran; behaviour moved in the intended direction |

**Stage 4 is not judged by the score.** 100 questions yields perhaps 150–250
samples after filtering, and the measured noise floor on this set is ±0.03
recall. A flat score is the expected outcome and does not mean SFT failed.
What to read instead: the `answered` rate, the tool-call count, and whether
trajectories look less aimless.

---

## The prompt

```text
仓库：/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases（分支 main，工作树应干净）。
代码、注释、提交信息用英文；跟我汇报用中文，简短。

## 任务一句话
用 deepseek-reasoner 生成的轨迹，SFT 一个 qwen3.5:4b，让它学会「找到证据就作答」而不是半途放弃。
这一轮是 **100 题的试点**，目的是跑通整条流水线，不是把分数打上去。

## 运行环境（这台机器，先看这段）
- 机器：Apple M4 Max，128 GB 统一内存，磁盘剩 250 GB。本地 LoRA 微调 4B 完全跑得动，不用租 GPU
- 项目用 .venv（uv 管理，有 uv.lock）。**训练依赖不要装进它**——会让锁文件和实际环境不一致，
  下次 uv sync 就把你装的删了。训练用单独的 venv（比如 .venv-train），只读 jsonl、写模型文件
- Ollama：http://127.0.0.1:11434，已有 qwen3.5:4b / 9b / 27b、qwen3-embedding
- Qdrant：**http://127.0.0.1:6340**。CLI 默认值是 6333，那个端口没有服务——
  凡是索引或检索的命令都要显式带 --qdrant-url http://127.0.0.1:6340
- match 依赖 rg（/opt/homebrew/bin/rg）。命令因 PATH 失败就用
  PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin 重试
- 本机没有别的任务占 GPU；但不要 kill 任何不是你自己启动的进程
- 真实 vault：/Users/daboluo/ObsidianVault/MyObsidian（1,555 篇），已索引在
  .arkb/obsidian.sqlite，vault_id=obsidian
- DEEPSEEK_API_KEY 在仓库根目录 .env 里（src/arkb/config.py 的 load_env_file 会读）

## 预算
这一轮**批准的上限是 $5**。超过之前停下来问我。不要用 Anthropic 的 key（没配）。
按实测数据，100 题 × 4 采样大约 $1.2–3。

## 背景：这个系统是什么
ARKB 是一个 agentic 检索系统。一个有界的 agent 循环（src/arkb/agent/loop.py）通过五个工具
访问 Markdown 知识库：list（浏览）、match（rg 字面匹配）、search（bm25/semantic/hybrid）、
read（按引用或文件名读）、finish（交最终答案）。
预算：12 次工具 / 10 次查询 / 6 次读取 / 8,000 证据 token / 300 秒 / 8 轮。
证据引用（ev_ 前缀）绑定 source + revision，finish 只能引用已交付的证据，否则被拒。

## 先读这些
- src/arkb/agent/loop.py：SYSTEM_INSTRUCTION、run_agent 的参数、_AgentRun 每轮怎么构造请求
- src/arkb/agent/session.py：ToolSession 怎么校验工具参数、_finish 怎么校验引用
- src/arkb/agent/transports.py：make_client 按模型名选传输层；PRICES 价目表；ChatUsage 记账
- src/arkb/agent/chat_completions_client.py：DeepSeek 的传输层（OpenAI 兼容），
  **注意它接受 base_url 参数**——阶段 0 可能用得上
- evaluation/eval.py：run() 的主循环、score() 的每个指标、load_questions 的数据格式
- evaluation/README.md：指标定义和读数注意事项
- docs/search-agent-roadmap.md 的「What the measurements said」一节：五条已知结论，别重复踩

## 已有的 rollout 数据长什么样
evaluation/results/<label>/results.jsonl，每行一题，含：
- question：{id, type, question, expected_sources, reference}
- result.observation.models[]：每次模型请求的完整 request（含 messages、tools）和 response
- result.observation.tools[]：每次工具调用的 arguments、raw_result、是否交付
- result.final：{answer, status, citations}
- scores：source_recall / source_precision / answered / tool_calls / ... （score() 算出来的）
这就是完整轨迹。生成新数据时保持同样的结构，下游才能复用 score()。

## 硬约束：不能在评估集上训练
evaluation/questions.json 的 114 道题和 evaluation/notes/ 的 58 篇笔记是**唯一的测量工具**。
拿它们生成训练数据，之后所有数字都失去意义。
训练数据一律从真实 vault（/Users/daboluo/ObsidianVault/MyObsidian）生成。
跨语料（训练在 vault、测试在 58 篇开发集）是**有意为之**：它测的是「学会了检索行为」
而不是「背下了这些题」。

## 已经定好的决策，不要重新讨论
1. 学生 qwen3.5:4b，教师 deepseek-reasoner
2. 训练数据从真实 vault 生成，评估仍用现有 114 题
3. 试点规模 100 题 × 4 采样
4. 训练代码放 training/ 目录；脚本提交，数据和模型 gitignore（它们很大）
5. 奖励函数复用 evaluation/eval.py 的 score()，不要另写一套指标

## 分阶段做，每阶段做完先汇报再继续

### 阶段 0：证明能微调并服务（$0，先做这个）
拿 10 条随便造的样本 LoRA 跑几十步，然后让 **agent 循环真的调用到这个微调后的模型**。
质量完全不看，只看通路。

服务路径有几条，按你判断的稳妥程度依次试，**报告哪条成功了**：
- mlx_lm.server 起一个 OpenAI 兼容端点，用 ChatCompletionsClient(base_url=...) 接上去
  （这条最省事，但要确认它支持 tools 参数并能返回 tool_calls）
- Ollama 直接导入 safetensors（新版本的 Modelfile 支持 FROM 一个目录）
- 转 GGUF 再 ollama create（最传统，但 qwen3.5 架构的转换器支持情况要自己确认）

**这一步失败就停下来告诉我**，不要硬上——整个项目的可行性系于此。

### 阶段 1：造题（$0 用本地 27B，或花 ~$0.2 用 DeepSeek，你选并说明理由）
从 vault 抽 100 篇笔记，每篇生成 1 道题：
- 题目要像人问的，**刻意换说法**，不要照抄笔记里的原词——否则训出来的是关键词匹配练习。
  用 evaluation/questions.json 里现有的题当 few-shot 例子，那批是这个风格
- expected_sources 就是那篇笔记的 vault 相对路径（这是语料自产标签，不需要人工标注）
- 题型参考现有的 8 类，但试点阶段以 knowledge_qa / semantic_discovery 为主就够
- **反向验证**：每道题用检索引擎跑一次，目标笔记进不了前 20 就丢掉（题出得太离谱）
- 人工挑 10 道贴给我看

输出格式和 evaluation/questions.json 一致，存成 training/data/questions-pilot.json。

### 阶段 2：并发生成 rollout（$1–3）
100 题 × 4 采样（temperature 调高一点制造多样性，比如 0.7）= 400 条轨迹。
- **要并发**。evaluation/eval.py 是串行的，那是评估该有的性质，数据生成没必要继承。
  注意每个 worker 需要自己的 Runtime / SQLite 句柄 / rg 文本缓存——现在这些是共享的，
  并发直接跑会出问题。多进程或每 worker 一份，你选
- 用 score() 打分，过滤出高质量轨迹（建议 source_recall == 1.0 且 status 为 answered/partial；
  阈值你定，写清理由）
- 报告：留存率、花了多少钱（用 transports.estimate_cost）、并发数、墙钟时间

### 阶段 3：轨迹 → 训练样本（$0，最容易静默出错的一步）
把轨迹渲染成 SFT 样本。两个必须做对的地方：

1. **loss masking**：只对模型自己产出的 token 算损失（thinking、tool_calls、最终答案）。
   工具返回的观察是环境输入，**必须 mask 掉**。写错了不会报错，模型会学着自己编造工具返回值，
   要到评估时才发现
2. **train/serve 模板一致**：渲染训练样本用的 chat template，必须和阶段 0 确认的服务路径
   实际使用的模板**逐字节一致**。不一致的话训练和推理对不上，这是第二个静默杀手

验收方式只有一个：**把渲染后的样本原样打印出来，人眼检查**，贴 2 条给我看，
标出哪些 token 算损失、哪些被 mask。不要只说「跑通了」。

### 阶段 4：LoRA + 评估（$0）
- 训 4B（LoRA，超参你定并说明理由）
- 用 evaluation/eval.py 在 114 题上测，label 叫 sft-pilot-4b
- 和基线对比：evaluation/results/sft-base-4b
  （已存在：recall 0.833 / precision 0.960 / answered 0.763 / tool_calls 5.17 / 16.5 秒每题）

**判断标准不是分数。** 150–250 条样本移动不了分数，这个集合的噪声底是 ±0.03 召回。
要看的是：
- answered 率有没有往上走（基线 0.763，这是目标行为）
- tool_calls 有没有往下走（基线 5.17，瞎搜是症状）
- 轨迹读起来是不是没那么漫无目的了（挑几条贴给我）
- 有没有出现训练特有的坏味道：编造工具返回、重复调用、输出格式崩坏

分数不动是**预期结果**，如实报告即可，不要粉饰，也不要因此下「SFT 没用」的结论——
试点规模本来就不足以移动分数。

## 不要做
- 不要用 evaluation/questions.json 或 evaluation/notes/ 生成训练数据
- 不要修改 evaluation/ 下的题目、标签、指标定义
- 不要把训练依赖装进项目的 .venv
- 不要为了让分数好看而挑题、调阈值、或改评估口径
- 不要超过 $5 预算
- 不要扩大到 1,000 题——那是试点通过之后的事

## 验收
1. 项目的 make test、make lint 仍然全绿（你不该动到产品代码，但确认一下）
2. training/ 下的脚本能从头复现：造题 → rollout → 数据集 → 训练 → 评估
3. 每个阶段的通过标准都达到，阶段 0 和阶段 3 必须有人工确认的证据
4. 提交：英文信息，风格照 git log。数据和模型文件 gitignore。提交后不要 push，我来推

## 汇报
每个阶段做完先汇报再继续，不要一口气做到底。最终总结用中文，包含：
- 阶段 0 哪条服务路径成功了、为什么
- 造题的实际样例（10 道）和反向验证的淘汰率
- rollout 的留存率、实际花费、并发数、墙钟时间
- 阶段 3 渲染样本的人工检查证据（2 条，标出 mask 边界）
- 阶段 4 的对比表（answered / tool_calls / recall / precision）和几条轨迹
- 你认为扩到 1,000 题之前还需要改什么

遇到需要我决策的事先停下来问，不要自己扩大范围。
```

## What this pilot does not settle

- Whether SFT helps at a useful scale. That needs 1,000–2,000 questions and
  costs $12–30; the pilot only shows the machinery works.
- Whether the same pipeline would teach vault-specific navigation. That needs
  an evaluation set built on the real vault, which does not exist yet.
- GRPO. It needs concurrent sampling during training, which is a different
  order of compute than LoRA; revisit only if SFT shows the behaviour moves.

# Search agent roadmap (2026-09-22)

The goal is a search agent that works on the author's own Markdown knowledge
base. This document states what exists, what is missing in priority order, and
carries one self-contained prompt per task so each can be executed in a fresh
session.

## Where the system stands

Product (`src/arkb`): a bounded agent loop (`agent/loop.py`) with five tools —
`list`, `match` (literal, via `rg`), `search` (bm25 / semantic / hybrid),
`read`, `finish` — over a deterministic retrieval engine and a SQLite + Qdrant
snapshot index. Budgets: 12 tool, 10 query, 6 read calls, 8,000 evidence
tokens, 300 s, 8 turns. Evidence references bind source and revision; the final
object must cite delivered evidence.

Transports: Ollama (product default), `agent/claude_client.py`,
`agent/chat_completions_client.py` (DeepSeek). The latter two are wired only
into `evaluation/eval.py`, not into the product CLI.

Evaluation (`evaluation/`): 58 notes, 114 questions, one script
(`run` / `rescore` / `compare`). Measured 2026-09-22 on all 114:

| | deepseek-reasoner | qwen3.5:27b | qwen3.5:9b |
| --- | ---: | ---: | ---: |
| cited source recall | 0.991 | 0.973 | 0.886 |
| cited source precision | 0.795 | 0.906 | 0.944 |
| browse recall | 1.00 | 0.90 | 0.63 |
| seconds per question | 8.8 | 57.2 | 21.1 |
| whole run | 17 min | 109 min | 40 min |

Readings that drive the plan: a hosted model is both the best and the fastest
here, so the product should be able to use one; DeepSeek hit the 12-call tool
budget on 14 questions; citation precision, not recall, is the weak metric on
the strong models; `list` was used on 61 of 114 questions by DeepSeek.

Local run labels for regression comparisons (gitignored, on this machine):
`evaluation/results/v2-deepseek-reasoner`, `v2-9b-think`, `v2-27b-think`.

## Gaps in priority order

| # | Gap | Why it matters now |
| ---: | --- | --- |
| 1 | **Only a flat directory is supported.** `note_paths` iterates one directory; `source` is a bare filename; `DocumentAccess._paths` rejects any path with a separator; the exact-match text cache keys files by `path.name`. No frontmatter handling. | The author's vault has 1,603 `.md` files and **none** at the top level. The product indexes nothing there. Everything else is blocked on this. |
| 2 | **Nothing exposes the vault to an outer agent.** No MCP server. | The shortest path to daily use: Claude Code and the desktop app become the agent, ARKB stays the retrieval layer. No LLM needed inside ARKB for this mode. |
| 3 | **`ask` is one-shot and Ollama-only.** No session, no follow-ups, no hosted-model selection in the product, and no prompt caching on the hosted transports. | Measurement says a hosted model is better and faster; a search agent people use has follow-up turns, and every turn resends the whole history — uncached that is the dominant cost. |
| 4 | **No Obsidian-native navigation.** No wikilink/backlink traversal, no tag or folder filters, no date filters. `list` filters by filename only. | Links and tags are the strongest structural signal in a real vault, and they are free to index. |
| 5 | **No context engineering.** No vault map at startup, no whole-corpus bypass for small scopes, no compaction of old observations in long runs. | A 1,603-note vault needs orientation; a 20-note folder does not need retrieval at all. |
| 6 | **Citation precision and no verification step.** The final object is the model's own claim; nothing checks that each cited note supports the answer. | Measured: 0.795 precision on the strongest model, 0.58 on single-note discovery questions. |

Not worth a session yet: reranking in the agent path (code exists, switched
off), multilingual handling, writing to the vault.

## Plan

```
Task 1 (P0) real vault support ──┬── Task 2 (P1) MCP server
                                 ├── Task 3 (P1) hosted models + multi-turn
                                 ├── Task 4 (P2) links, tags, filters
                                 └── Task 5 (P2) context engineering
Task 6 (P2) citation precision and verification  (independent, any time)
```

Rules that hold for every task: `make test` and `make lint` stay green; the
evaluation must not regress (compare against the run labels above); the
evaluation corpus `evaluation/notes/` and `evaluation/questions.json` are not
edited unless the task says so; commits are English and match the repository's
history style.

---

## Task 1 prompt — real vault support (P0)

```text
仓库：/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases（分支 main，工作树应干净）。
代码、注释、提交信息用英文；跟我汇报用中文，简短。

## 背景
ARKB 是一个 agentic 检索系统：一个有界的 agent 循环（src/arkb/agent/loop.py）通过五个工具
（list / match / search / read / finish）访问一个 Markdown 知识库，下层是确定性的检索引擎
（src/arkb/retrieval/）和 SQLite + Qdrant 快照索引（src/arkb/knowledge/）。

问题：整个知识层假设「一个扁平目录里的 .md 文件」。我的真实 vault 是
/Users/daboluo/ObsidianVault/MyObsidian，1,603 篇 .md，顶层目录零篇，全部在子目录里
（00-ObsSys/、01-Journal/、02-Zettelkasten/、03-Clippings/、04-Areas/、90-Templates/、
99-Archive/、Attachments/、Excalidraw/ 等）。所以产品现在对它索引结果是空的。

## 先读这些
- src/arkb/knowledge/documents.py：note_paths（第 45 行，单层 iterdir）、load_notes、
  _load_note（source=path.name，无 frontmatter 处理）、DocumentAccess._paths（os.scandir 单层，
  且要求 Path(source).name == source）、_document_id(vault_id, path.name)
- src/arkb/retrieval/text_cache.py：scan() 和 rg_path() 用 path.name 作 key，并把文件物化到
  cache.directory/<target>/<name>——子目录同名文件会互相覆盖，这是必须修的地方
- src/arkb/knowledge/indexing.py：build_index 要求 note.source 唯一
- src/arkb/agent/session.py 与 src/arkb/agent/tools.py：read(source=...)、match(source=...)、
  search(source=...) 的校验
- docs/project-map.md：目录地图与约定

## 任务
让整条链路支持嵌套目录的真实 vault。已经定好的决策，不要重新讨论：

1. source 的规范形式是 **vault 相对 POSIX 路径**，例如 "04-Areas/Career Development/note.md"。
   反斜杠、绝对路径、盘符、".." 一律拒绝（text_cache.py 里已有这条契约的一半，照它来）。
2. 递归扫描，默认排除：任何以 "." 开头的目录（.obsidian、.trash 等）、Attachments/、
   Excalidraw/。排除规则要能配置（CLI 一个 --exclude，可重复；RuntimeConfig 或 index 参数里存下来）。
3. _load_note 处理 YAML frontmatter：如果文件以 "---" 开头，解析到下一个 "---"，把它从正文里剥掉。
   解析出的 tags / aliases / 日期字段先放进 Note 的一个 metadata 字段存起来（Task 4 会用），
   正文的 title 规则不变（第一行 "# "，否则文件名 stem）。frontmatter 解析失败不要让整次索引失败：
   跳过该文件的 metadata，正文照常处理，并在索引报告里记一笔。
4. document_id 由 vault 相对路径导出，不是文件名。
5. text_cache 的物化要避免子目录同名冲突：镜像目录结构，或用相对路径的哈希做文件名——你选一个，
   在注释里说明理由。rg 的输出到 source 的映射必须仍然精确。
6. 单个坏文件（非 UTF-8、读取失败）不要让 1,603 篇的索引整体失败：跳过并在返回的报告里列出，
   数量也要记进 manifest 或报告。

## 约束
- 扁平目录必须继续工作：evaluation/notes/ 是扁平的，evaluation 是回归基线，不要改动它。
- 不要引入新的第三方依赖来解析 frontmatter；标准库够用（YAML 子集：键值、列表即可，
  解析不了就当没有 frontmatter）。
- 不要顺手重构无关代码。

## 验收
1. make test、make lint 全绿；为嵌套目录、排除规则、frontmatter、同名冲突、坏文件各补测试
   （tests/knowledge/test_documents.py、tests/retrieval/ 下的 exact 测试、tests/agent/）。
2. 真实 vault 端到端跑通，把数字告诉我：
   .venv/bin/python -m arkb.interfaces.cli index --db .arkb/obsidian.sqlite --vault-id obsidian \
     --notes-dir /Users/daboluo/ObsidianVault/MyObsidian --offline
   记录：索引耗时、文档数、chunk 数、跳过的文件数。注意这会用本地 Ollama 做 embedding，
   1,603 篇可能要几十分钟，先用 --notes-dir 指向其中一个子目录小规模验证再跑全量。
3. 索引完后跑一个真实问题并把输出贴给我：
   .venv/bin/python -m arkb.interfaces.cli ask "<随便一个关于我笔记内容的问题>" \
     --db .arkb/obsidian.sqlite --vault-id obsidian --notes-dir /Users/daboluo/ObsidianVault/MyObsidian --trace
4. 评估不退化：
   make eval LABEL=t1-9b-think MODEL=qwen3.5:9b THINK=--think
   .venv/bin/python -m evaluation.eval compare evaluation/results/v2-9b-think evaluation/results/t1-9b-think
   逐项差异应在噪声内（几题的胜负），若有系统性下降，先解释再继续。
5. 提交：改动一次提交，英文信息，风格照仓库历史。

## 不要做
不要做 MCP、多轮会话、链接/标签工具、上下文工程、引用验证——那些是后续任务。
```

---

## Task 2 prompt — MCP server (P1, after Task 1)

```text
仓库：/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases（分支 main）。
代码、注释、提交信息用英文；跟我汇报用中文，简短。

## 背景
ARKB 是一个 Markdown 知识库的检索层：确定性的检索引擎（bm25 / semantic / hybrid）、
基于 rg 的字面匹配、有边界的读取，外加一个自己的 agent 循环。工具契约在
src/arkb/agent/tools.py（TOOL_DEFINITIONS：list / match / search / read / finish），
执行边界在 src/arkb/agent/session.py，组合在 src/arkb/runtime.py。

我想让 Claude Code 和 Claude 桌面版直接把我的 vault 当工具用。这种模式下外层的 Claude 才是 agent，
ARKB 只提供工具，内部不需要任何 LLM。

## 先读这些
- src/arkb/agent/tools.py（TOOL_DEFINITIONS 的 JSON schema、AgentTools 的方法）
- src/arkb/agent/session.py（ToolSession.invoke 的参数校验、证据引用的绑定）
- src/arkb/runtime.py（Runtime._snapshot、retrieval_engine、agent_tools、ask）
- src/arkb/interfaces/cli.py（五个命令的构造方式）
- docs/project-map.md

## 任务
新增一个只读的 MCP stdio server，把 list / match / search / read 暴露出去（不暴露 finish）。

已经定好的决策：
1. 入口是新的 CLI 子命令 `arkb mcp`，参数沿用现有风格：--db、--vault-id、--notes-dir、
   --qdrant-url、--offline 等；stdio 传输。
2. 不要复用 ToolSession 的 ev_* 引用机制。那套引用是为「一次 agent 运行内绑定证据」设计的，
   MCP 客户端是另一个 agent，跨请求无状态。改为每条结果直接带 source（vault 相对路径）、
   document_revision、start_char/end_char、title、content。read 可以直接按 source 读。
3. 工具的 name / description / 参数 schema 尽量复用 TOOL_DEFINITIONS，避免两套描述漂移；
   read 的 ref 参数去掉，只保留 source + expand（section/document）。
4. 只读：不提供任何写入、删除、索引重建的工具。
5. 依赖：用官方 `mcp` Python SDK，加成 pyproject 的可选 extra（例如 [project.optional-dependencies] 的 mcp），
   不要塞进主依赖。用 uv lock / uv sync --extra mcp 安装，别破坏现有的 claude / rerank extra。
6. 错误按 MCP 的错误响应返回，不要让 server 因为一次坏参数退出。

## 验收
1. make test、make lint 全绿。测试用脚本化的 stdio 客户端（或直接调用 server 的请求处理函数）
   覆盖：工具列表、一次 search、一次 match、一次 read、一个非法参数、一个不存在的 source。
   不要在测试里起真实的 Qdrant / Ollama。
2. 手动验证并把结果贴给我：用 Task 1 建好的 .arkb/obsidian.sqlite 起 server，
   在 Claude Code 里用 `claude mcp add` 接上（给我具体命令），列出工具，做一次 search 和一次 read。
3. README 或 docs 里加一小节：怎么起 server、怎么接到 Claude Code/桌面版、暴露了哪些工具、
   为什么是只读的。
4. 提交：英文信息，风格照仓库历史。

## 不要做
不要改 agent 循环、不要动 evaluation、不要加写入能力、不要顺手重构 tools.py 之外的东西。
```

---

## Task 3 prompt — hosted models and multi-turn (P1, after Task 1)

```text
仓库：/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases（分支 main）。
代码、注释、提交信息用英文；跟我汇报用中文，简短。

## 背景
ARKB 的 agent 循环（src/arkb/agent/loop.py）通过一个 chat(**request) 契约调用模型，请求形状是
Ollama 的：messages（assistant 带 content/thinking/tool_calls，工具结果是 role=tool）、tools、
format（终结用的 JSON schema）、think。已经有三个传输层实现了这个契约：
- Ollama：evaluation/eval.py 里的 OllamaClient（产品侧走 ollama SDK，见 runtime.py）
- src/arkb/agent/claude_client.py（Claude Messages API）
- src/arkb/agent/chat_completions_client.py（OpenAI 兼容，默认 DeepSeek）

两个问题：
1. 后两个传输层只接到了 evaluation/eval.py，产品的 arkb ask 只能用本地 Ollama。
   实测 deepseek-reasoner 在 114 题上的引用召回 0.991、每题 8.8 秒，本地 27B 是 0.973、57 秒——
   产品应该能用托管模型。
2. arkb ask 是一次性的：问一句答一句，没有会话，没有追问。

## 先读这些
- src/arkb/runtime.py：ask()、run_agent()
- src/arkb/agent/loop.py：_AgentRun 的状态（AgentState.messages / turn）、run_agent 的参数
- src/arkb/agent/claude_client.py 和 chat_completions_client.py（两者的 chat 契约与 usage 记录）
- evaluation/eval.py 的 make_client()（现在的选择逻辑：claude-* / deepseek-* / 其余走 Ollama）
- src/arkb/config.py 的 load_env_file（.env 读取，已存在）
- src/arkb/interfaces/cli.py 的 ask 子命令

## 任务
A. 把传输层选择搬进产品：
   1. 新建 src/arkb/agent/transports.py，提供 make_client(model, *, options, think, effort='high')，
      逻辑就是 evaluation/eval.py::make_client 现在那套（claude-* / deepseek-* / Ollama），
      让 evaluation/eval.py 改成 import 它，不要留两份。
   2. CLI 的 ask 与新的 chat 命令启动时读仓库根目录 .env（用 config.load_env_file），
      --generation-model 支持 claude-* 和 deepseek-* 名字；缺 key 时报一句清楚的错。
   3. 每次运行把 usage（prompt/output token、请求数）汇总出来，--trace 时打印；
      托管模型另外打印一行估算成本（用你查到的官方价格，写进常量并注明查询日期）。
   4. **prompt caching**：agent 循环每一轮都把整段历史重发一次，多轮会话更是如此，不缓存的话
      输入 token 是主要成本（实测 114 题一次跑掉 2,245k prompt token）。
      - Claude（claude_client.py）：请求的前缀顺序是 tools → system → messages，给稳定前缀打
        cache_control 断点（system 和 tools 各一个；多轮时在历史的边界再打一个），
        最多 4 个断点。变动的内容必须排在最后一个断点之后。
        ClaudeClient 的 usage 已经记录了 cache_read_input_tokens / cache_creation_input_tokens，
        直接报告它们。动手前先读一下 Anthropic 的 prompt caching 文档确认断点语义，不要凭记忆写。
      - DeepSeek（chat_completions_client.py）：服务端自动做硬盘缓存，不需要显式标记；
        它的 usage 里有 prompt_cache_hit_tokens / prompt_cache_miss_tokens，
        现在是整体透传的，把它们纳入成本估算即可。
      - Ollama 没有这个概念，跳过。

B. 多轮会话：
   1. 新增 arkb chat 子命令：一个 REPL，保留跨轮的 AgentState.messages，
      每轮跑一次完整的 agent 循环（预算按轮独立计算）。
   2. 追问不需要单独的查询改写模块——模型自己能从历史里解析指代；你要做的是把历史正确地带进下一轮，
      并保证工具结果和证据引用在跨轮之后仍然有效（注意 session.py 的引用是按 run 的前缀生成的，
      跨轮要么沿用同一个 ToolSession，要么明确说明旧引用失效并让模型重新取证——选一个并在注释里说明）。
   3. 历史变长时要有上限：超过阈值就丢弃最早的工具观察（保留它们的 source 摘要），
      不要让上下文无声地溢出。阈值可配置。
   4. 命令：空行退出，/reset 清空会话，/trace 切换轨迹打印。

## 约束
- 不要改 agent 循环的工具契约或预算语义。
- 不要把 anthropic / httpx 之外的新依赖引进来。
- evaluation 的行为不能变（它现在依赖 make_client 的选择逻辑）。

## 验收
1. make test、make lint 全绿；给 transports.make_client 和 chat 的多轮状态各补测试
   （用假 client，不要真的调 API）。
2. 手动验证并贴给我：
   - arkb ask "..." --generation-model deepseek-reasoner（用 Task 1 建的 obsidian 索引）
   - arkb chat 里连问三轮，第三轮用代词指代第一轮的内容，看它是否答对
   - 两者的 usage / 成本输出
   - **缓存确实命中**：多轮会话从第二轮起，Claude 的 cache_read_input_tokens 应该大于零，
     DeepSeek 的 prompt_cache_hit_tokens 应该大于零；把这几行贴给我。
     如果一直是零，说明前缀被什么东西打破了（时间戳、变动的工具顺序、未排序的 JSON），
     先找出来再说「做完了」。
3. 评估不退化：make eval LABEL=t3-9b-think MODEL=qwen3.5:9b THINK=--think，与
   evaluation/results/v2-9b-think 对比。
4. 提交：英文信息，可以分成「transports」和「chat」两次。

## 不要做
不要实现 MCP、不要碰链接/标签工具、不要改引用验证逻辑。
```

---

## Task 4 prompt — links, tags and filters (P2, after Task 1)

```text
仓库：/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases（分支 main）。
代码、注释、提交信息用英文；跟我汇报用中文，简短。

## 背景
ARKB 的 agent 有五个工具：list（浏览文件名/标题/小节标题）、match（rg 字面匹配）、
search（bm25/semantic/hybrid）、read、finish。在真实 Obsidian vault 上，最强的结构信号——
[[wikilink]] 链接图、标签、文件夹层级、修改时间——完全没有被用起来。list 现在只能按文件名过滤。

前置：Task 1（嵌套目录支持）必须已经完成，source 已经是 vault 相对路径，
_load_note 已经解析 frontmatter 并把 tags/aliases 存进了 Note 的 metadata。

## 先读这些
- src/arkb/knowledge/documents.py（Task 1 之后的 _load_note、DocumentAccess.list、_paths）
- src/arkb/agent/tools.py（TOOL_DEFINITIONS、AgentTools.list/match/search/read）
- src/arkb/agent/session.py（invoke 的分发与参数校验）
- src/arkb/knowledge/sqlite.py 与 indexing.py（快照里能存什么）
- src/arkb/agent/loop.py 顶部的 SYSTEM_INSTRUCTION（工具的用法说明写在这里）

## 任务
1. 索引时解析并存下每篇笔记的出链：[[wikilink]]（含 [[note|alias]] 与 [[note#heading]]）和
   Markdown 链接里指向库内 .md 的部分；解析失败或指向库外的忽略。存进 SQLite 快照
   （新表或已有表的新列，跟现有 schema 演进方式保持一致）。反链由出链反查得到，不单独存。
2. 新工具 links(source, direction='both'|'out'|'in', limit)：返回该笔记的出链与反链，
   每条带 source、title，以及链接出现处的一小段上下文（够判断为什么链过去即可）。
   和 list 一样，links 的返回不是证据：不产生证据引用、不计入证据预算、算一次工具调用但不算 query/read。
   （list 现在就是这么做的，照它实现。）
3. list 扩展过滤条件：
   - pattern 现在只匹配文件名，改成匹配 vault 相对路径（这样 "04-Areas/*" 能用）
   - tag：frontmatter 的 tags 与正文里的 #tag，任一命中即可
   - modified_after / modified_before：ISO 日期，按文件 mtime
   多个条件是「与」的关系。返回里补上每篇的 tags 和 mtime。
4. 在 SYSTEM_INSTRUCTION 里加一句说明 links 的用途（照现有几句的写法，一句，不要写长）。

## 约束
- 不要改 evaluation/notes/ 和 evaluation/questions.json：那是回归基线，改了就没法和历史对比。
  links/tags 的功能用 tests/ 下的新 fixture 覆盖（一个带 wikilink、标签、子目录的小语料）。
- 不要引入 Markdown 解析的第三方依赖，正则够用；把你处理和不处理的链接形式写进 docstring。
- 不要改 search / match / read 的既有行为。

## 验收
1. make test、make lint 全绿；新 fixture 覆盖：出链、反链、别名链接、带 heading 的链接、
   指向库外的链接被忽略、标签过滤、路径过滤、日期过滤。
2. 在真实 vault（Task 1 建好的 obsidian 索引）上手动验证并贴给我：
   - 一次 links 调用的输出（挑一篇链接多的笔记）
   - 一次带 tag 过滤的 list
   - 一个需要顺链接才能答的问题：arkb ask "..." --trace，看它是否真的用了 links
3. 评估不退化：make eval LABEL=t4-9b-think MODEL=qwen3.5:9b THINK=--think，与
   evaluation/results/v2-9b-think 对比。多一个工具会多占 prompt token，这是预期的；
   但引用召回/精确率不应系统性下降。
4. 提交：英文信息，风格照仓库历史。

## 不要做
不要实现上下文工程（vault map、小库旁路、压缩）、不要碰引用验证、不要做 MCP。
```

---

## Task 5 prompt — context engineering (P2, after Task 1)

```text
仓库：/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases（分支 main）。
代码、注释、提交信息用英文；跟我汇报用中文，简短。

## 背景
agent 现在每次都是「冷启动」：不知道库长什么样，不管库有多小都要先检索，长轨迹里早期的工具观察
一直占着上下文。在 1,603 篇的真实 vault 上需要先有方向感；在一个 20 篇的文件夹里根本不需要检索。

前置：Task 1（嵌套目录支持）必须已经完成。

## 先读这些
- src/arkb/agent/loop.py：SYSTEM_INSTRUCTION、_AgentRun 的每轮请求构造、
  _search_stalled 与 _SEARCH_STALLED_INSTRUCTION（已有的一个「运行时提示」先例）
- src/arkb/agent/observation.py：预算与证据计量（remaining_evidence、end_tool 的交付逻辑）
- src/arkb/runtime.py：ask() 怎么组装 tools 与 engine
- src/arkb/knowledge/documents.py：DocumentAccess.list（Task 1 之后的版本）

## 任务
三件事，各自可以独立开关，默认值要在提交信息里说明理由：

1. **库地图**：运行开始时，如果库里存在一篇「布局说明」笔记（默认候选名可配置，
   我的 vault 里是 00-ObsSys/Vault Layout.md），把它的正文作为一条额外的 system 消息放进对话开头，
   并注明这是库的结构说明、不是证据。找不到就跳过，不要报错。
   配置项：Runtime/CLI 一个 --map-note（可重复或逗号分隔），默认空。

2. **小范围旁路**：运行开始时统计当前范围（整库，或调用方给定的路径前缀）的正文 token 数；
   如果低于阈值（默认给一个你测过的值，比如 20,000），就跳过检索，直接把全部正文作为证据交付，
   让模型一次作答。要点：
   - 交付仍然走 observer 的证据计量，不能绕过预算；超过 8,000 证据 token 的部分按现有规则处理
   - 这条路径必须仍然产生可引用的证据引用（引用绑定 source + revision），否则 finish 会失败
   - 记录在轨迹里：这次运行走的是旁路
   配置项：--small-scope-tokens，0 表示关闭。

3. **长轨迹压缩**：当对话消息的 token 数超过阈值（默认给一个值，并说明怎么估的），
   把最早的工具结果替换成一行摘要（保留 source 与条数，丢掉正文），保留证据引用表不变——
   被压缩掉的引用仍然可以被 finish 引用，因为引用表在 session 里而不在消息里。确认这一点后再动手；
   如果实际上做不到，就改成「压缩后同时把这些引用标记为不可引用」，并在注释里说明。
   配置项：--history-tokens，0 表示关闭。

## 约束
- 三个功能默认状态由你决定，但必须能一键关掉，且评估要分别测量开与关。
- 不要改工具契约、不要改预算语义、不要动 evaluation/notes/ 与 questions.json。
- 不要为了让评估变好而调整题目或标签。

## 验收
1. make test、make lint 全绿；三件事各有测试（用假 client 和临时目录，不起真实服务）。
2. 评估：至少跑两组并把对比贴给我
   make eval LABEL=t5-off-9b MODEL=qwen3.5:9b THINK=--think   （三个功能全关）
   make eval LABEL=t5-on-9b  MODEL=qwen3.5:9b THINK=--think   （按你选的默认值打开）
   .venv/bin/python -m evaluation.eval compare evaluation/results/t5-off-9b evaluation/results/t5-on-9b
   注意 evaluation/notes/ 只有 58 篇、约 84k 字符，很可能整体落进「小范围旁路」——
   如果是，就分别报告「阈值低于语料」和「阈值高于语料」两种设置，不要只报一个数。
3. 真实 vault 上手动验证库地图与压缩：arkb ask "..." --trace，贴出关键片段。
4. 提交：英文信息，三件事可以分三次提交。

## 不要做
不要做 MCP、多轮会话、链接工具、引用验证。
```

---

## Task 6 prompt — citation precision and verification (P2, independent)

```text
仓库：/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases（分支 main）。
代码、注释、提交信息用英文；跟我汇报用中文，简短。

## 背景
2026-09-22 在 114 题上实测：

| | deepseek-reasoner | qwen3.5:27b | qwen3.5:9b |
| 引用召回 | 0.991 | 0.973 | 0.886 |
| 引用精确率 | 0.795 | 0.906 | 0.944 |
| semantic_discovery 精确率 | 0.58 | 0.69 | 0.86 |

强模型的问题是「引得太多」：一篇就够的题引三四篇。产品现在对引用没有任何事后检查——
finish 里模型说引哪几篇就是哪几篇。

运行记录在本机（gitignored）：evaluation/results/v2-deepseek-reasoner、v2-9b-think、v2-27b-think。

## 先读这些
- src/arkb/agent/loop.py：SYSTEM_INSTRUCTION、_FINAL_INSTRUCTION、finalize()
- src/arkb/agent/session.py：_finish()（引用校验：必须是已交付的引用、不能重复）
- src/arkb/agent/tools.py：finish 的 schema 与描述
- evaluation/eval.py：score() 里的 source_precision / source_recall / premise_flagged
- evaluation/README.md：指标定义与读数注意事项

## 任务
A. 两个候选改动，各自独立可开关，用评估分别测量：
   1. **提示约束**：在 _FINAL_INSTRUCTION 或 finish 的工具描述里，明确要求只引用答案实际依赖的笔记
      （「每条引用都应对应答案里的一个具体断言；不要因为一篇笔记相关就引用它」），
      措辞照现有几句的克制风格写，一到两句。
   2. **验证步**：finish 之后、返回之前，对每条引用做一次廉价检查——把答案与该条引用的正文一起
      送一次模型请求，问「这条证据是否支持答案里的某个具体断言」，不支持的引用剔除。
      要点：这多花一次或多次模型请求，必须计入 observer 的账（models 列表与 usage），
      不能悄悄多花钱；剔除后如果一条引用都不剩，状态要降级（answered → insufficient_evidence），
      不要返回一个无引用的 answered。做成可开关，默认状态由测量结果决定。

B. 评估侧一个小修正：false_premise 这个题型现在的判定是「状态必须是 partial 或 insufficient_evidence」
   （evaluation/eval.py 的 premise_flagged）。实测三个模型都有 7/10 标成 answered，但答案内容是对的
   （明确指出前提不成立）且 10/10 引用了能反驳前提的笔记。把判定改成「引用了期望的那篇笔记」
   （即 source_recall == 1），并用 rescore 重算已有的三个运行，不要重跑。
   在 evaluation/README.md 里把这条改动和理由写清楚。

## 约束
- 不要改 evaluation/notes/ 和 evaluation/questions.json 的题目与标签。
- 不要为了分数好看而挑题、挑模型、或调整阈值到刚好通过。
- 验证步不要变成第二个 agent 循环：它只做「这条引用支不支持答案」这一件事。

## 验收
1. make test、make lint 全绿；提示改动与验证步各有测试（假 client，检查请求数、剔除逻辑、
   全部剔除时的状态降级）。
2. 至少四组评估，把对比表贴给我：
   - 基线：evaluation/results/v2-deepseek-reasoner、v2-9b-think（已存在，rescore 后使用）
   - 只开提示约束：deepseek 与 9b 各一次
   - 开验证步：deepseek 与 9b 各一次
   用 .venv/bin/python -m evaluation.eval compare 两两对比。
   判断标准：**精确率上升且召回不下降**才算有效；如果召回掉了，如实说，不要只报精确率。
   同时报告成本变化（model_requests、prompt_tokens、elapsed_s）。
3. 根据测量结果决定两个功能的默认状态，把理由写进提交信息。
4. 提交：英文信息，可以分成「scoring 修正」和「引用约束/验证」两次。

## 不要做
不要做 MCP、多轮会话、链接工具、上下文工程、嵌套目录支持。
```

---

## What is deliberately not in this plan

- Rewriting the agent loop or the tool contract. Both hold up; the measured
  problems are elsewhere.
- Reranking in the agent path: the code exists and is switched off; turning it
  on is a one-line experiment once something else is stable.
- Writing to the vault. The knowledge base stays read-only until the read path
  is good enough to be used daily.
- **Cross-session memory** (remembering preferences, or what was asked before).
  An earlier version of this list put it beside multi-turn conversation. That
  was wrong: without daily use on the real vault there is no evidence about
  what is worth remembering, and a memory layer built on guesses is harder to
  remove than to add. Revisit after Tasks 1 to 3 have produced real usage.
- Any new registered study, benchmark or statistics machinery. The 114-question
  set plus per-question wins and losses is the instrument.

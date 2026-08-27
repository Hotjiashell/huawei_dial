# ClarQ 部署模型评估

该目录提供从测试集跑对话轨迹、调用训练同款检索器、计算指标到生成报告的完整流程。
评估对象是已部署为 OpenAI-compatible 服务的训练后策略模型。

## 流程

每个测试样本先用原始问题执行一次 baseline 检索，再让策略模型在
`clarify_user`、`search_case` 和 `Complete` 之间决策。追问由训练时同款 Qwen3-32B
模拟用户回答，搜索直接复用
`workspace/verl_dial-main-h800/examples/clarq_grpo/retriever.py` 的 `CaseRetriever`。
轨迹结束后，先检查最后一次 Agent 检索的可配置 Top-K 是否命中标准案例 title（默认 Top-3）。
命中则直接成功；未命中时，独立 Success Judge 会比较原始问题、标准案例的 title/content 和
最终 Top-K 案例的 title/content，判断这些召回案例能否回答原始问题。没有得到非空的 Agent
检索结果则直接失败，不调用 Success Judge。之后还可选调用独立的轨迹质量 Judge，最后生成
总体及分领域指标。

测试数据默认为 `workspace/ClarQ/profile_split/test`，共 800 条，覆盖 `electronics`、
`money`、`superuser` 和 `travel` 四个领域。测试样本中的 `case_id` 会在评估启动时从
`workspace/ClarQ/case_answers_with_title.json` 解析为标准 `title` 和 `answer`；Recall/MRR
只比较 title，不直接比较 case_id。正式指标定义见 [metric.md](metric.md)。

## 环境准备

评估程序要求 Python 3.9+。模型/检索服务所在的训练环境通常已包含 `requests`；
独立环境可执行：

```bash
python3 -m pip install -r workspace/eval/requirements.txt
cp workspace/eval/config.example.env workspace/eval/.env
```

编辑 `.env` 中的模型名、服务地址和认证信息。`.env` 已被忽略，不会进入版本控制。
运行产物的 `run_config.json` 也不会保存 API Key、Elasticsearch 密码或 URL 查询参数。

案例标题文档可以通过 `CASE_DOCUMENT_PATH` 或命令行 `--case-document` 指定。文档可以是
`[{"case_id": "case0001", "title": "...", "answer": "..."}, ...]` 数组，也可以将
`answer` 写为 `content` 或 `text`。每个测试样本的 case_id 必须能解析到非空 title 和内容，
因为未命中时 Success Judge 会使用标准案例内容作为参考。这个文档不是 Elasticsearch index，
也不替代 `--elasticsearch-index`。

必须可访问以下组件：

- 训练后策略模型服务；
- Qwen3-32B 用户模拟器服务，用于回答追问；
- 必需的 Success Judge 服务。可通过 `SUCCESS_JUDGE_*` 单独配置；未配置时依次复用
  `JUDGE_*`、用户模拟器服务；
- Elasticsearch 与 embedding 服务；
- 可选的独立轨迹质量 Judge。未配置时默认复用用户模拟器，使用 `--skip-judge` 可关闭；
  该参数不会关闭正式 Success Judge。

### 用户模拟器模型模式

`--model-mode` 只影响用户模拟器（包括随机用户模拟器的额外调用），用于按模型家族关闭
thinking；默认是 `qwen3_5`：

- `qwen3_5`：请求 `/chat/completions`，并在请求体中传入
  `chat_template_kwargs: {"enable_thinking": false}`；
- `qwen3`：本地使用 Qwen3 tokenizer 执行
  `tokenizer.apply_chat_template(messages, enable_thinking=False)`，再把渲染后的 prompt 请求到
  `/completions`。该模式需要安装 `transformers`，并能从 `--simulator-tokenizer-path`（若未设置则
  使用 `--simulator-model`）加载 tokenizer。

例如，使用本地 Qwen3 tokenizer：

```bash
workspace/eval/run_evaluation.sh \
  --model-mode qwen3 \
  --simulator-tokenizer-path /models/Qwen3-32B \
  --output-dir workspace/eval/outputs/qwen3-simulator-smoke
```

### 策略模型模式

`--policy-model-mode` 独立控制策略模型的 thinking 和请求协议，默认是 `qwen3_5`。它不影响
用户模拟器的 `--model-mode`：

- `qwen3_5`：请求 `/chat/completions`，带标准 OpenAI `tools`、`tool_choice=auto`、
  `parallel_tool_calls=false`，并传入
  `chat_template_kwargs: {"enable_thinking": <policy-enable-thinking>}`；
- `qwen3`：本地 Qwen3 tokenizer 通过
  `tokenizer.apply_chat_template(messages, tools=TOOLS, enable_thinking=<policy-enable-thinking>)`
  渲染消息和工具定义后，请求 `/completions`。该模式需要安装 `transformers`，并能从
  `--policy-tokenizer-path`（若未设置则使用 `--policy-model`）加载 tokenizer。

例如，策略模型和用户模拟器都使用本地 Qwen3 tokenizer：

```bash
workspace/eval/run_evaluation.sh \
  --policy-model-mode qwen3 \
  --policy-tokenizer-path /models/Qwen3-Policy \
  --model-mode qwen3 \
  --simulator-tokenizer-path /models/Qwen3-32B \
  --output-dir workspace/eval/outputs/qwen3-policy-and-simulator-smoke
```

## 运行

先做连通性检查。用户模拟器不会访问 `/models`：程序会对它发起一次真实的、不会写入评测
轨迹的澄清请求，验证当前 `--model-mode`、无 thinking 配置和返回协议都可用。策略模型、
独立 Success Judge 和可选轨迹 Judge 仍会检查 `/models`；若其中任一服务与用户模拟器完全复用
同一个 endpoint/model/key，则也会跳过 `/models`，以该次用户模拟器推理探针作为连通性检查。
此外，程序会用测试集第一条问题执行一次真实检索探针。以上检查都不会生成评估产物：

```bash
workspace/eval/run_evaluation.sh --check-only
```

建议先跑小样本验证工具调用协议与吞吐：

```bash
workspace/eval/run_evaluation.sh \
  --limit 20 \
  --workers 4 \
  --success-top-k 3 \
  --output-dir workspace/eval/outputs/smoke
```

确认后运行完整测试集：

```bash
workspace/eval/run_evaluation.sh \
  --workers 8 \
  --output-dir workspace/eval/outputs/checkpoint-1000
```

不指定 `--output-dir` 时会创建时间戳目录。已有评估文件的目录不会被覆盖。并发数应根据
策略、模拟器、embedding 和 Elasticsearch 中最弱服务的吞吐设置；比较耗时指标时应固定
并发数。

只跑某些领域或一段样本：

```bash
workspace/eval/run_evaluation.sh \
  --domains electronics,travel \
  --offset 0 \
  --limit 100 \
  --output-dir workspace/eval/outputs/subset
```

### 用户模拟器模式

默认的 `grounded` 模式与训练期口径一致：模型只能从直接回答追问的 `known_info`、`UNKNOWN`
或 `INVALID_QUESTION` 中选择。使用随机模式可评估策略在多样用户表达下的稳健性：

```bash
workspace/eval/run_evaluation.sh \
  --user-simulator random \
  --random-user-simulator-seed 42 \
  --output-dir workspace/eval/outputs/random-user-smoke
```

随机模式针对每个“样本 + 追问”以稳定随机种子独立采样，因此改变 worker 数量不会改变采样的
行为类别。默认采样规则为：

- 16%：不回答 Agent 的追问，只用短句改述原始用户问题；
- 其余情况先由训练同款受约束选择器判断该追问是否存在直接对应的 `known_info`；
- 若存在对应事实，79% 的概率额外调用一次模型，把这条 `known_info` 压缩成回答追问的短句；
- 若不存在对应事实，且启用主动反馈选项，47% 的概率额外调用一次模型，从所有 `known_info`
  中随机选一条主动反馈；否则返回 `I don't know.`。

选项 `--random-user-rephrase-probability`、`--random-user-compress-known-probability`、
`--random-user-proactive-known-probability` 和
`--[no-]random-user-proactive-known-on-unknown` 可调整这些行为。每个追问事件会在
`trajectories.jsonl` 记录 `user_simulator_behavior`；压缩回答和主动反馈仍标记为
`known_info`，因此不会被误计为未知回答。上述命令行参数也可通过 `config.example.env` 中的
`EVAL_RANDOM_USER_*` 环境变量设置。

## 断点续跑与离线重算

轨迹按完成顺序实时追加并 `fsync` 到 `trajectories.jsonl`。中断后使用原参数和原目录：

```bash
workspace/eval/run_evaluation.sh \
  --resume \
  --output-dir workspace/eval/outputs/checkpoint-1000
```

续跑会跳过成功样本，并重试之前的基础设施失败。若数据、模型、检索或轨迹参数与原运行
不一致，程序会拒绝混写。`errors.jsonl` 保留历史错误用于排查，最终聚合按每个
`sample_id` 的最新轨迹计算。

已经有符合当前 schema 的轨迹时，可以改变 Recall K 后离线重算，不调用任何服务：

```bash
python3 workspace/eval/evaluate.py \
  --aggregate-only \
  --output-dir workspace/eval/outputs/checkpoint-1000 \
  --k-values 1,3,5
```

程序会从原运行配置校验已保存的 `top_k` 和 `success_top_k`，拒绝计算超出轨迹保留深度的 K；
`--max-turns` 未指定时也会沿用原运行配置。正常在线运行出现基础设施失败时退出码为 2，质量指标会
排除这些样本；临时实验可用 `--allow-infrastructure-failures` 允许零退出码，但正式报告应先
恢复失败样本。缺少 `success_judgment` 或 `target_case_title` 的旧版轨迹无法离线推导当前
Success/Recall。早于 schema 2.4 的 LLM Success Judge 成功记录还缺少 Judge 选中的案例标题和
标准案例内容，无法按当前协议聚合或重建 `judge_success.json`，需要重新执行评估。更换案例文档后
也必须使用新的输出目录，或重新完整评估。

## 产物

- `trajectories.jsonl`：完整对话、工具调用、检索列表、停止原因、
  标准 `target_case_title` / `target_case_content`、结构化 `success_judgment`、耗时和可选轨迹 Judge 结果；
- `errors.jsonl`：有基础设施失败时生成的错误历史；
- `metrics.json`：机器可读总体、分领域指标及 Wilson 95% 置信区间；
- `report.md`：中文汇总报告；
- `judge_success.json`：仅收录 LLM Success Judge 判定成功的样本；每项包含原始问题、
  标准案例 title/content、Judge 选中的可回答案例 title/content 和 Judge 理由。标准案例
  直接命中的成功不会调用 LLM，因此不在该文件中；
- `run_config.json`：不含凭据的可复现配置。

策略模型默认关闭 Qwen thinking，具体请求格式由 `--policy-model-mode` 决定。程序同时兼容标准
`message.tool_calls` 与正文中的 Hermes `<tool_call>` 格式。用户模拟器不要求支持 `/models`。若策略模型、
独立 Success Judge 或独立轨迹 Judge 也不支持 `/models`，可在确认其他组件可用后添加
`--skip-preflight`；这不会跳过实际评估中的错误检测。

## 测试

无网络测试不会访问任何模型、embedding 或 Elasticsearch：

```bash
PYTHONPYCACHEPREFIX=/tmp/clarq_eval_pycache \
  python3 -m unittest discover -s workspace/eval/tests -v
PYTHONPYCACHEPREFIX=/tmp/clarq_eval_pycache \
  python3 -m compileall -q workspace/eval
```

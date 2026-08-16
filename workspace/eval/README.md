# ClarQ 部署模型评估

该目录提供从测试集跑对话轨迹、调用训练同款检索器、计算指标到生成报告的完整流程。
评估对象是已部署为 OpenAI-compatible Chat Completions 服务的训练后策略模型。

## 流程

每个测试样本先用原始问题执行一次 baseline 检索，再让策略模型在
`clarify_user`、`search_case` 和 `Complete` 之间决策。追问由训练时同款 Qwen3-32B
模拟用户回答，搜索直接复用
`workspace/verl_dial-main-h800/examples/clarq_grpo/retriever.py` 的 `CaseRetriever`。
轨迹结束后使用训练时完全相同的 case-judge 提示词，只对最后一次 Agent 检索调用一次
模拟器并取得 `<SATISFIED_DONE>/<FAILED_DONE>`，以此计算 Success；没有得到非空的 Agent
检索结果则直接失败。之后还可选调用独立的轨迹质量 Judge，最后生成总体及分领域指标。

测试数据默认为 `workspace/ClarQ/profile_split/test`，共 800 条，覆盖 `electronics`、
`money`、`superuser` 和 `travel` 四个领域。正式指标定义见 [metric.md](metric.md)。

## 环境准备

评估程序要求 Python 3.9+。模型/检索服务所在的训练环境通常已包含 `requests`；
独立环境可执行：

```bash
python3 -m pip install -r workspace/eval/requirements.txt
cp workspace/eval/config.example.env workspace/eval/.env
```

编辑 `.env` 中的模型名、服务地址和认证信息。`.env` 已被忽略，不会进入版本控制。
运行产物的 `run_config.json` 也不会保存 API Key、Elasticsearch 密码或 URL 查询参数。

必须可访问以下组件：

- 训练后策略模型服务；
- Qwen3-32B 用户模拟器服务，用于回答追问和必需的终局满意度判定；
- Elasticsearch 与 embedding 服务；
- 可选的独立轨迹质量 Judge。未配置时默认复用用户模拟器，使用 `--skip-judge` 可关闭；
  该参数不会关闭正式 Success 所需的终局满意度判定。

## 运行

先做连通性检查。该命令会检查各模型的 `/models`，并用测试集第一条问题执行一次
真实检索探针，但不会生成评估产物：

```bash
workspace/eval/run_evaluation.sh --check-only
```

建议先跑小样本验证工具调用协议与吞吐：

```bash
workspace/eval/run_evaluation.sh \
  --limit 20 \
  --workers 4 \
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

程序会从原运行配置校验已保存的 `top_k`，拒绝计算超出轨迹保留深度的 K；
`--max-turns` 未指定时也会沿用原运行配置。正常在线运行出现基础设施失败时退出码为 2，质量指标会
排除这些样本；临时实验可用 `--allow-infrastructure-failures` 允许零退出码，但正式报告应先
恢复失败样本。旧版轨迹没有保存 `simulator_feedback`，无法离线推导训练口径的 Success，
聚合时会明确报错，需要重新执行评估。

## 产物

- `trajectories.jsonl`：完整对话、工具调用、检索列表、停止原因、
  `<SATISFIED_DONE>/<FAILED_DONE>`、耗时和可选 Judge 结果；
- `errors.jsonl`：有基础设施失败时生成的错误历史；
- `metrics.json`：机器可读总体、分领域指标及 Wilson 95% 置信区间；
- `report.md`：中文汇总报告；
- `run_config.json`：不含凭据的可复现配置。

策略请求默认关闭 Qwen thinking，并发送标准 OpenAI tools、`tool_choice=auto` 和
`parallel_tool_calls=false`。程序同时兼容标准 `message.tool_calls` 与正文中的 Hermes
`<tool_call>` 格式。若服务不支持 `/models`，可在确认其他组件可用后添加
`--skip-preflight`；这不会跳过实际评估中的错误检测。

## 测试

无网络测试不会访问任何模型、embedding 或 Elasticsearch：

```bash
PYTHONPYCACHEPREFIX=/tmp/clarq_eval_pycache \
  python3 -m unittest discover -s workspace/eval/tests -v
PYTHONPYCACHEPREFIX=/tmp/clarq_eval_pycache \
  python3 -m compileall -q workspace/eval
```

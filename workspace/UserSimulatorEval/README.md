# 澄清反问用户回答评估

## 用户模拟器类人性测试集与评测

新增的两段式流程用于科学比较真实用户与用户模拟器的澄清回复行为：

```text
真实 chat_content
  → build_test_set.py
  → user_simulator_test_set.jsonl
  → evaluate_user_simulator.py --mode both
  → 真实用户 / 模拟器的并列指标与逐条审计结果
```

### 1. 构建测试集

[`build_test_set.py`](build_test_set.py) 从 `explore/diveUserData/dialog.json` 抽取一条条评测样本。
输入支持 JSON 数组、正常 JSONL，以及历史导出中出现的多个 JSON 对象直接拼接（`}{`）的格式。

每一条原始记录都会完整提交给 LLM。是否存在、以及如何抽取下列结构，全部由 LLM 判断：

```text
用户初始问题 → 客服澄清反问 → 紧接的用户真实回复
```

脚本不再解析角色、检查问号、要求开场用户发言，也不自行检查“客服提问后是否紧接用户回答”。它将
完整的 `chat_content`、`context`、`core_intent` 和已有 `known_info` 发送给模型，由模型决定是否合格、
选择哪组澄清问答，并输出 `initial_question`、`clarification_question` 和 `human_response`。用户
“我不知道”或“只重申需求”的真实回复也会保留，因为它们正是不回复率要测量的对象。

抽取默认会调用 LLM 做二次质检：确认客服问题确为澄清反问，并仅从原始对话和用户已有事实中
补齐支持真实回复所需的 `known_info`。所有新提示词均为中文，请求固定携带：

```json
"chat_template_kwargs": {"enable_thinking": false}
```

示例：

```bash
export USER_SIMULATOR_EVAL_JUDGE_URL='http://127.0.0.1:8000/v1'
export USER_SIMULATOR_EVAL_JUDGE_MODEL='your-judge-model'

python3 workspace/UserSimulatorEval/build_test_set.py \
  workspace/UserSimulatorEval/explore/diveUserData/dialog.json \
  --output workspace/UserSimulatorEval/data/user_simulator_test_set.jsonl \
  --report workspace/UserSimulatorEval/data/user_simulator_test_set.report.json
```

输出的每一行都是一个 JSON 样本，主要字段为：

```json
{
  "sample_id": "dialog-1-q1",
  "initial_question": "连不上网",
  "known_info": ["电脑连不上网", "电脑位于红区"],
  "original_known_info": ["电脑连不上网", "电脑位于红区"],
  "added_known_info": [],
  "clarification_question": "请问你在哪个工作区",
  "human_response": "红区",
  "source_chat_content": "用户：连不上网\\n客服：请问你在哪个工作区\\n用户：红区"
}
```

`source_chat_content`、`source` 和 `extraction.review_reason` 用于审计每条数据的来源和事实补齐理由。

### 2. 评测真实用户与模拟器

如果测试集中的 `human_response` 含有多行，只想保留第一次换行前的用户回复，可以使用
[`trim_human_responses.py`](trim_human_responses.py)：

```bash
python3 workspace/UserSimulatorEval/trim_human_responses.py \
  workspace/UserSimulatorEval/data/user_simulator_test_set.jsonl \
  --output workspace/UserSimulatorEval/data/user_simulator_test_set.first_line.jsonl
```

脚本只修改 `human_response`，其他字段、样本顺序和 JSONL 行数保持不变。默认不会覆盖已有文件；
如果确认要直接修改原测试集，可以显式使用：

```bash
python3 workspace/UserSimulatorEval/trim_human_responses.py \
  workspace/UserSimulatorEval/data/user_simulator_test_set.jsonl \
  --in-place
```

`--in-place` 使用临时文件原子替换输入文件；也可以用 `--overwrite` 覆盖指定的已有输出文件。

如果已经有旧版评测输出、只想按当前指标定义重新统计而不重新调用模型，可以使用：

```bash
python3 workspace/UserSimulatorEval/aggregate_user_simulator_output.py \
  workspace/UserSimulatorEval/outputs/compare.json \
  --output workspace/UserSimulatorEval/outputs/compare.reaggregated.json
```

该脚本读取原输出 `records` 中已经保存的 Judge 结果；信息提供效率只使用
`non_response=false` 的回复，其他指标沿用原有统计口径。

### 3. 统计真实用户的回复长度与四类行为

如果要直接分析测试集中的真实用户 `human_response`，可以运行
[`evaluate_human_user_behavior.py`](evaluate_human_user_behavior.py)：

```bash
python3 workspace/UserSimulatorEval/evaluate_human_user_behavior.py \
  workspace/UserSimulatorEval/data/user_simulator_test_set.jsonl \
  --url "$USER_SIMULATOR_EVAL_JUDGE_URL" \
  --model "$USER_SIMULATOR_EVAL_JUDGE_MODEL" \
  --output workspace/UserSimulatorEval/outputs/human_user_behavior.json
```

该脚本的回复长度完全本地计算，不调用 LLM：`human_response` 每个非空换行片段算一句回复，先计算
每句去除空白后的字符数，再统计所有句子的平均长度。报告同时保存每个 case 的句子拆分和长度。

每个 case 额外调用一次中文 Judge，判断：

1. 是否使用短语/词组回复；
2. 是否主动提供了客服本次没有明确追问的信息；
3. 是否没有正面回应客服、只是重复或改述初始需求；
4. 是否在完整对话中修改了自己之前确认的信息。

结果中的 `behavior_metrics` 给出四项的数量和比例，分母是成功完成 Judge 判断的 case 数；`cases`
保留每个 case 的四个布尔值，`errors` 保存失败的 case。完整的
`source_chat_content` 会传给 Judge，以便判断第四项前后信息是否发生变化。请求默认携带
`chat_template_kwargs.enable_thinking=false`。

[`evaluate_user_simulator.py`](evaluate_user_simulator.py) 使用同一个中文 LLM Judge 对回复标注信息点、
是否未经追问而泄漏信息、以及是否完全无视追问。四项本地汇总指标的正式定义见
[`metric.md`](metric.md)。

先只建立真实用户基线：

```bash
python3 workspace/UserSimulatorEval/evaluate_user_simulator.py \
  workspace/UserSimulatorEval/data/user_simulator_test_set.jsonl \
  --mode real \
  --judge-url "$USER_SIMULATOR_EVAL_JUDGE_URL" \
  --judge-model "$USER_SIMULATOR_EVAL_JUDGE_MODEL" \
  --output workspace/UserSimulatorEval/outputs/real_user_baseline.json
```

再让用户模拟器生成回复并与真实用户并列比较：

```bash
python3 workspace/UserSimulatorEval/evaluate_user_simulator.py \
  workspace/UserSimulatorEval/data/user_simulator_test_set.jsonl \
  --mode both \
  --judge-url "$USER_SIMULATOR_EVAL_JUDGE_URL" \
  --judge-model "$USER_SIMULATOR_EVAL_JUDGE_MODEL" \
  --simulator-url 'http://127.0.0.1:8005/v1' \
  --simulator-model 'clarq-user-simulator' \
  --output workspace/UserSimulatorEval/outputs/compare.json
```

默认的 `--simulator-adapter clarq_random` 会**直接调用**
`workspace/eval/clarq_eval/random_user_simulator.py` 中当前的随机用户模拟器（16% 重述、79% 压缩已知
事实、未知时 47% 主动反馈等默认行为），因此测到的是实际评测链路中的用户模拟器，而非另写一个近似
prompt。可用 `--simulator-adapter clarq_grounded` 测训练兼容的基础模拟器；两种方式均沿用
`--model-mode {qwen3,qwen3_5}` 的关闭思考逻辑。

如果被测对象是一个独立的、自然语言输出的用户模拟器服务，可改用：

```bash
--simulator-adapter natural_prompt \
--simulator-system-prompt-file path/to/your_simulator_prompt.txt
```

该 adapter 的默认 system prompt 为中文，要求模型仅基于 `known_info` 自然回答当前澄清问题。所有接口
请求仍默认关闭思考。输出会保留每条生成回复、模拟器行为元数据、Judge 原始 JSON、信息点、未被追问
的信息点、长度和不回复判断，并在 `metrics` 中给出真实用户、模拟器及四项主要指标的差值。不会将它们
擅自合成为单一类人分数。

其中，信息提供效率只统计 Judge 判定为回应了客服澄清问题的回复（`non_response=false`）。
被判定为不回复的样本仍参与不回复率、泄漏率等其他指标，但不进入信息提供效率的分子和分母。
如果已有旧版评测结果，需要按当前口径重算，可运行：

```bash
python3 workspace/UserSimulatorEval/aggregate_user_simulator_output.py \
  workspace/UserSimulatorEval/outputs/compare.json \
  --output workspace/UserSimulatorEval/outputs/compare.reaggregated.json
```

`clarification_eval.py` 使用一个可配置的 LLM Judge，从客服对话中识别：

1. 客服提出了澄清反问，且用户随后确实回答的配对数量；
2. 每次回答包含的独立信息点数量；
3. 用户是否顺带透露了客服本次没有明确、准确追问的信息点（信息泄漏）。

输入支持当前使用的 JSONL 格式（每行一条记录）：

```json
{"call_sno": 1, "chat_content": "用户：连不上网\n客服：请问你在哪个工作区\n用户：红区"}
```

也支持顶层 JSON 数组，以及 `chat_content` 为 `[{"role": "user", "content": "..."}]` 的已拆分对话。

## 运行

评估模型接口按 OpenAI Chat Completions 协议调用。`--url` 可以填完整的
`.../chat/completions` 地址，也可以填会自动补上该路径的 base URL。

```bash
python3 clarification_eval.py \
  diveUserData/dialog.json \
  --url https://api.example.com/v1 \
  --model-name your-judge-model \
  --api-key "$YOUR_API_KEY" \
  --output clarification_eval.json
```

也可以使用环境变量，适合在脚本或 CI 中运行：

```bash
export LLM_JUDGE_URL=https://api.example.com/v1
export LLM_JUDGE_MODEL_NAME=your-judge-model
export LLM_JUDGE_API_KEY=...
python3 clarification_eval.py diveUserData/dialog.json
```

本地 OpenAI-compatible 服务如果不需要鉴权，可以省略 `--api-key`。请求默认使用
`temperature=0`，并向兼容 Qwen/vLLM 的接口发送
`chat_template_kwargs={"enable_thinking": false}` 以关闭思考模式；先尝试
`response_format={"type":"json_object"}`，若服务不支持会自动回退到普通请求。网络失败或服务返回非 JSON 时会重试；可用 `--timeout` 和
`--max-retries` 调整。

默认最多同时发起 4 个 Judge 请求；可用 `--workers N`（或 `--concurrency N`）调整，
例如 `--workers 8`。调度器只会保留最多 N 条执行中的请求：每完成或失败一条才启动下一条，
不会一次把全部输入预先提交到线程池。传入 `--workers 1` 时退回串行。即使并发完成顺序不同，最终
`dialogues` 和 `errors` 仍按原始输入顺序输出。

运行时会在 stderr 显示每条对话的提交、完成或失败进度，例如：

```text
[clarification-eval] 已启动 12/100 call_sno=103 开始评估
[clarification-eval] 已完成 12/100 (12.0%) call_sno=103 完成，澄清回答 1 次
```

默认单条 Judge 请求、数据解析或 Judge 输出异常时，会记录该条错误并继续评估其他对话；
最终 JSON 的 `errors` 保存 `record_index`、`call_sno`、异常类型和错误信息，
`summary.error_count` 是失败条数。使用 `--fail-fast` 可改回首次失败即退出，使用
`--no-progress` 可关闭进度输出。

## 输出

输出是一个 JSON 对象，统计数字由脚本根据 Judge 返回的明细重新计算，而不是让模型直接生成：

```json
{
  "summary": {
    "dialogue_count": 2,
    "successful_dialogue_count": 2,
    "error_count": 0,
    "dialogues_with_answered_clarification": 1,
    "answered_clarification_pair_count": 1,
    "clarification_question_count": 1,
    "total_information_point_count": 1,
    "average_information_points_per_answered_clarification": 1.0,
    "answers_with_unasked_information_count": 0,
    "unasked_information_leakage_rate": 0.0
  },
  "dialogues": [
    {
      "call_sno": 1,
      "clarification_pairs": [
        {
          "question_turn_index": 1,
          "answer_turn_indices": [2],
          "question": "请问你在哪个工作区",
          "answer": "红区",
          "information_points": [
            {"text": "红区", "requested_by_agent": true}
          ],
          "information_point_count": 1,
          "unasked_information_points": [],
          "has_unasked_information": false
        }
      ]
    }
  ]
}
```

其中：

- `answered_clarification_pair_count` 是“客服澄清反问且用户回答了”的总次数；
- `clarification_question_count` 是这些已回答配对中的客服澄清问题次数（通常与上一个字段相同；
  若后续扩展为一问多答，可作为问题 turn 去重后的数量）；
- `information_point_count` 是该次回答中独立事实/属性等信息点的数量；
- `requested_by_agent=false` 的信息点会进入 `unasked_information_points`，并使
  `has_unasked_information=true`；
- `unasked_information_leakage_rate` 的分母是已回答的澄清配对数。

模型被要求使用 0-based turn index，并且脚本会校验配对确实是客服 turn 后紧接的连续用户
turn，过滤掉越界或角色不匹配的模型输出。

## 作为 Python 模块使用

```python
from clarification_eval import LLMJudge, evaluate_dialogues, load_dialogues

dialogues = load_dialogues("diveUserData/dialog.json")
judge = LLMJudge(
    url="https://api.example.com/v1",
    model_name="your-judge-model",
    api_key="...",
)
report = evaluate_dialogues(dialogues, judge)
print(report["summary"])
```

Python 调用时同样可以指定并发数：

```python
report = evaluate_dialogues(dialogues, judge, workers=8)
```

## 类人用户特征挖掘

`clarification_feature_mining.py` 读取 `clarification_eval.py` 的 JSON 输出，且**只处理**其中已确认
`clarification_pairs` 非空的对话。它按输入顺序串行调用模型，让模型在每一段对话中：

1. 自主归纳可帮助模拟器表现得更像人的用户行为；
2. 将每个澄清问题-用户回答配对归入此前已总结的行为类别，或在确实无法复用时创建新类别；
3. 对每个归类和模拟建议提供问题/回答原文证据与理由。

模型不会重新识别澄清配对，而是使用前一阶段确认好的问题/回答配对和完整上下文。类别库会随对话逐条积累，常见的行为包括但不预设为限：回答具体程度、部分回答、补充背景、表达不确定、纠正问题前提、反问、回避、重复与情绪化表达。

```bash
python3 clarification_feature_mining.py \
  clarification_eval.json \
  --url https://api.example.com/v1 \
  --model-name Qwen/Qwen3.6-27B \
  --api-key "$YOUR_API_KEY" \
  --output clarification_feature_mining.json
```

该脚本固定为串行处理，不提供 `--workers`。每次请求都发送
`chat_template_kwargs={"enable_thinking": false}`，并首先尝试 JSON Object 模式；服务不支持时会自动回退到普通请求。

默认输出会在每段对话结束或失败后立即以原子方式写盘。结果文件保存：

- `category_catalog`：累计的行为类别、定义、模拟建议、出现次数和示例证据；
- `dialogue_analyses`：每段对话的原始澄清配对、调用前类别库、完整请求、所有请求尝试、原始模型响应、模型判断、规范化判断与校验警告；
- `errors`：失败对话的请求内容、请求尝试和错误原因；
- `summary`：可处理对话/配对数、已完成数、类别数、归类次数和错误数。

中断后可使用 `--resume` 从同一输出文件继续，已经完成的 `record_index` 会跳过；失败过但尚未成功的对话会重试：

```bash
python3 clarification_feature_mining.py clarification_eval.json \
  --url https://api.example.com/v1 \
  --model-name Qwen/Qwen3.6-27B \
  --api-key "$YOUR_API_KEY" \
  --output clarification_feature_mining.json \
  --resume
```

已有输出文件时，第一次重跑会拒绝覆盖，避免丢失过程记录；可以明确使用 `--overwrite` 重新开始。`--max-categories-in-prompt`（默认 100）限制每次给模型的历史类别数量，类别过多时优先提供出现频率高且近期出现的类别。单次请求的 `--timeout`、`--max-retries`、`--temperature` 与 `--max-tokens` 均可调整；单条失败默认记录并继续，`--fail-fast` 可改为首次失败停止。

## 澄清对话分类统计

`clarification_category_stats.py` 读取 `clarification_eval.py` 的 JSON 输出，并只选择已经确认有
`clarification_pairs` 的对话。`clarification_pairs` 只用于脚本筛选入选对话，绝不会传给模型；模型输入中的对话数据仅为完整的 `turns`，从而可以判断“否定此前信息、修正需求”这类需要前后文的类别。

分类定义运行时从 [clarification.md](clarification.md) 的编号列表读取。当前文件的第 1 条会成为
`category_1`，第 2 条会成为 `category_2`，以此类推。模型会为每个对话选择至少一个类别，可以多选，并且必须为每个类别输出理由。

```bash
python3 clarification_category_stats.py \
  clarification_eval.json \
  --url https://api.example.com/v1 \
  --model-name Qwen/Qwen3.6-27B \
  --api-key "$YOUR_API_KEY" \
  --output clarification_category_stats.json
```

脚本固定为串行处理，并在每段对话完成或失败后原子写入结果。请求默认关闭 Qwen/vLLM 思考模式，发送
`chat_template_kwargs={"enable_thinking": false}`。可用 `--categories path/to/clarification.md` 指定其他类别文件。

输出 JSON **仅有两块**：

- `category_distribution`：每个类别的 `dialogue_count` 和 `dialogue_rate_among_classified`。同一对话同一类别只计一次；类别允许重叠，因此各计数之和可以大于已成功分类的对话数。比例的分母为成功获得模型分类的入选对话数。
- `dialogue_classifications`：每个入选对话的完整文本、分类类别和理由。单条调用失败也会在此列表保留 `status: "failed"` 和错误原因，但不会计入第一块的类别数量。

如果中断或有单条失败，可以从已有输出继续；已经成功分类的对话会跳过，失败项会重新尝试：

```bash
python3 clarification_category_stats.py clarification_eval.json \
  --url https://api.example.com/v1 \
  --model-name Qwen/Qwen3.6-27B \
  --api-key "$YOUR_API_KEY" \
  --output clarification_category_stats.json \
  --resume
```

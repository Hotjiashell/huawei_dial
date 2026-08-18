# 澄清反问用户回答评估

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
`temperature=0`，先尝试 `response_format={"type":"json_object"}`，若服务不支持会自动
回退到普通请求。网络失败或服务返回非 JSON 时会重试；可用 `--timeout` 和
`--max-retries` 调整。

默认最多同时发起 4 个 Judge 请求；可用 `--workers N`（或 `--concurrency N`）调整，
例如 `--workers 8`。传入 `--workers 1` 时退回串行。即使并发完成顺序不同，最终
`dialogues` 和 `errors` 仍按原始输入顺序输出。

运行时会在 stderr 显示每条对话的提交、完成或失败进度，例如：

```text
[clarification-eval] 已提交 12/100 call_sno=103 已提交
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

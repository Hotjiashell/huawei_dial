#!/usr/bin/env python3
"""Evaluate human-user reply style in a user-simulator test set.

Reply length is calculated locally.  Every non-empty line in ``human_response``
is one reply unit, so a newline starts a new unit.  A Chinese LLM Judge then
labels four behaviors for every case:

1. phrase/fragment-style reply;
2. volunteering information not explicitly requested by the agent;
3. ignoring the clarification and repeating the original need;
4. changing information the user had previously confirmed during the dialogue.

The complete ``source_chat_content`` is sent when available, which is needed
for the fourth behavior.  All Judge calls disable Qwen/vLLM thinking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from user_simulator_eval_common import OpenAIChatClient, as_string_list, load_jsonl


DEFAULT_INPUT = Path(__file__).with_name("data") / "user_simulator_test_set.jsonl"
DEFAULT_OUTPUT = Path(__file__).with_name("outputs") / "human_user_behavior.json"


BEHAVIOR_JUDGE_SYSTEM_PROMPT = """你是严格、可复核的客服对话行为分析专家。请评估一条测试集中的真实用户回复，并判断下面四种行为是否出现。

判定规则：
1. phrase_reply：用户是否主要使用短语、词组或非常简短的片段来回答，而不是完整的自然句子。
2. volunteered_unasked_information：用户是否在这次回复中主动提供了客服本次澄清问题没有明确、准确索取的信息。只看用户这次回复中实际出现的信息；客服明确问到的信息不算主动提供。
3. nonresponsive_repeats_request：用户是否没有正面回应客服的澄清问题，而只是重复、改述或强调自己的初始需求。用户说“不知道”“不清楚”、只回答一部分、回答是/否、纠正客服前提，都算回应客服，不属于该项；必须是明显完全无视这次追问并重申需求才为 true。
4. changed_previous_confirmation：结合完整原始对话，用户是否先明确确认/陈述了某个事实，后来又明确修改、否定或更正了同一个事实。只有前后矛盾且确实是用户自己的信息才算。

请严格依据输入文字，不要猜测用户没有说过的事实。`known_info` 仅作背景；信息点必须出现在用户回复或完整对话中。只能输出一个合法 JSON 对象，不要 Markdown、代码围栏或额外文字。格式必须严格为：
{
  "phrase_reply": false/true,
  "volunteered_unasked_information": false/true,
  "nonresponsive_repeats_request": false/true,
  "changed_previous_confirmation": false/true
}"""


BEHAVIOR_FIELDS = (
    "phrase_reply",
    "volunteered_unasked_information",
    "nonresponsive_repeats_request",
    "changed_previous_confirmation",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"true", "1", "yes", "y", "是", "有"}


def reply_lines(text: Any) -> list[str]:
    """Split a response into non-empty newline-delimited reply units."""

    if text is None:
        return []
    # splitlines handles LF, CRLF, CR and Unicode line separators.
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def line_length(line: str) -> int:
    """Count visible characters in one reply unit, excluding whitespace."""

    return len("".join(str(line).split()))


def reply_length_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate global and per-case mean reply-unit lengths locally."""

    all_lengths: list[int] = []
    per_case: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        lines = reply_lines(record.get("human_response"))
        lengths = [line_length(line) for line in lines]
        all_lengths.extend(lengths)
        per_case.append(
            {
                "sample_id": str(record.get("sample_id") or f"row-{index + 1}"),
                "reply_line_count": len(lines),
                "reply_lines": lines,
                "reply_line_lengths": lengths,
                "average_reply_length_characters": sum(lengths) / len(lengths) if lengths else 0.0,
            }
        )
    case_averages = [item["average_reply_length_characters"] for item in per_case if item["reply_line_count"]]
    return {
        "case_count": len(records),
        "reply_line_count": len(all_lengths),
        "total_reply_length_characters": sum(all_lengths),
        "average_reply_length_characters_per_line": (
            sum(all_lengths) / len(all_lengths) if all_lengths else 0.0
        ),
        # This gives each case equal weight, even if some responses contain
        # multiple newline-delimited units.
        "average_of_case_average_reply_lengths": (
            sum(case_averages) / len(case_averages) if case_averages else 0.0
        ),
        "per_case": per_case,
    }


def normalise_behavior_judgement(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the four boolean labels from Judge output."""

    return {field: _as_bool(value.get(field)) for field in BEHAVIOR_FIELDS}


def build_judge_input(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the full context sent to the behavior Judge."""

    return {
        "sample_id": record.get("sample_id"),
        "initial_question": str(record.get("initial_question") or ""),
        "known_info": as_string_list(record.get("known_info")),
        "clarification_question": str(record.get("clarification_question") or ""),
        "human_response": str(record.get("human_response") or ""),
        "source_chat_content": record.get("source_chat_content", ""),
    }


def judge_record(record: Mapping[str, Any], client: OpenAIChatClient) -> dict[str, Any]:
    response = client.chat(
        [
            {"role": "system", "content": BEHAVIOR_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "请评估下面这个 case，并严格按系统消息输出 JSON。\n\n"
                + json.dumps(build_judge_input(record), ensure_ascii=False, indent=2),
            },
        ],
        temperature=0.0,
        max_tokens=1024,
        json_object=True,
    )
    result = normalise_behavior_judgement(response["json"])
    result.update(
        {
            "judge_raw_response": response["content"],
            "judge_request_mode": response["request_mode"],
        }
    )
    return result


def aggregate_behavior_judgements(judgements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate the four behavior proportions over successfully judged cases."""

    total = len(judgements)
    result: dict[str, Any] = {"judged_case_count": total}
    for field in BEHAVIOR_FIELDS:
        count = sum(1 for judgement in judgements if judgement.get(field))
        result[f"{field}_count"] = count
        result[f"{field}_rate"] = count / total if total else 0.0
    return result


def _sample_id(record: Mapping[str, Any], index: int) -> str:
    return str(record.get("sample_id") or f"row-{index + 1}").strip() or f"row-{index + 1}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统计测试集用户回复长度并用 LLM 评估四类用户行为。")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="用户模拟器测试集 JSONL")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出 JSON 报告路径")
    parser.add_argument("--url", default=os.getenv("USER_SIMULATOR_EVAL_JUDGE_URL", os.getenv("LLM_JUDGE_URL")), help="OpenAI-compatible Judge 地址")
    parser.add_argument("--model", "--model-name", dest="model", default=os.getenv("USER_SIMULATOR_EVAL_JUDGE_MODEL", os.getenv("LLM_JUDGE_MODEL_NAME")), help="Judge 模型名称")
    parser.add_argument("--api-key", default=os.getenv("USER_SIMULATOR_EVAL_JUDGE_API_KEY", os.getenv("LLM_JUDGE_API_KEY", "")), help="Judge API key")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=2, help="每种请求格式的重试次数")
    parser.add_argument("--limit", type=int, help="最多处理多少条 case")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出")
    parser.add_argument("--fail-fast", action="store_true", help="单条 Judge 失败时立即退出")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"error: 输出已存在：{output_path}；如需覆盖请加 --overwrite", file=sys.stderr)
        return 2
    if not args.url or not args.model:
        print("error: 需要 --url 和 --model", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit <= 0:
        print("error: --limit 必须大于 0", file=sys.stderr)
        return 2

    try:
        records = load_jsonl(args.input)
        if args.limit is not None:
            records = records[: args.limit]
        client = OpenAIChatClient(args.url, args.model, args.api_key, args.timeout, args.max_retries)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    length_stats = reply_length_stats(records)
    judged_records: list[dict[str, Any]] = []
    judgements: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sample_id = _sample_id(record, index)
        try:
            judgement = judge_record(record, client)
            judgements.append(judgement)
            judged_records.append(
                {
                    "sample_id": sample_id,
                    "human_response": str(record.get("human_response") or ""),
                    "reply_line_lengths": [line_length(line) for line in reply_lines(record.get("human_response"))],
                    "judgement": judgement,
                }
            )
            print(f"[human-user-behavior] {index + 1}/{len(records)} 完成 {sample_id}", file=sys.stderr)
        except Exception as error:  # noqa: BLE001 - keep successful cases in batch mode
            failure = {"sample_id": sample_id, "error_type": type(error).__name__, "error": str(error)}
            errors.append(failure)
            print(f"[human-user-behavior] {index + 1}/{len(records)} 失败 {sample_id}：{failure['error']}", file=sys.stderr)
            if args.fail_fast:
                return 1

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "metric_definition": "workspace/UserSimulatorEval/metric.md",
        "thinking_disabled": True,
        "case_count": len(records),
        "judged_case_count": len(judgements),
        "error_count": len(errors),
        "reply_length": length_stats,
        "behavior_metrics": aggregate_behavior_judgements(judgements),
        "cases": judged_records,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reply_length": length_stats, "behavior_metrics": report["behavior_metrics"], "error_count": len(errors)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

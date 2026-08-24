#!/usr/bin/env python3
"""Build an auditable user-simulator test set from real customer dialogues.

Every raw dialogue is judged by the LLM directly.  The LLM, rather than local
turn parsing or question heuristics, decides whether a dialogue contains one
usable ``initial_question -> clarification_question -> human_response``
interaction and extracts those three fields.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from user_simulator_eval_common import (
    OpenAIChatClient,
    as_string_list,
    load_json_records,
    write_jsonl,
)


DEFAULT_INPUT = Path(__file__).with_name("explore") / "diveUserData" / "dialog.json"
DEFAULT_OUTPUT = Path(__file__).with_name("data") / "user_simulator_test_set.jsonl"


REVIEW_SYSTEM_PROMPT = """你是客服对话数据质检与用户画像补全专家。你只能依据给出的原始对话和已有用户事实工作，绝不能猜测、推理或补造事实。

任务：判断原始对话是否能构成一条用于评估“用户模拟器”的数据。合格数据必须同时满足：
1. 聊天记录中包含用户最初提出的问题；
2. 客服对用户的问题进行了澄清反问（非寒暄、建议、回答、催促或普通确认）；
3. 澄清反问后紧接着有用户的实际回复。该回复即使回答“不知道”、只回答一部分，或完全无视追问而重申需求等，只要用户说了话，就仍然是有效的真实用户回复，必须保留。

请只从原始对话和已有用户事实中抽取字段，不得猜测、推理或补造事实。已有 known_info 会由程序原样保留。
added_known_info 只写确有必要补充、且在用户原话或已有事实中明确出现的新客观事实；不得写客服的建议、推测、诊断结论、解决方案或原文中没有的信息。没有需要补充的事实时使用空数组。每条补充事实要简洁、独立、可由原文核对。

如果对话有多组问答，只选择最适合作为用户模拟器测试样本的一组，且 `initial_question` 必须是该组澄清问答所服务的用户最初问题。

只能输出一个合法 JSON 对象，不要 Markdown、代码围栏或其他文字。格式必须严格为：
{
  "eligible": true,
  "reason": "简短说明是否合格及补充依据",
  "added_known_info": ["用户明确表达的事实"],
  "initial_question": "用户最初提出的问题",
  "clarification_question": "客服的澄清反问",
  "human_response": "用户的实际回复"
}"""


def dialogue_inputs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build one LLM input per raw record without semantic pre-filtering.

    In particular, this function does not parse roles, inspect punctuation,
    require an initial turn, or require a consecutive user answer.  The model
    receives the complete raw ``chat_content`` and decides all of those facts.
    """

    inputs: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        source_id = str(record.get("call_sno", record_index + 1)).strip() or str(record_index + 1)
        inputs.append(
            {
                "sample_id": f"dialog-{source_id}",
                "source": {
                    "record_index": record_index,
                    "call_sno": record.get("call_sno"),
                    "source_context": str(record.get("context") or "").strip(),
                    "source_core_intent": str(record.get("core_intent") or "").strip(),
                },
                "existing_known_info": as_string_list(record.get("known_info")),
                "raw_chat_content": record.get("chat_content", record.get("conversation", record.get("messages"))),
            }
        )
    return inputs


def normalise_review(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise only the LLM's declared output schema, not its judgement."""

    return {
        "eligible": value.get("eligible") is True,
        "reason": str(value.get("reason") or "").strip(),
        "added_known_info": as_string_list(value.get("added_known_info")),
        "initial_question": str(value.get("initial_question") or "").strip(),
        "clarification_question": str(value.get("clarification_question") or "").strip(),
        "human_response": str(value.get("human_response") or "").strip(),
    }


def review_dialogue(dialogue: Mapping[str, Any], client: OpenAIChatClient) -> dict[str, Any]:
    """Ask the LLM to decide eligibility and extract all semantic fields."""

    review_input = {
        "existing_known_info": dialogue["existing_known_info"],
        "source_context": dialogue["source"]["source_context"],
        "source_core_intent": dialogue["source"]["source_core_intent"],
        "raw_chat_content": dialogue["raw_chat_content"],
    }
    response = client.chat(
        [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "请直接审查下面完整的原始对话，自行判断并按规定 JSON 输出。\n\n"
                + json.dumps(review_input, ensure_ascii=False, indent=2),
            },
        ],
        temperature=0.0,
        max_tokens=1024,
        json_object=True,
    )
    result = normalise_review(response["json"])
    result.update(
        {
            "raw_response": response["content"],
            "request_mode": response["request_mode"],
        }
    )
    return result


def make_test_record(dialogue: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    """Create the stable test-set row, preserving evidence for later audit."""

    missing = [
        field
        for field in ("initial_question", "clarification_question", "human_response")
        if not str(review.get(field) or "").strip()
    ]
    if missing:
        raise ValueError(f"模型将对话判为 eligible，但没有按协议提供字段：{', '.join(missing)}")

    existing = as_string_list(dialogue.get("existing_known_info"))
    added = [fact for fact in as_string_list(review.get("added_known_info")) if fact not in existing]
    return {
        "schema_version": "1.0",
        "sample_id": dialogue["sample_id"],
        "source": dialogue["source"],
        "initial_question": str(review["initial_question"]).strip(),
        "known_info": [*existing, *added],
        "original_known_info": existing,
        "added_known_info": added,
        "clarification_question": str(review["clarification_question"]).strip(),
        "human_response": str(review["human_response"]).strip(),
        "source_chat_content": dialogue["raw_chat_content"],
        "extraction": {
            "method": "llm_reviewed",
            "review_reason": str(review.get("reason") or ""),
            "review_raw_response": str(review.get("raw_response") or ""),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从真实客服对话抽取用户模拟器评测集。")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="原始 dialog.json / JSONL / JSON 对象流")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出评测集 JSONL 路径")
    parser.add_argument("--report", help="可选：输出抽取统计与跳过原因的 JSON 报告")
    parser.add_argument("--url", default=os.getenv("USER_SIMULATOR_EVAL_JUDGE_URL", os.getenv("LLM_JUDGE_URL")), help="用于质检和补齐 known_info 的 OpenAI-compatible 地址")
    parser.add_argument("--model", "--model-name", dest="model", default=os.getenv("USER_SIMULATOR_EVAL_JUDGE_MODEL", os.getenv("LLM_JUDGE_MODEL_NAME")), help="质检模型名称")
    parser.add_argument("--api-key", default=os.getenv("USER_SIMULATOR_EVAL_JUDGE_API_KEY", os.getenv("LLM_JUDGE_API_KEY", "")), help="质检接口 API key")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次 LLM 请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=2, help="每种请求格式的重试次数")
    parser.add_argument("--limit", type=int, help="最多交给模型判断多少条原始对话，便于小批试跑")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 --output")
    parser.add_argument("--fail-fast", action="store_true", help="LLM 单条失败时立即停止；默认记录后继续")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(f"error: 输出已存在：{output}；如需覆盖请加 --overwrite", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit <= 0:
        print("error: --limit 必须大于 0", file=sys.stderr)
        return 2
    if not args.url or not args.model:
        print("error: 需要 --url 和 --model；所有抽取判断均由 LLM 完成", file=sys.stderr)
        return 2

    try:
        dialogues = dialogue_inputs(load_json_records(args.input))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.limit is not None:
        dialogues = dialogues[: args.limit]

    client = OpenAIChatClient(args.url, args.model, args.api_key, args.timeout, args.max_retries)
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for position, dialogue in enumerate(dialogues, 1):
        try:
            review = review_dialogue(dialogue, client)
            if review["eligible"]:
                accepted.append(make_test_record(dialogue, review))
                print(f"[build-test-set] {position}/{len(dialogues)} 收录 {dialogue['sample_id']}", file=sys.stderr)
            else:
                skipped.append({"sample_id": dialogue["sample_id"], "reason": review.get("reason", "LLM 判为不合格")})
                print(f"[build-test-set] {position}/{len(dialogues)} 跳过 {dialogue['sample_id']}：{review.get('reason', '')}", file=sys.stderr)
        except Exception as error:  # noqa: BLE001 - batch extraction records isolated failures
            skipped.append({"sample_id": dialogue["sample_id"], "error_type": type(error).__name__, "reason": str(error)})
            print(f"[build-test-set] {position}/{len(dialogues)} 失败 {dialogue['sample_id']}：{type(error).__name__}: {error}", file=sys.stderr)
            if args.fail_fast:
                return 1

    write_jsonl(output, accepted)
    report = {
        "schema_version": "1.0",
        "input": str(args.input),
        "dialogue_count": len(dialogues),
        "accepted_count": len(accepted),
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("dialogue_count", "accepted_count", "skipped_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build an auditable user-simulator test set from real customer dialogues.

Each output row is one real clarification interaction:
``initial_question -> clarification_question -> human_response``.  Existing
``known_info`` is retained and an LLM may add only facts explicitly present in
the source dialogue, so the simulator has enough information to reproduce the
real reply without gaining invented knowledge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from user_simulator_eval_common import (
    OpenAIChatClient,
    Turn,
    as_string_list,
    load_json_records,
    parse_turns,
    write_jsonl,
)


DEFAULT_INPUT = Path(__file__).with_name("explore") / "diveUserData" / "dialog.json"
DEFAULT_OUTPUT = Path(__file__).with_name("data") / "user_simulator_test_set.jsonl"


REVIEW_SYSTEM_PROMPT = """你是客服对话数据质检与用户画像补全专家。你只能依据给出的原始对话和已有用户事实工作，绝不能猜测、推理或补造事实。

任务：判断候选片段是否能构成一条用于评估“用户模拟器”的数据。合格数据必须同时满足：
1. 聊天记录中包含用户最初提出的问题；
2. 客服的这句话确实是在为解决问题而询问缺失的具体信息（澄清反问），而非寒暄、建议、回答、催促或普通确认；
3. 澄清反问后紧接着有用户的实际回复。该回复即使回答“不知道”、只回答一部分，或完全无视追问而重申需求，也仍然是有效的真实用户回复，必须保留；
4. 可以从已有 known_info 与原始对话中整理出支持该用户回复的用户已知事实。

请仅补充“用户明确说过或已有事实明确写出”的客观事实。不得把客服的建议、推测、诊断结论、解决方案，或对话中没有出现的信息，写入事实。已有 known_info 会由程序原样保留，因此 added_known_info 只写确有必要补充的、新的事实；没有则使用空数组。每条补充事实要简洁、独立、可由原文核对。

只能输出一个合法 JSON 对象，不要 Markdown、代码围栏或其他文字。格式必须严格为：
{
  "eligible": true,
  "reason": "简短说明是否合格及补充依据",
  "added_known_info": ["用户明确表达的事实"]
}"""


QUESTION_CUE = re.compile(
    r"[？?]|请问|麻烦(?:问|告知|提供)|方便(?:说|告知|提供)|能否|是否|什么|怎么|怎样|哪里|哪[个些]|几[个位天]|为何|为什么|有没有|能不能"
)


def looks_like_question(text: str) -> bool:
    """Conservative structural pre-filter; LLM review makes the final call."""

    return bool(QUESTION_CUE.search(str(text or "")))


def _opening_user_turns(turns: Sequence[Turn]) -> list[Turn]:
    """Return the first consecutive user utterances in chat_content.

    A harmless opening greeting from the agent is allowed.  What matters is
    that the source transcript itself (rather than metadata ``context``)
    contains a user utterance before the clarification interaction.
    """

    opening: list[Turn] = []
    started = False
    for turn in turns:
        if not started:
            if turn.role == "user":
                opening.append(turn)
                started = True
            continue
        if turn.role != "user":
            break
        opening.append(turn)
    return opening


def candidate_interactions(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Find structurally complete candidate clarification interactions.

    Requiring an actual user opening in ``chat_content`` prevents a source
    whose ``context`` metadata exists but whose chat transcript lost the
    initial request from entering the evaluation set.  A preceding agent
    greeting is permitted.
    """

    candidates: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        chat_content = record.get("chat_content", record.get("conversation", record.get("messages", "")))
        turns = parse_turns(chat_content)
        opening = _opening_user_turns(turns)
        if not opening:
            continue
        initial_question = "\n".join(turn.text for turn in opening).strip()
        if not initial_question:
            continue
        for question_turn in turns:
            if question_turn.role != "assistant" or question_turn.index <= opening[-1].index:
                continue
            if not looks_like_question(question_turn.text):
                continue
            answer_start = question_turn.index + 1
            if answer_start >= len(turns) or turns[answer_start].role != "user":
                continue
            answer_turns: list[Turn] = []
            for turn in turns[answer_start:]:
                if turn.role != "user":
                    break
                answer_turns.append(turn)
            human_response = "\n".join(turn.text for turn in answer_turns).strip()
            if not human_response:
                continue
            prefix = [turn.as_dict() for turn in turns[: answer_turns[-1].index + 1]]
            source_id = str(record.get("call_sno", record_index + 1)).strip() or str(record_index + 1)
            candidates.append(
                {
                    "sample_id": f"dialog-{source_id}-q{question_turn.index}",
                    "source": {
                        "record_index": record_index,
                        "call_sno": record.get("call_sno"),
                        "source_context": str(record.get("context") or "").strip(),
                        "source_core_intent": str(record.get("core_intent") or "").strip(),
                    },
                    "initial_question": initial_question,
                    "existing_known_info": as_string_list(record.get("known_info")),
                    "clarification_question": question_turn.text,
                    "human_response": human_response,
                    "evidence_turns": prefix,
                }
            )
    return candidates


def normalise_review(value: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    """Accept only the narrow review contract used to form the test set."""

    eligible = value.get("eligible") is True
    reason = str(value.get("reason") or "").strip()
    additions = as_string_list(value.get("added_known_info"))
    return eligible, reason, additions


def review_candidate(candidate: Mapping[str, Any], client: OpenAIChatClient) -> dict[str, Any]:
    """LLM-review one structural candidate and return its source-auditable result."""

    review_input = {
        "initial_question": candidate["initial_question"],
        "existing_known_info": candidate["existing_known_info"],
        "clarification_question": candidate["clarification_question"],
        "human_response": candidate["human_response"],
        "dialogue_evidence": candidate["evidence_turns"],
    }
    response = client.chat(
        [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "请对下面候选数据做质检，并按规定 JSON 输出。\n\n" + json.dumps(review_input, ensure_ascii=False, indent=2),
            },
        ],
        temperature=0.0,
        max_tokens=1024,
        json_object=True,
    )
    eligible, reason, additions = normalise_review(response["json"])
    return {
        "eligible": eligible,
        "reason": reason,
        "added_known_info": additions,
        "raw_response": response["content"],
        "request_mode": response["request_mode"],
    }


def make_test_record(candidate: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    """Create the stable test-set row, preserving evidence for later audit."""

    existing = as_string_list(candidate.get("existing_known_info"))
    added = [fact for fact in as_string_list(review.get("added_known_info")) if fact not in existing]
    return {
        "schema_version": "1.0",
        "sample_id": candidate["sample_id"],
        "source": candidate["source"],
        "initial_question": candidate["initial_question"],
        "known_info": [*existing, *added],
        "original_known_info": existing,
        "added_known_info": added,
        "clarification_question": candidate["clarification_question"],
        "human_response": candidate["human_response"],
        "evidence_turns": candidate["evidence_turns"],
        "extraction": {
            "method": "llm_reviewed" if review.get("method") != "structural_only" else "structural_only",
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
    parser.add_argument("--limit", type=int, help="最多处理多少个结构候选，便于小批试跑")
    parser.add_argument("--skip-llm-review", action="store_true", help="仅按结构抽取；不校验澄清问题，也不补充 known_info")
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
    if not args.skip_llm_review and (not args.url or not args.model):
        print("error: 需要 --url 和 --model 才能完成 LLM 质检及 known_info 补齐；仅做结构抽取可加 --skip-llm-review", file=sys.stderr)
        return 2

    try:
        candidates = candidate_interactions(load_json_records(args.input))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.limit is not None:
        candidates = candidates[: args.limit]

    client = None if args.skip_llm_review else OpenAIChatClient(args.url, args.model, args.api_key, args.timeout, args.max_retries)
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates, 1):
        try:
            if client is None:
                review: dict[str, Any] = {
                    "eligible": True,
                    "reason": "未使用 LLM 质检：仅通过聊天结构筛选。",
                    "added_known_info": [],
                    "method": "structural_only",
                }
            else:
                review = review_candidate(candidate, client)
            if review["eligible"]:
                accepted.append(make_test_record(candidate, review))
                print(f"[build-test-set] {position}/{len(candidates)} 收录 {candidate['sample_id']}", file=sys.stderr)
            else:
                skipped.append({"sample_id": candidate["sample_id"], "reason": review.get("reason", "LLM 判为不合格")})
                print(f"[build-test-set] {position}/{len(candidates)} 跳过 {candidate['sample_id']}：{review.get('reason', '')}", file=sys.stderr)
        except Exception as error:  # noqa: BLE001 - batch extraction records isolated failures
            skipped.append({"sample_id": candidate["sample_id"], "error_type": type(error).__name__, "reason": str(error)})
            print(f"[build-test-set] {position}/{len(candidates)} 失败 {candidate['sample_id']}：{type(error).__name__}: {error}", file=sys.stderr)
            if args.fail_fast:
                return 1

    write_jsonl(output, accepted)
    report = {
        "schema_version": "1.0",
        "input": str(args.input),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "skipped_count": len(skipped),
        "skip_llm_review": bool(args.skip_llm_review),
        "skipped": skipped,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("candidate_count", "accepted_count", "skipped_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

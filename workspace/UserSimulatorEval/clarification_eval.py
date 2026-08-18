#!/usr/bin/env python3
"""Evaluate information supplied in answers to customer-service clarifications.

The input is JSONL (one object per line), for example::

    {"call_sno": 1, "chat_content": "用户：连不上网\n客服：请问你在哪个工作区\n用户：红区"}

An OpenAI-compatible chat-completions endpoint is used as the LLM Judge.  The
judge identifies clarification-question/user-answer pairs and labels every
distinct information point in the answer as either requested by the agent or
not requested.  Aggregate metrics are calculated locally from those labels.

No third-party Python package is required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest


ROLE_ALIASES = {
    "用户": "user",
    "客户": "user",
    "顾客": "user",
    "客服": "assistant",
    "坐席": "assistant",
    "机器人": "assistant",
    "agent": "assistant",
    "assistant": "assistant",
    "user": "user",
}


@dataclass(frozen=True)
class Turn:
    """A single utterance in a dialogue."""

    index: int
    role: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "role": self.role, "text": self.text}


def _normalise_role(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return ROLE_ALIASES.get(value) or ROLE_ALIASES.get(value.capitalize())


def parse_turns(chat_content: Any) -> list[Turn]:
    """Parse common dialogue representations into role-labelled turns.

    The supplied data uses one ``用户：...``/``客服：...`` utterance per line.
    Lists of ``{"role": ..., "content": ...}`` objects are accepted as well,
    which makes the evaluator usable with already-tokenised conversations.
    Unlabelled lines are retained as ``unknown`` so that the judge can see the
    original text, but they cannot be treated as a clarification answer by the
    local validation step.
    """

    turns: list[Turn] = []
    if isinstance(chat_content, (list, tuple)):
        for item in chat_content:
            if isinstance(item, Mapping):
                role = _normalise_role(item.get("role") or item.get("speaker")) or "unknown"
                text = item.get("content", item.get("text", ""))
            else:
                role, text = "unknown", item
            if text is not None and str(text).strip():
                turns.append(Turn(len(turns), role, str(text).strip()))
        return turns

    if not isinstance(chat_content, str):
        return turns

    # Accept both newline-separated turns and CRLF input. Empty lines are not
    # meaningful utterances and are omitted.
    for raw_line in chat_content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^\s*(用户|客户|顾客|客服|坐席|机器人|user|assistant|agent)\s*[:：]\s*(.*)$", line, re.I)
        if match:
            role = _normalise_role(match.group(1)) or "unknown"
            text = match.group(2).strip()
        else:
            role, text = "unknown", line
        if text:
            turns.append(Turn(len(turns), role, text))
    return turns


def load_dialogues(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL, a JSON array, or one JSON object from *path*."""

    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    stripped = raw.lstrip("\ufeff \t\r\n")
    if not stripped:
        return []

    if stripped.startswith("["):
        value = json.loads(stripped)
        if not isinstance(value, list):
            raise ValueError("JSON input beginning with '[' must contain an array")
        records = value
    else:
        records = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc

    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"dialogue #{index + 1} must be a JSON object")
        result.append(dict(record))
    return result


JUDGE_SYSTEM_PROMPT = """你是严谨的客服对话评估器（LLM Judge）。你的任务是识别客服澄清反问以及用户对该反问的回答，并分析回答提供的信息。

定义：
1. “澄清反问”是客服为了诊断问题、缩小范围或完成用户请求而向用户索取缺失的、具体的事实/选项。普通寒暄、告知、建议、确认已经明确的事实，不算澄清反问。
2. 只保留确实有用户回答的配对。回答通常是该客服问题之后、下一次客服发言之前的连续用户发言；如果没有回答，不要输出配对。
3. “信息点”是回答中一个可独立识别的事实、属性、地点、时间、设备、症状、选择或其他对处理问题有用的内容。把同一事实只算一次，不按字数或句子数计算。
4. 对每个信息点判断客服是否在这次澄清反问中明确、准确地追问了它：requested_by_agent=true 仅表示该信息点是问题明确索取的内容；问题宽泛、无法确定是否索取的，按 false 处理。
5. 用户礼貌用语、情绪、重复内容不算信息点；如果回答完全没有可识别信息点，返回空数组。

必须只输出一个合法 JSON 对象，不要 Markdown、解释文字或代码围栏。JSON 结构必须是：
{
  "clarification_pairs": [
    {
      "question_turn_index": 0,
      "answer_turn_indices": [1],
      "question": "客服原文",
      "answer": "用户原文（连续用户发言合并）",
      "information_points": [
        {"text": "信息点原文或简短概括", "requested_by_agent": true}
      ],
      "unasked_information_points": ["只列 requested_by_agent=false 的信息点文本"],
      "has_unasked_information": false
    }
  ]
}

question_turn_index 和 answer_turn_indices 必须使用输入中给出的 0-based turn index。question 和 answer 必须忠实摘录输入，不要臆造。"""


def build_judge_prompt(turns: Sequence[Turn]) -> str:
    rendered = "\n".join(f"[{turn.index}] {turn.role}: {turn.text}" for turn in turns)
    return "请评估下面这一段对话，并严格按系统消息要求输出 JSON。\n\n对话：\n" + rendered


def resolve_endpoint(url: str) -> str:
    """Resolve a base URL to an OpenAI-compatible chat-completions endpoint."""

    value = url.strip().rstrip("/")
    if not value:
        raise ValueError("LLM Judge URL cannot be empty")
    if value.endswith("/chat/completions"):
        return value
    # A URL ending in /v1 (or /api/v1) is conventionally a base endpoint.
    return value + "/chat/completions"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping) and isinstance(content.get("text"), str):
        return content["text"]
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON returned by a model, tolerating a Markdown fence/preamble."""

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # Find the first balanced JSON object. This handles models that prepend a
    # short sentence despite the instruction not to do so.
    start = candidate.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for pos in range(start, len(candidate)):
            char = candidate[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(candidate[start : pos + 1])
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        break
        start = candidate.find("{", start + 1)
    raise ValueError("LLM Judge did not return a JSON object")


class LLMJudge:
    """Small OpenAI-compatible HTTP client for the clarification judge."""

    def __init__(self, url: str, model_name: str, api_key: str = "", timeout: float = 60.0, max_retries: int = 2):
        self.endpoint = resolve_endpoint(url)
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    def evaluate(self, turns: Sequence[Turn]) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_prompt(turns)},
            ],
        }
        # Most OpenAI-compatible servers accept this hint; if a server rejects
        # it, the retry path below sends the same request without the hint.
        attempts = [dict(payload, response_format={"type": "json_object"}), payload]
        last_error: Exception | None = None
        for attempt_index, body in enumerate(attempts):
            for retry_index in range(self.max_retries + 1):
                try:
                    return self._post(body)
                except Exception as exc:  # noqa: BLE001 - rethrown with context below
                    last_error = exc
                    if attempt_index == len(attempts) - 1 and retry_index >= self.max_retries:
                        break
                    # Retry transient network/server errors with a short,
                    # bounded backoff. A rejected response_format immediately
                    # falls through to the plain payload on the next attempt.
                    if retry_index < self.max_retries:
                        time.sleep(min(2.0, 0.5 * (2**retry_index)))
        raise RuntimeError(f"LLM Judge request failed: {last_error}") from last_error

    def _post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urlrequest.Request(self.endpoint, data=data, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {self.endpoint}: {detail[:1000]}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"cannot reach {self.endpoint}: {exc.reason}") from exc
        try:
            response_json = json.loads(raw)
            content = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unexpected LLM response: {raw[:1000]}") from exc
        return extract_json_object(_content_to_text(content))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是", "有"}
    return False


def _as_index_list(value: Any) -> list[int]:
    if isinstance(value, (int, float)) and int(value) == value:
        return [int(value)]
    if isinstance(value, str):
        found = re.findall(r"-?\d+", value)
        return [int(item) for item in found]
    if isinstance(value, list):
        result: list[int] = []
        for item in value:
            result.extend(_as_index_list(item))
        return result
    return []


def _point_texts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("text", "")
        if item is not None and str(item).strip():
            texts.append(str(item).strip())
    return texts


def normalize_judgement(raw: Mapping[str, Any], turns: Sequence[Turn]) -> dict[str, Any]:
    """Validate/normalise a model judgement against the source turns."""

    raw_pairs = raw.get("clarification_pairs", [])
    if not isinstance(raw_pairs, list):
        raw_pairs = []
    pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, tuple[int, ...]]] = set()
    for item in raw_pairs:
        if not isinstance(item, Mapping):
            continue
        question_indices = _as_index_list(item.get("question_turn_index", item.get("question_turn")))
        answer_indices = _as_index_list(item.get("answer_turn_indices", item.get("answer_turn_index", item.get("answer_turn"))))
        if not question_indices or not answer_indices:
            continue
        q_index = question_indices[0]
        answer_indices = list(dict.fromkeys(answer_indices))
        if q_index < 0 or q_index >= len(turns) or any(i < 0 or i >= len(turns) for i in answer_indices):
            continue
        # A valid pair must preserve the dialogue order and be an assistant ->
        # user transition. We allow consecutive user turns as one answer.
        if turns[q_index].role != "assistant" or min(answer_indices) != q_index + 1:
            continue
        if any(turns[i].role != "user" for i in answer_indices):
            continue
        if any(i <= q_index for i in answer_indices):
            continue
        answer_indices.sort()
        if answer_indices != list(range(answer_indices[0], answer_indices[-1] + 1)):
            continue
        pair_key = (q_index, tuple(answer_indices))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        all_points: list[dict[str, Any]] = []
        seen_point_texts: set[str] = set()
        raw_points = item.get("information_points", [])
        if isinstance(raw_points, list):
            for point in raw_points:
                if isinstance(point, Mapping):
                    text = str(point.get("text", "")).strip()
                    if not text or text in seen_point_texts:
                        continue
                    seen_point_texts.add(text)
                    requested = _as_bool(point.get("requested_by_agent", point.get("requested")))
                    all_points.append({"text": text, "requested_by_agent": requested})
                elif point is not None and str(point).strip():
                    # Older/looser judge output: treat bare points as asked
                    # only when they are absent from the explicit unasked list.
                    text = str(point).strip()
                    if text not in seen_point_texts:
                        seen_point_texts.add(text)
                        all_points.append({"text": text, "requested_by_agent": True})

        unasked = _point_texts(item.get("unasked_information_points"))
        if unasked:
            unasked_set = set(unasked)
            for point in all_points:
                if point["text"] in unasked_set:
                    point["requested_by_agent"] = False
            for text in unasked:
                if not any(p["text"] == text for p in all_points):
                    seen_point_texts.add(text)
                    all_points.append({"text": text, "requested_by_agent": False})
        unasked = [p["text"] for p in all_points if not p["requested_by_agent"]]
        # Always use source text for the persisted transcript. The model only
        # decides which turns form a pair and how to label their information;
        # it must not be able to alter the evidence shown in the report.
        question_text = turns[q_index].text
        answer_text = "\n".join(turns[i].text for i in answer_indices)
        pairs.append(
            {
                "question_turn_index": q_index,
                "answer_turn_indices": answer_indices,
                "question": question_text,
                "answer": answer_text,
                "information_points": all_points,
                "information_point_count": len(all_points),
                "unasked_information_points": unasked,
                "has_unasked_information": bool(unasked) or _as_bool(item.get("has_unasked_information")),
            }
        )
    return {"clarification_pairs": pairs}


def aggregate_results(dialogue_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = [pair for result in dialogue_results for pair in result.get("clarification_pairs", [])]
    total_points = sum(int(pair.get("information_point_count", len(pair.get("information_points", [])))) for pair in pairs)
    leakage_pairs = sum(1 for pair in pairs if pair.get("has_unasked_information"))
    dialogue_count = len(dialogue_results)
    answered_dialogues = sum(1 for result in dialogue_results if result.get("clarification_pairs"))
    question_count = sum(int(result.get("clarification_question_count", 0)) for result in dialogue_results)
    return {
        "dialogue_count": dialogue_count,
        "dialogues_with_answered_clarification": answered_dialogues,
        "answered_clarification_pair_count": len(pairs),
        "clarification_question_count": question_count,
        "total_information_point_count": total_points,
        "average_information_points_per_answered_clarification": (total_points / len(pairs)) if pairs else 0.0,
        "answers_with_unasked_information_count": leakage_pairs,
        "unasked_information_leakage_rate": (leakage_pairs / len(pairs)) if pairs else 0.0,
    }


def evaluate_dialogues(dialogues: Sequence[Mapping[str, Any]], judge: LLMJudge) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for record_index, record in enumerate(dialogues):
        turns = parse_turns(record.get("chat_content", record.get("conversation", record.get("messages", ""))))
        raw_judgement = judge.evaluate(turns) if turns else {"clarification_pairs": []}
        normalized = normalize_judgement(raw_judgement, turns)
        # The model returns answered pairs; question count is kept separately
        # for the useful denominator in answer-rate analyses. It is inferred
        # only from assistant turns that the judge identified as clarifications.
        question_indices = {p["question_turn_index"] for p in normalized["clarification_pairs"]}
        result = {
            "record_index": record_index,
            "call_sno": record.get("call_sno"),
            "turns": [turn.as_dict() for turn in turns],
            "clarification_question_count": len(question_indices),
            **normalized,
        }
        results.append(result)
    return {"summary": aggregate_results(results), "dialogues": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use an LLM Judge to measure user answers to customer-service clarifications.")
    parser.add_argument("input", help="JSONL/JSON dialogue file, e.g. diveUserData/dialog.json")
    parser.add_argument("--url", default=os.getenv("LLM_JUDGE_URL"), help="OpenAI-compatible base URL or /chat/completions endpoint (env: LLM_JUDGE_URL)")
    parser.add_argument("--model-name", "--model_name", dest="model_name", default=os.getenv("LLM_JUDGE_MODEL_NAME"), help="Judge model name (env: LLM_JUDGE_MODEL_NAME)")
    parser.add_argument("--api-key", "--api_key", dest="api_key", default=os.getenv("LLM_JUDGE_API_KEY", ""), help="API key (env: LLM_JUDGE_API_KEY); may be empty for local endpoints")
    parser.add_argument("--output", help="Output JSON path; defaults to stdout")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds (default: 60)")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per request (default: 2)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("--url is required (or set LLM_JUDGE_URL)")
    if not args.model_name:
        parser.error("--model-name is required (or set LLM_JUDGE_MODEL_NAME)")
    try:
        dialogues = load_dialogues(args.input)
        judge = LLMJudge(args.url, args.model_name, args.api_key, args.timeout, args.max_retries)
        report = evaluate_dialogues(dialogues, judge)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

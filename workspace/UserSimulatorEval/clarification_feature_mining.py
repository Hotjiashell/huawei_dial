#!/usr/bin/env python3
"""Serial LLM-led mining of human-like user behaviours from clarification pairs.

The input is the JSON report produced by ``clarification_eval.py``. Only
dialogues that contain an already-confirmed ``clarification_pairs`` entry are
sent to the model. The model processes source dialogues in order, reusing or
extending a persistent catalogue of behavioural categories. Every request,
response, parsed judgement, catalogue snapshot and error is saved to the
output JSON after each source dialogue.

No third-party package is required. The endpoint follows the OpenAI Chat
Completions protocol and supports Qwen/vLLM's ``chat_template_kwargs``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

from clarification_eval import _content_to_text, extract_json_object, resolve_endpoint


FORMAT_VERSION = 1


MINING_SYSTEM_PROMPT = """你是一名用户模拟器研究员。你的目标是从“客服澄清追问且用户已回答”的真实对话中，归纳能够帮助后续模拟器表现得更像人的用户行为特征。

工作原则：
1. 只分析输入列出的 eligible_pairs；完整对话仅用于理解上下文。不要把其他轮次单独作为分析对象。
2. 只在原文有证据时归纳，不要臆造人格、动机或事实。
3. 每个 pair 可以归入零个、一个或多个已有类别。类别表示可在模拟器中复现的行为模式，而不是本次对话的业务主题。优先复用已给类别；只有没有语义等价类别时才新增。禁止仅因措辞不同而创建同义类别。
4. 新类别必须足够稳定、可复用且可操作；定义说明什么行为属于该类；simulation_guidance 说明模拟器应如何表现。每次最多新增 3 个类别。
5. 每个归类和每条模拟器建议都必须给出理由，并引用对应 pair_index 及问题/回答中的简短原文证据。证据必须来自输入，不要改写成不存在的对话。

只能返回一个合法 JSON 对象，禁止 Markdown、解释性文字或代码围栏。返回结构必须为：
{
  "dialogue_summary": "这段对话中与用户模拟有关的简短总结；没有显著特征时说明原因",
  "new_categories": [
    {
      "proposed_key": "new_1",
      "name": "简洁、可复用的类别名",
      "definition": "该行为模式的边界和定义",
      "simulation_guidance": "模拟器可执行的生成/采样建议",
      "why_new": "为何已有类别不能覆盖",
      "evidence_pair_indices": [0]
    }
  ],
  "pair_assessments": [
    {
      "pair_index": 0,
      "category_refs": [
        {
          "kind": "existing",
          "category_id": "category_001",
          "reason": "归入该类别的理由",
          "question_evidence": "问题中的短原文",
          "answer_evidence": "回答中的短原文"
        },
        {
          "kind": "new",
          "proposed_key": "new_1",
          "reason": "归入新类别的理由",
          "question_evidence": "问题中的短原文",
          "answer_evidence": "回答中的短原文"
        }
      ]
    }
  ]
}

category_refs.kind 只能是 existing 或 new。existing 必须引用给定的 category_id；new 必须引用本次 new_categories 中的 proposed_key。每一个 eligible pair 都必须返回一项 pair_assessments；如果没有合适类别，也返回空的 category_refs，并在 simulator_considerations 说明没有足够证据的原因。"""


class JudgeRequestError(RuntimeError):
    """A request failure carrying the retry/fallback trace for persistence."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


class FeatureMiningJudge:
    """OpenAI-compatible JSON client that preserves raw request/response data."""

    def __init__(
        self,
        url: str,
        model_name: str,
        api_key: str = "",
        timeout: float = 120.0,
        max_retries: int = 2,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ):
        self.endpoint = resolve_endpoint(url)
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyse(self, user_prompt: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": MINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        base_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Equivalent to OpenAI SDK's extra_body={...}; keep Qwen thinking
            # disabled so that the output remains concise, parsable JSON.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        payloads = [
            ("json_object", dict(base_payload, response_format={"type": "json_object"})),
            ("plain", base_payload),
        ]
        attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for mode, payload in payloads:
            for retry_index in range(self.max_retries + 1):
                trace = {
                    "mode": mode,
                    "retry_index": retry_index,
                    "request_body": copy.deepcopy(payload),
                }
                raw_response: str | None = None
                try:
                    raw_response = self._post_for_content(payload)
                    parsed = extract_json_object(raw_response)
                    trace["status"] = "response_received"
                    trace["raw_response"] = raw_response
                    attempts.append(trace)
                    return {
                        "messages": messages,
                        "request_attempts": attempts,
                        "raw_response": raw_response,
                        "parsed_response": parsed,
                    }
                except Exception as exc:  # noqa: BLE001 - retry/fallback is intentional
                    last_error = exc
                    trace["status"] = "failed"
                    trace["error_type"] = type(exc).__name__
                    trace["error"] = str(exc)
                    if raw_response is not None:
                        trace["raw_response"] = raw_response
                    attempts.append(trace)
                    if retry_index < self.max_retries:
                        time.sleep(min(2.0, 0.5 * (2**retry_index)))
        raise JudgeRequestError(f"LLM Judge request failed: {last_error}", attempts) from last_error

    def _post_for_content(self, payload: Mapping[str, Any]) -> str:
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
            response = json.loads(raw)
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unexpected LLM response: {raw[:1000]}") from exc
        return _content_to_text(content)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_text(value: Any, maximum: int = 2000) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text[:maximum]


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def as_string_list(value: Any, maximum: int = 100) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = as_text(item)
        if text and text not in result:
            result.append(text)
        if len(result) >= maximum:
            break
    return result


def as_integer_list(value: Any, maximum: int = 100) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        parsed = as_int(item)
        if parsed is not None and parsed not in result:
            result.append(parsed)
        if len(result) >= maximum:
            break
    return result


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: expected a top-level JSON object")
    return dict(value)


def eligible_dialogues(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    dialogues = report.get("dialogues", [])
    if not isinstance(dialogues, list):
        raise ValueError("input report field 'dialogues' must be a list")
    selected: list[dict[str, Any]] = []
    for source_position, dialogue in enumerate(dialogues):
        if not isinstance(dialogue, Mapping):
            continue
        pairs = dialogue.get("clarification_pairs", [])
        if isinstance(pairs, list) and pairs:
            selected.append(
                {
                    "source_position": source_position,
                    "record_index": dialogue.get("record_index", source_position),
                    "call_sno": dialogue.get("call_sno"),
                    "turns": dialogue.get("turns", []),
                    "clarification_pairs": pairs,
                }
            )
    return selected


def serialise_dialogue_for_prompt(dialogue: Mapping[str, Any]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    raw_pairs = dialogue.get("clarification_pairs", [])
    if isinstance(raw_pairs, list):
        for pair_index, pair in enumerate(raw_pairs):
            if not isinstance(pair, Mapping):
                continue
            pairs.append(
                {
                    "pair_index": pair_index,
                    "question_turn_index": pair.get("question_turn_index"),
                    "answer_turn_indices": pair.get("answer_turn_indices", []),
                    "question": pair.get("question", ""),
                    "answer": pair.get("answer", ""),
                }
            )
    turns = dialogue.get("turns", [])
    if not isinstance(turns, list):
        turns = []
    return {
        "record_index": dialogue.get("record_index"),
        "call_sno": dialogue.get("call_sno"),
        "eligible_pairs": pairs,
        "full_dialogue_context": turns,
    }


def categories_for_prompt(categories: Sequence[Mapping[str, Any]], maximum: int) -> list[dict[str, Any]]:
    # Keep frequent/recent categories when the accumulated taxonomy becomes
    # large. The exact catalogue sent is still saved with the request process.
    ranked = sorted(
        categories,
        key=lambda item: (
            -int(item.get("occurrence_count", 0)),
            -int(item.get("last_seen_sequence", -1)),
            str(item.get("category_id", "")),
        ),
    )[:maximum]
    return [
        {
            "category_id": item.get("category_id"),
            "name": item.get("name"),
            "definition": item.get("definition"),
            "simulation_guidance": item.get("simulation_guidance"),
            "occurrence_count": item.get("occurrence_count", 0),
        }
        for item in ranked
    ]


def build_user_prompt(dialogue: Mapping[str, Any], categories: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "existing_category_catalog": categories,
        "source_dialogue": serialise_dialogue_for_prompt(dialogue),
    }
    return (
        "请按系统要求挖掘下面 source_dialogue 中 eligible_pairs 所体现的类人用户行为。"
        "existing_category_catalog 是此前对话已总结出的类别库；请优先复用其中 category_id。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def next_category_id(categories: Sequence[Mapping[str, Any]]) -> str:
    largest = 0
    for category in categories:
        category_id = as_text(category.get("category_id"))
        if category_id.startswith("category_"):
            suffix = as_int(category_id.removeprefix("category_"))
            if suffix is not None:
                largest = max(largest, suffix)
    return f"category_{largest + 1:03d}"


def normalise_model_judgement(
    raw: Mapping[str, Any],
    *,
    visible_categories: Sequence[Mapping[str, Any]],
    all_categories: Sequence[Mapping[str, Any]],
    pair_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Validate references, assign new IDs, and retain model output separately."""

    warnings: list[str] = []
    visible_ids = {as_text(item.get("category_id")) for item in visible_categories}
    all_ids = {as_text(item.get("category_id")) for item in all_categories}
    raw_new = raw.get("new_categories", [])
    proposed: dict[str, dict[str, Any]] = {}
    if isinstance(raw_new, list):
        for index, item in enumerate(raw_new[:3], 1):
            if not isinstance(item, Mapping):
                warnings.append(f"new_categories[{index - 1}] is not an object")
                continue
            key = as_text(item.get("proposed_key"))
            name = as_text(item.get("name"), 160)
            definition = as_text(item.get("definition"), 1200)
            guidance = as_text(item.get("simulation_guidance"), 1200)
            if not key or not name or not definition or not guidance:
                warnings.append(f"new_categories[{index - 1}] is missing proposed_key/name/definition/simulation_guidance")
                continue
            if key in proposed:
                warnings.append(f"duplicate new category key: {key}")
                continue
            proposed[key] = {
                "proposed_key": key,
                "name": name,
                "definition": definition,
                "simulation_guidance": guidance,
                "why_new": as_text(item.get("why_new"), 1200),
                "evidence_pair_indices": [
                    value for value in as_integer_list(item.get("evidence_pair_indices")) if 0 <= value < pair_count
                ],
            }

    raw_assessments = raw.get("pair_assessments", [])
    by_pair: dict[int, Mapping[str, Any]] = {}
    if isinstance(raw_assessments, list):
        for item in raw_assessments:
            if not isinstance(item, Mapping):
                continue
            pair_index = as_int(item.get("pair_index"))
            if pair_index is None or not 0 <= pair_index < pair_count:
                warnings.append("pair_assessments contains an invalid pair_index")
                continue
            if pair_index in by_pair:
                warnings.append(f"duplicate pair_assessments entry for pair_index {pair_index}")
                continue
            by_pair[pair_index] = item
    else:
        warnings.append("pair_assessments is not a list")

    referenced_new_keys: set[str] = set()
    for item in by_pair.values():
        refs = item.get("category_refs", [])
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if isinstance(ref, Mapping) and as_text(ref.get("kind")).lower() == "new":
                key = as_text(ref.get("proposed_key"))
                if key in proposed:
                    referenced_new_keys.add(key)
    new_categories: list[dict[str, Any]] = []
    new_key_to_id: dict[str, str] = {}
    known_categories = list(all_categories)
    for key, category in proposed.items():
        if key not in referenced_new_keys:
            warnings.append(f"new category {key} was not used by any pair and was not registered")
            continue
        category_id = next_category_id(known_categories + new_categories)
        new_key_to_id[key] = category_id
        new_categories.append(
            {
                "category_id": category_id,
                "name": category["name"],
                "definition": category["definition"],
                "simulation_guidance": category["simulation_guidance"],
                "why_new": category["why_new"],
                "created_at": utc_now(),
                "first_seen": None,
                "last_seen_sequence": -1,
                "occurrence_count": 0,
                "example_assignments": [],
                "source_proposed_key": key,
            }
        )

    assessments: list[dict[str, Any]] = []
    for pair_index in range(pair_count):
        item = by_pair.get(pair_index, {})
        refs: list[dict[str, Any]] = []
        raw_refs = item.get("category_refs", []) if isinstance(item, Mapping) else []
        if isinstance(raw_refs, list):
            seen_ids: set[str] = set()
            for ref in raw_refs:
                if not isinstance(ref, Mapping):
                    continue
                kind = as_text(ref.get("kind")).lower()
                category_id = ""
                if kind == "existing":
                    category_id = as_text(ref.get("category_id"))
                    if category_id not in visible_ids:
                        if category_id in all_ids:
                            warnings.append(f"pair {pair_index} referenced omitted category {category_id}")
                        else:
                            warnings.append(f"pair {pair_index} referenced unknown category {category_id}")
                        continue
                elif kind == "new":
                    key = as_text(ref.get("proposed_key"))
                    category_id = new_key_to_id.get(key, "")
                    if not category_id:
                        warnings.append(f"pair {pair_index} referenced invalid new category {key}")
                        continue
                else:
                    warnings.append(f"pair {pair_index} has invalid category ref kind")
                    continue
                if category_id in seen_ids:
                    continue
                seen_ids.add(category_id)
                refs.append(
                    {
                        "category_id": category_id,
                        "source": kind,
                        "reason": as_text(ref.get("reason"), 1600),
                        "question_evidence": as_text(ref.get("question_evidence"), 1000),
                        "answer_evidence": as_text(ref.get("answer_evidence"), 1000),
                    }
                )

        considerations: list[dict[str, str]] = []
        raw_considerations = item.get("simulator_considerations", []) if isinstance(item, Mapping) else []
        if isinstance(raw_considerations, list):
            for consideration in raw_considerations[:10]:
                if not isinstance(consideration, Mapping):
                    continue
                aspect = as_text(consideration.get("aspect"), 240)
                behavior = as_text(consideration.get("behavior"), 1600)
                reason = as_text(consideration.get("reason"), 1600)
                if not aspect or not behavior or not reason:
                    warnings.append(f"pair {pair_index} has incomplete simulator consideration")
                    continue
                considerations.append(
                    {
                        "aspect": aspect,
                        "behavior": behavior,
                        "reason": reason,
                        "question_evidence": as_text(consideration.get("question_evidence"), 1000),
                        "answer_evidence": as_text(consideration.get("answer_evidence"), 1000),
                    }
                )
        if pair_index not in by_pair:
            warnings.append(f"model omitted pair_assessments for pair_index {pair_index}")
        assessments.append(
            {
                "pair_index": pair_index,
                "category_assignments": refs,
                "simulator_considerations": considerations,
            }
        )
    return (
        {
            "dialogue_summary": as_text(raw.get("dialogue_summary"), 3000),
            "pair_assessments": assessments,
        },
        new_categories,
        warnings,
    )


def apply_category_assignments(
    categories: list[dict[str, Any]],
    new_categories: list[dict[str, Any]],
    assessment: Mapping[str, Any],
    dialogue: Mapping[str, Any],
    sequence_index: int,
) -> list[dict[str, Any]]:
    categories.extend(new_categories)
    category_by_id = {as_text(item.get("category_id")): item for item in categories}
    pairs = dialogue.get("clarification_pairs", [])
    if not isinstance(pairs, list):
        pairs = []
    for pair_assessment in assessment.get("pair_assessments", []):
        if not isinstance(pair_assessment, Mapping):
            continue
        pair_index = as_int(pair_assessment.get("pair_index"))
        if pair_index is None or not 0 <= pair_index < len(pairs):
            continue
        refs = pair_assessment.get("category_assignments", [])
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, Mapping):
                continue
            category = category_by_id.get(as_text(ref.get("category_id")))
            if category is None:
                continue
            category["occurrence_count"] = int(category.get("occurrence_count", 0)) + 1
            category["last_seen_sequence"] = sequence_index
            source_ref = {
                "record_index": dialogue.get("record_index"),
                "call_sno": dialogue.get("call_sno"),
                "pair_index": pair_index,
                "reason": ref.get("reason", ""),
                "question_evidence": ref.get("question_evidence", ""),
                "answer_evidence": ref.get("answer_evidence", ""),
            }
            if category.get("first_seen") is None:
                category["first_seen"] = source_ref
            examples = category.setdefault("example_assignments", [])
            if len(examples) < 5:
                examples.append(source_ref)
    return categories


def update_summary(output: dict[str, Any], source_report: Mapping[str, Any], eligible: Sequence[Mapping[str, Any]]) -> None:
    analyses = output.get("dialogue_analyses", [])
    errors = output.get("errors", [])
    categories = output.get("category_catalog", [])
    completed_pair_count = sum(
        len(item.get("source_dialogue", {}).get("eligible_pairs", []))
        for item in analyses
        if isinstance(item, Mapping)
    )
    output["summary"] = {
        "source_dialogue_count": len(source_report.get("dialogues", [])) if isinstance(source_report.get("dialogues", []), list) else 0,
        "eligible_dialogue_count": len(eligible),
        "eligible_clarification_pair_count": sum(len(item.get("clarification_pairs", [])) for item in eligible),
        "completed_dialogue_count": len(analyses),
        "completed_clarification_pair_count": completed_pair_count,
        "error_count": len(errors),
        "category_count": len(categories),
        "category_assignment_count": sum(int(item.get("occurrence_count", 0)) for item in categories),
    }
    output["updated_at"] = utc_now()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def new_output(input_path: Path, judge: FeatureMiningJudge) -> dict[str, Any]:
    return {
        "format": "clarification_feature_mining",
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "run": {
            "input_path": str(input_path),
            "endpoint": judge.endpoint,
            "model_name": judge.model_name,
            "serial": True,
            "temperature": judge.temperature,
            "max_tokens": judge.max_tokens,
            "thinking_disabled": True,
            "status": "running",
        },
        "category_catalog": [],
        "dialogue_analyses": [],
        "errors": [],
        "summary": {},
    }


def validate_resume_output(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("format") != "clarification_feature_mining":
        raise ValueError("--resume output is not a clarification_feature_mining result")
    output = dict(value)
    for field in ("category_catalog", "dialogue_analyses", "errors"):
        if not isinstance(output.get(field), list):
            raise ValueError(f"--resume output field {field!r} must be a list")
    return output


def completed_record_indices(output: Mapping[str, Any]) -> set[int]:
    completed: set[int] = set()
    analyses = output.get("dialogue_analyses", [])
    if not isinstance(analyses, list):
        return completed
    for analysis in analyses:
        if not isinstance(analysis, Mapping):
            continue
        source = analysis.get("source_dialogue", {})
        if isinstance(source, Mapping):
            index = as_int(source.get("record_index"))
            if index is not None:
                completed.add(index)
    return completed


def remove_old_errors(output: dict[str, Any], record_index: Any) -> None:
    errors = output.get("errors", [])
    if not isinstance(errors, list):
        return
    output["errors"] = [
        error
        for error in errors
        if not isinstance(error, Mapping) or error.get("record_index") != record_index
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serially mine LLM-discovered human-like user behaviours from clarification_eval.py output."
    )
    parser.add_argument("input", help="JSON output produced by clarification_eval.py")
    parser.add_argument("--url", default=os.getenv("LLM_JUDGE_URL"), help="OpenAI-compatible base URL or /chat/completions endpoint (env: LLM_JUDGE_URL)")
    parser.add_argument("--model-name", "--model_name", dest="model_name", default=os.getenv("LLM_JUDGE_MODEL_NAME"), help="Model name (env: LLM_JUDGE_MODEL_NAME)")
    parser.add_argument("--api-key", "--api_key", dest="api_key", default=os.getenv("LLM_JUDGE_API_KEY", ""), help="API key (env: LLM_JUDGE_API_KEY); may be empty for local endpoints")
    parser.add_argument("--output", help="Output JSON path; default: <input stem>.feature_mining.json")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file and skip completed dialogues")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds (default: 120)")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per request mode (default: 2)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Model temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum output tokens per dialogue (default: 8192)")
    parser.add_argument("--max-categories-in-prompt", type=int, default=100, help="Maximum previous categories supplied to each model call (default: 100)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed dialogue instead of recording and continuing")
    parser.add_argument("--no-progress", action="store_true", help="Disable serial progress output on stderr")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("--url is required (or set LLM_JUDGE_URL)")
    if not args.model_name:
        parser.error("--model-name is required (or set LLM_JUDGE_MODEL_NAME)")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.max_categories_in_prompt < 1:
        parser.error("--max-categories-in-prompt must be at least 1")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + ".feature_mining.json")
    try:
        source_report = load_json_object(input_path)
        eligible = eligible_dialogues(source_report)
        judge = FeatureMiningJudge(
            args.url,
            args.model_name,
            args.api_key,
            args.timeout,
            args.max_retries,
            args.temperature,
            args.max_tokens,
        )
        if output_path.exists() and args.resume:
            output = validate_resume_output(load_json_object(output_path))
            output["run"] = {
                **dict(output.get("run", {})),
                "endpoint": judge.endpoint,
                "model_name": judge.model_name,
                "serial": True,
                "temperature": judge.temperature,
                "max_tokens": judge.max_tokens,
                "thinking_disabled": True,
                "status": "running",
                "resumed_at": utc_now(),
            }
        elif output_path.exists() and not args.overwrite:
            raise ValueError(f"output already exists: {output_path}; use --resume or --overwrite")
        else:
            output = new_output(input_path, judge)
        done = completed_record_indices(output)
        pending = [dialogue for dialogue in eligible if as_int(dialogue.get("record_index")) not in done]
        if not args.no_progress:
            print(
                f"[clarification-feature-mining] eligible={len(eligible)} completed={len(done)} pending={len(pending)} serial=true",
                file=sys.stderr,
                flush=True,
            )
        update_summary(output, source_report, eligible)
        atomic_write_json(output_path, output)

        for pending_index, dialogue in enumerate(pending, 1):
            record_index = dialogue.get("record_index")
            call_sno = dialogue.get("call_sno")
            sequence_index = len(output["dialogue_analyses"])
            visible_catalogue = categories_for_prompt(output["category_catalog"], args.max_categories_in_prompt)
            user_prompt = build_user_prompt(dialogue, visible_catalogue)
            if not args.no_progress:
                print(
                    f"[clarification-feature-mining] {pending_index}/{len(pending)} call_sno={call_sno} 开始分析",
                    file=sys.stderr,
                    flush=True,
                )
            judge_trace: dict[str, Any] | None = None
            model_raw: Mapping[str, Any] | None = None
            try:
                judge_trace = judge.analyse(user_prompt)
                model_raw = judge_trace["parsed_response"]
                if not isinstance(model_raw, Mapping):
                    raise ValueError("LLM response JSON must be an object")
                normalised, new_categories, warnings = normalise_model_judgement(
                    model_raw,
                    visible_categories=visible_catalogue,
                    all_categories=output["category_catalog"],
                    pair_count=len(dialogue["clarification_pairs"]),
                )
                output["category_catalog"] = apply_category_assignments(
                    output["category_catalog"],
                    new_categories,
                    normalised,
                    dialogue,
                    sequence_index,
                )
                output["dialogue_analyses"].append(
                    {
                        "sequence_index": sequence_index,
                        "completed_at": utc_now(),
                        "source_dialogue": serialise_dialogue_for_prompt(dialogue),
                        "catalogue_before": visible_catalogue,
                        "process": judge_trace,
                        "model_judgement": model_raw,
                        "normalised_judgement": normalised,
                        "new_category_ids": [item["category_id"] for item in new_categories],
                        "normalisation_warnings": warnings,
                    }
                )
                remove_old_errors(output, record_index)
                update_summary(output, source_report, eligible)
                atomic_write_json(output_path, output)
                if not args.no_progress:
                    print(
                        f"[clarification-feature-mining] {pending_index}/{len(pending)} call_sno={call_sno} 完成，新增类别 {len(new_categories)} 个，当前类别 {len(output['category_catalog'])} 个",
                        file=sys.stderr,
                        flush=True,
                    )
            except KeyboardInterrupt:
                output["run"]["status"] = "interrupted"
                output["run"]["interrupted_at"] = utc_now()
                update_summary(output, source_report, eligible)
                atomic_write_json(output_path, output)
                print(
                    f"[clarification-feature-mining] 已中断；已完成结果已保存到 {output_path}",
                    file=sys.stderr,
                    flush=True,
                )
                return 130
            except Exception as exc:  # noqa: BLE001 - one dialogue must not discard accumulated taxonomy
                error_entry: dict[str, Any] = {
                    "sequence_index": sequence_index,
                    "record_index": record_index,
                    "call_sno": call_sno,
                    "failed_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "catalogue_before": visible_catalogue,
                    "user_prompt": user_prompt,
                }
                if isinstance(exc, JudgeRequestError):
                    error_entry["request_attempts"] = exc.attempts
                if judge_trace is not None:
                    error_entry["process"] = judge_trace
                if model_raw is not None:
                    error_entry["model_judgement"] = model_raw
                remove_old_errors(output, record_index)
                output["errors"].append(error_entry)
                update_summary(output, source_report, eligible)
                atomic_write_json(output_path, output)
                print(
                    f"[clarification-feature-mining] {pending_index}/{len(pending)} call_sno={call_sno} 失败（{type(exc).__name__}: {exc}），已保存并继续",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    output["run"]["status"] = "failed"
                    update_summary(output, source_report, eligible)
                    atomic_write_json(output_path, output)
                    return 1

        output["run"]["status"] = "completed"
        output["run"]["completed_at"] = utc_now()
        update_summary(output, source_report, eligible)
        atomic_write_json(output_path, output)
        if not args.no_progress:
            print(
                f"[clarification-feature-mining] 完成：对话 {output['summary']['completed_dialogue_count']}/{len(eligible)}，类别 {output['summary']['category_count']}，错误 {output['summary']['error_count']}；结果：{output_path}",
                file=sys.stderr,
                flush=True,
            )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

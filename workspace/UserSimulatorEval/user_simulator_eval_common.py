#!/usr/bin/env python3
"""Shared, dependency-free helpers for user-simulator evaluation tools.

The source dialogue export is not always valid JSONL: some exports concatenate
multiple JSON objects without a newline.  ``load_json_records`` intentionally
accepts JSON arrays, JSONL, one JSON object, and such concatenated objects.
All OpenAI-compatible requests made here explicitly disable Qwen thinking.
"""

from __future__ import annotations

import json
import re
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


class LLMRequestError(RuntimeError):
    """An OpenAI-compatible request failed after its configured retries."""


@dataclass(frozen=True)
class Turn:
    index: int
    role: str
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "role": self.role, "text": self.text}


def resolve_chat_endpoint(url: str) -> str:
    """Turn an OpenAI-compatible base URL into a chat-completions URL."""

    value = str(url or "").strip().rstrip("/")
    if not value:
        raise ValueError("LLM URL cannot be empty")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def content_to_text(content: Any) -> str:
    """Normalise OpenAI message content variants to visible text."""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Mapping) and isinstance(content.get("text"), str):
        return content["text"].strip()
    if isinstance(content, list):
        values: list[str] = []
        for item in content:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                values.append(item["text"])
        return "".join(values).strip()
    return str(content or "").strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object while tolerating a short model preamble/fence."""

    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        try:
            parsed, _ = decoder.raw_decode(candidate[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("模型没有返回合法 JSON 对象")


class OpenAIChatClient:
    """Small OpenAI-compatible client with JSON-object fallback support."""

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        if not str(model or "").strip():
            raise ValueError("模型名称不能为空")
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0")
        self.endpoint = resolve_chat_endpoint(url)
        self.model = str(model).strip()
        self.api_key = str(api_key or "")
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_object: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Return content and request metadata from one no-thinking call.

        JSON mode first uses ``response_format=json_object``.  A number of
        OpenAI-compatible serving stacks do not implement that extension, so
        it safely retries the same prompt without it.
        """

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if seed is not None:
            payload["seed"] = int(seed)
        variants: list[tuple[str, dict[str, Any]]] = [("plain", payload)]
        if json_object:
            variants.insert(0, ("json_object", {**payload, "response_format": {"type": "json_object"}}))

        failures: list[str] = []
        for mode, body in variants:
            for retry_index in range(self.max_retries + 1):
                try:
                    content = self._post(body)
                    result: dict[str, Any] = {
                        "content": content,
                        "request_mode": mode,
                        "request_body": body,
                    }
                    if json_object:
                        result["json"] = extract_json_object(content)
                    return result
                except Exception as error:  # noqa: BLE001 - retries intentionally include parse failures
                    failures.append(f"{mode}/第{retry_index + 1}次: {type(error).__name__}: {error}")
                    if retry_index < self.max_retries:
                        time.sleep(min(2.0, 0.5 * (2**retry_index)))
        raise LLMRequestError("; ".join(failures[-4:]))

    def _post(self, payload: Mapping[str, Any]) -> str:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urlrequest.Request(self.endpoint, data=request_data, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urlerror.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise LLMRequestError(f"HTTP {error.code}：{detail[:1000]}") from error
        except urlerror.URLError as error:
            raise LLMRequestError(f"无法访问 {self.endpoint}：{error.reason}") from error
        try:
            parsed = json.loads(raw)
            choice = parsed["choices"][0]
            if not isinstance(choice, Mapping):
                raise TypeError("choices[0] 不是对象")
            message = choice.get("message")
            content = message.get("content") if isinstance(message, Mapping) else choice.get("text")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise LLMRequestError(f"模型响应格式不符合 OpenAI Chat Completions：{raw[:1000]}") from error
        visible = content_to_text(content)
        if not visible:
            raise LLMRequestError("模型返回了空文本")
        return visible


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON arrays, JSONL, or whitespace-separated JSON object streams."""

    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    stripped = raw.lstrip("\ufeff \t\r\n")
    if not stripped:
        return []

    records: list[Any]
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError(f"{source}: 顶层 JSON 数组格式错误")
        records = parsed
    else:
        # ``raw_decode`` correctly handles regular JSONL as well as a source
        # that accidentally joins objects as ``}{`` without a newline.
        decoder = json.JSONDecoder()
        records = []
        position = 0
        while position < len(raw):
            while position < len(raw) and raw[position].isspace():
                position += 1
            if position >= len(raw):
                break
            try:
                record, position = decoder.raw_decode(raw, position)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}: 位置 {position} 不是合法 JSON 对象：{error}") from error
            records.append(record)

    converted: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{source}: 第 {index + 1} 条不是 JSON 对象")
        converted.append(dict(record))
    return converted


def _normalise_role(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return ROLE_ALIASES.get(normalized)


def parse_turns(chat_content: Any) -> list[Turn]:
    """Parse list-style or ``用户：...`` / ``客服：...`` dialogue content."""

    turns: list[Turn] = []
    if isinstance(chat_content, (list, tuple)):
        for item in chat_content:
            if isinstance(item, Mapping):
                role = _normalise_role(item.get("role") or item.get("speaker")) or "unknown"
                content = item.get("content", item.get("text", ""))
            else:
                role, content = "unknown", item
            text = str(content or "").strip()
            if text:
                turns.append(Turn(len(turns), role, text))
        return turns
    if not isinstance(chat_content, str):
        return turns
    pattern = re.compile(r"^\s*(用户|客户|顾客|客服|坐席|机器人|agent|assistant|user)\s*[:：]\s*(.*)$", re.I)
    for raw_line in chat_content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            role = _normalise_role(match.group(1)) or "unknown"
            text = match.group(2).strip()
        else:
            role, text = "unknown", line
        if text:
            turns.append(Turn(len(turns), role, text))
    return turns


def as_string_list(value: Any) -> list[str]:
    """Return de-duplicated non-empty strings while preserving first order."""

    source = value if isinstance(value, (list, tuple)) else []
    values: list[str] = []
    seen: set[str] = set()
    for item in source:
        text = str(item or "").strip()
        if text and text not in seen:
            values.append(text)
            seen.add(text)
    return values


def write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    temporary.replace(output)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL test set and reject accidental non-object rows."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: JSONL 解析失败：{error}") from error
        if not isinstance(record, Mapping):
            raise ValueError(f"{source}:{line_number}: 每条测试样本必须是 JSON 对象")
        records.append(dict(record))
    return records

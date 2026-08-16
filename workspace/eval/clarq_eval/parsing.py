from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


class PolicyProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ParsedPolicyTurn:
    content: str
    cleaned_content: str
    finish_reason: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.tool_calls and self.cleaned_content.lower() == "complete"


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_THINK_PATTERN = re.compile(r"<(?:think|analysis)>.*?</(?:think|analysis)>", re.DOTALL | re.IGNORECASE)


def _parse_arguments(value: Any, context: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise PolicyProtocolError(f"{context} arguments must be a JSON object or string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise PolicyProtocolError(f"{context} arguments are not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise PolicyProtocolError(f"{context} arguments must decode to a JSON object")
    return parsed


def _parse_standard_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls, start=1):
        function = raw_call.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            raise PolicyProtocolError(f"tool call {index} has no function name")
        calls.append(
            ToolCall(
                call_id=str(raw_call.get("id") or f"call_{index}"),
                name=name,
                arguments=_parse_arguments(function.get("arguments", {}), f"tool call {index}"),
            )
        )
    return calls


def _parse_hermes_tool_calls(content: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, payload in enumerate(_TOOL_CALL_PATTERN.findall(content), start=1):
        try:
            raw_call = json.loads(payload)
        except json.JSONDecodeError as error:
            raise PolicyProtocolError(f"Hermes tool call {index} is not valid JSON: {error}") from error
        if not isinstance(raw_call, dict):
            raise PolicyProtocolError(f"Hermes tool call {index} must be a JSON object")
        name = str(raw_call.get("name") or "").strip()
        if not name:
            raise PolicyProtocolError(f"Hermes tool call {index} has no name")
        calls.append(
            ToolCall(
                call_id=f"call_{index}",
                name=name,
                arguments=_parse_arguments(raw_call.get("arguments", {}), f"Hermes tool call {index}"),
            )
        )
    return calls


def parse_policy_response(response: dict[str, Any]) -> ParsedPolicyTurn:
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise PolicyProtocolError("Policy response has no choices[0].message") from error

    raw_content = message.get("content")
    content = "" if raw_content is None else str(raw_content)
    raw_tool_calls = message.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list):
        raise PolicyProtocolError("message.tool_calls must be a list")

    tool_calls = _parse_standard_tool_calls(raw_tool_calls) if raw_tool_calls else _parse_hermes_tool_calls(content)
    content_without_calls = _TOOL_CALL_PATTERN.sub("", content)
    cleaned_content = _THINK_PATTERN.sub("", content_without_calls).strip()
    violations: list[str] = []
    if len(tool_calls) > 1:
        violations.append("multiple_tool_calls")
    if tool_calls and cleaned_content:
        violations.append("text_with_tool_call")

    return ParsedPolicyTurn(
        content=content,
        cleaned_content=cleaned_content,
        finish_reason=choice.get("finish_reason"),
        tool_calls=tool_calls,
        violations=violations,
    )


def canonical_assistant_message(turn: ParsedPolicyTurn) -> dict[str, Any]:
    # Raw Hermes responses embed the call in content. Once normalized into the
    # OpenAI tool_calls field, retaining that markup would replay the call twice
    # through Qwen's chat template on the next policy request.
    content = turn.cleaned_content if turn.tool_calls else turn.content
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in turn.tool_calls
        ]
    return message


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise PolicyProtocolError("Response does not contain a JSON object")
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as error:
            raise PolicyProtocolError(f"Response contains invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise PolicyProtocolError("Response JSON must be an object")
    return value

from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

from .clients import (
    INVALID_CLARIFICATION_RESPONSE,
    UNKNOWN_CLARIFICATION_RESPONSE,
    OpenAIChatClient,
    TrajectoryJudge,
    UserSimulator,
)
from .models import EvaluationSample
from .parsing import canonical_assistant_message, parse_policy_response


SYSTEM_PROMPT = """You are a conversational case-retrieval agent.

Use exactly one action at a time:
- Call clarify_user when one missing user-known fact would materially change
  which cases are relevant. Ask one concise, discriminative question.
- Call search_case when the request is specific enough. Build a focused query
  from the original request and confirmed clarifications without assumptions.
- Output exactly Complete when the latest retrieved cases are sufficient.

Do not answer the user's technical question yourself. Search before completing.
If retrieved cases differ substantially from the request, clarify one useful
missing detail or reformulate the query. Do not repeat answered questions.
You may call search_case at most four times. Output no explanation with Complete.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "clarify_user",
            "description": "Ask the simulated user one concise question for a missing user-known fact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "One concise and discriminative clarification question.",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_case",
            "description": "Retrieve the five most relevant cases from the hybrid ClarQ case index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A focused retrieval query grounded in the user request and confirmed clarifications."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
]


class Retriever(Protocol):
    def search(self, query: str, max_results: int | None = None) -> list[Any]: ...


def _result_value(result: Any, attribute: str, *dictionary_keys: str, default: Any = "") -> Any:
    if hasattr(result, attribute):
        return getattr(result, attribute)
    if isinstance(result, dict):
        for key in dictionary_keys:
            if key in result:
                return result[key]
    return default


def serialize_search_results(results: list[Any], content_chars: int) -> list[dict[str, Any]]:
    serialized = []
    for rank, result in enumerate(results, start=1):
        serialized.append(
            {
                "rank": rank,
                "case_id": str(_result_value(result, "url", "case_id", "caseID")),
                "title": str(_result_value(result, "title", "title", "case_name")),
                "content": str(_result_value(result, "content", "content", "text"))[:content_chars],
                "score": _result_value(result, "score", "score", default=None),
            }
        )
    return serialized


def visible_search_response(cases: list[dict[str, Any]], notice: str | None = None) -> str:
    body: dict[str, Any] = {
        "cases": [
            {
                "rank": case["rank"],
                "case_id": case["case_id"],
                "title": case["title"],
                "content": case["content"],
            }
            for case in cases
        ]
    }
    if notice:
        body["notice"] = notice
    return json.dumps(body, ensure_ascii=False)


def _normalized_question(question: str) -> str:
    return re.sub(r"\W+", "", question, flags=re.UNICODE).lower()


class EvaluationRunner:
    def __init__(
        self,
        *,
        policy_client: OpenAIChatClient,
        user_simulator: UserSimulator,
        retriever: Retriever,
        judge: TrajectoryJudge | None = None,
        max_turns: int = 6,
        max_searches: int = 4,
        top_k: int = 5,
        case_content_chars: int = 1000,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = 42,
        enable_thinking: bool = False,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if max_turns <= 0 or max_searches <= 0 or top_k <= 0:
            raise ValueError("max_turns, max_searches, and top_k must be positive")
        self.policy_client = policy_client
        self.user_simulator = user_simulator
        self.retriever = retriever
        self.judge = judge
        self.max_turns = max_turns
        self.max_searches = max_searches
        self.top_k = top_k
        self.case_content_chars = case_content_chars
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.system_prompt = system_prompt

    def run(self, sample: EvaluationSample) -> dict[str, Any]:
        started_at = time.monotonic()
        baseline_results = serialize_search_results(
            self.retriever.search(sample.initial_question, self.top_k),
            self.case_content_chars,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": sample.initial_question},
        ]
        events: list[dict[str, Any]] = []
        last_search_results = baseline_results
        final_results: list[dict[str, Any]] = []
        clarification_count = 0
        useful_clarification_count = 0
        unknown_clarification_count = 0
        invalid_clarification_count = 0
        duplicate_clarification_count = 0
        search_count = 0
        search_attempt_count = 0
        pending_clarifications = 0
        clarification_before_ids: list[str] | None = None
        asked_questions: set[str] = set()
        stop_reason = "max_turns"
        terminal_text = ""

        for turn_number in range(1, self.max_turns + 1):
            response = self.policy_client.chat(
                messages,
                tools=TOOLS,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                extra_payload={
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                },
            )
            parsed = parse_policy_response(response)
            event: dict[str, Any] = {
                "turn": turn_number,
                "assistant_content": parsed.content,
                "finish_reason": parsed.finish_reason,
                "violations": list(parsed.violations),
            }

            if len(parsed.tool_calls) > 1:
                parsed.tool_calls = parsed.tool_calls[:1]
            messages.append(canonical_assistant_message(parsed))

            if parsed.is_complete:
                event["action"] = {"type": "complete"}
                events.append(event)
                stop_reason = "complete"
                terminal_text = parsed.cleaned_content
                break

            if not parsed.tool_calls:
                event["action"] = {"type": "unexpected_text", "text": parsed.cleaned_content}
                event["violations"].append("unexpected_terminal_text")
                events.append(event)
                stop_reason = "unexpected_text"
                terminal_text = parsed.cleaned_content
                break

            call = parsed.tool_calls[0]
            event["action"] = {"type": call.name, "arguments": call.arguments}

            if call.name == "clarify_user":
                question = call.arguments.get("question")
                if not isinstance(question, str) or not question.strip():
                    event["violations"].append("invalid_clarify_arguments")
                    events.append(event)
                    stop_reason = "invalid_tool_arguments"
                    break

                question = question.strip()
                normalized = _normalized_question(question)
                is_duplicate = bool(normalized and normalized in asked_questions)
                asked_questions.add(normalized)
                duplicate_clarification_count += int(is_duplicate)
                clarification_count += 1
                if pending_clarifications == 0:
                    clarification_before_ids = [case["case_id"] for case in last_search_results]
                pending_clarifications += 1

                reply = self.user_simulator.answer(sample, question)
                if reply in sample.known_info:
                    useful_clarification_count += 1
                    clarification_type = "known_info"
                elif reply == INVALID_CLARIFICATION_RESPONSE:
                    invalid_clarification_count += 1
                    clarification_type = "invalid"
                else:
                    unknown_clarification_count += 1
                    clarification_type = "unknown"

                event.update(
                    {
                        "question": question,
                        "tool_response": reply,
                        "clarification_type": clarification_type,
                        "duplicate_question": is_duplicate,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "content": reply,
                    }
                )
                events.append(event)
                continue

            if call.name == "search_case":
                query = call.arguments.get("query")
                if not isinstance(query, str) or not query.strip():
                    event["violations"].append("invalid_search_arguments")
                    events.append(event)
                    stop_reason = "invalid_tool_arguments"
                    break

                query = query.strip()
                search_attempt_count += 1
                event["query"] = query
                event["search_attempt"] = search_attempt_count
                if search_count >= self.max_searches:
                    response_text = visible_search_response(
                        final_results,
                        notice=(
                            f"Search limit reached ({self.max_searches}). "
                            "These are the latest cases; use them and finish."
                        ),
                    )
                    event.update(
                        {
                            "tool_response": response_text,
                            "search_results": final_results,
                            "search_executed": False,
                            "search_limit_reached": True,
                        }
                    )
                else:
                    search_count += 1
                    cases = serialize_search_results(
                        self.retriever.search(query, self.top_k),
                        self.case_content_chars,
                    )
                    response_text = visible_search_response(cases)
                    event.update(
                        {
                            "tool_response": response_text,
                            "search_results": cases,
                            "search_executed": True,
                            "search_number": search_count,
                            "clarifications_since_previous_search": pending_clarifications,
                        }
                    )
                    if pending_clarifications:
                        event["pre_clarification_case_ids"] = clarification_before_ids or []
                    final_results = cases
                    last_search_results = cases
                    pending_clarifications = 0
                    clarification_before_ids = None

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "content": response_text,
                    }
                )
                events.append(event)
                continue

            event["violations"].append("unsupported_tool")
            events.append(event)
            stop_reason = "unsupported_tool"
            break

        result: dict[str, Any] = {
            "schema_version": "1.0",
            "sample_id": sample.sample_id,
            "domain": sample.domain,
            "target_case_id": sample.target_case_id,
            "initial_question": sample.initial_question,
            "core_intent": sample.core_intent,
            "known_info": list(sample.known_info),
            "baseline_results": baseline_results,
            "events": events,
            "final_results": final_results,
            "stop_reason": stop_reason,
            "terminal_text": terminal_text,
            "turn_count": len(events),
            "clarification_count": clarification_count,
            "useful_clarification_count": useful_clarification_count,
            "unknown_clarification_count": unknown_clarification_count,
            "invalid_clarification_count": invalid_clarification_count,
            "duplicate_clarification_count": duplicate_clarification_count,
            "search_count": search_count,
            "search_attempt_count": search_attempt_count,
            "protocol_violation_count": sum(len(event.get("violations", [])) for event in events),
            "elapsed_seconds": time.monotonic() - started_at,
            "runner_config": {
                "max_turns": self.max_turns,
                "max_searches": self.max_searches,
                "top_k": self.top_k,
                "case_content_chars": self.case_content_chars,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                "enable_thinking": self.enable_thinking,
            },
            "error": None,
        }
        if self.judge is not None:
            try:
                result["judge"] = self.judge.judge(sample, result)
                result["judge_error"] = None
            except Exception as error:  # Keep deterministic metrics even when the optional judge fails.
                result["judge"] = None
                result["judge_error"] = f"{type(error).__name__}: {error}"
        else:
            result["judge"] = None
            result["judge_error"] = None
        return result

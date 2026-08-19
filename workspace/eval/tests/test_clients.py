from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from clarq_eval.clients import (
    ChatAPIError,
    SuccessJudge,
    TrajectoryJudge,
    UserSimulator,
)
from clarq_eval.models import EvaluationSample


SAMPLE = EvaluationSample(
    sample_id="sample-1",
    domain="money",
    target_case_id="target",
    initial_question="initial question",
    core_intent="core intent",
    known_info=(),
    target_case_title="Canonical title",
    target_case_content="Canonical answer content",
)


class FakeChatClient:
    def __init__(self, content: str = "UNKNOWN", error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[tuple[list[dict], dict]] = []

    def chat(self, messages: list[dict], **kwargs) -> dict:
        self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return {"choices": [{"message": {"content": self.content}}]}

class UserSimulatorTests(unittest.TestCase):
    def test_clarification_reply_uses_allowed_selection(self) -> None:
        client = FakeChatClient(content="KNOWN_INFO_1")
        simulator = UserSimulator(client)
        sample = EvaluationSample(
            sample_id="sample-1",
            domain="money",
            target_case_id="target",
            initial_question="initial question",
            core_intent="core intent",
            known_info=("Version 2",),
        )

        reply = simulator.answer(sample, "Which version?")

        self.assertEqual(reply, "Version 2")
        messages, kwargs = client.calls[0]
        self.assertIn("Initial user question", messages[0]["content"])
        self.assertEqual(kwargs["max_tokens"], 16)


class SuccessJudgeTests(unittest.TestCase):
    def test_prompt_contains_question_reference_and_retrieved_case_content(self) -> None:
        client = FakeChatClient(content='{"can_answer": true, "reason": "The retrieved solution applies."}')

        judgment = SuccessJudge(client).judge(
            SAMPLE,
            [{"case_id": "ignored", "title": "Retrieved title", "content": "Retrieved solution"}],
        )

        self.assertEqual(judgment, {"can_answer": True, "reason": "The retrieved solution applies."})
        messages, kwargs = client.calls[0]
        prompt = messages[0]["content"]
        self.assertIn("initial question", prompt)
        self.assertIn("Canonical title", prompt)
        self.assertIn("Canonical answer content", prompt)
        self.assertIn("Retrieved title", prompt)
        self.assertIn("Retrieved solution", prompt)
        self.assertNotIn("ignored", prompt)
        self.assertEqual(kwargs["max_tokens"], 256)
        self.assertEqual(kwargs["extra_payload"]["response_format"]["type"], "json_schema")

    def test_http_400_retries_without_json_schema(self) -> None:
        class SchemaRejectingClient(FakeChatClient):
            def chat(self, messages: list[dict], **kwargs) -> dict:
                self.calls.append((messages, kwargs))
                if len(self.calls) == 1:
                    raise ChatAPIError("unsupported response format", status_code=400)
                return {"choices": [{"message": {"content": '{"can_answer": false, "reason": "No match."}'}}]}

        client = SchemaRejectingClient()
        judgment = SuccessJudge(client).judge(SAMPLE, [{"title": "Other", "content": "Other answer"}])

        self.assertFalse(judgment["can_answer"])
        self.assertIn("response_format", client.calls[0][1]["extra_payload"])
        self.assertNotIn("response_format", client.calls[1][1]["extra_payload"])

    def test_invalid_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(Exception, "can_answer"):
            SuccessJudge(FakeChatClient(content='{"can_answer": "yes", "reason": "bad type"}')).judge(
                SAMPLE,
                [{"title": "Other", "content": "Other answer"}],
            )


class TrajectoryJudgeTests(unittest.TestCase):
    def test_prompt_excludes_final_retrieved_cases(self) -> None:
        client = FakeChatClient(
            content=(
                '{"irrelevant_question": false, "repeated_question": false, '
                '"unclear_question": false, "nonstandard_language": false, '
                '"score": 4, "reason": "good"}'
            )
        )
        result = {
            "events": [
                {
                    "turn": 1,
                    "action": {"type": "search_case", "query": "query"},
                    "search_results": [{"rank": 1, "case_id": "case-1", "title": "Case title"}],
                }
            ],
            "final_results": [
                {"case_id": "case-1", "title": "Case title", "content": "final case content"}
            ],
        }

        judgment = TrajectoryJudge(client).judge(SAMPLE, result)

        self.assertEqual(judgment["score"], 4)
        prompt = client.calls[0][0][0]["content"]
        self.assertNotIn("Final retrieved cases", prompt)
        self.assertNotIn("final case content", prompt)
        self.assertIn("Return exactly one valid JSON object", prompt)
        self.assertIn('"reason": "A brief justification based only on the trajectory."', prompt)


if __name__ == "__main__":
    unittest.main()

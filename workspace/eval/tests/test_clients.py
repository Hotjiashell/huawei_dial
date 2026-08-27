from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from clarq_eval.clients import (
    ChatAPIError,
    OpenAIChatClient,
    SuccessJudge,
    TrajectoryJudge,
    UserSimulator,
    response_content,
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


class RecordingOpenAIChatClient(OpenAIChatClient):
    def __init__(self) -> None:
        super().__init__("http://service/v1", "Qwen/Qwen3-32B")
        self.requests: list[tuple[str, str, dict | None]] = []

    def _request(self, method: str, url: str, json_body: dict | None = None) -> dict:
        self.requests.append((method, url, json_body))
        return {"choices": [{"text": "KNOWN_INFO_1"}]}


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages: list[dict], **kwargs) -> str:
        self.calls.append((messages, kwargs))
        return "<qwen3-rendered-prompt>"


class PolicyModelModeTests(unittest.TestCase):
    def test_qwen3_policy_renders_tools_locally_and_calls_completions(self) -> None:
        client = RecordingOpenAIChatClient()
        tokenizer = FakeTokenizer()
        client._tokenizer_cache["/models/policy-qwen3"] = tokenizer
        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "search_case"}}]

        response = client.policy_chat(
            messages,
            tools=tools,
            model_mode="qwen3",
            tokenizer_path="/models/policy-qwen3",
            enable_thinking=False,
            max_tokens=16,
        )

        self.assertEqual(response_content(response), "KNOWN_INFO_1")
        self.assertEqual(tokenizer.calls[0][0], messages)
        self.assertEqual(
            tokenizer.calls[0][1],
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
                "tools": tools,
            },
        )
        _, url, body = client.requests[0]
        self.assertTrue(url.endswith("/completions"))
        self.assertEqual(body["prompt"], "<qwen3-rendered-prompt>")
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)
        self.assertNotIn("parallel_tool_calls", body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_qwen3_policy_passes_enabled_thinking_to_local_template(self) -> None:
        client = RecordingOpenAIChatClient()
        tokenizer = FakeTokenizer()
        client._tokenizer_cache["/models/policy-qwen3"] = tokenizer

        client.policy_chat(
            [{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "search_case"}}],
            model_mode="qwen3",
            tokenizer_path="/models/policy-qwen3",
            enable_thinking=True,
        )

        self.assertTrue(tokenizer.calls[0][1]["enable_thinking"])

    def test_qwen3_5_policy_uses_chat_completions_with_tools(self) -> None:
        client = RecordingOpenAIChatClient()
        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "search_case"}}]

        client.policy_chat(
            messages,
            tools=tools,
            model_mode="qwen3_5",
            enable_thinking=False,
            max_tokens=16,
        )

        _, url, body = client.requests[0]
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertEqual(body["messages"], messages)
        self.assertEqual(body["tools"], tools)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertFalse(body["parallel_tool_calls"])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})


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

    def test_qwen3_uses_tokenizer_template_and_completions_endpoint(self) -> None:
        client = RecordingOpenAIChatClient()
        tokenizer = FakeTokenizer()
        client._tokenizer_cache["/models/qwen3"] = tokenizer

        response = client.user_simulator_chat(
            [{"role": "user", "content": "hello"}],
            model_mode="qwen3",
            tokenizer_path="/models/qwen3",
            max_tokens=16,
            extra_payload={
                "chat_template_kwargs": {"enable_thinking": True},
                "structured_outputs": {"choice": ["KNOWN_INFO_1"]},
            },
        )

        self.assertEqual(response_content(response), "KNOWN_INFO_1")
        self.assertEqual(tokenizer.calls[0][0], [{"role": "user", "content": "hello"}])
        self.assertEqual(tokenizer.calls[0][1]["enable_thinking"], False)
        self.assertTrue(tokenizer.calls[0][1]["add_generation_prompt"])
        _, url, body = client.requests[0]
        self.assertTrue(url.endswith("/completions"))
        self.assertEqual(body["prompt"], "<qwen3-rendered-prompt>")
        self.assertNotIn("chat_template_kwargs", body)
        self.assertEqual(body["structured_outputs"], {"choice": ["KNOWN_INFO_1"]})

    def test_qwen3_user_simulator_uses_the_model_mode_request_path(self) -> None:
        client = RecordingOpenAIChatClient()
        tokenizer = FakeTokenizer()
        client._tokenizer_cache["/models/qwen3"] = tokenizer
        sample = EvaluationSample(
            sample_id="sample-1",
            domain="money",
            target_case_id="target",
            initial_question="initial question",
            core_intent="core intent",
            known_info=("Version 2",),
        )

        reply = UserSimulator(
            client,
            model_mode="qwen3",
            tokenizer_path="/models/qwen3",
        ).answer(sample, "Which version?")

        self.assertEqual(reply, "Version 2")
        self.assertEqual(tokenizer.calls[0][1]["enable_thinking"], False)
        _, url, body = client.requests[0]
        self.assertTrue(url.endswith("/completions"))
        self.assertEqual(body["structured_outputs"], {"choice": ["KNOWN_INFO_1", "UNKNOWN", "INVALID_QUESTION"]})

    def test_qwen3_5_uses_chat_template_kwargs_on_chat_completions(self) -> None:
        client = RecordingOpenAIChatClient()

        client.user_simulator_chat(
            [{"role": "user", "content": "hello"}],
            model_mode="qwen3_5",
            max_tokens=16,
            extra_payload={"chat_template_kwargs": {"enable_thinking": True}},
        )

        _, url, body = client.requests[0]
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})


class SuccessJudgeTests(unittest.TestCase):
    def test_prompt_contains_question_reference_and_retrieved_case_content(self) -> None:
        client = FakeChatClient(
            content=(
                '{"can_answer": true, "answer_case_title": "Retrieved title", '
                '"reason": "The retrieved solution applies."}'
            )
        )

        judgment = SuccessJudge(client).judge(
            SAMPLE,
            [{"case_id": "ignored", "title": "Retrieved title", "content": "Retrieved solution"}],
        )

        self.assertEqual(
            judgment,
            {
                "can_answer": True,
                "answer_case_title": "Retrieved title",
                "reason": "The retrieved solution applies.",
            },
        )
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
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"can_answer": false, "answer_case_title": null, '
                                    '"reason": "No match."}'
                                )
                            }
                        }
                    ]
                }

        client = SchemaRejectingClient()
        judgment = SuccessJudge(client).judge(SAMPLE, [{"title": "Other", "content": "Other answer"}])

        self.assertFalse(judgment["can_answer"])
        self.assertIn("response_format", client.calls[0][1]["extra_payload"])
        self.assertNotIn("response_format", client.calls[1][1]["extra_payload"])

    def test_invalid_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(Exception, "can_answer"):
            SuccessJudge(
                FakeChatClient(
                    content=(
                        '{"can_answer": "yes", "answer_case_title": null, "reason": "bad type"}'
                    )
                )
            ).judge(
                SAMPLE,
                [{"title": "Other", "content": "Other answer"}],
            )

    def test_selected_case_title_must_match_a_retrieved_case(self) -> None:
        with self.assertRaisesRegex(Exception, "answer_case_title"):
            SuccessJudge(
                FakeChatClient(
                    content=(
                        '{"can_answer": true, "answer_case_title": "Invented title", '
                        '"reason": "bad selection"}'
                    )
                )
            ).judge(SAMPLE, [{"title": "Other", "content": "Other answer"}])


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

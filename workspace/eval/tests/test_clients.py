from __future__ import annotations

import ast
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from clarq_eval.clients import (
    FAILED_DONE_TOKEN,
    SATISFIED_DONE_TOKEN,
    ChatAPIError,
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
)


class FakeChatClient:
    def __init__(self, content: str = SATISFIED_DONE_TOKEN, error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[tuple[list[dict], dict]] = []

    def chat(self, messages: list[dict], **kwargs) -> dict:
        self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return {"choices": [{"message": {"content": self.content}}]}


def training_case_judge_prompt() -> str:
    workspace = Path(__file__).resolve().parents[2]
    source = workspace / "verl_dial-main-h800" / "examples" / "clarq_grpo" / "agent.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "UserSimulatorClient":
            for statement in node.body:
                if not isinstance(statement, ast.Assign):
                    continue
                if any(
                    isinstance(target, ast.Name) and target.id == "CASE_JUDGE_PROMPT_TEMPLATE"
                    for target in statement.targets
                ):
                    return ast.literal_eval(statement.value)
    raise AssertionError("Training CASE_JUDGE_PROMPT_TEMPLATE was not found")


class UserSimulatorTests(unittest.TestCase):
    def test_final_case_judge_matches_training_prompt_and_protocol(self) -> None:
        client = FakeChatClient()
        simulator = UserSimulator(client)

        feedback = simulator.judge_cases(
            SAMPLE,
            [{"case_id": "ignored", "title": "A title", "content": "A solution"}],
        )

        self.assertEqual(feedback, SATISFIED_DONE_TOKEN)
        self.assertEqual(UserSimulator.CASE_JUDGE_PROMPT, training_case_judge_prompt())
        messages, kwargs = client.calls[0]
        self.assertIn('"title": "A title"', messages[0]["content"])
        self.assertNotIn("ignored", messages[0]["content"])
        self.assertEqual(kwargs["max_tokens"], 16)
        self.assertEqual(
            kwargs["extra_payload"]["structured_outputs"]["choice"],
            [SATISFIED_DONE_TOKEN, FAILED_DONE_TOKEN],
        )

    def test_invalid_output_and_http_400_fall_back_to_failed(self) -> None:
        cases = [{"title": "title", "content": "content"}]
        invalid = UserSimulator(FakeChatClient(content="maybe"))
        rejected = UserSimulator(FakeChatClient(error=ChatAPIError("bad request", status_code=400)))

        self.assertEqual(invalid.judge_cases(SAMPLE, cases), FAILED_DONE_TOKEN)
        self.assertEqual(rejected.judge_cases(SAMPLE, cases), FAILED_DONE_TOKEN)

    def test_empty_cases_fail_without_a_model_call(self) -> None:
        client = FakeChatClient()
        simulator = UserSimulator(client)

        self.assertEqual(simulator.judge_cases(SAMPLE, []), FAILED_DONE_TOKEN)
        self.assertEqual(client.calls, [])


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

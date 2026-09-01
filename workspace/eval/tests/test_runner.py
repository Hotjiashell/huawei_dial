from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from clarq_eval.models import EvaluationSample
from clarq_eval.random_user_simulator import RandomUserSimulatorReply
from clarq_eval.runner import EvaluationRunner


def tool_response(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
            }
        ]
    }


def complete_response() -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {"content": "Complete"}}]}


class FakePolicy:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.message_history: list[list[dict]] = []
        self.policy_calls: list[dict] = []

    def chat(self, messages: list[dict], **kwargs) -> dict:
        self.message_history.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("policy response sequence exhausted")
        return self.responses.pop(0)

    def policy_chat(self, messages: list[dict], **kwargs) -> dict:
        self.policy_calls.append(dict(kwargs))
        return self.chat(messages, **kwargs)


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list[dict]]):
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, max_results: int | None = None) -> list[dict]:
        self.calls.append((query, max_results))
        return self.results_by_query.get(query, [])[:max_results]


class FakeSimulator:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses

    def answer(self, sample: EvaluationSample, question: str) -> str:
        return self.responses[question]


class FakeRandomSimulator:
    def answer(self, sample: EvaluationSample, question: str) -> RandomUserSimulatorReply:
        return RandomUserSimulatorReply(
            text="Android 14.",
            clarification_type="known_info",
            behavior="compressed_known_info",
            source_known_info="Version 2",
        )


class FakeSuccessJudge:
    def __init__(
        self,
        can_answer: bool = False,
        answer_case_title: str | None = None,
        reason: str = "No applicable case.",
    ):
        self.judgment = {
            "can_answer": can_answer,
            "answer_case_title": answer_case_title,
            "reason": reason,
        }
        self.calls: list[tuple[EvaluationSample, list[dict]]] = []

    def judge(self, sample: EvaluationSample, cases: list[dict]) -> dict:
        self.calls.append((sample, cases))
        return dict(self.judgment)


def case(case_id: str, title: str = "title") -> dict:
    return {"caseID": case_id, "case_name": title, "text": f"content for {case_id}", "score": 1.0}


SAMPLE = EvaluationSample(
    sample_id="sample-1",
    domain="electronics",
    target_case_id="target",
    target_case_title="title",
    target_case_content="canonical answer",
    initial_question="initial question",
    core_intent="intent",
    known_info=("Version 2", "Linux"),
)


class EvaluationRunnerTests(unittest.TestCase):
    def test_clarify_search_complete_and_top_k_hit_skips_success_judge(self) -> None:
        policy = FakePolicy(
            [
                tool_response("clarify_user", {"question": "Which version?"}),
                tool_response("search_case", {"query": "initial question Version 2"}, "call_2"),
                complete_response(),
            ]
        )
        retriever = FakeRetriever(
            {
                "initial question": [case("baseline-miss")],
                "initial question Version 2": [case("target"), case("other")],
            }
        )
        success_judge = FakeSuccessJudge(can_answer=False)
        runner = EvaluationRunner(
            policy_client=policy,
            user_simulator=FakeSimulator({"Which version?": "Version 2"}),
            retriever=retriever,
            success_judge=success_judge,
            max_turns=4,
        )

        result = runner.run(SAMPLE)

        self.assertEqual(result["stop_reason"], "complete")
        self.assertEqual(result["turn_count"], 3)
        self.assertEqual(result["clarification_count"], 1)
        self.assertEqual(result["useful_clarification_count"], 1)
        self.assertEqual(result["search_count"], 1)
        self.assertEqual(result["final_results"][0]["case_id"], "target")
        self.assertEqual(
            result["success_judgment"],
            {
                "success": True,
                "method": "ground_truth_top_k",
                "top_k": 3,
                "ground_truth_rank": 1,
                "judge": None,
            },
        )
        self.assertEqual(success_judge.calls, [])
        search_event = result["events"][1]
        self.assertEqual(search_event["clarifications_since_previous_search"], 1)
        self.assertEqual(search_event["pre_clarification_case_ids"], ["baseline-miss"])
        self.assertEqual(retriever.calls, [("initial question", 5), ("initial question Version 2", 5)])

    def test_complete_tool_call_ends_interaction(self) -> None:
        runner = EvaluationRunner(
            policy_client=FakePolicy([tool_response("Complete", {})]),
            user_simulator=FakeSimulator({}),
            retriever=FakeRetriever({"initial question": [case("target")]}),
            success_judge=FakeSuccessJudge(can_answer=False),
            max_turns=1,
        )

        result = runner.run(SAMPLE)

        self.assertEqual("complete", result["stop_reason"])
        self.assertEqual("Complete", result["terminal_text"])
        self.assertEqual([{"type": "complete"}], [event["action"] for event in result["events"]])

    def test_gt_miss_calls_success_judge_with_only_configured_top_k(self) -> None:
        retrieved = [
            case("first", "First"),
            case("second", "Second"),
            case("third", "Third"),
            case("fourth", "Fourth"),
        ]
        success_judge = FakeSuccessJudge(
            can_answer=True,
            answer_case_title="Third",
            reason="Third case gives the needed fix.",
        )
        runner = EvaluationRunner(
            policy_client=FakePolicy([tool_response("search_case", {"query": "final"}), complete_response()]),
            user_simulator=FakeSimulator({}),
            retriever=FakeRetriever({"initial question": [case("baseline")], "final": retrieved}),
            success_judge=success_judge,
            top_k=4,
            success_top_k=3,
            max_turns=3,
        )

        result = runner.run(SAMPLE)

        self.assertTrue(result["success_judgment"]["success"])
        self.assertEqual(result["success_judgment"]["method"], "llm_judge")
        self.assertEqual(result["success_judgment"]["ground_truth_rank"], 0)
        self.assertEqual(result["success_judgment"]["judge"], success_judge.judgment)
        self.assertEqual(len(success_judge.calls), 1)
        judged_sample, judged_cases = success_judge.calls[0]
        self.assertEqual(judged_sample.initial_question, "initial question")
        self.assertEqual(judged_sample.target_case_title, "title")
        self.assertEqual(judged_sample.target_case_content, "canonical answer")
        self.assertEqual([item["case_id"] for item in judged_cases], ["first", "second", "third"])

    def test_gt_miss_with_negative_success_judge_is_failure(self) -> None:
        success_judge = FakeSuccessJudge(can_answer=False)
        runner = EvaluationRunner(
            policy_client=FakePolicy([tool_response("search_case", {"query": "final"}), complete_response()]),
            user_simulator=FakeSimulator({}),
            retriever=FakeRetriever({"initial question": [case("baseline")], "final": [case("other", "Other")]}),
            success_judge=success_judge,
            max_turns=3,
        )

        result = runner.run(SAMPLE)

        self.assertFalse(result["success_judgment"]["success"])
        self.assertEqual(result["success_judgment"]["method"], "llm_judge")
        self.assertEqual(len(success_judge.calls), 1)

    def test_no_agent_search_fails_without_success_judge_call(self) -> None:
        success_judge = FakeSuccessJudge(can_answer=True)
        runner = EvaluationRunner(
            policy_client=FakePolicy([complete_response()]),
            user_simulator=FakeSimulator({}),
            retriever=FakeRetriever({"initial question": [case("target")]}),
            success_judge=success_judge,
            max_turns=2,
        )

        result = runner.run(SAMPLE)

        self.assertEqual(result["stop_reason"], "complete")
        self.assertEqual(result["success_judgment"]["method"], "no_final_results")
        self.assertFalse(result["success_judgment"]["success"])
        self.assertEqual(success_judge.calls, [])

    def test_final_turn_tool_call_uses_previous_retrieval_for_success(self) -> None:
        success_judge = FakeSuccessJudge(can_answer=False)
        retriever = FakeRetriever(
            {
                "initial question": [case("baseline")],
                "query one": [case("target")],
                "query two": [case("must-not-be-called")],
            }
        )
        runner = EvaluationRunner(
            policy_client=FakePolicy(
                [
                    tool_response("search_case", {"query": "query one"}),
                    tool_response("search_case", {"query": "query two"}, "call_2"),
                ]
            ),
            user_simulator=FakeSimulator({}),
            retriever=retriever,
            success_judge=success_judge,
            max_turns=2,
        )

        result = runner.run(SAMPLE)

        self.assertEqual(result["stop_reason"], "max_turns")
        self.assertEqual(result["search_count"], 1)
        self.assertEqual(result["success_judgment"]["method"], "ground_truth_top_k")
        self.assertEqual([query for query, _ in retriever.calls], ["initial question", "query one"])
        self.assertEqual(success_judge.calls, [])

    def test_random_simulator_metadata_preserves_known_info_metric_classification(self) -> None:
        runner = EvaluationRunner(
            policy_client=FakePolicy(
                [
                    tool_response("clarify_user", {"question": "Which version?"}),
                    complete_response(),
                ]
            ),
            user_simulator=FakeRandomSimulator(),
            retriever=FakeRetriever({"initial question": [case("baseline")]}),
            success_judge=FakeSuccessJudge(can_answer=False),
            max_turns=3,
        )

        result = runner.run(SAMPLE)

        self.assertEqual(result["useful_clarification_count"], 1)
        self.assertEqual(result["unknown_clarification_count"], 0)
        event = result["events"][0]
        self.assertEqual(event["tool_response"], "Android 14.")
        self.assertEqual(event["clarification_type"], "known_info")
        self.assertEqual(event["user_simulator_behavior"], "compressed_known_info")
        self.assertEqual(event["user_simulator_source_known_info"], "Version 2")

    def test_policy_model_mode_and_tokenizer_path_are_forwarded(self) -> None:
        policy = FakePolicy([complete_response()])
        runner = EvaluationRunner(
            policy_client=policy,
            user_simulator=FakeSimulator({}),
            retriever=FakeRetriever({"initial question": [case("baseline")]}),
            success_judge=FakeSuccessJudge(can_answer=False),
            policy_model_mode="qwen3",
            policy_tokenizer_path="/models/policy-qwen3",
            enable_thinking=False,
            max_turns=1,
        )

        runner.run(SAMPLE)

        self.assertEqual(policy.policy_calls[0]["model_mode"], "qwen3")
        self.assertEqual(policy.policy_calls[0]["tokenizer_path"], "/models/policy-qwen3")
        self.assertFalse(policy.policy_calls[0]["enable_thinking"])
        self.assertTrue(policy.policy_calls[0]["tools"])

    def test_runner_validates_success_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "success_top_k"):
            EvaluationRunner(
                policy_client=FakePolicy([]),
                user_simulator=FakeSimulator({}),
                retriever=FakeRetriever({}),
                success_judge=FakeSuccessJudge(),
                top_k=2,
                success_top_k=3,
            )


if __name__ == "__main__":
    unittest.main()

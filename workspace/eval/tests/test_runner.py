from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from clarq_eval.clients import (
    FAILED_DONE_TOKEN,
    INVALID_CLARIFICATION_RESPONSE,
    SATISFIED_DONE_TOKEN,
    UNKNOWN_CLARIFICATION_RESPONSE,
)
from clarq_eval.models import EvaluationSample
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

    def chat(self, messages: list[dict], **kwargs) -> dict:
        self.message_history.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("policy response sequence exhausted")
        return self.responses.pop(0)


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list[dict]]):
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, max_results: int | None = None) -> list[dict]:
        self.calls.append((query, max_results))
        return self.results_by_query.get(query, [])[:max_results]


class FakeSimulator:
    def __init__(
        self,
        responses: dict[str, str],
        final_feedback: str = SATISFIED_DONE_TOKEN,
    ):
        self.responses = responses
        self.final_feedback = final_feedback
        self.judge_calls: list[list[dict]] = []

    def answer(self, sample: EvaluationSample, question: str) -> str:
        return self.responses[question]

    def judge_cases(self, sample: EvaluationSample, cases: list[dict]) -> str:
        self.judge_calls.append(cases)
        return self.final_feedback


def case(case_id: str, title: str = "title") -> dict:
    return {"caseID": case_id, "case_name": title, "text": f"content for {case_id}", "score": 1.0}


SAMPLE = EvaluationSample(
    sample_id="sample-1",
    domain="electronics",
    target_case_id="target",
    target_case_title="title",
    initial_question="initial question",
    core_intent="intent",
    known_info=("Version 2", "Linux"),
)


class EvaluationRunnerTests(unittest.TestCase):
    def test_clarify_search_complete_and_pairing(self) -> None:
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
        simulator = FakeSimulator({"Which version?": "Version 2"})
        runner = EvaluationRunner(
            policy_client=policy,
            user_simulator=simulator,
            retriever=retriever,
            max_turns=4,
        )

        result = runner.run(SAMPLE)

        self.assertEqual(result["stop_reason"], "complete")
        self.assertEqual(result["turn_count"], 3)
        self.assertEqual(result["clarification_count"], 1)
        self.assertEqual(result["useful_clarification_count"], 1)
        self.assertEqual(result["search_count"], 1)
        self.assertEqual(result["final_results"][0]["case_id"], "target")
        self.assertEqual(result["simulator_feedback"], SATISFIED_DONE_TOKEN)
        self.assertTrue(result["satisfaction_judge_called"])
        self.assertEqual(simulator.judge_calls[0][0]["case_id"], "target")
        search_event = result["events"][1]
        self.assertEqual(search_event["clarifications_since_previous_search"], 1)
        self.assertEqual(search_event["pre_clarification_case_ids"], ["baseline-miss"])
        self.assertEqual(search_event["pre_clarification_case_titles"], ["title"])
        self.assertEqual(result["target_case_title"], "title")
        self.assertEqual(retriever.calls, [("initial question", 5), ("initial question Version 2", 5)])

    def test_search_limit_reuses_latest_results_without_retrieval(self) -> None:
        policy = FakePolicy(
            [
                tool_response("search_case", {"query": "query one"}),
                tool_response("search_case", {"query": "query two"}, "call_2"),
                complete_response(),
            ]
        )
        retriever = FakeRetriever(
            {
                "initial question": [case("baseline")],
                "query one": [case("target")],
                "query two": [case("must-not-be-called")],
            }
        )
        runner = EvaluationRunner(
            policy_client=policy,
            user_simulator=FakeSimulator({}),
            retriever=retriever,
            max_turns=3,
            max_searches=1,
        )

        result = runner.run(SAMPLE)

        self.assertEqual(result["search_count"], 1)
        self.assertEqual(result["search_attempt_count"], 2)
        self.assertTrue(result["events"][1]["search_limit_reached"])
        self.assertFalse(result["events"][1]["search_executed"])
        self.assertEqual(result["final_results"][0]["case_id"], "target")
        self.assertEqual([query for query, _ in retriever.calls], ["initial question", "query one"])

    def test_duplicate_unknown_and_invalid_clarifications(self) -> None:
        questions = ["Which version?", "Which version?", "What color?", "Version and OS?"]
        responses = [
            tool_response("clarify_user", {"question": question}, f"call_{index}")
            for index, question in enumerate(questions, start=1)
        ]
        responses.append(tool_response("search_case", {"query": "final query"}, "call_5"))
        simulator = FakeSimulator(
            {
                "Which version?": "Version 2",
                "What color?": UNKNOWN_CLARIFICATION_RESPONSE,
                "Version and OS?": INVALID_CLARIFICATION_RESPONSE,
            }
        )
        runner = EvaluationRunner(
            policy_client=FakePolicy(responses),
            user_simulator=simulator,
            retriever=FakeRetriever(
                {"initial question": [case("baseline")], "final query": [case("target")]}
            ),
            max_turns=5,
        )

        result = runner.run(SAMPLE)

        self.assertEqual(result["clarification_count"], 4)
        self.assertEqual(result["useful_clarification_count"], 2)
        self.assertEqual(result["unknown_clarification_count"], 1)
        self.assertEqual(result["invalid_clarification_count"], 1)
        self.assertEqual(result["duplicate_clarification_count"], 1)
        self.assertEqual(result["stop_reason"], "max_turns")
        self.assertEqual(result["search_count"], 0)
        self.assertTrue(result["events"][-1]["tool_not_executed_due_to_turn_limit"])
        self.assertEqual(result["simulator_feedback"], FAILED_DONE_TOKEN)

    def test_final_turn_tool_call_is_not_executed_and_previous_cases_are_judged(self) -> None:
        simulator = FakeSimulator({}, final_feedback=SATISFIED_DONE_TOKEN)
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
            user_simulator=simulator,
            retriever=retriever,
            max_turns=2,
        )

        result = runner.run(SAMPLE)

        self.assertEqual(result["stop_reason"], "max_turns")
        self.assertEqual(result["search_count"], 1)
        self.assertEqual(result["search_attempt_count"], 1)
        self.assertEqual(result["final_results"][0]["case_id"], "target")
        self.assertEqual(result["simulator_feedback"], SATISFIED_DONE_TOKEN)
        self.assertEqual([query for query, _ in retriever.calls], ["initial question", "query one"])

    def test_training_aligned_success_does_not_require_complete(self) -> None:
        policy = FakePolicy(
            [
                tool_response("search_case", {"query": "final query"}),
                {"choices": [{"finish_reason": "stop", "message": {"content": "Done"}}]},
            ]
        )
        simulator = FakeSimulator({}, final_feedback=SATISFIED_DONE_TOKEN)
        runner = EvaluationRunner(
            policy_client=policy,
            user_simulator=simulator,
            retriever=FakeRetriever(
                {"initial question": [case("baseline")], "final query": [case("other")]}
            ),
            max_turns=3,
        )

        result = runner.run(SAMPLE)

        self.assertEqual(result["stop_reason"], "unexpected_text")
        self.assertEqual(result["turn_count"], 2)
        self.assertEqual(result["simulator_feedback"], SATISFIED_DONE_TOKEN)

    def test_no_agent_search_is_failed_without_final_judge_call(self) -> None:
        simulator = FakeSimulator({}, final_feedback=SATISFIED_DONE_TOKEN)
        runner = EvaluationRunner(
            policy_client=FakePolicy([complete_response()]),
            user_simulator=simulator,
            retriever=FakeRetriever({"initial question": [case("target")]}),
            max_turns=2,
        )

        result = runner.run(SAMPLE)

        self.assertEqual(result["stop_reason"], "complete")
        self.assertEqual(result["simulator_feedback"], FAILED_DONE_TOKEN)
        self.assertFalse(result["satisfaction_judge_called"])
        self.assertEqual(simulator.judge_calls, [])

    def test_empty_final_search_is_failed_without_final_judge_call(self) -> None:
        simulator = FakeSimulator({}, final_feedback=SATISFIED_DONE_TOKEN)
        runner = EvaluationRunner(
            policy_client=FakePolicy(
                [tool_response("search_case", {"query": "empty query"}), complete_response()]
            ),
            user_simulator=simulator,
            retriever=FakeRetriever({"initial question": [case("target")], "empty query": []}),
            max_turns=3,
        )

        result = runner.run(SAMPLE)

        self.assertEqual(result["search_count"], 1)
        self.assertEqual(result["final_results"], [])
        self.assertEqual(result["simulator_feedback"], FAILED_DONE_TOKEN)
        self.assertFalse(result["satisfaction_judge_called"])
        self.assertEqual(simulator.judge_calls, [])


if __name__ == "__main__":
    unittest.main()

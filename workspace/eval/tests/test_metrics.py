import unittest

import _bootstrap  # noqa: F401

from clarq_eval.clients import JUDGE_FIELDS
from clarq_eval.metrics import summarize_results


def result_template(sample_id: str, target: str) -> dict:
    return {
        "sample_id": sample_id,
        "domain": "domain-a",
        "target_case_id": target,
        "baseline_results": [],
        "events": [],
        "final_results": [],
        "stop_reason": "complete",
        "turn_count": 2,
        "clarification_count": 0,
        "useful_clarification_count": 0,
        "unknown_clarification_count": 0,
        "invalid_clarification_count": 0,
        "duplicate_clarification_count": 0,
        "search_count": 1,
        "search_attempt_count": 1,
        "protocol_violation_count": 0,
        "elapsed_seconds": 1.0,
        "judge": None,
        "judge_error": None,
        "error": None,
    }


class MetricTests(unittest.TestCase):
    def test_recall_gain_success_turn_judge_and_failure_denominator(self) -> None:
        first = result_template("first", "target-a")
        first["baseline_results"] = [{"case_id": "miss"}]
        first["final_results"] = [{"case_id": "target-a"}, {"case_id": "other"}]
        first["events"] = [
            {"turn": 1, "action": {"type": "clarify_user"}},
            {
                "turn": 2,
                "action": {"type": "search_case"},
                "search_executed": True,
                "search_results": first["final_results"],
                "clarifications_since_previous_search": 1,
                "pre_clarification_case_ids": ["miss"],
            },
        ]
        first["clarification_count"] = 1
        first["useful_clarification_count"] = 1
        first["judge"] = {field: False for field in JUDGE_FIELDS}
        first["judge"].update(
            {"final_cases_satisfy_intent": True, "overall_satisfied": True, "score": 4, "reason": "good"}
        )

        second = result_template("second", "target-b")
        second["baseline_results"] = [{"case_id": "target-b"}]
        second["final_results"] = [
            {"case_id": "x"},
            {"case_id": "y"},
            {"case_id": "target-b"},
        ]
        second["events"] = [
            {
                "turn": 1,
                "action": {"type": "search_case"},
                "search_executed": True,
                "search_results": second["final_results"],
            }
        ]
        second["turn_count"] = 1
        second["judge_error"] = "judge unavailable"

        failure = result_template("failed", "target-c")
        failure["error"] = {"type": "Timeout", "message": "service unavailable"}

        metrics = summarize_results(
            [first, second, failure],
            k_values=(1, 3, 5),
            success_k=1,
            max_turns=3,
        )
        overall = metrics["overall"]

        self.assertEqual(overall["sample_counts"]["requested"], 3)
        self.assertEqual(overall["sample_counts"]["completed"], 2)
        self.assertEqual(overall["sample_counts"]["infrastructure_failures"], 1)
        self.assertEqual(overall["result"]["success_rate"], 0.5)
        self.assertEqual(overall["result"]["final_recall_at_k"], {"1": 0.5, "3": 1.0, "5": 1.0})
        self.assertEqual(overall["result"]["baseline_recall_at_k"]["1"], 0.5)
        self.assertEqual(overall["process"]["clarification_gain_at_k"]["1"], 1)
        self.assertEqual(
            overall["process"]["clarified_episode_final_vs_baseline_gain_at_k"]["1"],
            1,
        )
        self.assertEqual(overall["efficiency"]["success_at_turn"], {"1": 0.0, "2": 0.5, "3": 0.5})
        self.assertEqual(overall["user_satisfaction"]["coverage"], 0.5)
        self.assertEqual(overall["user_satisfaction"]["mean_score"], 4)
        self.assertEqual(overall["user_satisfaction"]["overall_satisfied_rate"], 1.0)
        self.assertEqual(overall["sample_counts"]["judge_failures"], 1)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            summarize_results([])
        with self.assertRaises(ValueError):
            summarize_results([result_template("one", "target")], k_values=(0,))


if __name__ == "__main__":
    unittest.main()

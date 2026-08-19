from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from clarq_eval.clients import JUDGE_FIELDS
from clarq_eval.metrics import summarize_results


def success_judgment(
    *,
    success: bool,
    method: str,
    top_k: int = 3,
    ground_truth_rank: int = 0,
    reason: str = "Not applicable.",
) -> dict:
    judge = None
    if method == "llm_judge":
        judge = {"can_answer": success, "reason": reason}
    return {
        "success": success,
        "method": method,
        "top_k": top_k,
        "ground_truth_rank": ground_truth_rank,
        "judge": judge,
    }


def result_template(sample_id: str, target: str) -> dict:
    return {
        "sample_id": sample_id,
        "domain": "domain-a",
        "target_case_id": target,
        "target_case_title": f"title-{target}",
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
        "success_judgment": success_judgment(success=False, method="llm_judge"),
        "error": None,
    }


class MetricTests(unittest.TestCase):
    def test_recall_success_turn_and_judgment_breakdown(self) -> None:
        first = result_template("first", "target-a")
        first["baseline_results"] = [{"case_id": "miss"}]
        first["final_results"] = [
            {"case_id": "target-a", "title": "title-target-a"},
            {"case_id": "other", "title": "other-title"},
        ]
        first["events"] = [
            {"turn": 1, "action": {"type": "clarify_user"}},
            {
                "turn": 2,
                "action": {"type": "search_case"},
                "search_executed": True,
                "search_results": first["final_results"],
                "clarifications_since_previous_search": 1,
                "pre_clarification_case_ids": ["miss"],
                "pre_clarification_case_titles": ["miss-title"],
            },
        ]
        first["clarification_count"] = 1
        first["useful_clarification_count"] = 1
        first["judge"] = {field: False for field in JUDGE_FIELDS}
        first["judge"].update({"score": 4, "reason": "good"})
        first["success_judgment"] = success_judgment(
            success=True,
            method="ground_truth_top_k",
            ground_truth_rank=1,
        )

        second = result_template("second", "target-b")
        second["baseline_results"] = [{"case_id": "target-b", "title": "title-target-b"}]
        second["final_results"] = [
            {"case_id": "x", "title": "x-title"},
            {"case_id": "y", "title": "y-title"},
            {"case_id": "target-b", "title": "title-target-b"},
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
        second["success_judgment"] = success_judgment(
            success=True,
            method="llm_judge",
            reason="The first retrieved case answers the question equivalently.",
        )

        failure = result_template("failed", "target-c")
        failure["error"] = {"type": "Timeout", "message": "service unavailable"}
        failure["success_judgment"] = None

        metrics = summarize_results(
            [first, second, failure],
            k_values=(1, 3, 5),
            max_turns=3,
            success_top_k=3,
        )
        overall = metrics["overall"]

        self.assertEqual(overall["sample_counts"]["requested"], 3)
        self.assertEqual(overall["sample_counts"]["completed"], 2)
        self.assertEqual(overall["sample_counts"]["infrastructure_failures"], 1)
        self.assertEqual(overall["result"]["success_rate"], 1.0)
        self.assertEqual(
            overall["result"]["success_judgment_counts"],
            {
                "ground_truth_top_k_successes": 1,
                "llm_judge_successes": 1,
                "llm_judge_failures": 0,
                "no_final_results_failures": 0,
            },
        )
        self.assertEqual(overall["result"]["success_judge_call_rate"], 0.5)
        self.assertEqual(overall["result"]["final_recall_at_k"], {"1": 0.5, "3": 1.0, "5": 1.0})
        self.assertEqual(overall["result"]["baseline_recall_at_k"]["1"], 0.5)
        self.assertEqual(overall["process"]["clarification_gain_at_k"]["1"], 1)
        self.assertEqual(
            overall["process"]["clarified_episode_final_vs_baseline_gain_at_k"]["1"],
            1,
        )
        self.assertEqual(overall["efficiency"]["success_at_turn"], {"1": 0.5, "2": 1.0, "3": 1.0})
        self.assertEqual(overall["trajectory_judge"]["coverage"], 0.5)
        self.assertEqual(overall["trajectory_judge"]["mean_score"], 4)
        self.assertEqual(overall["sample_counts"]["judge_failures"], 1)

    def test_success_uses_stored_llm_decision_not_final_title_rank(self) -> None:
        llm_success = result_template("llm-success", "target")
        llm_success["final_results"] = [{"case_id": "other", "title": "Other title"}]
        llm_success["success_judgment"] = success_judgment(
            success=True,
            method="llm_judge",
            reason="Equivalent answer.",
        )
        llm_success["stop_reason"] = "unexpected_text"
        llm_success["turn_count"] = 2

        top_k_success = result_template("title-hit", "target")
        top_k_success["final_results"] = [{"case_id": "target", "title": "title-target"}]
        top_k_success["success_judgment"] = success_judgment(
            success=True,
            method="ground_truth_top_k",
            ground_truth_rank=1,
        )
        top_k_success["turn_count"] = 1

        llm_failure = result_template("llm-failure", "target")
        llm_failure["final_results"] = [{"case_id": "different", "title": "Different title"}]
        llm_failure["success_judgment"] = success_judgment(success=False, method="llm_judge")

        overall = summarize_results(
            [llm_success, top_k_success, llm_failure],
            k_values=(1,),
            max_turns=3,
            success_top_k=3,
        )["overall"]

        self.assertEqual(overall["result"]["success_rate"], 2 / 3)
        self.assertEqual(overall["result"]["final_recall_at_k"]["1"], 1 / 3)
        self.assertEqual(overall["efficiency"]["success_at_turn"], {"1": 1 / 3, "2": 2 / 3, "3": 2 / 3})

    def test_no_final_results_is_a_stored_failure(self) -> None:
        result = result_template("empty", "target")
        result["search_count"] = 0
        result["success_judgment"] = success_judgment(success=False, method="no_final_results")

        overall = summarize_results([result], k_values=(1,), max_turns=2, success_top_k=3)["overall"]

        self.assertEqual(overall["result"]["success_rate"], 0.0)
        self.assertEqual(overall["result"]["success_judge_call_rate"], 0.0)
        self.assertEqual(overall["result"]["success_judgment_counts"]["no_final_results_failures"], 1)

    def test_missing_success_judgment_is_rejected(self) -> None:
        result = result_template("legacy", "target")
        result.pop("success_judgment")
        with self.assertRaisesRegex(ValueError, "legacy trajectories"):
            summarize_results([result])

    def test_mismatched_success_top_k_is_rejected(self) -> None:
        result = result_template("one", "target")
        with self.assertRaisesRegex(ValueError, "does not match"):
            summarize_results([result], success_top_k=5)

    def test_recall_matches_title_instead_of_case_id(self) -> None:
        result = result_template("title-match", "target-id")
        result["target_case_title"] = "Canonical title"
        result["final_results"] = [{"case_id": "different-id", "title": "Canonical title"}]
        result["success_judgment"] = success_judgment(
            success=True,
            method="ground_truth_top_k",
            ground_truth_rank=1,
        )
        metrics = summarize_results([result], k_values=(1,), max_turns=2, success_top_k=3)
        self.assertEqual(metrics["overall"]["result"]["final_recall_at_k"]["1"], 1.0)

        result["final_results"] = [{"case_id": "target-id", "title": "Different title"}]
        result["success_judgment"] = success_judgment(success=False, method="llm_judge")
        metrics = summarize_results([result], k_values=(1,), max_turns=2, success_top_k=3)
        self.assertEqual(metrics["overall"]["result"]["final_recall_at_k"]["1"], 0.0)

    def test_missing_target_title_is_rejected(self) -> None:
        result = result_template("legacy", "target")
        result.pop("target_case_title")
        with self.assertRaisesRegex(ValueError, "target_case_title"):
            summarize_results([result])

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            summarize_results([])
        with self.assertRaises(ValueError):
            summarize_results([result_template("one", "target")], k_values=(0,))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from evaluate_trajectory import main, read_latest_trajectories, render_report, summarize_trajectories


def record(
    sample_id: str,
    target_title: str,
    *,
    domain: str = "money",
    baseline_results: list[dict] | None = None,
    events: list[dict] | None = None,
    final_results: list[dict] | None = None,
    error: dict | None = None,
) -> dict:
    return {
        "sample_id": sample_id,
        "domain": domain,
        "target_case_title": target_title,
        "baseline_results": baseline_results or [],
        "events": events or [],
        "final_results": final_results or [],
        "stop_reason": "complete",
        "turn_count": len(events or []),
        "elapsed_seconds": 0.5,
        "success_judgment": {
            "success": True,
            "method": "llm_judge",
            "reason": "THIS_JUDGE_REASON_MUST_NOT_APPEAR_IN_OUTPUT",
        },
        "judge": {"score": 5, "reason": "THIS_TRAJECTORY_JUDGE_MUST_NOT_APPEAR_IN_OUTPUT"},
        "error": error,
    }


class TrajectoryMetricsTests(unittest.TestCase):
    def test_direct_metrics_ignore_saved_judgments_and_reasoning_text(self) -> None:
        target = {"case_id": "target", "title": "Target title"}
        first = record(
            "one",
            "Target title",
            baseline_results=[{"case_id": "miss", "title": "Miss title"}],
            events=[
                {
                    "turn": 1,
                    "assistant_content": "THIS_ASSISTANT_REASONING_MUST_NOT_APPEAR_IN_OUTPUT",
                    "action": {"type": "clarify_user"},
                    "clarification_type": "known_info",
                    "violations": [],
                },
                {
                    "turn": 2,
                    "action": {"type": "search_case"},
                    "search_executed": True,
                    "search_results": [target],
                    "violations": [],
                },
            ],
            final_results=[target],
        )
        second = record(
            "two",
            "Other target",
            domain="travel",
            baseline_results=[{"case_id": "other", "title": "Other target"}],
            events=[
                {
                    "turn": 1,
                    "action": {"type": "search_case"},
                    "search_executed": True,
                    "search_results": [{"case_id": "miss", "title": "Miss title"}],
                    "violations": ["unexpected_text"],
                }
            ],
            final_results=[{"case_id": "miss", "title": "Miss title"}],
        )
        failed = record(
            "failed",
            "Failed target",
            error={"type": "Timeout", "message": "service unavailable"},
        )

        metrics = summarize_trajectories([first, second, failed], k_values=(1, 3), success_top_k=1)
        overall = metrics["overall"]

        self.assertEqual(
            overall["sample_counts"],
            {"requested": 3, "completed": 2, "infrastructure_failures": 1},
        )
        self.assertEqual(overall["result"]["success_rate"], 0.5)
        self.assertEqual(overall["result"]["final_recall_at_k"], {"1": 0.5, "3": 0.5})
        self.assertEqual(overall["result"]["baseline_recall_at_k"], {"1": 0.5, "3": 0.5})
        self.assertEqual(overall["process"]["clarification_gain_at_k"]["1"], 1)
        self.assertEqual(overall["process"]["useful_clarification_rate"], 1.0)
        self.assertEqual(overall["process"]["protocol_violation_rate"], 0.5)
        self.assertFalse(metrics["metric_config"]["uses_live_inference"])
        self.assertFalse(metrics["metric_config"]["uses_saved_llm_judgments"])
        self.assertFalse(metrics["metric_config"]["uses_trajectory_judge"])
        metrics_text = json.dumps(metrics, ensure_ascii=False)
        report_text = render_report(metrics)
        for forbidden_text in (
            "THIS_ASSISTANT_REASONING_MUST_NOT_APPEAR_IN_OUTPUT",
            "THIS_JUDGE_REASON_MUST_NOT_APPEAR_IN_OUTPUT",
            "THIS_TRAJECTORY_JUDGE_MUST_NOT_APPEAR_IN_OUTPUT",
        ):
            self.assertNotIn(forbidden_text, metrics_text)
            self.assertNotIn(forbidden_text, report_text)

    def test_cli_keeps_only_latest_record_and_writes_aggregate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory_path = root / "trajectories.jsonl"
            stale = record("one", "Target title", final_results=[])
            latest = record(
                "one",
                "Target title",
                events=[
                    {
                        "turn": 1,
                        "action": {"type": "search_case"},
                        "search_executed": True,
                        "search_results": [{"case_id": "target", "title": "Target title"}],
                    }
                ],
                final_results=[{"case_id": "target", "title": "Target title"}],
            )
            trajectory_path.write_text(
                "\n".join(json.dumps(item) for item in (stale, latest)) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(len(read_latest_trajectories(trajectory_path)), 1)
            self.assertEqual(main(["--trajectory-file", str(trajectory_path)]), 0)
            metrics_text = (root / "trajectory_metrics.json").read_text(encoding="utf-8")
            report_text = (root / "trajectory_report.md").read_text(encoding="utf-8")
            metrics = json.loads(metrics_text)

            self.assertEqual(metrics["overall"]["result"]["success_rate"], 1.0)
            for forbidden_text in (
                "THIS_JUDGE_REASON_MUST_NOT_APPEAR_IN_OUTPUT",
                "THIS_TRAJECTORY_JUDGE_MUST_NOT_APPEAR_IN_OUTPUT",
            ):
                self.assertNotIn(forbidden_text, metrics_text)
                self.assertNotIn(forbidden_text, report_text)

    def test_completed_trajectory_requires_target_title(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing target_case_title"):
            summarize_trajectories([record("one", "")])

    def test_rejects_k_beyond_explicitly_retained_depth(self) -> None:
        result = record("one", "Target title")
        result["runner_config"] = {"top_k": 1}
        with self.assertRaisesRegex(ValueError, "only retain Top 1"):
            summarize_trajectories([result], k_values=(1, 3), success_top_k=1)


if __name__ == "__main__":
    unittest.main()

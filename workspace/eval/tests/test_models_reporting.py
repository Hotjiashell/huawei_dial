from __future__ import annotations

import json
import io
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

import _bootstrap  # noqa: F401

from clarq_eval.metrics import summarize_results
from clarq_eval.models import EvaluationSample, load_samples
from clarq_eval.reporting import append_jsonl, read_jsonl, render_markdown, write_json, write_report
from evaluate import (
    _prepare_output,
    _run_config,
    aggregate,
    latest_records,
    parse_args,
    preflight,
    run_evaluation,
)


def successful_result(sample: EvaluationSample) -> dict:
    cases = [{"case_id": sample.target_case_id}]
    return {
        "sample_id": sample.sample_id,
        "domain": sample.domain,
        "target_case_id": sample.target_case_id,
        "baseline_results": [],
        "events": [
            {
                "turn": 1,
                "action": {"type": "search_case"},
                "search_executed": True,
                "search_results": cases,
            }
        ],
        "final_results": cases,
        "stop_reason": "complete",
        "turn_count": 1,
        "clarification_count": 0,
        "search_count": 1,
        "search_attempt_count": 1,
        "protocol_violation_count": 0,
        "elapsed_seconds": 0.01,
        "judge": None,
        "judge_error": None,
        "error": None,
    }


class FakeEvaluationRunner:
    def __init__(self, failures: set[str] | None = None):
        self.failures = failures or set()
        self.calls: list[str] = []

    def run(self, sample: EvaluationSample) -> dict:
        self.calls.append(sample.sample_id)
        if sample.sample_id in self.failures:
            raise TimeoutError("fake policy timeout")
        return successful_result(sample)


class FakeHealthClient:
    def __init__(self):
        self.base_url = "http://service/v1"
        self.model = "model"
        self.api_key = "key"
        self.calls = 0

    def health_check(self) -> dict:
        self.calls += 1
        return {"data": []}


class ModelAndReportingTests(unittest.TestCase):
    def test_dataset_loading_filter_offset_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "id": f"sample-{index}",
                    "case_id": f"case-{index}",
                    "context": f"question {index}",
                    "core_intent": f"intent {index}",
                    "known_info": [f"fact {index}"],
                }
                for index in range(3)
            ]
            (root / "money.json").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            (root / "travel.json").write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

            samples = load_samples(root, domains=("money",), offset=1, limit=1)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "sample-1")
            self.assertEqual(samples[0].domain, "money")
            with self.assertRaises(ValueError):
                load_samples(root, domains=("unknown",))

    def test_append_read_latest_and_atomic_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl_path = root / "trajectories.jsonl"
            append_jsonl(jsonl_path, {"sample_id": "one", "error": {"type": "Timeout"}})
            append_jsonl(jsonl_path, {"sample_id": "one", "error": None, "value": 2})
            append_jsonl(jsonl_path, {"sample_id": "two", "error": None})

            records = read_jsonl(jsonl_path)
            latest = latest_records(records)

            self.assertEqual(len(records), 3)
            self.assertEqual(latest["one"]["value"], 2)
            output = root / "value.json"
            write_json(output, {"ok": True})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ok": True})

    def test_report_and_offline_aggregate(self) -> None:
        result = {
            "sample_id": "one",
            "domain": "money",
            "target_case_id": "target",
            "baseline_results": [{"case_id": "other"}],
            "events": [
                {
                    "turn": 1,
                    "action": {"type": "search_case"},
                    "search_executed": True,
                    "search_results": [{"case_id": "target"}],
                }
            ],
            "final_results": [{"case_id": "target"}],
            "stop_reason": "complete",
            "turn_count": 2,
            "clarification_count": 0,
            "search_count": 1,
            "search_attempt_count": 1,
            "protocol_violation_count": 0,
            "elapsed_seconds": 0.5,
            "judge": None,
            "judge_error": None,
            "error": None,
        }
        metrics = summarize_results([result], k_values=(1, 3), success_k=1, max_turns=2)
        markdown = render_markdown(metrics)
        self.assertIn("Success Rate：100.00%", markdown)
        self.assertIn("| money | 1 / 1 | 0 | 100.00%", markdown)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            append_jsonl(output_dir / "trajectories.jsonl", result)
            rebuilt = aggregate(output_dir, (1, 3), 1, 2)
            self.assertEqual(rebuilt["overall"]["result"]["success_rate"], 1.0)
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "report.md").is_file())
            write_report(output_dir / "manual.md", metrics)
            self.assertTrue((output_dir / "manual.md").is_file())

    def test_aggregate_cli_does_not_require_online_services(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--aggregate-only", "--output-dir", directory])
        self.assertTrue(args.aggregate_only)
        self.assertIsNone(args.policy_base_url)
        self.assertIsNone(args.max_turns)
        self.assertIsNone(args.top_k)

    def test_offline_aggregate_rejects_k_beyond_stored_depth(self) -> None:
        sample = EvaluationSample("one", "money", "target", "question", "intent", ())
        result = successful_result(sample)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            append_jsonl(output_dir / "trajectories.jsonl", result)
            write_json(
                output_dir / "run_config.json",
                {"trajectory": {"top_k": 1, "max_turns": 2}},
            )
            with self.assertRaisesRegex(ValueError, "only retain Top 1"):
                aggregate(output_dir, (1, 3), 1, None)

    def test_run_config_redacts_secrets_and_resume_checks_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(
                [
                    "--output-dir",
                    directory,
                    "--policy-base-url",
                    "https://user:url-password@example.com/v1?token=url-secret",
                    "--policy-model",
                    "policy",
                    "--policy-api-key",
                    "policy-secret",
                    "--simulator-model",
                    "simulator",
                    "--simulator-api-key",
                    "simulator-secret",
                    "--elasticsearch-password",
                    "elastic-secret",
                    "--embedding-api-key",
                    "embedding-secret",
                    "--skip-judge",
                ]
            )
            config = _run_config(args)
            serialized = json.dumps(config)
            for secret in (
                "url-password",
                "url-secret",
                "policy-secret",
                "simulator-secret",
                "elastic-secret",
                "embedding-secret",
            ):
                self.assertNotIn(secret, serialized)

            _prepare_output(args, config)
            args.resume = True
            _prepare_output(args, _run_config(args))
            incompatible = deepcopy(_run_config(args))
            incompatible["trajectory"]["top_k"] = 10
            with self.assertRaises(ValueError):
                _prepare_output(args, incompatible)

    def test_execution_resume_retries_only_latest_failure(self) -> None:
        samples = [
            EvaluationSample(
                sample_id=f"sample-{index}",
                domain="money",
                target_case_id=f"case-{index}",
                initial_question=f"question {index}",
                core_intent=f"intent {index}",
                known_info=(),
            )
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                output_dir=Path(directory),
                workers=2,
                k_values=(1, 3, 5),
                success_k=1,
                max_turns=2,
            )
            first_runner = FakeEvaluationRunner(failures={"sample-1"})
            with redirect_stdout(io.StringIO()):
                first_metrics = run_evaluation(args, samples, first_runner)
            self.assertEqual(first_metrics["overall"]["sample_counts"]["completed"], 1)
            self.assertEqual(first_metrics["overall"]["sample_counts"]["infrastructure_failures"], 1)

            retry_runner = FakeEvaluationRunner()
            with redirect_stdout(io.StringIO()):
                retry_metrics = run_evaluation(args, samples, retry_runner)
            self.assertEqual(retry_runner.calls, ["sample-1"])
            self.assertEqual(retry_metrics["overall"]["sample_counts"]["completed"], 2)
            self.assertEqual(retry_metrics["overall"]["sample_counts"]["infrastructure_failures"], 0)
            self.assertEqual(retry_metrics["overall"]["result"]["success_rate"], 1.0)
            self.assertEqual(len(read_jsonl(Path(directory) / "trajectories.jsonl")), 3)

    def test_preflight_deduplicates_shared_model_and_rejects_empty_index(self) -> None:
        sample = EvaluationSample("one", "money", "target", "question", "intent", ())
        client = FakeHealthClient()

        class EmptyRetriever:
            def search(self, query: str, max_results: int) -> list:
                return []

        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "zero cases"):
            preflight([sample], [("simulator", client), ("judge", client)], EmptyRetriever())
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()

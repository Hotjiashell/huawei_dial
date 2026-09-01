from __future__ import annotations

import json
import io
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path

import _bootstrap  # noqa: F401

from clarq_eval.metrics import summarize_results
from clarq_eval.clients import OpenAIChatClient
from clarq_eval.models import EvaluationSample, load_case_documents, load_case_titles, load_samples
from clarq_eval.reporting import append_jsonl, read_jsonl, render_markdown, write_json, write_report
from evaluate import (
    _prepare_output,
    _load_retriever,
    _run_config,
    aggregate,
    latest_records,
    judge_success_records,
    parse_args,
    preflight,
    run_evaluation,
)


def successful_result(sample: EvaluationSample) -> dict:
    cases = [{"case_id": sample.target_case_id, "title": sample.target_case_title}]
    return {
        "sample_id": sample.sample_id,
        "domain": sample.domain,
        "target_case_id": sample.target_case_id,
        "target_case_title": sample.target_case_title,
        "target_case_content": sample.target_case_content,
        "initial_question": sample.initial_question,
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
        "success_judgment": {
            "success": True,
            "method": "ground_truth_top_k",
            "top_k": 3,
            "ground_truth_rank": 1,
            "judge": None,
        },
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


class FakeOpenAIHealthClient(OpenAIChatClient):
    def __init__(self, base_url: str = "http://service/v1", model: str = "model"):
        super().__init__(base_url, model, "key")
        self.calls = 0

    def health_check(self) -> dict:
        self.calls += 1
        return {"data": []}


class FakeUserSimulatorProbe:
    def __init__(self, reply: str = "I don't know.", health_client: OpenAIChatClient | None = None):
        self.reply = reply
        self.health_client = health_client
        self.calls: list[EvaluationSample] = []

    def probe(self, sample: EvaluationSample) -> str:
        self.calls.append(sample)
        return self.reply


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

    def test_case_title_document_and_sample_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "test"
            data_root.mkdir()
            document = root / "cases.json"
            document.write_text(
                json.dumps(
                    [{"case_id": "case-1", "title": "Canonical title", "answer": "Canonical content"}]
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_case_titles(document), {"case-1": "Canonical title"})
            self.assertEqual(
                load_case_documents(document),
                {"case-1": {"title": "Canonical title", "content": "Canonical content"}},
            )
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({"case-1": "Mapped title"}), encoding="utf-8")
            self.assertEqual(load_case_titles(mapping), {"case-1": "Mapped title"})

            (data_root / "money.json").write_text(
                json.dumps(
                    {
                        "id": "sample-1",
                        "case_id": "case-1",
                        "context": "question",
                        "core_intent": "intent",
                        "known_info": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            samples = load_samples(
                data_root,
                case_documents={"case-1": {"title": "Canonical title", "content": "Canonical content"}},
            )
            self.assertEqual(samples[0].target_case_title, "Canonical title")
            self.assertEqual(samples[0].target_case_content, "Canonical content")
            with self.assertRaisesRegex(ValueError, "missing from the case title document"):
                load_samples(data_root, case_titles={"other": "Other title"})

            duplicate = root / "duplicate.json"
            duplicate.write_text(
                json.dumps([
                    {"case_id": "case-1", "title": "one"},
                    {"case_id": "case-1", "title": "two"},
                ]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
                load_case_titles(duplicate)

    def test_case_title_document_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_case_titles(root / "missing.json")

            missing_title = root / "missing-title.json"
            missing_title.write_text(json.dumps([{"case_id": "case-1"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty case_id and title"):
                load_case_titles(missing_title)

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
            "target_case_title": "Target title",
            "baseline_results": [{"case_id": "other", "title": "Other title"}],
            "events": [
                {
                    "turn": 1,
                    "action": {"type": "search_case"},
                    "search_executed": True,
                    "search_results": [{"case_id": "target", "title": "Target title"}],
                }
            ],
            "final_results": [{"case_id": "target", "title": "Target title"}],
            "stop_reason": "complete",
            "turn_count": 2,
            "clarification_count": 0,
            "search_count": 1,
            "search_attempt_count": 1,
            "protocol_violation_count": 0,
            "elapsed_seconds": 0.5,
            "judge": None,
            "judge_error": None,
            "success_judgment": {
                "success": True,
                "method": "ground_truth_top_k",
                "top_k": 3,
                "ground_truth_rank": 1,
                "judge": None,
            },
            "error": None,
        }
        metrics = summarize_results([result], k_values=(1, 3), max_turns=2)
        markdown = render_markdown(metrics)
        self.assertIn("Success Rate：100.00%", markdown)
        self.assertIn("| money | 1 / 1 | 0 | 100.00%", markdown)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            append_jsonl(output_dir / "trajectories.jsonl", result)
            rebuilt = aggregate(output_dir, (1, 3), 2)
            self.assertEqual(rebuilt["overall"]["result"]["success_rate"], 1.0)
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "report.md").is_file())
            self.assertEqual(
                json.loads((output_dir / "judge_success.json").read_text(encoding="utf-8")), []
            )
            write_report(output_dir / "manual.md", metrics)
            self.assertTrue((output_dir / "manual.md").is_file())

    def test_judge_success_output_contains_selected_case_and_reason(self) -> None:
        sample = EvaluationSample(
            "one",
            "money",
            "target",
            "How do I solve this?",
            "intent",
            (),
            "Canonical title",
            "Canonical answer",
        )
        result = successful_result(sample)
        result["final_results"] = [
            {
                "case_id": "equivalent",
                "title": "Equivalent title",
                "content": "Equivalent solution content",
            }
        ]
        result["success_judgment"] = {
            "success": True,
            "method": "llm_judge",
            "top_k": 3,
            "ground_truth_rank": 0,
            "judge": {
                "can_answer": True,
                "answer_case_title": "Equivalent title",
                "reason": "It gives the same applicable procedure.",
            },
        }

        expected = [
            {
                "sample_id": "one",
                "initial_question": "How do I solve this?",
                "ground_truth_case": {
                    "title": "Canonical title",
                    "content": "Canonical answer",
                },
                "answer_case": {
                    "title": "Equivalent title",
                    "content": "Equivalent solution content",
                },
                "reason": "It gives the same applicable procedure.",
            }
        ]
        self.assertEqual(judge_success_records([result]), expected)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            append_jsonl(output_dir / "trajectories.jsonl", result)
            aggregate(output_dir, (1, 3), 2)
            self.assertEqual(
                json.loads((output_dir / "judge_success.json").read_text(encoding="utf-8")), expected
            )

    def test_aggregate_cli_does_not_require_online_services(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--aggregate-only", "--output-dir", directory])
        self.assertTrue(args.aggregate_only)
        self.assertIsNone(args.policy_base_url)
        self.assertIsNone(args.max_turns)
        self.assertIsNone(args.top_k)
        self.assertEqual(args.success_top_k, 3)

    def test_offline_aggregate_rejects_k_beyond_stored_depth(self) -> None:
        sample = EvaluationSample("one", "money", "target", "question", "intent", (), "Target title")
        result = successful_result(sample)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            append_jsonl(output_dir / "trajectories.jsonl", result)
            write_json(
                output_dir / "run_config.json",
                {"trajectory": {"top_k": 1, "max_turns": 2}, "metrics": {"success_top_k": 3}},
            )
            with self.assertRaisesRegex(ValueError, "only retain Top 1"):
                aggregate(output_dir, (1, 3), None)

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
                    "--success-judge-api-key",
                    "success-judge-secret",
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
                "success-judge-secret",
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
            legacy = deepcopy(_run_config(args))
            legacy["schema_version"] = "1.0"
            with self.assertRaises(ValueError):
                _prepare_output(args, legacy)

    def test_new_retrive_backend_loads_module_and_records_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module_path = root / "new_retrive.py"
            module_path.write_text(
                "def retrieve_case(query, top_k=20, strategy='lexical&semantic', index='document_12', "
                "use_similar_question=False, use_chat=False, chat_content=''):\n"
                "    return [{'caseID': 'case-1', 'case_name': query, 'text': index, 'score': top_k}]\n",
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--output-dir",
                    directory,
                    "--policy-base-url",
                    "http://policy/v1",
                    "--policy-model",
                    "policy",
                    "--simulator-model",
                    "simulator",
                    "--skip-judge",
                    "--retriever-backend",
                    "new-retrive",
                    "--new-retrive-module",
                    str(module_path),
                    "--elasticsearch-index",
                    "custom-index",
                    "--top-k",
                    "5",
                ]
            )
            retriever = _load_retriever(args)
            results = retriever.search("question")

        self.assertEqual(results[0]["case_name"], "question")
        self.assertEqual(results[0]["text"], "custom-index")
        self.assertEqual(results[0]["score"], 5)
        retrieval = _run_config(args)["retrieval"]
        self.assertEqual(retrieval["backend"], "new-retrive")
        self.assertEqual(retrieval["new_retrive_module"], str(module_path.resolve()))

    def test_online_success_top_k_must_fit_retrieval_depth(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(
                [
                    "--policy-base-url",
                    "http://policy/v1",
                    "--policy-model",
                    "policy",
                    "--simulator-model",
                    "simulator",
                    "--top-k",
                    "2",
                    "--k-values",
                    "1",
                    "--success-top-k",
                    "3",
                ]
            )

    def test_random_user_simulator_cli_and_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(
                [
                    "--output-dir",
                    directory,
                    "--policy-base-url",
                    "http://policy/v1",
                    "--policy-model",
                    "policy",
                    "--simulator-model",
                    "simulator",
                    "--user-simulator",
                    "random",
                    "--seed",
                    "99",
                    "--random-user-rephrase-probability",
                    "0.16",
                    "--random-user-compress-known-probability",
                    "0.79",
                    "--random-user-proactive-known-probability",
                    "0.47",
                ]
            )

        self.assertEqual(args.user_simulator_mode, "random")
        self.assertEqual(args.model_mode, "qwen3_5")
        self.assertEqual(args.policy_model_mode, "qwen3_5")
        self.assertEqual(args.random_user_simulator_seed, 99)
        self.assertFalse(args.random_user_proactive_known_on_unknown)
        simulation = _run_config(args)["services"]["user_simulator"]
        self.assertEqual(simulation["mode"], "random")
        self.assertEqual(simulation["model_mode"], "qwen3_5")
        self.assertIsNone(simulation["tokenizer_path"])
        policy = _run_config(args)["services"]["policy"]
        self.assertEqual(policy["model_mode"], "qwen3_5")
        self.assertIsNone(policy["tokenizer_path"])
        self.assertEqual(
            simulation["random_sampling"],
            {
                "seed": 99,
                "rephrase_question_probability": 0.16,
                "compress_known_info_probability": 0.79,
                "enable_proactive_known_info_on_unknown": False,
                "proactive_known_info_probability": 0.47,
            },
        )

    def test_qwen3_model_mode_records_the_tokenizer_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(
                [
                    "--output-dir",
                    directory,
                    "--policy-base-url",
                    "http://policy/v1",
                    "--policy-model",
                    "Qwen/Qwen3-32B",
                    "--policy-model-mode",
                    "qwen3",
                    "--policy-tokenizer-path",
                    "/models/policy-qwen3-tokenizer",
                    "--simulator-model",
                    "Qwen/Qwen3-32B",
                    "--model-mode",
                    "qwen3",
                    "--simulator-tokenizer-path",
                    "/models/qwen3-tokenizer",
                ]
            )

        simulation = _run_config(args)["services"]["user_simulator"]
        self.assertEqual(simulation["model_mode"], "qwen3")
        self.assertEqual(simulation["tokenizer_path"], "/models/qwen3-tokenizer")
        policy = _run_config(args)["services"]["policy"]
        self.assertEqual(policy["model_mode"], "qwen3")
        self.assertEqual(policy["tokenizer_path"], "/models/policy-qwen3-tokenizer")

    def test_execution_resume_retries_only_latest_failure(self) -> None:
        samples = [
            EvaluationSample(
                sample_id=f"sample-{index}",
                domain="money",
                target_case_id=f"case-{index}",
                target_case_title=f"title-{index}",
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
                max_turns=2,
                success_top_k=3,
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
            self.assertEqual(
                json.loads((Path(directory) / "judge_success.json").read_text(encoding="utf-8")), []
            )

    def test_preflight_uses_one_user_simulator_inference_probe_and_rejects_empty_index(self) -> None:
        sample = EvaluationSample("one", "money", "target", "question", "intent", (), "Target title")
        client = FakeHealthClient()
        simulator = FakeUserSimulatorProbe()

        class EmptyRetriever:
            def search(self, query: str, max_results: int) -> list:
                return []

        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "zero cases"):
            preflight([sample], [("policy", client), ("judge", client)], EmptyRetriever(), simulator)
        self.assertEqual(client.calls, 1)
        self.assertEqual(simulator.calls, [sample])

    def test_preflight_reports_user_simulator_probe_failures(self) -> None:
        sample = EvaluationSample("one", "money", "target", "question", "intent", (), "Target title")

        class FailingUserSimulator:
            def probe(self, probe_sample: EvaluationSample) -> str:
                raise TimeoutError("simulator timeout")

        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "inference probe"):
            preflight([sample], [], object(), FailingUserSimulator())

    def test_preflight_skips_models_check_for_services_shared_with_user_simulator(self) -> None:
        sample = EvaluationSample("one", "money", "target", "question", "intent", (), "Target title")
        shared_client = FakeOpenAIHealthClient()
        policy_client = FakeOpenAIHealthClient("http://policy/v1", "policy")
        simulator = FakeUserSimulatorProbe(health_client=shared_client)

        class NonemptyRetriever:
            def search(self, query: str, max_results: int) -> list:
                return [{"case_id": "case-1"}]

        with redirect_stdout(io.StringIO()):
            preflight(
                [sample],
                [("policy", policy_client), ("success judge", shared_client)],
                NonemptyRetriever(),
                simulator,
            )

        self.assertEqual(simulator.calls, [sample])
        self.assertEqual(policy_client.calls, 1)
        self.assertEqual(shared_client.calls, 0)


if __name__ == "__main__":
    unittest.main()

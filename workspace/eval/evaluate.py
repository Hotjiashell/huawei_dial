#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

from clarq_eval.clients import OpenAIChatClient, SuccessJudge, TrajectoryJudge, UserSimulator
from clarq_eval.metrics import summarize_results
from clarq_eval.models import EvaluationSample, load_case_documents, load_samples
from clarq_eval.random_user_simulator import RandomUserSimulator, RandomUserSimulatorConfig
from clarq_eval.reporting import append_jsonl, read_jsonl, write_json, write_report
from clarq_eval.runner import EvaluationRunner


EVAL_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = EVAL_DIR.parent
DEFAULT_TEST_ROOT = WORKSPACE_DIR / "ClarQ" / "profile_split" / "test"
DEFAULT_CASE_DOCUMENT = WORKSPACE_DIR / "ClarQ" / "case_answers_with_title.json"
DEFAULT_VERL_ROOT = WORKSPACE_DIR / "verl_dial-main-h800"
OUTPUT_FILENAMES = (
    "trajectories.jsonl",
    "errors.jsonl",
    "metrics.json",
    "report.md",
    "judge_success.json",
    "run_config.json",
)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be true or false, got {value!r}")


def _csv_strings(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive comma-separated integers")
    return tuple(sorted(set(values)))


def _redact_url(value: str | None) -> str | None:
    if not value:
        return value
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, "", ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a deployed ClarQ conversational retrieval policy on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument(
        "--case-document",
        type=Path,
        default=_env(
            "CASE_DOCUMENT_PATH",
            _env("CASE_ANSWERS_PATH", str(DEFAULT_CASE_DOCUMENT)),
        ),
        help="JSON document used to resolve each test sample case_id to its canonical title and content.",
    )
    parser.add_argument("--verl-root", type=Path, default=DEFAULT_VERL_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--domains", default=_env("EVAL_DOMAINS", ""), help="Comma-separated domains.")
    parser.add_argument("--offset", type=int, default=int(_env("EVAL_OFFSET", "0")))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=int(_env("EVAL_WORKERS", "4")))

    max_turns_env = _env("EVAL_MAX_TURNS")
    top_k_env = _env("EVAL_TOP_K")
    parser.add_argument("--max-turns", type=int, default=int(max_turns_env) if max_turns_env else None)
    parser.add_argument("--max-searches", type=int, default=int(_env("EVAL_MAX_SEARCHES", "4")))
    parser.add_argument("--top-k", type=int, default=int(top_k_env) if top_k_env else None)
    parser.add_argument(
        "--success-top-k",
        type=int,
        default=int(_env("EVAL_SUCCESS_TOP_K", "3")),
        help="Final retrieval depth used for the Success title-hit check and Success Judge input.",
    )
    parser.add_argument("--k-values", type=_csv_ints, default=_csv_ints(_env("EVAL_K_VALUES", "1,3,5")))
    parser.add_argument("--case-content-chars", type=int, default=int(_env("EVAL_CASE_CONTENT_CHARS", "1000")))
    parser.add_argument("--policy-temperature", type=float, default=float(_env("POLICY_TEMPERATURE", "0")))
    parser.add_argument("--policy-max-tokens", type=int, default=int(_env("POLICY_MAX_TOKENS", "512")))
    parser.add_argument("--seed", type=int, default=int(_env("EVAL_SEED", "42")))
    parser.add_argument(
        "--policy-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("POLICY_ENABLE_THINKING", False),
    )

    parser.add_argument("--policy-base-url", default=_env("POLICY_BASE_URL"))
    parser.add_argument("--policy-model", default=_env("POLICY_MODEL"))
    parser.add_argument(
        "--policy-model-mode",
        choices=("qwen3", "qwen3_5"),
        default=_env("POLICY_MODEL_MODE", "qwen3_5"),
        help=(
            "Policy model family used to configure thinking and request format: qwen3 applies the tokenizer "
            "chat template locally; qwen3_5 uses chat_template_kwargs."
        ),
    )
    parser.add_argument(
        "--policy-tokenizer-path",
        default=_env("POLICY_TOKENIZER_PATH"),
        help=(
            "Local Qwen3 policy tokenizer directory or Hugging Face identifier. Used only when "
            "--policy-model-mode qwen3; defaults to --policy-model."
        ),
    )
    parser.add_argument("--policy-api-key", default=_env("POLICY_API_KEY", "EMPTY"))
    parser.add_argument("--policy-timeout", type=float, default=float(_env("POLICY_TIMEOUT", "120")))
    parser.add_argument("--policy-max-retries", type=int, default=int(_env("POLICY_MAX_RETRIES", "3")))

    parser.add_argument(
        "--simulator-base-url",
        default=_env("USER_SIMULATOR_BASE_URL", "http://127.0.0.1:8005/v1"),
    )
    parser.add_argument("--simulator-model", default=_env("USER_SIMULATOR_MODEL"))
    parser.add_argument("--simulator-api-key", default=_env("USER_SIMULATOR_API_KEY", "EMPTY"))
    parser.add_argument(
        "--simulator-timeout",
        type=float,
        default=float(_env("USER_SIMULATOR_TIMEOUT", "120")),
    )
    parser.add_argument(
        "--simulator-max-retries",
        type=int,
        default=int(_env("USER_SIMULATOR_MAX_RETRIES", "3")),
    )
    parser.add_argument(
        "--model-mode",
        "--user-simulator-model-mode",
        dest="model_mode",
        choices=("qwen3", "qwen3_5"),
        default=_env("USER_SIMULATOR_MODEL_MODE", "qwen3_5"),
        help=(
            "User simulator model family used to disable thinking: qwen3 applies the tokenizer "
            "chat template locally; qwen3_5 uses chat_template_kwargs."
        ),
    )
    parser.add_argument(
        "--simulator-tokenizer-path",
        default=_env("USER_SIMULATOR_TOKENIZER_PATH"),
        help=(
            "Local Qwen3 tokenizer directory or Hugging Face identifier. Used only when --model-mode qwen3; "
            "defaults to --simulator-model."
        ),
    )
    parser.add_argument(
        "--user-simulator-mode",
        "--user-simulator",
        dest="user_simulator_mode",
        choices=("grounded", "random"),
        default=_env("EVAL_USER_SIMULATOR_MODE", _env("EVAL_USER_SIMULATOR", "grounded")),
        help="User simulator behavior: grounded uses the training-compatible selector; random samples variants.",
    )
    random_simulator_seed = _env("EVAL_RANDOM_USER_SIMULATOR_SEED")
    parser.add_argument(
        "--random-user-simulator-seed",
        type=int,
        default=int(random_simulator_seed) if random_simulator_seed else None,
        help="Sampling seed for random user simulation; defaults to --seed.",
    )
    parser.add_argument(
        "--random-user-rephrase-probability",
        type=float,
        default=float(_env("EVAL_RANDOM_USER_REPHRASE_PROBABILITY", "0.16")),
        help="Chance that a random simulated user rephrases the original question instead of answering.",
    )
    parser.add_argument(
        "--random-user-compress-known-probability",
        type=float,
        default=float(_env("EVAL_RANDOM_USER_COMPRESS_KNOWN_PROBABILITY", "0.79")),
        help="Chance to make a second LLM call that compresses a known fact into a short reply.",
    )
    parser.add_argument(
        "--random-user-proactive-known-on-unknown",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("EVAL_RANDOM_USER_PROACTIVE_KNOWN_ON_UNKNOWN", False),
        help="Allow unknown answers to volunteer one randomly selected known fact.",
    )
    parser.add_argument(
        "--random-user-proactive-known-probability",
        type=float,
        default=float(_env("EVAL_RANDOM_USER_PROACTIVE_KNOWN_PROBABILITY", "0.47")),
        help="Chance to make a second LLM call that selects a known fact to volunteer after UNKNOWN.",
    )

    parser.add_argument("--judge-base-url", default=_env("JUDGE_BASE_URL"))
    parser.add_argument("--judge-model", default=_env("JUDGE_MODEL"))
    parser.add_argument("--judge-api-key", default=_env("JUDGE_API_KEY"))
    parser.add_argument("--judge-timeout", type=float, default=float(_env("JUDGE_TIMEOUT", "120")))
    parser.add_argument("--judge-max-retries", type=int, default=int(_env("JUDGE_MAX_RETRIES", "3")))
    success_judge_timeout = _env("SUCCESS_JUDGE_TIMEOUT")
    success_judge_max_retries = _env("SUCCESS_JUDGE_MAX_RETRIES")
    parser.add_argument("--success-judge-base-url", default=_env("SUCCESS_JUDGE_BASE_URL"))
    parser.add_argument("--success-judge-model", default=_env("SUCCESS_JUDGE_MODEL"))
    parser.add_argument("--success-judge-api-key", default=_env("SUCCESS_JUDGE_API_KEY"))
    parser.add_argument(
        "--success-judge-timeout",
        type=float,
        default=float(success_judge_timeout) if success_judge_timeout else None,
    )
    parser.add_argument(
        "--success-judge-max-retries",
        type=int,
        default=int(success_judge_max_retries) if success_judge_max_retries else None,
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the optional trajectory-quality Judge; the independent Success Judge remains enabled.",
    )

    parser.add_argument(
        "--search-server-host",
        default=_env("SEARCH_SERVER_HOST", _env("ELASTICSEARCH_URL", "http://127.0.0.1:9200")),
    )
    parser.add_argument("--search-strategy", default=_env("SEARCH_SERVER_STRATEGY", "hybrid"))
    parser.add_argument("--elasticsearch-index", default=_env("ELASTICSEARCH_INDEX", "clarq_cases"))
    parser.add_argument("--elasticsearch-api-key", default=_env("ELASTICSEARCH_API_KEY", ""))
    parser.add_argument("--elasticsearch-user", default=_env("ELASTICSEARCH_USER", "elastic"))
    parser.add_argument("--elasticsearch-password", default=_env("ELASTICSEARCH_PASSWORD", ""))
    parser.add_argument("--elasticsearch-ca-cert", default=_env("ELASTICSEARCH_CA_CERT", ""))
    parser.add_argument(
        "--elasticsearch-verify-certs",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ELASTICSEARCH_VERIFY_CERTS", True),
    )
    parser.add_argument(
        "--embedding-url",
        default=_env("EMBEDDING_URL", "http://127.0.0.1:8001/v1/embeddings"),
    )
    parser.add_argument("--embedding-model", default=_env("EMBEDDING_MODEL", "clarq-embedding"))
    parser.add_argument("--embedding-api-key", default=_env("EMBEDDING_API_KEY", "EMPTY"))
    parser.add_argument("--rank-window", type=int, default=int(_env("SEARCH_RANK_WINDOW", "50")))
    parser.add_argument(
        "--vector-num-candidates",
        type=int,
        default=int(_env("VECTOR_NUM_CANDIDATES", "200")),
    )
    parser.add_argument("--bm25-title-boost", type=float, default=float(_env("BM25_TITLE_BOOST", "3")))
    parser.add_argument("--rrf-rank-constant", type=int, default=int(_env("RRF_RANK_CONSTANT", "60")))
    parser.add_argument("--rrf-bm25-weight", type=float, default=float(_env("RRF_BM25_WEIGHT", "1")))
    parser.add_argument(
        "--rrf-title-vector-weight",
        type=float,
        default=float(_env("RRF_TITLE_VECTOR_WEIGHT", "1")),
    )
    parser.add_argument(
        "--rrf-answer-vector-weight",
        type=float,
        default=float(_env("RRF_ANSWER_VECTOR_WEIGHT", "1")),
    )
    parser.add_argument("--search-timeout", type=float, default=float(_env("SEARCH_TIMEOUT", "30")))
    parser.add_argument("--embedding-timeout", type=float, default=float(_env("EMBEDDING_TIMEOUT", "60")))

    parser.add_argument("--resume", action="store_true", help="Resume and retry prior infrastructure failures.")
    parser.add_argument("--aggregate-only", action="store_true", help="Rebuild metrics/report from trajectories.")
    parser.add_argument("--check-only", action="store_true", help="Check data and external services, then exit.")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--allow-infrastructure-failures", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        if args.resume or args.aggregate_only:
            parser.error("--output-dir is required with --resume or --aggregate-only")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = EVAL_DIR / "outputs" / stamp
    if not args.aggregate_only:
        if args.max_turns is None:
            args.max_turns = 6
        if args.top_k is None:
            args.top_k = 5
    positive_fields = (
        "workers",
        "max_turns",
        "max_searches",
        "top_k",
        "success_top_k",
        "case_content_chars",
        "policy_max_tokens",
        "policy_max_retries",
        "simulator_max_retries",
        "judge_max_retries",
        "success_judge_max_retries",
        "rank_window",
        "vector_num_candidates",
        "rrf_rank_constant",
    )
    for field in positive_fields:
        value = getattr(args, field)
        if value is not None and value <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    for field in (
        "random_user_rephrase_probability",
        "random_user_compress_known_probability",
        "random_user_proactive_known_probability",
    ):
        value = getattr(args, field)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{field.replace('_', '-')} must be between 0 and 1")
    if args.random_user_simulator_seed is None:
        args.random_user_simulator_seed = args.seed
    required_depth = max(args.k_values)
    if not args.aggregate_only and args.top_k < required_depth:
        parser.error(f"--top-k must be at least {required_depth} for the configured metrics")
    if not args.aggregate_only and args.success_top_k > args.top_k:
        parser.error("--success-top-k must not exceed --top-k")
    if args.aggregate_only and args.check_only:
        parser.error("--aggregate-only and --check-only cannot be combined")
    if not args.aggregate_only:
        for option, value in (
            ("--policy-base-url", args.policy_base_url),
            ("--policy-model", args.policy_model),
            ("--simulator-base-url", args.simulator_base_url),
            ("--simulator-model", args.simulator_model),
        ):
            if not value:
                parser.error(f"{option} is required (or set the corresponding environment variable)")
        args.judge_base_url = args.judge_base_url or args.simulator_base_url
        args.judge_model = args.judge_model or args.simulator_model
        args.judge_api_key = args.judge_api_key or args.simulator_api_key
        args.success_judge_base_url = args.success_judge_base_url or args.judge_base_url
        args.success_judge_model = args.success_judge_model or args.judge_model
        args.success_judge_api_key = args.success_judge_api_key or args.judge_api_key
        args.success_judge_timeout = args.success_judge_timeout or args.judge_timeout
        args.success_judge_max_retries = (
            args.success_judge_max_retries or args.judge_max_retries
        )
    args.domains = _csv_strings(args.domains)
    args.test_root = args.test_root.resolve()
    args.case_document = args.case_document.resolve()
    args.verl_root = args.verl_root.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def retriever_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "search_server_host": args.search_server_host,
        "search_strategy": args.search_strategy,
        "index": args.elasticsearch_index,
        "elasticsearch_api_key": args.elasticsearch_api_key,
        "elasticsearch_user": args.elasticsearch_user,
        "elasticsearch_password": args.elasticsearch_password,
        "elasticsearch_ca_cert": args.elasticsearch_ca_cert,
        "elasticsearch_verify_certs": args.elasticsearch_verify_certs,
        "embedding_url": args.embedding_url,
        "embedding_model": args.embedding_model,
        "embedding_api_key": args.embedding_api_key,
        "rank_window": args.rank_window,
        "vector_num_candidates": args.vector_num_candidates,
        "bm25_title_boost": args.bm25_title_boost,
        "rrf_rank_constant": args.rrf_rank_constant,
        "rrf_bm25_weight": args.rrf_bm25_weight,
        "rrf_title_vector_weight": args.rrf_title_vector_weight,
        "rrf_answer_vector_weight": args.rrf_answer_vector_weight,
        "search_timeout": args.search_timeout,
        "embedding_timeout": args.embedding_timeout,
        "top_k": args.top_k,
    }


def _load_retriever(args: argparse.Namespace) -> Any:
    if not (args.verl_root / "examples" / "clarq_grpo" / "retriever.py").is_file():
        raise FileNotFoundError(f"ClarQ retriever not found under {args.verl_root}")
    root = str(args.verl_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from examples.clarq_grpo.retriever import CaseRetriever
    except ModuleNotFoundError as error:
        if error.name == "requests":
            raise RuntimeError(
                "The requests package is required for online evaluation; "
                "install workspace/eval/requirements.txt"
            ) from error
        raise

    return CaseRetriever(retriever_config(args))


def _client(base_url: str, model: str, api_key: str, timeout: float, retries: int) -> OpenAIChatClient:
    return OpenAIChatClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        max_retries=retries,
    )


def build_components(args: argparse.Namespace) -> tuple[EvaluationRunner, list[tuple[str, OpenAIChatClient]], Any]:
    policy_client = _client(
        args.policy_base_url,
        args.policy_model,
        args.policy_api_key,
        args.policy_timeout,
        args.policy_max_retries,
    )
    simulator_client = _client(
        args.simulator_base_url,
        args.simulator_model,
        args.simulator_api_key,
        args.simulator_timeout,
        args.simulator_max_retries,
    )
    health_clients = [("policy", policy_client)]
    success_judge_client = _client(
        args.success_judge_base_url,
        args.success_judge_model,
        args.success_judge_api_key,
        args.success_judge_timeout,
        args.success_judge_max_retries,
    )
    health_clients.append(("success judge", success_judge_client))
    judge = None
    if not args.skip_judge:
        judge_client = _client(
            args.judge_base_url,
            args.judge_model,
            args.judge_api_key,
            args.judge_timeout,
            args.judge_max_retries,
        )
        judge = TrajectoryJudge(judge_client)
        health_clients.append(("judge", judge_client))

    retriever = _load_retriever(args)
    runner = EvaluationRunner(
        policy_client=policy_client,
        user_simulator=(
            UserSimulator(
                simulator_client,
                model_mode=args.model_mode,
                tokenizer_path=args.simulator_tokenizer_path,
            )
            if args.user_simulator_mode == "grounded"
            else RandomUserSimulator(
                simulator_client,
                RandomUserSimulatorConfig(
                    seed=args.random_user_simulator_seed,
                    rephrase_question_probability=args.random_user_rephrase_probability,
                    compress_known_info_probability=args.random_user_compress_known_probability,
                    enable_proactive_known_info_on_unknown=args.random_user_proactive_known_on_unknown,
                    proactive_known_info_probability=args.random_user_proactive_known_probability,
                ),
                model_mode=args.model_mode,
                tokenizer_path=args.simulator_tokenizer_path,
            )
        ),
        retriever=retriever,
        success_judge=SuccessJudge(success_judge_client),
        judge=judge,
        max_turns=args.max_turns,
        max_searches=args.max_searches,
        top_k=args.top_k,
        success_top_k=args.success_top_k,
        case_content_chars=args.case_content_chars,
        temperature=args.policy_temperature,
        max_tokens=args.policy_max_tokens,
        seed=args.seed,
        enable_thinking=args.policy_enable_thinking,
        policy_model_mode=args.policy_model_mode,
        policy_tokenizer_path=args.policy_tokenizer_path,
    )
    return runner, health_clients, retriever


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "2.7",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "test_root": str(args.test_root),
            "case_document": str(args.case_document),
            "domains": list(args.domains),
            "offset": args.offset,
            "limit": args.limit,
        },
        "trajectory": {
            "max_turns": args.max_turns,
            "max_searches": args.max_searches,
            "top_k": args.top_k,
            "case_content_chars": args.case_content_chars,
            "temperature": args.policy_temperature,
            "max_tokens": args.policy_max_tokens,
            "seed": args.seed,
            "enable_thinking": args.policy_enable_thinking,
        },
        "metrics": {
            "k_values": list(args.k_values),
            "success_definition": (
                f"ground-truth title is in final top-{args.success_top_k}, or an "
                "independent Success Judge determines the final retrieved cases can answer "
                "the initial question"
            ),
            "success_top_k": args.success_top_k,
            "success_requires_complete": False,
            "ground_truth_match_field": "title",
            "ground_truth_case_source": "case_id resolved from data.case_document",
        },
        "services": {
            "policy": {
                "base_url": _redact_url(args.policy_base_url),
                "model": args.policy_model,
                "model_mode": args.policy_model_mode,
                "tokenizer_path": args.policy_tokenizer_path,
            },
            "user_simulator": {
                "base_url": _redact_url(args.simulator_base_url),
                "model": args.simulator_model,
                "model_mode": args.model_mode,
                "tokenizer_path": args.simulator_tokenizer_path,
                "mode": args.user_simulator_mode,
                "random_sampling": (
                    {
                        "seed": args.random_user_simulator_seed,
                        "rephrase_question_probability": args.random_user_rephrase_probability,
                        "compress_known_info_probability": args.random_user_compress_known_probability,
                        "enable_proactive_known_info_on_unknown": (
                            args.random_user_proactive_known_on_unknown
                        ),
                        "proactive_known_info_probability": (
                            args.random_user_proactive_known_probability
                        ),
                    }
                    if args.user_simulator_mode == "random"
                    else None
                ),
            },
            "judge": None
            if args.skip_judge
            else {"base_url": _redact_url(args.judge_base_url), "model": args.judge_model},
            "success_judge": {
                "base_url": _redact_url(args.success_judge_base_url),
                "model": args.success_judge_model,
            },
        },
        "retrieval": {
            "search_server_host": _redact_url(args.search_server_host),
            "search_strategy": args.search_strategy,
            "index": args.elasticsearch_index,
            "embedding_url": _redact_url(args.embedding_url),
            "embedding_model": args.embedding_model,
            "rank_window": args.rank_window,
            "vector_num_candidates": args.vector_num_candidates,
            "bm25_title_boost": args.bm25_title_boost,
            "rrf_rank_constant": args.rrf_rank_constant,
            "rrf_bm25_weight": args.rrf_bm25_weight,
            "rrf_title_vector_weight": args.rrf_title_vector_weight,
            "rrf_answer_vector_weight": args.rrf_answer_vector_weight,
        },
        "execution": {
            "workers": args.workers,
            "success_judge_enabled": True,
            "trajectory_judge_enabled": not args.skip_judge,
        },
    }


def _resume_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config.get("schema_version"),
        "data": config.get("data"),
        "trajectory": config.get("trajectory"),
        "services": config.get("services"),
        "retrieval": config.get("retrieval"),
        "success_metric": {
            "success_top_k": config.get("metrics", {}).get("success_top_k"),
            "success_definition": config.get("metrics", {}).get("success_definition"),
        },
    }


def _prepare_output(args: argparse.Namespace, config: dict[str, Any]) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    existing_outputs = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing_outputs and not args.resume:
        raise FileExistsError(
            f"Output directory already contains evaluation files: {output_dir}. "
            "Use --resume or choose a new --output-dir."
        )
    if args.resume:
        if not config_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {config_path}")
        with config_path.open(encoding="utf-8") as file:
            previous_config = json.load(file)
        if _resume_signature(previous_config) != _resume_signature(config):
            raise ValueError(
                "Resume configuration does not match the existing trajectory configuration. "
                "Use the original data/model/retrieval/trajectory settings or a new output directory."
            )
        config["created_at"] = previous_config.get("created_at", config["created_at"])
        config["resumed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(config_path, config)


def latest_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("Every trajectory record must contain a non-empty sample_id")
        latest[sample_id] = record
    return latest


def _error_record(sample: EvaluationSample, error: Exception, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": "2.7",
        "sample_id": sample.sample_id,
        "domain": sample.domain,
        "target_case_id": sample.target_case_id,
        "target_case_title": sample.target_case_title,
        "target_case_content": sample.target_case_content,
        "initial_question": sample.initial_question,
        "core_intent": sample.core_intent,
        "known_info": list(sample.known_info),
        "baseline_results": [],
        "events": [],
        "final_results": [],
        "stop_reason": "infrastructure_failure",
        "turn_count": 0,
        "clarification_count": 0,
        "search_count": 0,
        "elapsed_seconds": elapsed_seconds,
        "success_judgment": None,
        "judge": None,
        "judge_error": None,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def judge_success_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build audit records for cases selected by successful LLM Success Judges."""
    selected: list[dict[str, Any]] = []
    for record in records:
        if record.get("error"):
            continue
        judgment = record.get("success_judgment")
        if not isinstance(judgment, dict) or judgment.get("method") != "llm_judge":
            continue
        judge = judgment.get("judge")
        if not judgment.get("success") or not isinstance(judge, dict):
            continue

        answer_case_title = judge.get("answer_case_title")
        if not isinstance(answer_case_title, str) or not answer_case_title:
            raise ValueError("Successful LLM Success Judge record is missing answer_case_title")
        top_k = judgment.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("Successful LLM Success Judge record has invalid top_k")
        answer_case = next(
            (
                case
                for case in record.get("final_results", [])[:top_k]
                if str(case.get("title") or "") == answer_case_title
            ),
            None,
        )
        if not isinstance(answer_case, dict):
            raise ValueError(
                "Success Judge answer_case_title is not present in the final retrieved Top-K"
            )

        initial_question = record.get("initial_question")
        ground_truth_title = record.get("target_case_title")
        ground_truth_content = record.get("target_case_content")
        reason = judge.get("reason")
        if not all(
            isinstance(value, str)
            for value in (initial_question, ground_truth_title, ground_truth_content, reason)
        ):
            raise ValueError("Successful LLM Success Judge record lacks required audit text")

        selected.append(
            {
                "sample_id": str(record.get("sample_id") or ""),
                "initial_question": initial_question,
                "ground_truth_case": {
                    "title": ground_truth_title,
                    "content": ground_truth_content,
                },
                "answer_case": {
                    "title": str(answer_case.get("title") or ""),
                    "content": str(answer_case.get("content") or ""),
                },
                "reason": reason,
            }
        )
    return selected


def preflight(
    samples: list[EvaluationSample],
    health_clients: list[tuple[str, OpenAIChatClient]],
    retriever: Any,
    user_simulator: Any,
) -> None:
    if not samples:
        raise ValueError("Preflight requires at least one evaluation sample")
    try:
        user_simulator.probe(samples[0])
    except Exception as error:
        raise RuntimeError(
            f"Preflight failed for user simulator inference probe: {type(error).__name__}: {error}"
        ) from error
    print("[preflight] user simulator: OK (one inference probe; /models not required)")

    simulator_client = getattr(user_simulator, "health_client", None)
    simulator_identity = (
        (simulator_client.base_url, simulator_client.model, simulator_client.api_key)
        if isinstance(simulator_client, OpenAIChatClient)
        else None
    )
    checked: set[tuple[str, str, str]] = set()
    for name, client in health_clients:
        identity = (client.base_url, client.model, client.api_key)
        if identity == simulator_identity:
            print(
                f"[preflight] {name}: shared user-simulator endpoint already checked by inference probe "
                f"({client.model}; /models skipped)"
            )
            checked.add(identity)
            continue
        if identity in checked:
            print(f"[preflight] {name}: shared endpoint already checked ({client.model})")
            continue
        try:
            client.health_check()
        except Exception as error:
            raise RuntimeError(f"Preflight failed for {name}: {type(error).__name__}: {error}") from error
        checked.add(identity)
        print(f"[preflight] {name}: OK ({client.model})")
    try:
        retrieval = retriever.search(samples[0].initial_question, 1)
    except Exception as error:
        raise RuntimeError(f"Preflight failed for retriever: {type(error).__name__}: {error}") from error
    if not retrieval:
        raise RuntimeError(
            "Preflight failed for retriever: the probe query returned zero cases; "
            "check the index, search strategy, embedding model, and field mappings"
        )
    print(f"[preflight] retriever: OK ({len(retrieval)} result for probe query)")


def aggregate(
    output_dir: Path,
    k_values: tuple[int, ...],
    max_turns: int | None,
) -> dict[str, Any]:
    trajectory_path = output_dir / "trajectories.jsonl"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {trajectory_path}")
    records = list(latest_records(read_jsonl(trajectory_path)).values())
    run_config_path = output_dir / "run_config.json"
    run_config: dict[str, Any] = {}
    if run_config_path.is_file():
        with run_config_path.open(encoding="utf-8") as file:
            run_config = json.load(file)

    stored_top_k = run_config.get("trajectory", {}).get("top_k")
    if stored_top_k is None:
        stored_depths = [
            record.get("runner_config", {}).get("top_k")
            for record in records
            if not record.get("error")
        ]
        stored_depths = [int(depth) for depth in stored_depths if depth is not None]
        stored_top_k = min(stored_depths) if stored_depths else None
    required_depth = max(k_values)
    if stored_top_k is not None and required_depth > int(stored_top_k):
        raise ValueError(
            f"Cannot compute K={required_depth}: stored trajectories only retain Top {stored_top_k}"
        )

    stored_success_top_k = run_config.get("metrics", {}).get("success_top_k")
    if stored_success_top_k is None:
        stored_success_depths = [
            record.get("runner_config", {}).get("success_top_k")
            for record in records
            if not record.get("error")
        ]
        stored_success_depths = [int(depth) for depth in stored_success_depths if depth is not None]
        stored_success_top_k = min(stored_success_depths) if stored_success_depths else None
    if stored_success_top_k is not None and stored_top_k is not None and int(stored_success_top_k) > int(stored_top_k):
        raise ValueError(
            f"Stored Success Top-K {stored_success_top_k} exceeds retained retrieval Top-K {stored_top_k}"
        )

    if max_turns is None:
        max_turns = run_config.get("trajectory", {}).get("max_turns")
    if max_turns is None:
        stored_turn_limits = [
            record.get("runner_config", {}).get("max_turns")
            for record in records
            if not record.get("error")
        ]
        stored_turn_limits = [int(turns) for turns in stored_turn_limits if turns is not None]
        max_turns = max(stored_turn_limits) if stored_turn_limits else 6
    metrics = summarize_results(
        records,
        k_values=k_values,
        max_turns=int(max_turns),
        success_top_k=int(stored_success_top_k) if stored_success_top_k is not None else None,
    )
    write_json(output_dir / "metrics.json", metrics)
    write_report(output_dir / "report.md", metrics)
    write_json(output_dir / "judge_success.json", judge_success_records(records))
    return metrics


def run_evaluation(
    args: argparse.Namespace,
    samples: list[EvaluationSample],
    runner: EvaluationRunner,
) -> dict[str, Any]:
    trajectory_path = args.output_dir / "trajectories.jsonl"
    error_path = args.output_dir / "errors.jsonl"
    previous = latest_records(read_jsonl(trajectory_path))
    completed_ids = {sample_id for sample_id, record in previous.items() if not record.get("error")}
    pending = [sample for sample in samples if sample.sample_id not in completed_ids]
    if completed_ids:
        print(f"[resume] skipping {len(completed_ids)} completed samples; running {len(pending)} samples")

    processed = 0
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="clarq-eval")
    try:
        futures = {}
        for sample in pending:
            started_at = time.monotonic()
            future = executor.submit(runner.run, sample)
            futures[future] = (sample, started_at)
        for future in as_completed(futures):
            sample, started_at = futures[future]
            try:
                record = future.result()
            except Exception as error:
                record = _error_record(sample, error, time.monotonic() - started_at)
                append_jsonl(error_path, record)
            append_jsonl(trajectory_path, record)
            processed += 1
            state = "ERROR" if record.get("error") else "OK"
            print(f"[{processed}/{len(pending)}] {state} {sample.sample_id}", flush=True)
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        print("Interrupted; completed trajectories are durable. Resume with --resume.", file=sys.stderr)
        raise
    else:
        executor.shutdown(wait=True)

    selected_latest = latest_records(read_jsonl(trajectory_path))
    selected_records = [selected_latest[sample.sample_id] for sample in samples if sample.sample_id in selected_latest]
    metrics = summarize_results(
        selected_records,
        k_values=args.k_values,
        max_turns=args.max_turns,
        success_top_k=args.success_top_k,
    )
    write_json(args.output_dir / "metrics.json", metrics)
    write_report(args.output_dir / "report.md", metrics)
    write_json(args.output_dir / "judge_success.json", judge_success_records(selected_records))
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.aggregate_only:
        metrics = aggregate(args.output_dir, args.k_values, args.max_turns)
        print(f"Aggregated {metrics['overall']['sample_counts']['requested']} latest trajectories")
        print(f"Report: {args.output_dir / 'report.md'}")
        if (
            metrics["overall"]["sample_counts"]["infrastructure_failures"]
            and not args.allow_infrastructure_failures
        ):
            return 2
        return 0

    case_documents = load_case_documents(args.case_document)
    samples = load_samples(
        args.test_root,
        domains=args.domains,
        offset=args.offset,
        limit=args.limit,
        case_documents=case_documents,
    )
    missing_target_content = [sample.sample_id for sample in samples if not sample.target_case_content.strip()]
    if missing_target_content:
        raise ValueError(
            "The case document must provide non-empty content for every target case required by "
            f"the Success Judge; missing content for {len(missing_target_content)} sample(s), "
            f"starting with {missing_target_content[0]!r}"
        )
    print(f"Loaded {len(samples)} samples from {args.test_root}")
    runner, health_clients, retriever = build_components(args)
    if not args.skip_preflight:
        preflight(samples, health_clients, retriever, runner.user_simulator)
    if args.check_only:
        print("Preflight checks passed; no evaluation was run.")
        return 0

    config = _run_config(args)
    _prepare_output(args, config)
    metrics = run_evaluation(args, samples, runner)
    counts = metrics["overall"]["sample_counts"]
    print(
        f"Evaluation finished: {counts['completed']} completed, "
        f"{counts['infrastructure_failures']} infrastructure failures"
    )
    print(f"Report: {args.output_dir / 'report.md'}")
    if counts["infrastructure_failures"] and not args.allow_infrastructure_failures:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

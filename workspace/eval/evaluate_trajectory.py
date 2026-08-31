#!/usr/bin/env python3
"""Compute ClarQ metrics from saved trajectories without running inference.

This script has no model, retriever, or Judge dependencies. It reads only
structured trajectory fields: target titles, tool actions, ranked search
results, counters, and stop reasons. Success is a directly reproducible final
Top-K title-hit rate, so stored LLM Success Judge and trajectory Judge results
are intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


DEFAULT_K_VALUES = (1, 3, 5)


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive comma-separated integers")
    return tuple(sorted(set(values)))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[int | float]) -> float | None:
    return mean(values) if values else None


def _percentile(values: list[int | float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if not total:
        return {
            "successes": successes,
            "total": total,
            "value": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    value = successes / total
    denominator = 1 + z * z / total
    center = (value + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(value * (1 - value) / total + z * z / (4 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "value": value,
        "ci95_low": max(0.0, center - margin),
        "ci95_high": min(1.0, center + margin),
    }


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _cases(value: Any) -> list[dict[str, Any]]:
    return [case for case in value if isinstance(case, dict)] if isinstance(value, list) else []


def _events(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in record.get("events", []) if isinstance(event, dict)]


def _action_type(event: dict[str, Any]) -> str:
    action = event.get("action")
    return str(action.get("type") or "") if isinstance(action, dict) else ""


def _search_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in _events(record)
        if _action_type(event) == "search_case" and event.get("search_executed") is True
    ]


def _final_results(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Use the recorded final list, or the last executed search as a fallback."""
    if isinstance(record.get("final_results"), list):
        return _cases(record["final_results"])
    searches = _search_events(record)
    return _cases(searches[-1].get("search_results")) if searches else []


def _rank(cases: Iterable[dict[str, Any]], target_title: str) -> int:
    return next(
        (
            rank
            for rank, case in enumerate(cases, start=1)
            if str(case.get("title") or "").strip() == target_title
        ),
        0,
    )


def _required_target_title(record: dict[str, Any]) -> str:
    target_title = str(record.get("target_case_title") or "").strip()
    if not target_title:
        sample_id = str(record.get("sample_id") or "<unknown>")
        raise ValueError(
            f"Completed trajectory {sample_id!r} is missing target_case_title; "
            "title-based metrics cannot be computed"
        )
    return target_title


def _turn_count(record: dict[str, Any]) -> int:
    recorded = _int(record.get("turn_count"), -1)
    if recorded >= 0:
        return recorded
    events = _events(record)
    event_turns = [_int(event.get("turn"), 0) for event in events]
    return max(event_turns, default=len(events))


def _stored_top_k(record: dict[str, Any]) -> int | None:
    """Return the retained retrieval depth when the trajectory records it."""
    runner_config = record.get("runner_config")
    if not isinstance(runner_config, dict):
        return None
    top_k = _int(runner_config.get("top_k"), 0)
    return top_k if top_k > 0 else None


def _known_retained_top_k(records: list[dict[str, Any]]) -> int | None:
    completed = [record for record in records if not record.get("error")]
    depths = [_stored_top_k(record) for record in completed]
    if not completed or any(depth is None for depth in depths):
        return None
    return min(depth for depth in depths if depth is not None)


def _clarification_metrics(record: dict[str, Any]) -> dict[str, int]:
    clarifications = [event for event in _events(record) if _action_type(event) == "clarify_user"]
    types = [str(event.get("clarification_type") or "") for event in clarifications]
    return {
        "count": len(clarifications),
        "useful": sum(value == "known_info" for value in types),
        "unknown": sum(value == "unknown" for value in types),
        "invalid": sum(value == "invalid" for value in types),
        "duplicate": sum(bool(event.get("duplicate_question")) for event in clarifications),
    }


def _protocol_violation_count(record: dict[str, Any]) -> int:
    return sum(
        len(event.get("violations", []))
        for event in _events(record)
        if isinstance(event.get("violations"), list)
    )


def _clarification_pairs(record: dict[str, Any]) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Pair each clarification block with the next executed search from event order."""
    before = _cases(record.get("baseline_results"))
    pending_clarifications = 0
    pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for event in _events(record):
        action_type = _action_type(event)
        if action_type == "clarify_user":
            pending_clarifications += 1
            continue
        if action_type != "search_case" or event.get("search_executed") is not True:
            continue
        after = _cases(event.get("search_results"))
        if pending_clarifications:
            pairs.append((before, after))
        before = after
        pending_clarifications = 0
    return pairs


def _summarize_group(
    records: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...],
    max_turns: int,
    success_top_k: int,
) -> dict[str, Any]:
    completed = [record for record in records if not record.get("error")]
    failures = [record for record in records if record.get("error")]
    denominator = len(completed)

    final_ranks: list[int] = []
    baseline_ranks: list[int] = []
    first_search_ranks: list[int] = []
    best_search_ranks: list[int] = []
    search_executed_flags: list[bool] = []
    empty_final_search_flags: list[bool] = []
    turn_counts: list[int] = []
    elapsed_seconds: list[float] = []
    clarification_counts: list[dict[str, int]] = []
    search_counts: list[int] = []
    search_attempt_counts: list[int] = []
    protocol_violations: list[int] = []
    stop_reasons: Counter[str] = Counter()
    pair_ranks: list[tuple[int, int]] = []

    for record in completed:
        target_title = _required_target_title(record)
        searches = _search_events(record)
        final_results = _final_results(record)
        final_ranks.append(_rank(final_results, target_title))
        baseline_ranks.append(_rank(_cases(record.get("baseline_results")), target_title))
        first_search_ranks.append(
            _rank(_cases(searches[0].get("search_results")), target_title) if searches else 0
        )
        search_ranks = [_rank(_cases(event.get("search_results")), target_title) for event in searches]
        positive_search_ranks = [rank for rank in search_ranks if rank > 0]
        best_search_ranks.append(min(positive_search_ranks) if positive_search_ranks else 0)
        search_executed_flags.append(bool(searches))
        empty_final_search_flags.append(bool(searches) and not final_results)
        turn_counts.append(_turn_count(record))
        elapsed = _number(record.get("elapsed_seconds"))
        if elapsed is not None:
            elapsed_seconds.append(elapsed)
        clarification_counts.append(_clarification_metrics(record))
        search_counts.append(len(searches))
        search_attempt_counts.append(sum(_action_type(event) == "search_case" for event in _events(record)))
        protocol_violations.append(_protocol_violation_count(record))
        stop_reasons[str(record.get("stop_reason") or "unknown")] += 1
        pair_ranks.extend(
            (_rank(before, target_title), _rank(after, target_title))
            for before, after in _clarification_pairs(record)
        )

    success_flags = [1 <= rank <= success_top_k for rank in final_ranks]
    final_successes = sum(success_flags)
    final_recall = {
        str(k): _ratio(sum(1 <= rank <= k for rank in final_ranks), denominator) for k in k_values
    }
    baseline_recall = {
        str(k): _ratio(sum(1 <= rank <= k for rank in baseline_ranks), denominator) for k in k_values
    }
    first_search_recall = {
        str(k): _ratio(sum(1 <= rank <= k for rank in first_search_ranks), denominator)
        for k in k_values
    }
    any_search_recall = {
        str(k): _ratio(sum(1 <= rank <= k for rank in best_search_ranks), denominator)
        for k in k_values
    }

    pair_pre_recall: dict[str, float | None] = {}
    pair_post_recall: dict[str, float | None] = {}
    pair_gain: dict[str, float | None] = {}
    for k in k_values:
        pre_values = [int(1 <= before_rank <= k) for before_rank, _ in pair_ranks]
        post_values = [int(1 <= after_rank <= k) for _, after_rank in pair_ranks]
        pair_pre_recall[str(k)] = _mean(pre_values)
        pair_post_recall[str(k)] = _mean(post_values)
        pair_gain[str(k)] = _mean([post - pre for pre, post in zip(pre_values, post_values)])

    clarified_indices = [index for index, value in enumerate(clarification_counts) if value["count"] > 0]
    clarification_episode_gain = {
        str(k): _mean(
            [
                int(1 <= final_ranks[index] <= k) - int(1 <= baseline_ranks[index] <= k)
                for index in clarified_indices
            ]
        )
        for k in k_values
    }

    total_clarifications = sum(value["count"] for value in clarification_counts)
    success_at_turn = {
        str(turn): _ratio(
            sum(success and turn_count <= turn for success, turn_count in zip(success_flags, turn_counts)),
            denominator,
        )
        for turn in range(1, max_turns + 1)
    }

    return {
        "sample_counts": {
            "requested": len(records),
            "completed": denominator,
            "infrastructure_failures": len(failures),
        },
        "result": {
            "success_definition": (
                f"target title is in the final retrieved Top-{success_top_k}; no LLM judgment is used"
            ),
            "success_rate": _ratio(final_successes, denominator),
            "success_top_k": success_top_k,
            "final_recall_at_k": final_recall,
            "baseline_recall_at_k": baseline_recall,
            "first_search_recall_at_k": first_search_recall,
            "any_search_recall_at_k": any_search_recall,
            "final_mrr": _mean([1 / rank if rank else 0.0 for rank in final_ranks]),
            "baseline_mrr": _mean([1 / rank if rank else 0.0 for rank in baseline_ranks]),
            "any_search_mrr": _mean([1 / rank if rank else 0.0 for rank in best_search_ranks]),
            "no_agent_search_rate": _ratio(sum(not flag for flag in search_executed_flags), denominator),
            "empty_final_search_rate": _ratio(sum(empty_final_search_flags), denominator),
        },
        "process": {
            "clarification_pair_count": len(pair_ranks),
            "clarification_pair_pre_recall_at_k": pair_pre_recall,
            "clarification_pair_post_recall_at_k": pair_post_recall,
            "clarification_gain_at_k": pair_gain,
            "clarified_episode_count": len(clarified_indices),
            "clarified_episode_final_vs_baseline_gain_at_k": clarification_episode_gain,
            "useful_clarification_rate": _ratio(
                sum(value["useful"] for value in clarification_counts), total_clarifications
            ),
            "unknown_clarification_rate": _ratio(
                sum(value["unknown"] for value in clarification_counts), total_clarifications
            ),
            "invalid_clarification_rate": _ratio(
                sum(value["invalid"] for value in clarification_counts), total_clarifications
            ),
            "duplicate_clarification_rate": _ratio(
                sum(value["duplicate"] for value in clarification_counts), total_clarifications
            ),
            "protocol_violation_rate": _ratio(
                sum(value > 0 for value in protocol_violations), denominator
            ),
        },
        "efficiency": {
            "turn_definition": "one saved policy decision in the trajectory",
            "mean_turns": _mean(turn_counts),
            "median_turns": _percentile(turn_counts, 0.5),
            "p95_turns": _percentile(turn_counts, 0.95),
            "mean_clarifications": _ratio(total_clarifications, denominator),
            "mean_searches": _mean(search_counts),
            "mean_search_attempts": _mean(search_attempt_counts),
            "mean_elapsed_seconds": _mean(elapsed_seconds),
            "success_at_turn": success_at_turn,
            "stop_reason_counts": dict(sorted(stop_reasons.items())),
            "normal_completion_rate": _ratio(stop_reasons.get("complete", 0), denominator),
        },
        "confidence_intervals": {
            "success_rate": _wilson(final_successes, denominator),
            **{
                f"final_recall_at_{k}": _wilson(
                    sum(1 <= rank <= k for rank in final_ranks), denominator
                )
                for k in k_values
            },
        },
    }


def summarize_trajectories(
    records: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    success_top_k: int = 3,
    max_turns: int | None = None,
) -> dict[str, Any]:
    """Summarize the latest trajectory for every sample without model output."""
    if not records:
        raise ValueError("No trajectories to aggregate")
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError("k_values must contain positive integers")
    if success_top_k <= 0:
        raise ValueError("success_top_k must be positive")
    if max_turns is not None and max_turns <= 0:
        raise ValueError("max_turns must be positive")

    normalized_k = tuple(sorted(set(k_values)))
    retained_top_k = _known_retained_top_k(records)
    required_depth = max(*normalized_k, success_top_k)
    if retained_top_k is not None and required_depth > retained_top_k:
        raise ValueError(
            f"Cannot compute K={required_depth}: saved trajectories only retain Top {retained_top_k}"
        )
    observed_turns = [_turn_count(record) for record in records if not record.get("error")]
    resolved_max_turns = max_turns if max_turns is not None else max(observed_turns, default=1)
    overall = _summarize_group(
        records,
        k_values=normalized_k,
        max_turns=resolved_max_turns,
        success_top_k=success_top_k,
    )
    domains = sorted({str(record.get("domain") or "unknown") for record in records})
    return {
        "schema_version": "trajectory-metrics-v1",
        "metric_config": {
            "k_values": list(normalized_k),
            "max_turns": resolved_max_turns,
            "success_top_k": success_top_k,
            "success_definition": "final retrieved Top-K target-title hit only",
            "ground_truth_match_field": "title",
            "retained_top_k": retained_top_k,
            "uses_live_inference": False,
            "uses_saved_llm_judgments": False,
            "uses_trajectory_judge": False,
        },
        "overall": overall,
        "per_domain": {
            domain: _summarize_group(
                [record for record in records if str(record.get("domain") or "unknown") == domain],
                k_values=normalized_k,
                max_turns=resolved_max_turns,
                success_top_k=success_top_k,
            )
            for domain in domains
        },
    }


def read_latest_trajectories(path: Path) -> list[dict[str, Any]]:
    """Read JSONL and retain the final record for each non-empty sample ID."""
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {path}")
    latest: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            sample_id = str(record.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"Trajectory in {path}:{line_number} is missing a non-empty sample_id")
            latest[sample_id] = record
    if not latest:
        raise ValueError(f"Trajectory file contains no JSON records: {path}")
    return list(latest.values())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if percent else f"{value:.4f}"
    return str(value)


def render_report(metrics: dict[str, Any]) -> str:
    """Render a compact aggregate report without embedding trajectory text."""
    config = metrics["metric_config"]
    overall = metrics["overall"]
    success_ci = overall["confidence_intervals"]["success_rate"]
    lines = [
        "# ClarQ 轨迹离线指标",
        "",
        "仅使用保存轨迹的结构化检索与工具事件字段。不会执行模型推理、检索或 Judge 调用；"
        "不读取模型回复正文、推理文本或已保存的 LLM Judge 结论。",
        "",
        "## 指标口径",
        "",
        f"- Success Rate：目标案例 title 位于最终检索 Top-{config['success_top_k']}。",
        "- Recall/MRR：按目标案例 title 的精确匹配计算。",
        "- 基础设施失败不进入质量指标的分母。",
        "",
        "## 总体结果",
        "",
        f"- 请求样本：{overall['sample_counts']['requested']}",
        f"- 完成样本：{overall['sample_counts']['completed']}",
        f"- 基础设施失败：{overall['sample_counts']['infrastructure_failures']}",
        f"- Success Rate：{_fmt(overall['result']['success_rate'], percent=True)} "
        f"(Wilson 95% CI [{_fmt(success_ci['ci95_low'], percent=True)}, "
        f"{_fmt(success_ci['ci95_high'], percent=True)}])",
        f"- Final MRR：{_fmt(overall['result']['final_mrr'])}",
        f"- Baseline MRR：{_fmt(overall['result']['baseline_mrr'])}",
        f"- Any-search MRR：{_fmt(overall['result']['any_search_mrr'])}",
        f"- 平均轮数：{_fmt(overall['efficiency']['mean_turns'])}",
        f"- 平均追问次数：{_fmt(overall['efficiency']['mean_clarifications'])}",
        f"- 平均实际检索次数：{_fmt(overall['efficiency']['mean_searches'])}",
        "",
        "## Recall@K",
        "",
        "| K | Baseline | First search | Final | Any search |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in config["k_values"]:
        lines.append(
            f"| {k} | {_fmt(overall['result']['baseline_recall_at_k'][str(k)], percent=True)} "
            f"| {_fmt(overall['result']['first_search_recall_at_k'][str(k)], percent=True)} "
            f"| {_fmt(overall['result']['final_recall_at_k'][str(k)], percent=True)} "
            f"| {_fmt(overall['result']['any_search_recall_at_k'][str(k)], percent=True)} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute title-based metrics directly from saved ClarQ trajectories without inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trajectory-file", type=Path, required=True, help="Input trajectories.jsonl file.")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Output JSON path; defaults to trajectory_metrics.json next to the input file.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Output Markdown path; defaults to trajectory_report.md next to the input file.",
    )
    parser.add_argument("--k-values", type=_csv_ints, default=DEFAULT_K_VALUES)
    parser.add_argument(
        "--success-top-k",
        type=int,
        default=3,
        help="Final retrieval depth used for the pure title-hit Success Rate.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Success@Turn upper bound; defaults to the largest completed trajectory turn count.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.success_top_k <= 0:
        raise ValueError("--success-top-k must be positive")
    if args.max_turns is not None and args.max_turns <= 0:
        raise ValueError("--max-turns must be positive")

    trajectory_file = args.trajectory_file.resolve()
    metrics_output = (args.metrics_output or trajectory_file.with_name("trajectory_metrics.json")).resolve()
    report_output = (args.report_output or trajectory_file.with_name("trajectory_report.md")).resolve()
    if metrics_output == trajectory_file or report_output == trajectory_file:
        raise ValueError("Output paths must not overwrite the input trajectory file")

    records = read_latest_trajectories(trajectory_file)
    metrics = summarize_trajectories(
        records,
        k_values=args.k_values,
        success_top_k=args.success_top_k,
        max_turns=args.max_turns,
    )
    write_json(metrics_output, metrics)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(render_report(metrics), encoding="utf-8")
    print(f"Aggregated {metrics['overall']['sample_counts']['requested']} latest trajectories")
    print(f"Metrics: {metrics_output}")
    print(f"Report: {report_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

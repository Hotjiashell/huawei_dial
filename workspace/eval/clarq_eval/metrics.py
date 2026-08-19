from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Any, Iterable

from .clients import JUDGE_FIELDS


SUCCESS_METHODS = {"ground_truth_top_k", "llm_judge", "no_final_results"}


def _case_titles(cases: Iterable[dict[str, Any]]) -> list[str]:
    return [str(case.get("title") or "").strip() for case in cases]


def _rank(cases: Iterable[dict[str, Any]], target_case_title: str) -> int:
    return next(
        (
            rank
            for rank, title in enumerate(_case_titles(cases), start=1)
            if title == target_case_title
        ),
        0,
    )


def _hit(cases: Iterable[dict[str, Any]], target_case_title: str, k: int) -> int:
    rank = _rank(cases, target_case_title)
    return int(1 <= rank <= k)


def _target_title(result: dict[str, Any]) -> str:
    title = str(result.get("target_case_title") or "").strip()
    if not title:
        raise ValueError(
            "Completed trajectories must contain target_case_title; "
            "legacy trajectories must be evaluated again"
        )
    return title


def _success_judgment(result: dict[str, Any]) -> dict[str, Any]:
    judgment = result.get("success_judgment")
    if not isinstance(judgment, dict):
        raise ValueError(
            "Completed trajectories must contain a structured success_judgment; "
            "legacy trajectories must be evaluated again"
        )

    success = judgment.get("success")
    method = judgment.get("method")
    top_k = judgment.get("top_k")
    ground_truth_rank = judgment.get("ground_truth_rank")
    if not isinstance(success, bool):
        raise ValueError("success_judgment.success must be boolean")
    if method not in SUCCESS_METHODS:
        raise ValueError(f"Unknown success_judgment.method: {method!r}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("success_judgment.top_k must be a positive integer")
    if isinstance(ground_truth_rank, bool) or not isinstance(ground_truth_rank, int) or ground_truth_rank < 0:
        raise ValueError("success_judgment.ground_truth_rank must be a non-negative integer")

    judge = judgment.get("judge")
    if method == "llm_judge":
        if not isinstance(judge, dict):
            raise ValueError("LLM success judgments must contain a judge result")
        if set(judge) != {"can_answer", "reason"}:
            raise ValueError("LLM success judge result must contain only can_answer and reason")
        if not isinstance(judge.get("can_answer"), bool) or not isinstance(judge.get("reason"), str):
            raise ValueError("LLM success judge result has invalid field types")
        if success != judge["can_answer"]:
            raise ValueError("success_judgment.success must match judge.can_answer")
    elif judge is not None:
        raise ValueError("Non-LLM success judgments must not contain a judge result")

    if method == "ground_truth_top_k" and (not success or not 1 <= ground_truth_rank <= top_k):
        raise ValueError("Ground-truth Top-K success judgment is inconsistent")
    if method == "no_final_results" and (success or ground_truth_rank != 0):
        raise ValueError("No-final-results success judgment is inconsistent")
    return judgment


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float | int]) -> float | None:
    return mean(values) if values else None


def _percentile(values: list[float | int], quantile: float) -> float | None:
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
        return {"successes": successes, "total": total, "value": None, "ci95_low": None, "ci95_high": None}
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


def _executed_search_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in result.get("events", [])
        if event.get("action", {}).get("type") == "search_case" and event.get("search_executed")
    ]


def _summarize_group(
    results: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...],
    max_turns: int,
    configured_success_top_k: int | None,
) -> dict[str, Any]:
    completed = [result for result in results if not result.get("error")]
    failures = [result for result in results if result.get("error")]
    denominator = len(completed)

    final_ranks = [_rank(result.get("final_results", []), _target_title(result)) for result in completed]
    baseline_ranks = [
        _rank(result.get("baseline_results", []), _target_title(result)) for result in completed
    ]
    first_search_ranks: list[int] = []
    best_search_ranks: list[int] = []
    for result in completed:
        target = _target_title(result)
        events = _executed_search_events(result)
        first_search_ranks.append(_rank(events[0].get("search_results", []), target) if events else 0)
        positive_ranks = [
            rank
            for rank in (_rank(event.get("search_results", []), target) for event in events)
            if rank > 0
        ]
        best_search_ranks.append(min(positive_ranks) if positive_ranks else 0)

    final_recall = {
        str(k): _ratio(sum(1 <= rank <= k for rank in final_ranks), denominator) for k in k_values
    }
    baseline_recall = {
        str(k): _ratio(sum(1 <= rank <= k for rank in baseline_ranks), denominator) for k in k_values
    }
    first_search_recall = {
        str(k): _ratio(sum(1 <= rank <= k for rank in first_search_ranks), denominator) for k in k_values
    }
    any_search_recall = {
        str(k): _ratio(sum(1 <= rank <= k for rank in best_search_ranks), denominator) for k in k_values
    }
    success_judgments = [_success_judgment(result) for result in completed]
    judgment_top_ks = {judgment["top_k"] for judgment in success_judgments}
    if len(judgment_top_ks) > 1:
        raise ValueError("Completed trajectories use multiple Success Top-K values")
    if configured_success_top_k is not None and judgment_top_ks and judgment_top_ks != {configured_success_top_k}:
        raise ValueError("Stored success_judgment Top-K does not match the configured Success Top-K")
    success_top_k = configured_success_top_k or next(iter(judgment_top_ks), 3)
    success_flags = [judgment["success"] for judgment in success_judgments]
    final_successes = sum(success_flags)

    clarification_pairs: list[tuple[list[str], list[dict[str, Any]], str]] = []
    for result in completed:
        target = _target_title(result)
        for event in _executed_search_events(result):
            if int(event.get("clarifications_since_previous_search", 0)) > 0:
                clarification_pairs.append(
                    (
                        [str(title).strip() for title in event.get("pre_clarification_case_titles", [])],
                        event.get("search_results", []),
                        target,
                    )
                )

    pair_pre_recall: dict[str, float | None] = {}
    pair_post_recall: dict[str, float | None] = {}
    pair_gain: dict[str, float | None] = {}
    for k in k_values:
        pre_values = [int(target in before[:k]) for before, _, target in clarification_pairs]
        post_values = [_hit(after, target, k) for _, after, target in clarification_pairs]
        pair_pre_recall[str(k)] = _mean(pre_values)
        pair_post_recall[str(k)] = _mean(post_values)
        pair_gain[str(k)] = _mean([post - pre for pre, post in zip(pre_values, post_values)])

    clarified_results = [result for result in completed if int(result.get("clarification_count", 0)) > 0]
    clarification_episode_gain = {}
    for k in k_values:
        clarification_episode_gain[str(k)] = _mean(
            [
                _hit(result.get("final_results", []), _target_title(result), k)
                - _hit(result.get("baseline_results", []), _target_title(result), k)
                for result in clarified_results
            ]
        )

    success_turns = [
        int(result.get("turn_count", 0))
        if is_success
        else None
        for result, is_success in zip(completed, success_flags)
    ]
    success_at_turn = {
        str(turn): _ratio(
            sum(success_turn is not None and success_turn <= turn for success_turn in success_turns),
            denominator,
        )
        for turn in range(1, max_turns + 1)
    }

    total_clarifications = sum(int(result.get("clarification_count", 0)) for result in completed)
    total_searches = sum(int(result.get("search_count", 0)) for result in completed)
    total_search_attempts = sum(int(result.get("search_attempt_count", 0)) for result in completed)
    stop_reasons = Counter(str(result.get("stop_reason") or "unknown") for result in completed)
    judge_results = [result["judge"] for result in completed if isinstance(result.get("judge"), dict)]

    judge_summary: dict[str, Any] = {
        "coverage": _ratio(len(judge_results), denominator),
        "judged_samples": len(judge_results),
        "mean_score": _mean([int(judge["score"]) for judge in judge_results]),
    }
    for field in JUDGE_FIELDS:
        judge_summary[f"{field}_rate"] = _ratio(
            sum(bool(judge[field]) for judge in judge_results),
            len(judge_results),
        )

    success_ci = _wilson(final_successes, denominator)
    confidence_intervals = {"success_rate": success_ci}
    for k in k_values:
        confidence_intervals[f"final_recall_at_{k}"] = _wilson(
            sum(1 <= rank <= k for rank in final_ranks),
            denominator,
        )

    return {
        "sample_counts": {
            "requested": len(results),
            "completed": denominator,
            "infrastructure_failures": len(failures),
            "judge_failures": sum(bool(result.get("judge_error")) for result in completed),
        },
        "result": {
            "success_definition": (
                f"ground-truth title is in the final top-{success_top_k}, or an independent "
                "LLM Success Judge determines the final retrieved cases can answer the initial question; "
                "Complete is not required"
            ),
            "success_rate": _ratio(final_successes, denominator),
            "success_top_k": success_top_k,
            "success_judgment_counts": {
                "ground_truth_top_k_successes": sum(
                    judgment["method"] == "ground_truth_top_k" for judgment in success_judgments
                ),
                "llm_judge_successes": sum(
                    judgment["method"] == "llm_judge" and judgment["success"]
                    for judgment in success_judgments
                ),
                "llm_judge_failures": sum(
                    judgment["method"] == "llm_judge" and not judgment["success"]
                    for judgment in success_judgments
                ),
                "no_final_results_failures": sum(
                    judgment["method"] == "no_final_results" for judgment in success_judgments
                ),
            },
            "success_judge_call_rate": _ratio(
                sum(judgment["method"] == "llm_judge" for judgment in success_judgments),
                denominator,
            ),
            "final_recall_at_k": final_recall,
            "baseline_recall_at_k": baseline_recall,
            "first_search_recall_at_k": first_search_recall,
            "any_search_recall_at_k": any_search_recall,
            "final_mrr": _mean([1 / rank if rank else 0.0 for rank in final_ranks]),
            "baseline_mrr": _mean([1 / rank if rank else 0.0 for rank in baseline_ranks]),
            "any_search_mrr": _mean([1 / rank if rank else 0.0 for rank in best_search_ranks]),
            "no_agent_search_rate": _ratio(
                sum(not _executed_search_events(result) for result in completed),
                denominator,
            ),
            "empty_final_search_rate": _ratio(
                sum(
                    bool(_executed_search_events(result)) and not result.get("final_results")
                    for result in completed
                ),
                denominator,
            ),
        },
        "process": {
            "clarification_pair_count": len(clarification_pairs),
            "clarification_pair_pre_recall_at_k": pair_pre_recall,
            "clarification_pair_post_recall_at_k": pair_post_recall,
            "clarification_gain_at_k": pair_gain,
            "clarified_episode_count": len(clarified_results),
            "clarified_episode_final_vs_baseline_gain_at_k": clarification_episode_gain,
            "useful_clarification_rate": _ratio(
                sum(int(result.get("useful_clarification_count", 0)) for result in completed),
                total_clarifications,
            ),
            "unknown_clarification_rate": _ratio(
                sum(int(result.get("unknown_clarification_count", 0)) for result in completed),
                total_clarifications,
            ),
            "invalid_clarification_rate": _ratio(
                sum(int(result.get("invalid_clarification_count", 0)) for result in completed),
                total_clarifications,
            ),
            "duplicate_clarification_rate": _ratio(
                sum(int(result.get("duplicate_clarification_count", 0)) for result in completed),
                total_clarifications,
            ),
            "protocol_violation_rate": _ratio(
                sum(int(result.get("protocol_violation_count", 0)) > 0 for result in completed),
                denominator,
            ),
        },
        "efficiency": {
            "turn_definition": "one policy assistant decision, including Complete or invalid terminal text",
            "mean_turns": _mean([int(result.get("turn_count", 0)) for result in completed]),
            "median_turns": _percentile([int(result.get("turn_count", 0)) for result in completed], 0.5),
            "p95_turns": _percentile([int(result.get("turn_count", 0)) for result in completed], 0.95),
            "mean_clarifications": _ratio(total_clarifications, denominator),
            "mean_searches": _ratio(total_searches, denominator),
            "mean_search_attempts": _ratio(total_search_attempts, denominator),
            "mean_elapsed_seconds": _mean([float(result.get("elapsed_seconds", 0.0)) for result in completed]),
            "success_at_turn": success_at_turn,
            "stop_reason_counts": dict(sorted(stop_reasons.items())),
            "normal_completion_rate": _ratio(stop_reasons.get("complete", 0), denominator),
        },
        "trajectory_judge": judge_summary,
        "confidence_intervals": confidence_intervals,
    }


def summarize_results(
    results: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
    max_turns: int = 6,
    success_top_k: int | None = None,
) -> dict[str, Any]:
    if not results:
        raise ValueError("No evaluation results to aggregate")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if success_top_k is not None and success_top_k <= 0:
        raise ValueError("success_top_k must be positive")

    normalized_k = tuple(sorted(set(k_values)))
    overall = _summarize_group(
        results,
        k_values=normalized_k,
        max_turns=max_turns,
        configured_success_top_k=success_top_k,
    )
    domains = sorted({str(result.get("domain") or "unknown") for result in results})
    per_domain = {
        domain: _summarize_group(
            [result for result in results if str(result.get("domain") or "unknown") == domain],
            k_values=normalized_k,
            max_turns=max_turns,
            configured_success_top_k=success_top_k,
        )
        for domain in domains
    }
    resolved_success_top_k = overall["result"]["success_top_k"]
    return {
        "schema_version": "2.3",
        "metric_config": {
            "k_values": list(normalized_k),
            "max_turns": max_turns,
            "success_top_k": resolved_success_top_k,
            "success_requires_complete": False,
            "infrastructure_failures_excluded_from_quality_denominators": True,
            "ground_truth_match_field": "title",
            "ground_truth_title_source": "case_id resolved from the configured case document",
        },
        "overall": overall,
        "per_domain": per_domain,
    }

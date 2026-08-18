from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
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
            records.append(record)
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)


def _fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if percent else f"{value:.4f}"
    return str(value)


def _recall(summary: dict[str, Any], section: str, k: int) -> Any:
    return summary["result"][section].get(str(k))


def _fmt_ci(interval: dict[str, Any]) -> str:
    return (
        f"[{_fmt(interval.get('ci95_low'), percent=True)}, "
        f"{_fmt(interval.get('ci95_high'), percent=True)}]"
    )


def render_markdown(metrics: dict[str, Any]) -> str:
    config = metrics["metric_config"]
    k_values = config["k_values"]
    overall = metrics["overall"]
    lines = [
        "# ClarQ 测试集评估报告",
        "",
        "## 评估口径",
        "",
        "- Success Rate：终局 case judge 返回 `<SATISFIED_DONE>`，或标准案例 title 位于最终检索 Top-5；不要求输出 `Complete`。",
        "- Recall@K：标准案例的 title 位于对应检索结果前 K 名；标准 title 由测试样本的 case_id 从案例文档解析得到。",
        "- Success@Turn N：最终判定成功，且训练状态机在第 N 次策略决策内结束。",
        "- 没有非空的 Agent 检索结果时直接返回 `<FAILED_DONE>`，不调用终局 case judge。",
        "- 追问增益：追问前最近检索（没有则为原始问题 baseline）与追问后首次检索的 Recall@K 差值。",
        "- 基础设施失败不进入质量指标分母，单独报告失败数量。",
        "",
        "## 总体结果",
        "",
        f"- 请求样本：{overall['sample_counts']['requested']}",
        f"- 完成样本：{overall['sample_counts']['completed']}",
        f"- 基础设施失败：{overall['sample_counts']['infrastructure_failures']}",
        f"- 可选轨迹 Judge 失败：{overall['sample_counts']['judge_failures']}",
        f"- Success Rate：{_fmt(overall['result']['success_rate'], percent=True)}"
        f"（Wilson 95% CI {_fmt_ci(overall['confidence_intervals']['success_rate'])}）",
        f"- `<SATISFIED_DONE>` / `<FAILED_DONE>`："
        f"{overall['result']['simulator_feedback_counts']['<SATISFIED_DONE>']} / "
        f"{overall['result']['simulator_feedback_counts']['<FAILED_DONE>']}",
        f"- 终局 case judge 调用率："
        f"{_fmt(overall['result']['satisfaction_judge_call_rate'], percent=True)}",
        f"- 未执行 Agent 检索率：{_fmt(overall['result']['no_agent_search_rate'], percent=True)}",
        f"- 最终检索空结果率：{_fmt(overall['result']['empty_final_search_rate'], percent=True)}",
        "",
        "| 阶段 | " + " | ".join(f"Recall@{k}" for k in k_values) + " |",
        "| --- | " + " | ".join("---:" for _ in k_values) + " |",
    ]
    for label, key in (
        ("原始问题 baseline", "baseline_recall_at_k"),
        ("Agent 首次检索", "first_search_recall_at_k"),
        ("Agent 最终检索", "final_recall_at_k"),
        ("Agent 任意检索最佳命中", "any_search_recall_at_k"),
    ):
        values = " | ".join(_fmt(_recall(overall, key, k), percent=True) for k in k_values)
        lines.append(f"| {label} | {values} |")

    lines.extend(
        [
            "",
            "| MRR 阶段 | MRR |",
            "| --- | ---: |",
            f"| 原始问题 baseline | {_fmt(overall['result']['baseline_mrr'])} |",
            f"| Agent 最终检索 | {_fmt(overall['result']['final_mrr'])} |",
            f"| Agent 任意检索最佳排名 | {_fmt(overall['result']['any_search_mrr'])} |",
        ]
    )

    process = overall["process"]
    efficiency = overall["efficiency"]
    lines.extend(
        [
            "",
            "## 过程与效率",
            "",
            f"- 可配对追问区间：{process['clarification_pair_count']}",
            f"- 有效追问率：{_fmt(process['useful_clarification_rate'], percent=True)}",
            f"- 未知信息追问率：{_fmt(process['unknown_clarification_rate'], percent=True)}",
            f"- 无效追问率：{_fmt(process['invalid_clarification_rate'], percent=True)}",
            f"- 重复追问率：{_fmt(process['duplicate_clarification_rate'], percent=True)}",
            f"- 协议违规样本率：{_fmt(process['protocol_violation_rate'], percent=True)}",
            f"- 平均策略轮数：{_fmt(efficiency['mean_turns'])}",
            f"- 策略轮数中位数 / P95：{_fmt(efficiency['median_turns'])} / {_fmt(efficiency['p95_turns'])}",
            f"- 平均追问次数：{_fmt(efficiency['mean_clarifications'])}",
            f"- 平均检索次数：{_fmt(efficiency['mean_searches'])}",
            f"- 平均检索尝试次数：{_fmt(efficiency['mean_search_attempts'])}",
            f"- 平均单样本耗时：{_fmt(efficiency['mean_elapsed_seconds'])} 秒",
            f"- 正常 Complete 率：{_fmt(efficiency['normal_completion_rate'], percent=True)}",
            "",
            "| K | 追问前 Recall | 追问后 Recall | 配对追问增益 | 有追问样本 final-baseline 增益 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for k in k_values:
        key = str(k)
        lines.append(
            f"| {k} | {_fmt(process['clarification_pair_pre_recall_at_k'][key], percent=True)} "
            f"| {_fmt(process['clarification_pair_post_recall_at_k'][key], percent=True)} "
            f"| {_fmt(process['clarification_gain_at_k'][key], percent=True)} "
            f"| {_fmt(process['clarified_episode_final_vs_baseline_gain_at_k'][key], percent=True)} |"
        )

    lines.extend(["", "### Success@Turn", "", "| Turn | Success Rate |", "| ---: | ---: |"])
    for turn, value in efficiency["success_at_turn"].items():
        lines.append(f"| {turn} | {_fmt(value, percent=True)} |")

    lines.extend(["", "### 停止原因", "", "| 原因 | 样本数 |", "| --- | ---: |"])
    for reason, count in efficiency["stop_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")

    satisfaction = overall["trajectory_judge"]
    lines.extend(
        [
            "",
            "## 可选轨迹质量 Judge",
            "",
            f"- Judge 覆盖率：{_fmt(satisfaction['coverage'], percent=True)}",
            f"- 平均评分：{_fmt(satisfaction['mean_score'])} / 5",
            f"- 无关追问率：{_fmt(satisfaction.get('irrelevant_question_rate'), percent=True)}",
            f"- 重复追问率：{_fmt(satisfaction.get('repeated_question_rate'), percent=True)}",
            f"- 含糊追问率：{_fmt(satisfaction.get('unclear_question_rate'), percent=True)}",
            f"- 非规范用语率：{_fmt(satisfaction.get('nonstandard_language_rate'), percent=True)}",
            "",
            "## 分领域结果",
            "",
            "| 领域 | 完成 / 请求 | 基础设施失败 | Success Rate | "
            + " | ".join(f"Final R@{k}" for k in k_values)
            + " | 平均轮数 |",
            "| --- | ---: | ---: | ---: | "
            + " | ".join("---:" for _ in k_values)
            + " | ---: |",
        ]
    )
    for domain, summary in metrics["per_domain"].items():
        recalls = " | ".join(
            _fmt(summary["result"]["final_recall_at_k"].get(str(k)), percent=True) for k in k_values
        )
        lines.append(
            f"| {domain} | {summary['sample_counts']['completed']} / {summary['sample_counts']['requested']} "
            f"| {summary['sample_counts']['infrastructure_failures']} "
            f"| {_fmt(summary['result']['success_rate'], percent=True)} "
            f"| {recalls} | {_fmt(summary['efficiency']['mean_turns'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(render_markdown(metrics), encoding="utf-8")
    temporary.replace(path)

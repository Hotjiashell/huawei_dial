#!/usr/bin/env python3
"""Recompute user-simulator metrics from an existing evaluation JSON output.

This is useful after changing metric definitions: no model calls are made and
the per-reply Judge annotations already stored in ``records`` are reused.
Information efficiency follows the current rule that ``non_response=true``
replies are excluded from both its point total and its denominator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate_user_simulator import aggregate


def _extract_scores(records: Sequence[Mapping[str, Any]], stream: str) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if stream == "real_user":
            score = record.get("real_user")
        elif stream == "simulator":
            simulator = record.get("simulator")
            score = simulator.get("evaluation") if isinstance(simulator, Mapping) else None
        else:
            raise ValueError(f"未知结果流：{stream}")
        if isinstance(score, Mapping):
            scores.append(dict(score))
        else:
            # Missing scores are not silently treated as zero: the output
            # reports how many rows were actually available for aggregation.
            _ = index
    return scores


def reaggregate(report: Mapping[str, Any]) -> dict[str, Any]:
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("输入结果缺少 records 数组")
    result: dict[str, Any] = {}
    for stream in ("real_user", "simulator"):
        scores = _extract_scores(records, stream)
        if scores:
            result[stream] = aggregate(scores)
    if not result:
        raise ValueError("records 中没有 real_user 或 simulator 的评测结果")

    if "real_user" in result and "simulator" in result:
        result["simulator_minus_real"] = {
            key: result["simulator"][key] - result["real_user"][key]
            for key in (
                "information_efficiency_average_points_per_reply",
                "information_leakage_rate",
                "average_reply_length_characters",
                "non_response_rate",
            )
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="根据已有用户模拟器评测输出重新统计指标，不调用模型。")
    parser.add_argument("input", help="evaluate_user_simulator.py 生成的 JSON 输出")
    parser.add_argument("--output", help="输出重算后的 JSON；默认写入 <input>.reaggregated.json")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出文件")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}.reaggregated{input_path.suffix or '.json'}"
    )
    if output_path == input_path:
        print("error: 输出不能与输入相同；请指定其他 --output", file=sys.stderr)
        return 2
    if output_path.exists() and not args.overwrite:
        print(f"error: 输出已存在：{output_path}；如需覆盖请加 --overwrite", file=sys.stderr)
        return 2

    try:
        report = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise ValueError("输入结果必须是 JSON 对象")
        metrics = reaggregate(report)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    output = {
        "schema_version": report.get("schema_version", "1.0"),
        "source_output": str(input_path),
        "metric_definition": "workspace/UserSimulatorEval/metric.md",
        "reaggregated_without_model_calls": True,
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metrics": metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

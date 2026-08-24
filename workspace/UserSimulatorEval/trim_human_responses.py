#!/usr/bin/env python3
"""Keep only the first line of ``human_response`` in a test-set JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from user_simulator_eval_common import load_jsonl, write_jsonl


DEFAULT_INPUT = Path(__file__).with_name("data") / "user_simulator_test_set.jsonl"


def first_line(value: str) -> str:
    """Return *value* before its first LF/CRLF/CR newline."""

    for index, character in enumerate(value):
        if character in "\r\n":
            return value[:index]
    return value


def trim_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Copy records and trim only the ``human_response`` field."""

    trimmed: list[dict[str, Any]] = []
    changed_count = 0
    for record in records:
        item = dict(record)
        value = item.get("human_response")
        if isinstance(value, str):
            shortened = first_line(value)
            if shortened != value:
                changed_count += 1
            item["human_response"] = shortened
        trimmed.append(item)
    return trimmed, changed_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将用户模拟器测试集每条 human_response 截断为第一次换行之前的内容。"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="输入 JSONL 测试集（默认：workspace/UserSimulatorEval/data/user_simulator_test_set.jsonl）",
    )
    parser.add_argument(
        "--output",
        help="输出 JSONL 路径；未指定时写入输入文件旁的 <stem>.first_line.jsonl",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="直接覆盖输入文件；不能与 --output 同时使用",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出文件")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    if args.in_place and args.output:
        print("error: --in-place 不能与 --output 同时使用", file=sys.stderr)
        return 2

    output_path = input_path if args.in_place else Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}.first_line{input_path.suffix or '.jsonl'}"
    )
    if output_path == input_path and not args.in_place:
        print("error: 输出路径与输入路径相同；请使用 --in-place", file=sys.stderr)
        return 2
    if output_path.exists() and output_path != input_path and not args.overwrite:
        print(f"error: 输出已存在：{output_path}；如需覆盖请加 --overwrite", file=sys.stderr)
        return 2

    try:
        records = load_jsonl(input_path)
        trimmed, changed_count = trim_records(records)
        write_jsonl(output_path, trimmed)
    except (OSError, ValueError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "record_count": len(records),
                "trimmed_count": changed_count,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

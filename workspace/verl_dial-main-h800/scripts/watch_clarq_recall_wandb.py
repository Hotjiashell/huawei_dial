#!/usr/bin/env python3

import argparse
import json
import os
import signal
import time
from pathlib import Path

import wandb

METRIC_NAMES = {
    "recall_at_1": "val-aux/clarq/recall_at_1/mean@1",
    "recall_at_3": "val-aux/clarq/recall_at_3/mean@1",
    "recall_at_5": "val-aux/clarq/recall_at_5/mean@1",
    "miss_rate": "val-aux/clarq/miss_rate/mean@1",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Mirror ClarQ Recall@K validation metrics to W&B.")
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--training-pid-file", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def process_running(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return False
    return True


def calculate_metrics(path: Path) -> dict[str, float]:
    ranks = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                ranks.append(int(json.loads(line)["target_rank"]))
    if not ranks:
        raise ValueError(f"No validation records found in {path}")

    count = len(ranks)
    return {
        METRIC_NAMES["recall_at_1"]: sum(rank == 1 for rank in ranks) / count,
        METRIC_NAMES["recall_at_3"]: sum(1 <= rank <= 3 for rank in ranks) / count,
        METRIC_NAMES["recall_at_5"]: sum(1 <= rank <= 5 for rank in ranks) / count,
        METRIC_NAMES["miss_rate"]: sum(rank == 0 for rank in ranks) / count,
        "validation/num_examples": count,
    }


def validation_files(trajectory_dir: Path) -> list[tuple[int, Path]]:
    files = []
    for path in trajectory_dir.glob("*.jsonl"):
        try:
            step = int(path.stem)
        except ValueError:
            continue
        files.append((step, path))
    return sorted(files)


def main():
    args = parse_args()
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.run_name,
        job_type="validation-recall",
        group=args.source_run_id,
        resume="allow",
        config={"source_run_id": args.source_run_id},
    )
    run.define_metric("training/global_step")
    for metric_name in METRIC_NAMES.values():
        run.define_metric(metric_name, step_metric="training/global_step")

    seen_steps = set()
    stable_sizes = {}
    stopped_polls = 0
    try:
        while not stop_requested:
            for step, path in validation_files(args.trajectory_dir):
                if step in seen_steps:
                    continue
                size = path.stat().st_size
                if stable_sizes.get(path) != size:
                    stable_sizes[path] = size
                    continue
                metrics = calculate_metrics(path)
                metrics["training/global_step"] = step
                run.log(metrics)
                seen_steps.add(step)
                print(f"logged validation Recall@K for step {step}: {metrics}", flush=True)

            if process_running(args.training_pid_file):
                stopped_polls = 0
            else:
                stopped_polls += 1
                if stopped_polls >= 2:
                    break
            time.sleep(args.poll_seconds)
    finally:
        run.finish()


if __name__ == "__main__":
    main()

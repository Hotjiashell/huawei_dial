from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

SYSTEM_PROMPT = """You are a conversational case-retrieval agent.

Use exactly one action at a time:
- Call clarify_user when one missing user-known fact would materially change
  which cases are relevant. Ask one concise, discriminative question.
- Call search_case when the request is specific enough. Build a focused query
  from the original request and confirmed clarifications without assumptions.
- Output exactly Complete when the latest retrieved cases are sufficient.

Do not answer the user's technical question yourself. Search before completing.
If retrieved cases differ substantially from the request, clarify one useful
missing detail or reformulate the query. Do not repeat answered questions.
You may call search_case at most four times. Output no explanation with Complete.
"""


def convert_profile(
    profile: dict[str, Any],
    split: str,
    domain: str,
    index: int,
) -> dict[str, Any]:
    simulator_profile = {
        "initial_question": str(profile["context"]),
        "core_intent": str(profile["core_intent"]),
        "known_info": [str(item) for item in profile["known_info"]],
    }
    return {
        "data_source": "clarq",
        "agent_name": "clarq_agent",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(profile["context"])},
        ],
        "ability": "conversational_retrieval",
        "reward_model": {
            "style": "rule",
            "ground_truth": str(profile["case_id"]),
        },
        "extra_info": {
            "split": split,
            "index": index,
            "id": str(profile["id"]),
            "case_id": str(profile["case_id"]),
            "domain": domain,
            "profile": simulator_profile,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "clarify_user": {
                    "create_kwargs": {
                        "profile": simulator_profile,
                    }
                }
            },
        },
    }


def load_split(input_root: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for profile_file in sorted((input_root / split).glob("*.json")):
        domain = profile_file.stem
        with profile_file.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    profile = json.loads(line)
                    records.append(convert_profile(profile, split, domain, len(records)))
    return records


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Convert ClarQ simulator profiles to verl Parquet files.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=project_root / "ClarQ" / "profile_split",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        records = load_split(args.input_root, split)
        output_file = args.output_dir / f"{split}.parquet"
        Dataset.from_list(records).to_parquet(output_file)
        print(f"{split}: {len(records)} -> {output_file}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    domain: str
    target_case_id: str
    initial_question: str
    core_intent: str
    known_info: tuple[str, ...]

    @property
    def profile(self) -> dict[str, Any]:
        return {
            "initial_question": self.initial_question,
            "core_intent": self.core_intent,
            "known_info": list(self.known_info),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any], domain: str, line_number: int) -> "EvaluationSample":
        required = ("case_id", "context", "core_intent", "known_info")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"Missing fields {missing} in {domain}.json line {line_number}")

        sample_id = str(record.get("id") or f"{domain}:{line_number}").strip()
        known_info = record["known_info"]
        if not isinstance(known_info, list):
            raise ValueError(f"known_info must be a list in sample {sample_id}")

        return cls(
            sample_id=sample_id,
            domain=domain,
            target_case_id=str(record["case_id"]).strip(),
            initial_question=str(record["context"]).strip(),
            core_intent=str(record["core_intent"]).strip(),
            known_info=tuple(str(item).strip() for item in known_info if str(item).strip()),
        )


def load_samples(
    test_root: Path,
    domains: Iterable[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> list[EvaluationSample]:
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test data directory does not exist: {test_root}")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    selected_domains = {domain.strip() for domain in domains or [] if domain.strip()}
    files = sorted(test_root.glob("*.json"))
    if selected_domains:
        files = [path for path in files if path.stem in selected_domains]
        missing_domains = selected_domains - {path.stem for path in files}
        if missing_domains:
            raise ValueError(f"Unknown test domains: {sorted(missing_domains)}")
    if not files:
        raise ValueError(f"No test JSONL files found in {test_root}")

    samples: list[EvaluationSample] = []
    seen_ids: set[str] = set()
    for path in files:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
                sample = EvaluationSample.from_record(record, path.stem, line_number)
                if sample.sample_id in seen_ids:
                    raise ValueError(f"Duplicate sample id: {sample.sample_id}")
                seen_ids.add(sample.sample_id)
                samples.append(sample)

    return samples[offset : offset + limit if limit is not None else None]

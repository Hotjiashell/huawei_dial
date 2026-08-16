from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


def load_case_titles(path: Path) -> dict[str, str]:
    """Load the case-id-to-title mapping used by retrieval metrics.

    The ClarQ case document is a JSON array of objects containing ``case_id``
    and ``title``.  A mapping object is also accepted for small custom
    datasets, which makes the evaluation input easy to generate without
    duplicating the full case answers.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Case title document does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in case title document {path}: {error}") from error

    if isinstance(payload, dict):
        records: list[tuple[Any, Any]] = list(payload.items())
    elif isinstance(payload, list):
        records = []
        for index, record in enumerate(payload, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"Case document item {index} must be a JSON object")
            records.append((record.get("case_id"), record.get("title")))
    else:
        raise ValueError("Case title document must be a JSON array or object mapping case IDs to titles")

    titles: dict[str, str] = {}
    for index, (raw_case_id, raw_title) in enumerate(records, start=1):
        case_id = str(raw_case_id or "").strip()
        title = str(raw_title or "").strip()
        if not case_id or not title:
            raise ValueError(f"Case document item {index} must contain non-empty case_id and title")
        if case_id in titles:
            raise ValueError(f"Duplicate case_id in case title document: {case_id}")
        titles[case_id] = title
    if not titles:
        raise ValueError(f"Case title document contains no cases: {path}")
    return titles


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    domain: str
    target_case_id: str
    initial_question: str
    core_intent: str
    known_info: tuple[str, ...]
    target_case_title: str = ""

    @property
    def profile(self) -> dict[str, Any]:
        return {
            "initial_question": self.initial_question,
            "core_intent": self.core_intent,
            "known_info": list(self.known_info),
        }

    @classmethod
    def from_record(
        cls,
        record: dict[str, Any],
        domain: str,
        line_number: int,
        case_titles: Mapping[str, str] | None = None,
    ) -> "EvaluationSample":
        required = ("case_id", "context", "core_intent", "known_info")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"Missing fields {missing} in {domain}.json line {line_number}")

        sample_id = str(record.get("id") or f"{domain}:{line_number}").strip()
        known_info = record["known_info"]
        if not isinstance(known_info, list):
            raise ValueError(f"known_info must be a list in sample {sample_id}")

        target_case_id = str(record["case_id"]).strip()
        target_case_title = ""
        if case_titles is not None:
            try:
                target_case_title = str(case_titles[target_case_id]).strip()
            except KeyError as error:
                raise ValueError(
                    f"case_id {target_case_id!r} from {domain}.json line {line_number} "
                    "is missing from the case title document"
                ) from error
        return cls(
            sample_id=sample_id,
            domain=domain,
            target_case_id=target_case_id,
            initial_question=str(record["context"]).strip(),
            core_intent=str(record["core_intent"]).strip(),
            known_info=tuple(str(item).strip() for item in known_info if str(item).strip()),
            target_case_title=target_case_title,
        )


def load_samples(
    test_root: Path,
    domains: Iterable[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
    case_titles: Mapping[str, str] | None = None,
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
                sample = EvaluationSample.from_record(record, path.stem, line_number, case_titles)
                if sample.sample_id in seen_ids:
                    raise ValueError(f"Duplicate sample id: {sample.sample_id}")
                seen_ids.add(sample.sample_id)
                samples.append(sample)

    return samples[offset : offset + limit if limit is not None else None]

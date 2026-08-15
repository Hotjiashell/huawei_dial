import json
import random
from pathlib import Path

from tqdm import tqdm


INPUT_DIR = Path("ClarQ/dataset")
OUTPUT_ROOT = Path("ClarQ/dataset_split")
RANDOM_SEED = 20260730


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    input_paths = sorted(INPUT_DIR.glob("*.json"))
    split_dirs = {
        "train": OUTPUT_ROOT / "train",
        "validation": OUTPUT_ROOT / "validation",
        "test": OUTPUT_ROOT / "test",
    }

    for directory in split_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    for input_path in tqdm(input_paths, desc="Splitting ClarQ files", unit="file"):
        records = read_jsonl(input_path)
        random_generator = random.Random(f"{RANDOM_SEED}:{input_path.name}")
        random_generator.shuffle(records)

        train_end = len(records) * 6 // 8
        validation_end = train_end + len(records) // 8

        splits = {
            "train": records[:train_end],
            "validation": records[train_end:validation_end],
            "test": records[validation_end:],
        }

        for split_name, split_records in splits.items():
            write_jsonl(
                split_dirs[split_name] / input_path.name,
                split_records,
            )


if __name__ == "__main__":
    main()

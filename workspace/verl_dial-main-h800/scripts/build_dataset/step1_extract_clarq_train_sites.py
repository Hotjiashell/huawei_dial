import json
import random
from collections import Counter
from pathlib import Path

from tqdm import tqdm


INPUT_PATH = Path("ClarQ/train.json")
OUTPUT_DIR = Path("ClarQ/train_samples")
SITE_PREFIXES = ("superuser", "electronics", "travel", "money")
SAMPLE_SIZE = 1600
RANDOM_SEED = 20260730


def sample_sites() -> tuple[dict[str, list[tuple[int, dict]]], Counter[str]]:
    samples = {prefix: [] for prefix in SITE_PREFIXES}
    seen: Counter[str] = Counter()
    random_generator = random.Random(RANDOM_SEED)

    with INPUT_PATH.open("rb") as file, tqdm(
        total=INPUT_PATH.stat().st_size,
        desc="Sampling ClarQ train",
        unit="B",
        unit_scale=True,
    ) as progress:
        for line_number, line in enumerate(file):
            record = json.loads(line)
            prefix = record["id"].rsplit("_", 2)[0]

            if prefix in samples:
                seen[prefix] += 1
                item = (line_number, record)

                if len(samples[prefix]) < SAMPLE_SIZE:
                    samples[prefix].append(item)
                else:
                    index = random_generator.randrange(seen[prefix])
                    if index < SAMPLE_SIZE:
                        samples[prefix][index] = item

            progress.update(len(line))

    return samples, seen


def write_samples(
    samples: dict[str, list[tuple[int, dict]]],
    seen: Counter[str],
) -> None:
    insufficient_sites = {
        prefix: seen[prefix]
        for prefix in SITE_PREFIXES
        if seen[prefix] < SAMPLE_SIZE
    }
    if insufficient_sites:
        raise ValueError(
            f"Sites with fewer than {SAMPLE_SIZE} records: {insufficient_sites}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for prefix in SITE_PREFIXES:
        output_path = OUTPUT_DIR / f"{prefix}.json"
        ordered_records = sorted(samples[prefix], key=lambda item: item[0])

        with output_path.open("w", encoding="utf-8") as file:
            for _, record in ordered_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    samples, seen = sample_sites()
    write_samples(samples, seen)


if __name__ == "__main__":
    main()

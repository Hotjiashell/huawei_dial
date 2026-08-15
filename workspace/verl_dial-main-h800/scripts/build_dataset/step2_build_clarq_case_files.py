import json
from pathlib import Path

from tqdm import tqdm


INPUT_DIR = Path("ClarQ/train_samples")
OUTPUT_DIR = Path("ClarQ/train_samples_with_case_id")
ANSWERS_PATH = Path("ClarQ/case_answers.json")


def main() -> None:
    input_paths = sorted(INPUT_DIR.glob("*.json"))
    total_bytes = sum(path.stat().st_size for path in input_paths)
    answers = []
    case_number = 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tqdm(
        total=total_bytes,
        desc="Building ClarQ cases",
        unit="B",
        unit_scale=True,
    ) as progress:
        for input_path in input_paths:
            output_path = OUTPUT_DIR / input_path.name

            with (
                input_path.open("rb") as input_file,
                output_path.open("w", encoding="utf-8") as output_file,
            ):
                for line in input_file:
                    record = json.loads(line)
                    case_id = f"case{case_number:04d}"

                    answers.append(
                        {
                            "case_id": case_id,
                            "answer": record["answer"],
                        }
                    )
                    output_file.write(
                        json.dumps(
                            {"case_id": case_id, **record},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    case_number += 1
                    progress.update(len(line))

    ANSWERS_PATH.write_text(
        json.dumps(answers, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

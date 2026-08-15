import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm


INPUT_PATH = Path("ClarQ/case_answers.json")
CHECKPOINT_PATH = Path("ClarQ/case_titles_checkpoint.jsonl")
OUTPUT_PATH = Path("ClarQ/case_answers_with_title.json")

CHAT_URL = os.getenv(
    "VLLM_CHAT_URL",
    "http://127.0.0.1:8000/v1/chat/completions",
)
CHAT_MODEL = os.environ["VLLM_CHAT_MODEL"]
CHAT_API_KEY = os.getenv("VLLM_CHAT_API_KEY")
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "16000"))

SYSTEM_PROMPT = """
Create a concise retrieval title by inferring the original problem addressed by
the supplied answer. Return only the title.

Requirements:
- Use the same language as the answer.
- Use 6 to 16 words.
- Preserve important products, versions, errors, locations, and constraints.
- Describe the problem or information need, not the answer-writing process.
- Do not add quotation marks, labels, explanations, or ending punctuation.
""".strip()


def generate_title(answer: str) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Answer:\n{answer[:MAX_ANSWER_CHARS]}",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 48,
    }
    headers = {"Content-Type": "application/json"}
    if CHAT_API_KEY:
        headers["Authorization"] = f"Bearer {CHAT_API_KEY}"

    request = Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        result = json.load(response)

    return result["choices"][0]["message"]["content"].strip()


def load_completed_titles() -> dict[str, str]:
    if not CHECKPOINT_PATH.exists():
        return {}

    with CHECKPOINT_PATH.open(encoding="utf-8") as file:
        return {
            record["case_id"]: record["title"]
            for record in map(json.loads, file)
        }


def main() -> None:
    cases = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    titles = load_completed_titles()
    completed_count = sum(case["case_id"] in titles for case in cases)

    with (
        CHECKPOINT_PATH.open("a", encoding="utf-8") as checkpoint,
        tqdm(
            total=len(cases),
            initial=completed_count,
            desc="Generating case titles",
            unit="case",
        ) as progress,
    ):
        for case in cases:
            case_id = case["case_id"]
            if case_id in titles:
                continue

            title = generate_title(case["answer"])
            titles[case_id] = title
            checkpoint.write(
                json.dumps(
                    {"case_id": case_id, "title": title},
                    ensure_ascii=False,
                )
                + "\n"
            )
            checkpoint.flush()
            progress.update()

    titled_cases = [
        {
            "case_id": case["case_id"],
            "title": titles[case["case_id"]],
            "answer": case["answer"],
        }
        for case in cases
    ]
    OUTPUT_PATH.write_text(
        json.dumps(titled_cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

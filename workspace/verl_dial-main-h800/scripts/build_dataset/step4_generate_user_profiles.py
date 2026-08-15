import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tqdm import tqdm


DATASET_DIR = Path(os.getenv("DATASET_DIR", "ClarQ/dataset"))
PROFILE_DIR = Path(
    os.getenv("PROFILE_DIR", str(DATASET_DIR.parent / "profile"))
)

CHAT_URL = os.getenv(
    "VLLM_CHAT_URL",
    "http://127.0.0.1:8000/v1/chat/completions",
)
CHAT_MODEL = os.environ["VLLM_CHAT_MODEL"]
CHAT_API_KEY = os.getenv("VLLM_CHAT_API_KEY")

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
REGENERATE_EXISTING = os.getenv("REGENERATE_EXISTING", "0") == "1"

SYSTEM_PROMPT = """
You infer a user profile for a conversational search user simulator.

The input contains:
- context: the user's first utterance.
- answer: the answer eventually given to the user.
- cquestion: possibly a responder's clarifying question, or other
  dialogue-related information. It is not necessarily a user statement.

Infer the user's state before receiving the answer. Use answer and cquestion
only to disambiguate the underlying intent. Never leak facts, explanations,
recommendations, or solutions learned only from the answer into known_info.

Return one JSON object with exactly these fields:
{
  "core_intent": "one concise sentence",
  "known_info": ["one atomic fact or constraint", "..."]
}

Rules:
- Write both fields in the same language as context.
- core_intent states what the user ultimately wants to understand or achieve.
- Preserve important entities, products, versions, locations, errors, goals,
  and constraints.
- known_info contains only facts, observations, attempts, constraints,
  preferences, and uncertainties already expressed or strongly implied by
  context.
- Do not treat cquestion as something the user said or knows unless context
  independently supports it.
- Do not include the final answer, solution, hidden causes, or newly supplied
  expert knowledge in known_info.
- Use an empty list when context supplies no known information.
- Do not speculate and do not output Markdown or any text outside the JSON.
""".strip()


def request_profile(record: dict) -> dict:
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "context": record["context"],
                        "answer": record["answer"],
                        "cquestion": record["cquestion"],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if CHAT_API_KEY:
        headers["Authorization"] = f"Bearer {CHAT_API_KEY}"

    request = Request(
        CHAT_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            result = json.load(response)
    except HTTPError as error:
        body = error.read().decode("utf-8")
        raise RuntimeError(
            f"POST {CHAT_URL} returned HTTP {error.code}:\n{body}"
        ) from error

    profile = json.loads(result["choices"][0]["message"]["content"])
    assert isinstance(profile["core_intent"], str)
    assert isinstance(profile["known_info"], list)
    assert all(isinstance(item, str) for item in profile["known_info"])

    core_intent = profile["core_intent"].strip()
    known_info = [item.strip() for item in profile["known_info"]]

    assert core_intent
    assert all(known_info)
    return {
        "core_intent": core_intent,
        "known_info": known_info,
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_existing_profiles(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    return {
        record["case_id"]: {
            "core_intent": record["core_intent"],
            "known_info": record["known_info"],
        }
        for record in read_jsonl(path)
    }


def write_ordered_output(
    output_path: Path,
    records: list[dict],
    profiles: dict[str, dict],
) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            enriched_record = {
                **record,
                **profiles[record["case_id"]],
            }
            file.write(
                json.dumps(enriched_record, ensure_ascii=False) + "\n"
            )
    temporary_path.replace(output_path)


def process_file(input_path: Path) -> None:
    relative_path = input_path.relative_to(DATASET_DIR)
    output_path = PROFILE_DIR / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_path)
    assert len({record["case_id"] for record in records}) == len(records)

    profiles = load_existing_profiles(output_path)
    completed = 0 if REGENERATE_EXISTING else sum(
        record["case_id"] in profiles
        for record in records
    )
    pending_records = [
        record
        for record in records
        if REGENERATE_EXISTING or record["case_id"] not in profiles
    ]

    with (
        output_path.open("a", encoding="utf-8") as output_file,
        tqdm(
            total=len(records),
            initial=completed,
            desc=str(relative_path),
            unit="case",
        ) as progress,
    ):
        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        futures = {
            executor.submit(request_profile, record): record
            for record in pending_records
        }
        try:
            for future in as_completed(futures):
                record = futures[future]
                profile = future.result()
                profiles[record["case_id"]] = profile
                output_file.write(
                    json.dumps(
                        {**record, **profile},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_file.flush()
                progress.update()
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    write_ordered_output(output_path, records, profiles)


def main() -> None:
    input_paths = sorted(DATASET_DIR.rglob("*.json"))
    assert input_paths, f"No JSON files found in {DATASET_DIR}"

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    for input_path in input_paths:
        process_file(input_path)


if __name__ == "__main__":
    main()

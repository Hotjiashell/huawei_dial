import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tqdm import tqdm

INPUT_PATH = Path(
    os.getenv("CASE_DATA_PATH")
    or os.getenv("CASE_ANSWERS_PATH")
    or "../../dataset/case_answers_with_title.json"
)
ELASTICSEARCH_URL = os.getenv(
    "ELASTICSEARCH_URL",
    "http://127.0.0.1:9200",
).rstrip("/")
ELASTICSEARCH_INDEX = os.getenv(
    "ELASTICSEARCH_INDEX",
    "clarq_cases_v1",
)
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY") or os.getenv(
    "ELASTIC_API_KEY"
)
ELASTICSEARCH_USER = os.getenv("ELASTICSEARCH_USER", "elastic")

EMBEDDING_URL = os.getenv(
    "EMBEDDING_URL",
    "http://127.0.0.1:8001/v1/embeddings",
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EMBEDDING_DIM = int(
    os.getenv("EMBEDDING_DIM") or os.getenv("EMBEDDING_DIMS") or "1024"
)
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "8192"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))
SKIP_EXISTING = os.getenv("SKIP_EXISTING", "1") == "1"


def elasticsearch_authorization() -> str:
    if ELASTICSEARCH_API_KEY:
        return f"ApiKey {ELASTICSEARCH_API_KEY}"

    password = os.environ["ELASTICSEARCH_PASSWORD"]
    credentials = base64.b64encode(
        f"{ELASTICSEARCH_USER}:{password}".encode()
    ).decode("ascii")
    return f"Basic {credentials}"


def request_json(
    url: str,
    payload: dict | None,
    headers: dict[str, str],
    method: str = "POST",
) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return json.load(response)
    except HTTPError as error:
        body = error.read().decode("utf-8")
        raise RuntimeError(
            f"{method} {url} returned HTTP {error.code}:\n{body}"
        ) from error


def elasticsearch_json(
    path: str,
    payload: dict | None = None,
    method: str = "POST",
) -> dict:
    return request_json(
        f"{ELASTICSEARCH_URL}{path}",
        payload,
        {
            "Authorization": elasticsearch_authorization(),
            "Content-Type": "application/json",
        },
        method,
    )


def embed(texts: list[str]) -> list[list[float]]:
    headers = {"Content-Type": "application/json"}
    if EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"

    result = request_json(
        EMBEDDING_URL,
        {
            "model": EMBEDDING_MODEL,
            "input": texts,
            "encoding_format": "float",
            "truncate_prompt_tokens": MAX_INPUT_TOKENS,
        },
        headers,
    )
    vectors = [
        item["embedding"]
        for item in sorted(result["data"], key=lambda item: item["index"])
    ]

    assert len(vectors) == len(texts)
    assert all(len(vector) == EMBEDDING_DIM for vector in vectors)
    return vectors


def existing_case_ids(case_ids: list[str]) -> set[str]:
    result = elasticsearch_json(
        f"/{ELASTICSEARCH_INDEX}/_mget?_source=false",
        {"ids": case_ids},
    )
    return {
        document["_id"]
        for document in result["docs"]
        if document["found"]
    }


def bulk_index(
    cases: list[dict],
    title_vectors: list[list[float]],
    answer_vectors: list[list[float]],
) -> None:
    lines = []
    for case, title_vector, answer_vector in zip(
        cases,
        title_vectors,
        answer_vectors,
        strict=True,
    ):
        lines.append(
            json.dumps(
                {
                    "index": {
                        "_index": ELASTICSEARCH_INDEX,
                        "_id": case["case_id"],
                    }
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "answer": case["answer"],
                    "title_vector": title_vector,
                    "answer_vector": answer_vector,
                },
                ensure_ascii=False,
            )
        )

    body = ("\n".join(lines) + "\n").encode("utf-8")
    request = Request(
        f"{ELASTICSEARCH_URL}/_bulk?refresh=false",
        data=body,
        headers={
            "Authorization": elasticsearch_authorization(),
            "Content-Type": "application/x-ndjson",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            result = json.load(response)
    except HTTPError as error:
        body = error.read().decode("utf-8")
        raise RuntimeError(
            f"POST {request.full_url} returned HTTP {error.code}:\n{body}"
        ) from error

    if result["errors"]:
        failures = [
            operation
            for item in result["items"]
            for operation in item.values()
            if operation["status"] >= 300
        ]
        raise RuntimeError(
            json.dumps(failures[:3], ensure_ascii=False, indent=2)
        )


def restore_refresh_interval() -> None:
    elasticsearch_json(
        f"/{ELASTICSEARCH_INDEX}/_settings",
        {"index": {"refresh_interval": "1s"}},
        method="PUT",
    )
    elasticsearch_json(f"/{ELASTICSEARCH_INDEX}/_refresh")


def main() -> None:
    cases = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    assert len({case["case_id"] for case in cases}) == len(cases)

    with tqdm(
        total=len(cases),
        desc="Embedding and indexing cases",
        unit="case",
    ) as progress:
        for start in range(0, len(cases), BATCH_SIZE):
            batch = cases[start : start + BATCH_SIZE]

            if SKIP_EXISTING:
                existing = existing_case_ids(
                    [case["case_id"] for case in batch]
                )
                progress.update(len(existing))
                batch = [
                    case
                    for case in batch
                    if case["case_id"] not in existing
                ]

            if not batch:
                continue

            title_vectors = embed([case["title"] for case in batch])
            answer_vectors = embed([case["answer"] for case in batch])
            bulk_index(batch, title_vectors, answer_vectors)
            progress.update(len(batch))

    restore_refresh_interval()
    count = elasticsearch_json(
        f"/{ELASTICSEARCH_INDEX}/_count",
        method="GET",
    )["count"]
    assert count == len(cases), (
        f"Expected {len(cases)} documents, found {count}"
    )
    print(f"Indexed {count} documents into {ELASTICSEARCH_INDEX}")


if __name__ == "__main__":
    main()

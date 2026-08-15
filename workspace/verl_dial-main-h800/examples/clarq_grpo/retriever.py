from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class RetrieverError(RuntimeError):
    pass


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float | None = None


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return None if text.upper() in {"", "NONE", "NULL", "EMPTY"} else text


class CaseRetriever:
    """Self-contained BM25 + embedding weighted-RRF retriever."""

    def __init__(self, config: dict[str, Any]):
        self.elasticsearch_url = str(config["search_server_host"]).rstrip("/")
        self.strategy = str(config.get("search_strategy", "hybrid"))
        self.index = str(config.get("index", "clarq_cases"))
        self.default_top_k = int(config.get("top_k", 5))

        self.embedding_url = str(config["embedding_url"])
        self.embedding_model = str(config["embedding_model"])
        self.embedding_api_key = str(config.get("embedding_api_key") or "")

        self.rank_window = int(config.get("rank_window", 50))
        self.vector_num_candidates = int(config.get("vector_num_candidates", 200))
        self.rrf_rank_constant = int(config.get("rrf_rank_constant", 60))
        self.bm25_title_boost = float(config.get("bm25_title_boost", 3.0))
        self.channel_weights = {
            "bm25": float(config.get("rrf_bm25_weight", 1.0)),
            "title_vector": float(config.get("rrf_title_vector_weight", 1.0)),
            "answer_vector": float(config.get("rrf_answer_vector_weight", 1.0)),
        }
        self.search_timeout = float(config.get("search_timeout", 30))
        self.embedding_timeout = float(config.get("embedding_timeout", 60))

        api_key = _optional_string(config.get("elasticsearch_api_key"))
        password = _optional_string(config.get("elasticsearch_password"))
        self.elasticsearch_headers = {"Content-Type": "application/json"}
        if api_key:
            self.elasticsearch_headers["Authorization"] = f"ApiKey {api_key}"
        self.elasticsearch_auth = None
        if not api_key and password:
            self.elasticsearch_auth = (
                str(config.get("elasticsearch_user", "elastic")),
                password,
            )

        ca_cert = _optional_string(config.get("elasticsearch_ca_cert"))
        verify_certs = config.get("elasticsearch_verify_certs", True)
        if isinstance(verify_certs, str):
            verify_certs = verify_certs.lower() == "true"
        self.elasticsearch_verify: bool | str = ca_cert or bool(verify_certs)

    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        cases = self.retrieve_case(
            query=query,
            top_k=max_results or self.default_top_k,
            strategy=self.strategy,
        )
        return [self._case_to_result(case) for case in cases]

    def retrieve_case(
        self,
        query: str,
        top_k: int,
        strategy: str,
    ) -> list[dict[str, Any]]:
        query = str(query or "").strip()
        if not query:
            return []

        top_k = max(1, int(top_k))
        rank_window = max(top_k, self.rank_window)
        channels = self._strategy_channels(strategy)
        rankings: list[tuple[str, list[dict[str, Any]], float]] = []

        if "bm25" in channels:
            rankings.append(
                ("bm25", self._bm25_search(query, rank_window), self.channel_weights["bm25"])
            )

        vector_channels = channels & {"title_vector", "answer_vector"}
        if vector_channels:
            query_vector = self._embed_query(query)
            for channel in ("title_vector", "answer_vector"):
                if channel in vector_channels:
                    rankings.append(
                        (
                            channel,
                            self._vector_search(query_vector, channel, rank_window),
                            self.channel_weights[channel],
                        )
                    )

        return self._rrf(rankings)[:top_k]

    @staticmethod
    def _strategy_channels(strategy: str) -> set[str]:
        strategies = {
            "bm25": {"bm25"},
            "lexical": {"bm25"},
            "title_vector": {"title_vector"},
            "answer_vector": {"answer_vector"},
            "vector": {"title_vector", "answer_vector"},
            "semantic": {"title_vector", "answer_vector"},
            "hybrid": {"bm25", "title_vector", "answer_vector"},
            "lexical&semantic": {"bm25", "title_vector", "answer_vector"},
        }
        normalized = str(strategy or "hybrid").strip().lower()
        if normalized not in strategies:
            raise RetrieverError(f"Unsupported search strategy: {strategy}")
        return strategies[normalized]

    def _bm25_search(self, query: str, rank_window: int) -> list[dict[str, Any]]:
        result = self._elasticsearch_search(
            {
                "size": rank_window,
                "_source": ["case_id", "title", "answer"],
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": [f"title^{self.bm25_title_boost}", "answer"],
                        "type": "best_fields",
                    }
                },
            }
        )
        return self._hits(result)

    def _vector_search(
        self,
        query_vector: list[float],
        field: str,
        rank_window: int,
    ) -> list[dict[str, Any]]:
        result = self._elasticsearch_search(
            {
                "size": rank_window,
                "_source": ["case_id", "title", "answer"],
                "knn": {
                    "field": field,
                    "query_vector": query_vector,
                    "k": rank_window,
                    "num_candidates": max(rank_window, self.vector_num_candidates),
                },
            }
        )
        return self._hits(result)

    def _embed_query(self, query: str) -> list[float]:
        headers = {"Content-Type": "application/json"}
        if self.embedding_api_key:
            headers["Authorization"] = f"Bearer {self.embedding_api_key}"
        result = self._post_json(
            self.embedding_url,
            {
                "model": self.embedding_model,
                "input": [query],
                "encoding_format": "float",
            },
            headers=headers,
            timeout=self.embedding_timeout,
        )
        return sorted(result["data"], key=lambda item: item["index"])[0]["embedding"]

    def _elasticsearch_search(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._post_json(
            f"{self.elasticsearch_url}/{self.index}/_search",
            body,
            headers=self.elasticsearch_headers,
            timeout=self.search_timeout,
            auth=self.elasticsearch_auth,
            verify=self.elasticsearch_verify,
        )

    @staticmethod
    def _post_json(
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
        auth: tuple[str, str] | None = None,
        verify: bool | str = True,
    ) -> dict[str, Any]:
        response = requests.post(
            url,
            headers=headers,
            auth=auth,
            json=body,
            timeout=timeout,
            verify=verify,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _hits(result: dict[str, Any]) -> list[dict[str, Any]]:
        hits = result.get("hits", {}).get("hits", [])
        if not isinstance(hits, list):
            raise RetrieverError("Elasticsearch returned invalid hits.")
        return hits

    def _rrf(
        self,
        rankings: list[tuple[str, list[dict[str, Any]], float]],
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        sources: dict[str, dict[str, Any]] = {}
        channel_ranks: dict[str, dict[str, int]] = {}

        for channel, hits, weight in rankings:
            for rank, hit in enumerate(hits, start=1):
                source = hit.get("_source") or {}
                case_id = str(source.get("case_id") or hit.get("_id") or "")
                if not case_id:
                    continue
                scores[case_id] = scores.get(case_id, 0.0) + (
                    weight / (self.rrf_rank_constant + rank)
                )
                sources[case_id] = source
                channel_ranks.setdefault(case_id, {})[channel] = rank

        ranked_case_ids = sorted(scores, key=scores.get, reverse=True)
        return [
            {
                "caseID": case_id,
                "case_name": str(sources[case_id].get("title") or ""),
                "text": str(sources[case_id].get("answer") or ""),
                "score": scores[case_id],
                "metadata": {"retrieval_ranks": channel_ranks[case_id]},
            }
            for case_id in ranked_case_ids
        ]

    @staticmethod
    def _case_to_result(item: dict[str, Any]) -> SearchResult:
        case_id = str(item.get("caseID") or item.get("case_id") or "")
        return SearchResult(
            title=str(item.get("case_name") or case_id or "Untitled case"),
            url=case_id,
            content=str(item.get("text") or ""),
            score=item.get("score"),
        )

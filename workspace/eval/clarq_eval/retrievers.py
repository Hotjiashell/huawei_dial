"""Adapters for retrieval backends used by the ClarQ evaluator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class NewRetrieveCaseAdapter:
    """Expose ``new_retrive.retrieve_case`` through the evaluator's search API.

    The external interface owns its strategy and other optional controls.  The
    evaluator only supplies the per-call query and result depth plus the
    configured index name.
    """

    def __init__(
        self,
        retrieve_case: Callable[..., list[dict[str, Any]]],
        *,
        default_top_k: int,
        index: str,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be positive")
        if not index.strip():
            raise ValueError("index must be non-empty")
        self._retrieve_case = retrieve_case
        self._default_top_k = default_top_k
        self._index = index

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []
        top_k = int(max_results or self._default_top_k)
        if top_k <= 0:
            raise ValueError("max_results must be positive")
        results = self._retrieve_case(
            query=normalized_query,
            top_k=top_k,
            index=self._index,
        )
        if not isinstance(results, list):
            raise TypeError("new_retrive.retrieve_case must return a list of case dictionaries")
        return results

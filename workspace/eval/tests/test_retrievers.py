from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from clarq_eval.retrievers import NewRetrieveCaseAdapter


class NewRetrieveCaseAdapterTests(unittest.TestCase):
    def test_forwards_only_query_top_k_and_index(self) -> None:
        calls: list[dict] = []

        def retrieve_case(**kwargs: object) -> list[dict]:
            calls.append(dict(kwargs))
            return [{"caseID": "case-1", "case_name": "Case title", "text": "Case content"}]

        retriever = NewRetrieveCaseAdapter(
            retrieve_case,
            default_top_k=5,
            index="document_12",
        )

        results = retriever.search("  device issue  ", 3)

        self.assertEqual(
            calls,
            [{"query": "device issue", "top_k": 3, "index": "document_12"}],
        )
        self.assertEqual(results[0]["caseID"], "case-1")

    def test_uses_default_top_k_and_does_not_call_for_empty_query(self) -> None:
        calls: list[dict] = []

        def retrieve_case(**kwargs: object) -> list[dict]:
            calls.append(dict(kwargs))
            return []

        retriever = NewRetrieveCaseAdapter(retrieve_case, default_top_k=5, index="document_12")

        self.assertEqual(retriever.search(""), [])
        self.assertEqual(calls, [])
        retriever.search("question")
        self.assertEqual(calls, [{"query": "question", "top_k": 5, "index": "document_12"}])

    def test_rejects_non_list_result(self) -> None:
        retriever = NewRetrieveCaseAdapter(
            lambda **_kwargs: {"caseID": "wrong-type"},  # type: ignore[arg-type]
            default_top_k=5,
            index="document_12",
        )

        with self.assertRaisesRegex(TypeError, "must return a list"):
            retriever.search("question")


if __name__ == "__main__":
    unittest.main()

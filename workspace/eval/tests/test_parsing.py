import json
import unittest

import _bootstrap  # noqa: F401

from clarq_eval.parsing import (
    PolicyProtocolError,
    canonical_assistant_message,
    parse_json_object,
    parse_policy_response,
)


class PolicyParsingTests(unittest.TestCase):
    def test_standard_tool_call(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "search_case",
                                    "arguments": json.dumps({"query": "refined query"}),
                                },
                            }
                        ],
                    },
                }
            ]
        }

        turn = parse_policy_response(response)

        self.assertEqual(turn.tool_calls[0].name, "search_case")
        self.assertEqual(turn.tool_calls[0].arguments, {"query": "refined query"})
        self.assertEqual(turn.violations, [])
        canonical = canonical_assistant_message(turn)
        self.assertIsNone(canonical["content"])
        self.assertEqual(canonical["tool_calls"][0]["id"], "call_abc")

    def test_raw_hermes_call_is_not_replayed_in_content(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            "<think>private reasoning</think>"
                            '<tool_call>{"name":"clarify_user","arguments":'
                            '{"question":"Which version?"}}</tool_call>'
                        )
                    },
                }
            ]
        }

        turn = parse_policy_response(response)
        canonical = canonical_assistant_message(turn)

        self.assertEqual(turn.tool_calls[0].arguments["question"], "Which version?")
        self.assertEqual(turn.cleaned_content, "")
        self.assertIsNone(canonical["content"])
        self.assertNotIn("<tool_call>", json.dumps(canonical))

    def test_complete_with_thinking(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "<think>done</think>\nComplete"},
                }
            ]
        }

        turn = parse_policy_response(response)

        self.assertTrue(turn.is_complete)
        self.assertEqual(turn.cleaned_content, "Complete")

    def test_multiple_calls_and_text_are_flagged(self) -> None:
        calls = [
            {
                "id": f"call_{index}",
                "function": {"name": "search_case", "arguments": {"query": str(index)}},
            }
            for index in range(2)
        ]
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {"content": "also text", "tool_calls": calls},
                }
            ]
        }

        turn = parse_policy_response(response)

        self.assertEqual(turn.violations, ["multiple_tool_calls", "text_with_tool_call"])

    def test_invalid_arguments_raise_protocol_error(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "search_case", "arguments": "not-json"}}
                        ]
                    }
                }
            ]
        }
        with self.assertRaises(PolicyProtocolError):
            parse_policy_response(response)

    def test_judge_json_code_fence_and_embedded_object(self) -> None:
        self.assertEqual(parse_json_object('```json\n{"score": 4}\n```'), {"score": 4})
        self.assertEqual(parse_json_object('result: {"score": 5} done'), {"score": 5})


if __name__ == "__main__":
    unittest.main()

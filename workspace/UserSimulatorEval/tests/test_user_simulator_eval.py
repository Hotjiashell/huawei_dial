from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_test_set import dialogue_inputs, make_test_record, normalise_review  # noqa: E402
from evaluate_user_simulator import aggregate, normalise_metric_judgement  # noqa: E402
from trim_human_responses import first_line, trim_records  # noqa: E402
from user_simulator_eval_common import OpenAIChatClient, load_json_records  # noqa: E402


class UserSimulatorEvalTests(unittest.TestCase):
    def test_trim_human_response_at_first_newline(self) -> None:
        self.assertEqual("第一句", first_line("第一句\n第二句"))
        self.assertEqual("第一句", first_line("第一句\r\n第二句"))
        self.assertEqual("第一句", first_line("第一句\r第二句"))
        self.assertEqual("没有换行", first_line("没有换行"))

        records, changed = trim_records(
            [
                {"sample_id": "1", "human_response": "回答第一行\n回答第二行", "known_info": ["事实"]},
                {"sample_id": "2", "human_response": "单行回复"},
            ]
        )
        self.assertEqual(1, changed)
        self.assertEqual("回答第一行", records[0]["human_response"])
        self.assertEqual(["事实"], records[0]["known_info"])
        self.assertEqual("单行回复", records[1]["human_response"])

    def test_concatenated_json_objects_are_all_sent_to_the_llm(self) -> None:
        """The extractor must not structurally pre-filter incomplete chats."""

        raw = (
            '{"call_sno":1,"chat_content":"用户：无法登录\\n客服：请问您使用什么设备？\\n用户：Windows 电脑",'
            '"context":"无法登录","known_info":["无法登录"]}'
            '{"call_sno":2,"chat_content":"用户：还有问题","known_info":[]}'
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dialog.json"
            source.write_text(raw, encoding="utf-8")
            records = load_json_records(source)

        self.assertEqual(2, len(records))
        dialogues = dialogue_inputs(records)
        self.assertEqual(2, len(dialogues))
        self.assertEqual("用户：无法登录\n客服：请问您使用什么设备？\n用户：Windows 电脑", dialogues[0]["raw_chat_content"])
        # This source has no clarification interaction.  It is still passed
        # through, and only the LLM may mark it ineligible.
        self.assertEqual("用户：还有问题", dialogues[1]["raw_chat_content"])

    def test_test_record_keeps_original_and_added_known_info_separate(self) -> None:
        dialogue = {
            "sample_id": "dialog-1-q1",
            "source": {},
            "existing_known_info": ["无法登录", "无法登录"],
            "raw_chat_content": "用户：无法登录\n客服：使用什么设备？\n用户：Windows 电脑",
        }
        record = make_test_record(
            dialogue,
            {
                "added_known_info": ["用户使用 Windows 电脑", "无法登录"],
                "reason": "用户明确说使用 Windows 电脑",
                "initial_question": "无法登录",
                "clarification_question": "使用什么设备？",
                "human_response": "Windows 电脑",
            },
        )
        self.assertEqual(["无法登录"], record["original_known_info"])
        self.assertEqual(["用户使用 Windows 电脑"], record["added_known_info"])
        self.assertEqual(["无法登录", "用户使用 Windows 电脑"], record["known_info"])
        self.assertEqual("Windows 电脑", record["human_response"])
        self.assertIn("客服：使用什么设备？", record["source_chat_content"])

    def test_eligible_review_requires_only_protocol_fields(self) -> None:
        review = normalise_review(
            {
                "eligible": True,
                "reason": "模型判断该对话完整",
                "added_known_info": [],
                "initial_question": "无法登录",
                "clarification_question": "使用什么设备？",
                "human_response": "Windows 电脑",
            }
        )
        self.assertTrue(review["eligible"])
        self.assertEqual("无法登录", review["initial_question"])

    def test_metric_normalisation_and_aggregation(self) -> None:
        first = normalise_metric_judgement(
            {
                "non_response": False,
                "reason": "回答了设备，额外说了地点",
                "information_points": [
                    {"text": "使用 Windows 电脑", "requested_by_agent": True},
                    {"text": "位于红区", "requested_by_agent": False},
                    {"text": "使用 Windows 电脑", "requested_by_agent": True},
                ],
            }
        )
        first["reply_length_characters"] = 10
        second = normalise_metric_judgement(
            {"non_response": True, "information_points": []}
        )
        second["reply_length_characters"] = 4
        second["exclude_from_average_reply_length"] = True
        second["reply_length_exclusion_reason"] = "programmatic_invalid_clarification_response"
        third = normalise_metric_judgement(
            {"non_response": False, "information_points": []}
        )
        third["reply_length_characters"] = 12
        third["exclude_from_average_reply_length"] = True
        third["reply_length_exclusion_reason"] = "programmatic_unknown_response"
        summary = aggregate([first, second, third])
        self.assertEqual(3, summary["evaluated_reply_count"])
        self.assertEqual(2 / 3, summary["information_efficiency_average_points_per_reply"])
        self.assertEqual(1 / 3, summary["information_leakage_rate"])
        self.assertEqual(1, summary["reply_length_included_reply_count"])
        self.assertEqual(1, summary["programmatic_invalid_clarification_reply_excluded_from_length_count"])
        self.assertEqual(1, summary["programmatic_unknown_reply_excluded_from_length_count"])
        self.assertEqual(2, summary["programmatic_reply_excluded_from_length_count"])
        self.assertEqual(10.0, summary["average_reply_length_characters"])
        self.assertEqual(1 / 3, summary["non_response_rate"])

    def test_chat_request_disables_thinking_and_falls_back_json_mode(self) -> None:
        client = OpenAIChatClient("http://example.test/v1", "fake-model", max_retries=0)
        submitted: list[dict[str, object]] = []

        def fake_post(body: dict[str, object]) -> str:
            submitted.append(body)
            if "response_format" in body:
                raise RuntimeError("server does not support json mode")
            return json.dumps({"ok": True})

        client._post = fake_post  # type: ignore[method-assign]
        response = client.chat([{"role": "user", "content": "测试"}], json_object=True)
        self.assertEqual({"ok": True}, response["json"])
        self.assertEqual(2, len(submitted))
        for body in submitted:
            self.assertEqual({"enable_thinking": False}, body["chat_template_kwargs"])


if __name__ == "__main__":
    unittest.main()

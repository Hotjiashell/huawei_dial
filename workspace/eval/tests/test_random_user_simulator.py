from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from clarq_eval.random_user_simulator import RandomUserSimulator, RandomUserSimulatorConfig
from clarq_eval.models import EvaluationSample


SAMPLE = EvaluationSample(
    sample_id="sample-1",
    domain="money",
    target_case_id="target",
    initial_question="How can I fix the connection problem?",
    core_intent="fix connection",
    known_info=("The phone runs Android 14.", "The problem started after a system update."),
    target_case_title="Canonical title",
    target_case_content="Canonical answer",
)


class SequenceChatClient:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.calls: list[tuple[list[dict], dict]] = []

    def chat(self, messages: list[dict], **kwargs) -> dict:
        self.calls.append((messages, kwargs))
        if not self.contents:
            raise AssertionError("chat response sequence exhausted")
        return {"choices": [{"message": {"content": self.contents.pop(0)}}]}


class RandomUserSimulatorTests(unittest.TestCase):
    def test_rephrase_behavior_skips_grounded_selection(self) -> None:
        client = SequenceChatClient(["I need help resolving my connection issue."])
        simulator = RandomUserSimulator(
            client,
            RandomUserSimulatorConfig(
                seed=7,
                rephrase_question_probability=1.0,
                compress_known_info_probability=0.0,
                proactive_known_info_probability=0.0,
            ),
        )

        reply = simulator.answer(SAMPLE, "What Android version are you using?")

        self.assertEqual(reply.text, "I need help resolving my connection issue.")
        self.assertEqual(reply.clarification_type, "unknown")
        self.assertEqual(reply.behavior, "rephrase_initial_question")
        self.assertEqual(len(client.calls), 1)
        prompt = client.calls[0][0][0]["content"]
        self.assertIn("does NOT answer", prompt)
        self.assertIn(SAMPLE.initial_question, prompt)

    def test_known_fact_can_be_compressed_with_a_second_model_call(self) -> None:
        client = SequenceChatClient(["KNOWN_INFO_1", "Android 14."])
        simulator = RandomUserSimulator(
            client,
            RandomUserSimulatorConfig(
                seed=7,
                rephrase_question_probability=0.0,
                compress_known_info_probability=1.0,
                proactive_known_info_probability=0.0,
            ),
        )

        reply = simulator.answer(SAMPLE, "What Android version are you using?")

        self.assertEqual(reply.text, "Android 14.")
        self.assertEqual(reply.clarification_type, "known_info")
        self.assertEqual(reply.behavior, "compressed_known_info")
        self.assertEqual(reply.source_known_info, "The phone runs Android 14.")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0][1]["max_tokens"], 16)
        self.assertEqual(client.calls[1][1]["max_tokens"], 64)
        self.assertIn("The phone runs Android 14.", client.calls[1][0][0]["content"])

    def test_unknown_can_proactively_volunteer_a_random_known_fact(self) -> None:
        client = SequenceChatClient(["UNKNOWN", "KNOWN_INFO_2"])
        simulator = RandomUserSimulator(
            client,
            RandomUserSimulatorConfig(
                seed=7,
                rephrase_question_probability=0.0,
                compress_known_info_probability=0.0,
                enable_proactive_known_info_on_unknown=True,
                proactive_known_info_probability=1.0,
            ),
        )

        reply = simulator.answer(SAMPLE, "Which router firmware version is installed?")

        self.assertEqual(reply.text, "The problem started after a system update.")
        self.assertEqual(reply.clarification_type, "known_info")
        self.assertEqual(reply.behavior, "proactive_known_info")
        self.assertEqual(reply.source_known_info, reply.text)
        self.assertEqual(len(client.calls), 2)
        prompt = client.calls[1][0][0]["content"]
        self.assertIn("does not know the answer", prompt)
        self.assertIn("KNOWN_INFO_1", client.calls[1][1]["extra_payload"]["structured_outputs"]["choice"])
        self.assertIn("KNOWN_INFO_2", client.calls[1][1]["extra_payload"]["structured_outputs"]["choice"])

    def test_unknown_behavior_remains_unknown_when_proactive_option_is_disabled(self) -> None:
        client = SequenceChatClient(["UNKNOWN"])
        simulator = RandomUserSimulator(
            client,
            RandomUserSimulatorConfig(
                seed=7,
                rephrase_question_probability=0.0,
                compress_known_info_probability=0.0,
                enable_proactive_known_info_on_unknown=False,
                proactive_known_info_probability=1.0,
            ),
        )

        reply = simulator.answer(SAMPLE, "Which router firmware version is installed?")

        self.assertEqual(reply.text, "I don't know.")
        self.assertEqual(reply.clarification_type, "unknown")
        self.assertEqual(reply.behavior, "unknown")
        self.assertEqual(len(client.calls), 1)

    def test_probability_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            RandomUserSimulatorConfig(rephrase_question_probability=1.1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import requests

from examples.clarq_grpo.agent import (
    CLARIFICATION_COUNT_KEY,
    INVALID_CLARIFICATION_COUNT_KEY,
    INVALID_CLARIFICATION_RESPONSE,
    LAST_CASE_IDS_KEY,
    LAST_CASES_KEY,
    NOT_SATISFIED_DONE_TOKEN,
    SATISFIED_DONE_TOKEN,
    SEARCH_ATTEMPT_COUNT_KEY,
    SEARCH_COUNT_KEY,
    UNANSWERED_CLARIFICATION_COUNT_KEY,
    UNKNOWN_CLARIFICATION_COUNT_KEY,
    UNKNOWN_CLARIFICATION_RESPONSE,
    USEFUL_CLARIFICATION_COUNT_KEY,
    ClarifyUserTool,
    ClarQAgentLoop,
    SearchCaseTool,
    UserSimulatorClient,
    compute_result_reward,
)
from examples.clarq_grpo.prepare_data import convert_profile, load_split
from examples.clarq_grpo.retriever import CaseRetriever
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

REWARD_CONFIG = {
    "satisfied": 0.3,
    "not_satisfied": -1.0,
    "top1": 1.0,
    "top3": 0.6,
    "top5": 0.3,
    "miss": -0.5,
    "repeat_search_penalty": -0.05,
    "known_info_clarification_reward": 0.1,
    "unknown_clarification_penalty": -0.02,
    "invalid_clarification_penalty": -0.02,
}
PROFILE = {
    "case_id": "case0002",
    "id": "electronics_1_1",
    "context": "Why does this circuit fail?",
    "cquestion": "Hidden dialogue data",
    "answer": "Hidden reference answer",
    "core_intent": "Find the circuit failure.",
    "known_info": ["The supply is 5 V.", "The output is always low."],
    "initial_question": "Why does this circuit fail?",
}


async def direct_to_thread(function, *args, **kwargs):
    return function(*args, **kwargs)


class FakeSimulator:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[dict, object]] = []

    def respond(self, profile, system_input, clarification_count=0):
        self.calls.append((profile, system_input, clarification_count))
        return self.response


class FakeRetriever:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def search(self, query, top_k):
        self.calls += 1
        return self.responses.pop(0)


class ClarQDataTest(unittest.TestCase):
    def test_original_context_is_preserved_as_the_policy_prompt(self):
        record = convert_profile(PROFILE, split="train", domain="electronics", index=0)
        prompt_text = json.dumps(record["prompt"], ensure_ascii=False)

        self.assertEqual(record["data_source"], "clarq")
        self.assertIn(PROFILE["context"], prompt_text)
        self.assertNotIn(PROFILE["core_intent"], prompt_text)
        self.assertNotIn(PROFILE["answer"], prompt_text)
        self.assertNotIn(PROFILE["cquestion"], prompt_text)
        self.assertEqual(record["reward_model"]["ground_truth"], PROFILE["case_id"])
        self.assertEqual(record["extra_info"]["profile"]["initial_question"], PROFILE["context"])
        self.assertEqual(record["extra_info"]["profile"]["known_info"], PROFILE["known_info"])

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[3] / "ClarQ" / "profile_split").exists(),
        "ClarQ/profile_split is not available.",
    )
    def test_current_profile_split_counts(self):
        profile_root = Path(__file__).resolve().parents[3] / "ClarQ" / "profile_split"
        self.assertEqual(len(load_split(profile_root, "train")), 4800)
        self.assertEqual(len(load_split(profile_root, "validation")), 800)
        self.assertEqual(len(load_split(profile_root, "test")), 800)


class ClarQRewardTest(unittest.TestCase):
    def test_reward_table(self):
        cases = ["case0001", "case0002", "case0003", "case0004", "case0005"]
        retrieval_scenarios = {
            "case0001": 1.0,
            "case0003": 0.6,
            "case0005": 0.3,
            "case9999": -0.5,
        }

        for feedback, satisfaction_reward in (
            (SATISFIED_DONE_TOKEN, 0.3),
            (NOT_SATISFIED_DONE_TOKEN, -1.0),
        ):
            for target, retrieval_reward in retrieval_scenarios.items():
                with self.subTest(feedback=feedback, target=target):
                    reward = compute_result_reward(feedback, cases, target, REWARD_CONFIG)
                    self.assertEqual(reward["satisfaction_reward"], satisfaction_reward)
                    self.assertAlmostEqual(
                        reward["total_reward"],
                        satisfaction_reward + retrieval_reward,
                    )
                    target_rank = cases.index(target) + 1 if target in cases else 0
                    self.assertEqual(reward["recall_at_1"], float(target_rank == 1))
                    self.assertEqual(reward["recall_at_3"], float(1 <= target_rank <= 3))
                    self.assertEqual(reward["recall_at_5"], float(1 <= target_rank <= 5))
                    self.assertEqual(reward["miss_rate"], float(target_rank == 0))


class ClarQSimulatorPromptTest(unittest.TestCase):
    def test_case_judge_uses_a_separate_constrained_prompt(self):
        simulator = UserSimulatorClient(
            {
                "base_url": "http://127.0.0.1:8005/v1",
                "model": "mock-simulator",
                "max_retries": 1,
            }
        )
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": SATISFIED_DONE_TOKEN}}]},
        )
        cases = [{"title": "Target", "content": "Solution"}]

        with patch("examples.clarq_grpo.agent.requests.post", return_value=response) as post:
            feedback = simulator.respond(PROFILE, cases, clarification_count=2)

        body = post.call_args.kwargs["json"]
        prompt = body["messages"][0]["content"]
        self.assertEqual(feedback, SATISFIED_DONE_TOKEN)
        self.assertIn(f"Initial user question:\n{PROFILE['initial_question']}", prompt)
        self.assertNotIn("Clarification questions asked", prompt)
        self.assertNotIn("Facts known by the user", prompt)
        self.assertEqual(
            body["structured_outputs"]["choice"],
            [SATISFIED_DONE_TOKEN, NOT_SATISFIED_DONE_TOKEN],
        )

    def test_clarification_choice_maps_to_an_exact_known_fact(self):
        simulator = UserSimulatorClient(
            {
                "base_url": "http://127.0.0.1:8005/v1",
                "model": "mock-simulator",
            }
        )
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "KNOWN_INFO_1"}}]},
        )

        with patch("examples.clarq_grpo.agent.requests.post", return_value=response) as post:
            feedback = simulator.respond(PROFILE, "Another question?", clarification_count=3)

        body = post.call_args.kwargs["json"]
        prompt = body["messages"][0]["content"]
        self.assertEqual(feedback, PROFILE["known_info"][0])
        self.assertNotIn("Clarification questions asked", prompt)
        self.assertEqual(
            body["structured_outputs"]["choice"],
            ["KNOWN_INFO_1", "KNOWN_INFO_2", "UNKNOWN", "INVALID_QUESTION"],
        )

    def test_unknown_clarification_cannot_hallucinate(self):
        simulator = UserSimulatorClient({"base_url": "http://127.0.0.1:8005/v1", "model": "mock-simulator"})
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "UNKNOWN"}}]},
        )

        with patch("examples.clarq_grpo.agent.requests.post", return_value=response):
            feedback = simulator.respond(PROFILE, "Which regulator type is used?")

        self.assertEqual(feedback, UNKNOWN_CLARIFICATION_RESPONSE)

    def test_prompt_is_truncated_to_the_configured_token_budget(self):
        class CharacterTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return [ord(character) for character in text]

            @staticmethod
            def decode(token_ids, **kwargs):
                return "".join(chr(token_id) for token_id in token_ids)

        simulator = UserSimulatorClient(
            {
                "base_url": "http://127.0.0.1:8005/v1",
                "model": "mock-simulator",
                "max_input_tokens": 600,
                "tokenizer_path": "unused-in-test",
            }
        )
        simulator._tokenizer = CharacterTokenizer()
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": SATISFIED_DONE_TOKEN}}]},
        )
        cases = [{"title": "Target", "content": "long case content " * 500}]

        with patch("examples.clarq_grpo.agent.requests.post", return_value=response) as post:
            feedback = simulator.respond(PROFILE, cases)

        prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual(feedback, SATISFIED_DONE_TOKEN)
        self.assertLessEqual(len(simulator._tokenizer.encode(prompt)), 600)
        self.assertIn("Content truncated", prompt)

    def test_http_400_falls_back_for_only_the_failed_sample(self):
        simulator = UserSimulatorClient(
            {
                "base_url": "http://127.0.0.1:8005/v1",
                "model": "mock-simulator",
                "max_retries": 3,
            }
        )
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError(response=SimpleNamespace(status_code=400))

        with patch("examples.clarq_grpo.agent.requests.post", return_value=response) as post:
            feedback = simulator.respond(PROFILE, [{"title": "Target", "content": "Solution"}])

        self.assertEqual(feedback, NOT_SATISFIED_DONE_TOKEN)
        self.assertEqual(post.call_count, 1)


class ClarQRetrieverTest(unittest.TestCase):
    def test_hybrid_search_uses_all_channels_and_weighted_rrf(self):
        retriever = CaseRetriever(
            {
                "search_server_host": "http://127.0.0.1:9200",
                "search_strategy": "hybrid",
                "embedding_url": "http://127.0.0.1:8001/v1/embeddings",
                "embedding_model": "bge-m3",
                "top_k": 5,
            }
        )
        case_a = {"_source": {"case_id": "case-a", "title": "A", "answer": "answer A"}}
        case_b = {"_source": {"case_id": "case-b", "title": "B", "answer": "answer B"}}

        with (
            patch.object(retriever, "_bm25_search", return_value=[case_a, case_b]) as bm25,
            patch.object(retriever, "_embed_query", return_value=[0.1, 0.2]) as embed,
            patch.object(
                retriever,
                "_vector_search",
                side_effect=[[case_b, case_a], [case_b]],
            ) as vector,
        ):
            results = retriever.search("broken circuit", 2)

        self.assertEqual([result.url for result in results], ["case-b", "case-a"])
        bm25.assert_called_once_with("broken circuit", 50)
        embed.assert_called_once_with("broken circuit")
        self.assertEqual([call.args[1] for call in vector.call_args_list], ["title_vector", "answer_vector"])


class ClarQToolTest(unittest.IsolatedAsyncioTestCase):
    async def test_clarification_uses_only_assigned_profile(self):
        tool = object.__new__(ClarifyUserTool)
        tool.simulator = FakeSimulator(PROFILE["known_info"][0])
        tool.profiles = {}
        agent_data = SimpleNamespace(extra_fields={})

        with patch("examples.clarq_grpo.agent.asyncio.to_thread", new=direct_to_thread):
            instance_id, _ = await tool.create({"profile": PROFILE})
            response, reward, metrics = await tool.execute(
                instance_id,
                {"question": "What is the supply voltage?"},
                agent_data=agent_data,
            )
            await tool.release(instance_id)

        self.assertEqual(response.text, PROFILE["known_info"][0])
        self.assertEqual(reward, 0.0)
        self.assertEqual(
            metrics,
            {
                "useful_clarification": 1,
                "known_info_clarification": 1,
                "unknown_clarification": 0,
                "invalid_clarification": 0,
            },
        )
        self.assertEqual(tool.simulator.calls[0], (PROFILE, "What is the supply voltage?", 1))
        self.assertEqual(agent_data.extra_fields[CLARIFICATION_COUNT_KEY], 1)
        self.assertEqual(agent_data.extra_fields[USEFUL_CLARIFICATION_COUNT_KEY], 1)
        self.assertNotIn(UNANSWERED_CLARIFICATION_COUNT_KEY, agent_data.extra_fields)
        self.assertEqual(tool.profiles, {})

    async def test_unknown_clarification_is_tracked_as_unanswered(self):
        tool = object.__new__(ClarifyUserTool)
        tool.simulator = FakeSimulator(UNKNOWN_CLARIFICATION_RESPONSE)
        tool.profiles = {"instance": PROFILE}
        agent_data = SimpleNamespace(extra_fields={})

        with patch("examples.clarq_grpo.agent.asyncio.to_thread", new=direct_to_thread):
            response, _, metrics = await tool.execute(
                "instance",
                {"question": "Which regulator is used?"},
                agent_data=agent_data,
            )

        self.assertEqual(response.text, UNKNOWN_CLARIFICATION_RESPONSE)
        self.assertEqual(
            metrics,
            {
                "useful_clarification": 0,
                "known_info_clarification": 0,
                "unknown_clarification": 1,
                "invalid_clarification": 0,
            },
        )
        self.assertEqual(agent_data.extra_fields[UNANSWERED_CLARIFICATION_COUNT_KEY], 1)
        self.assertEqual(agent_data.extra_fields[UNKNOWN_CLARIFICATION_COUNT_KEY], 1)

    async def test_invalid_clarification_is_tracked_separately(self):
        tool = object.__new__(ClarifyUserTool)
        tool.simulator = FakeSimulator(INVALID_CLARIFICATION_RESPONSE)
        tool.profiles = {"instance": PROFILE}
        agent_data = SimpleNamespace(extra_fields={})

        with patch("examples.clarq_grpo.agent.asyncio.to_thread", new=direct_to_thread):
            response, _, metrics = await tool.execute(
                "instance",
                {"question": "What voltage and current are used?"},
                agent_data=agent_data,
            )

        self.assertEqual(response.text, INVALID_CLARIFICATION_RESPONSE)
        self.assertEqual(metrics["invalid_clarification"], 1)
        self.assertEqual(agent_data.extra_fields[UNANSWERED_CLARIFICATION_COUNT_KEY], 1)
        self.assertEqual(agent_data.extra_fields[INVALID_CLARIFICATION_COUNT_KEY], 1)

    async def test_search_uses_fourth_result_after_limit(self):
        search_results = [
            [SimpleNamespace(url=f"case000{index}", title=f"Result {index}", content=str(index) * 20, score=1.0)]
            for index in range(1, 5)
        ]
        tool = object.__new__(SearchCaseTool)
        tool.top_k = 5
        tool.max_searches = 4
        tool.case_content_chars = 10
        tool.retriever = FakeRetriever(search_results)
        agent_data = SimpleNamespace(extra_fields={})

        with patch("examples.clarq_grpo.agent.asyncio.to_thread", new=direct_to_thread):
            responses = []
            for index in range(1, 5):
                response, _, _ = await tool.execute(
                    "unused",
                    {"query": f"query {index}"},
                    agent_data=agent_data,
                )
                responses.append(response)
            blocked, _, blocked_metrics = await tool.execute(
                "unused",
                {"query": "query 5"},
                agent_data=agent_data,
            )

        self.assertEqual(json.loads(responses[0].text)["cases"][0]["content"], "1" * 10)
        self.assertEqual(json.loads(responses[-1].text)["cases"][0]["case_id"], "case0004")
        self.assertEqual(json.loads(blocked.text)["cases"][0]["case_id"], "case0004")
        self.assertEqual(agent_data.extra_fields[LAST_CASE_IDS_KEY], ["case0004"])
        self.assertEqual(agent_data.extra_fields[LAST_CASES_KEY][0]["content"], "4" * 10)
        self.assertEqual(agent_data.extra_fields[SEARCH_COUNT_KEY], 4)
        self.assertEqual(agent_data.extra_fields[SEARCH_ATTEMPT_COUNT_KEY], 5)
        self.assertIn("Search limit reached", json.loads(blocked.text)["notice"])
        self.assertEqual(blocked_metrics["search_limit_reached"], 1)
        self.assertEqual(tool.retriever.calls, 4)


class ClarQAgentLoopTest(unittest.IsolatedAsyncioTestCase):
    async def test_final_reward_and_response_mask(self):
        base_output = AgentLoopOutput(
            prompt_ids=[1, 2],
            response_ids=[3, 4, 5],
            response_mask=[1, 0, 1],
            num_turns=4,
            metrics=AgentLoopMetrics(),
            extra_fields={
                LAST_CASES_KEY: [
                    {"case_id": "case0002", "title": "Target", "content": "Solution"},
                ],
                LAST_CASE_IDS_KEY: ["case0002"],
                SEARCH_COUNT_KEY: 1,
                SEARCH_ATTEMPT_COUNT_KEY: 3,
                CLARIFICATION_COUNT_KEY: 3,
                USEFUL_CLARIFICATION_COUNT_KEY: 1,
                UNANSWERED_CLARIFICATION_COUNT_KEY: 2,
                UNKNOWN_CLARIFICATION_COUNT_KEY: 1,
                INVALID_CLARIFICATION_COUNT_KEY: 1,
            },
        )
        loop = object.__new__(ClarQAgentLoop)
        loop.simulator = FakeSimulator(SATISFIED_DONE_TOKEN)
        loop.reward_config = REWARD_CONFIG

        with (
            patch.object(ToolAgentLoop, "run", new=AsyncMock(return_value=base_output)),
            patch("examples.clarq_grpo.agent.asyncio.to_thread", new=direct_to_thread),
        ):
            output = await ClarQAgentLoop.run(
                loop,
                {},
                extra_info={"profile": PROFILE},
                reward_model={"ground_truth": "case0002"},
            )

        self.assertEqual(output.response_mask, [1, 0, 1])
        self.assertEqual(output.reward_score, 1.26)
        self.assertNotIn(LAST_CASES_KEY, output.extra_fields)
        self.assertEqual(output.extra_fields["reward_extra_info"]["target_rank"], 1.0)
        self.assertEqual(output.extra_fields["reward_extra_info"]["clarification_count"], 3.0)
        self.assertEqual(output.extra_fields["reward_extra_info"]["dialogue_turns"], 4.0)
        self.assertEqual(output.extra_fields["reward_extra_info"]["search_attempt_count"], 3.0)
        self.assertEqual(output.extra_fields["reward_extra_info"]["search_penalty"], -0.1)
        self.assertEqual(output.extra_fields["reward_extra_info"]["useful_clarification_count"], 1.0)
        self.assertEqual(output.extra_fields["reward_extra_info"]["unanswered_clarification_count"], 2.0)
        self.assertEqual(output.extra_fields["reward_extra_info"]["unknown_clarification_count"], 1.0)
        self.assertEqual(output.extra_fields["reward_extra_info"]["invalid_clarification_count"], 1.0)
        self.assertAlmostEqual(output.extra_fields["reward_extra_info"]["clarification_reward"], 0.06)
        self.assertEqual(loop.simulator.calls[0][2], 0)

        output_dict = output.as_dict()
        self.assertEqual(output_dict["response_mask"].tolist(), [1, 0, 1])
        self.assertAlmostEqual(output_dict["rm_scores"].tolist()[-1], 1.26)

    async def test_no_search_writes_zero_search_count(self):
        base_output = AgentLoopOutput(
            prompt_ids=[1, 2],
            response_ids=[3],
            response_mask=[1],
            num_turns=2,
            metrics=AgentLoopMetrics(),
            extra_fields={},
        )
        loop = object.__new__(ClarQAgentLoop)
        loop.simulator = FakeSimulator(SATISFIED_DONE_TOKEN)
        loop.reward_config = REWARD_CONFIG

        with patch.object(ToolAgentLoop, "run", new=AsyncMock(return_value=base_output)):
            output = await ClarQAgentLoop.run(
                loop,
                {},
                extra_info={"profile": PROFILE},
                reward_model={"ground_truth": "case0002"},
            )

        self.assertEqual(output.extra_fields[SEARCH_COUNT_KEY], 0)
        self.assertEqual(output.extra_fields[SEARCH_ATTEMPT_COUNT_KEY], 0)
        self.assertEqual(output.extra_fields[CLARIFICATION_COUNT_KEY], 0)
        self.assertEqual(output.extra_fields[USEFUL_CLARIFICATION_COUNT_KEY], 0)
        self.assertEqual(output.extra_fields[UNANSWERED_CLARIFICATION_COUNT_KEY], 0)
        self.assertEqual(output.extra_fields[UNKNOWN_CLARIFICATION_COUNT_KEY], 0)
        self.assertEqual(output.extra_fields[INVALID_CLARIFICATION_COUNT_KEY], 0)
        self.assertEqual(output.extra_fields["reward_extra_info"]["search_count"], 0.0)
        self.assertEqual(output.extra_fields["reward_extra_info"]["clarification_count"], 0.0)
        self.assertEqual(output.extra_fields["reward_extra_info"]["clarification_reward"], 0.0)
        self.assertEqual(output.extra_fields["reward_extra_info"]["dialogue_turns"], 2.0)
        self.assertEqual(output.reward_score, -1.5)
        self.assertEqual(loop.simulator.calls, [])


if __name__ == "__main__":
    unittest.main()

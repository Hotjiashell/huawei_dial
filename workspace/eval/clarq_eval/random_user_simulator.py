"""Probabilistic user simulator variants for robustness evaluation.

The regular :class:`UserSimulator` is intentionally deterministic in behavior:
it chooses a single fact that directly answers the agent's clarification, or
returns unknown/invalid.  This module keeps that selector as the grounding
step, then samples realistic response styles around it.  Sampling is stable
per sample/question/behavior when a seed is supplied, so concurrent evaluation
workers do not change which style a trajectory receives.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from random import Random
from typing import Any, Literal

from .clients import (
    INVALID_CLARIFICATION_RESPONSE,
    UNKNOWN_CLARIFICATION_RESPONSE,
    OpenAIChatClient,
    UserSimulator,
    response_content,
)
from .models import EvaluationSample
from .parsing import PolicyProtocolError


ClarificationType = Literal["known_info", "unknown", "invalid"]


@dataclass(frozen=True)
class RandomUserSimulatorConfig:
    """Sampling parameters for :class:`RandomUserSimulator`."""

    seed: int | None = 42
    rephrase_question_probability: float = 0.16
    compress_known_info_probability: float = 0.79
    enable_proactive_known_info_on_unknown: bool = True
    proactive_known_info_probability: float = 0.47

    def __post_init__(self) -> None:
        for name, probability in (
            ("rephrase_question_probability", self.rephrase_question_probability),
            ("compress_known_info_probability", self.compress_known_info_probability),
            ("proactive_known_info_probability", self.proactive_known_info_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class RandomUserSimulatorReply:
    """A simulator reply plus the semantic class used by evaluation metrics."""

    text: str
    clarification_type: ClarificationType
    behavior: str
    source_known_info: str | None = None


class RandomUserSimulator:
    """Sample varied yet grounded user behavior for clarification evaluation.

    Sampling order for each clarification is:

    1. With ``rephrase_question_probability`` (default 16%), ask the LLM to
       rephrase the original user question and deliberately not answer the
       Agent's clarification.
    2. Otherwise use the regular constrained selector to determine whether a
       known fact answers the clarification.
    3. If a fact is known, with ``compress_known_info_probability`` (79%) make
       one additional LLM request for a concise rendering of that fact.
    4. If the answer is unknown, with
       ``proactive_known_info_probability`` (47%, when enabled), make one
       additional LLM request to select a random known fact to volunteer.

    The standard selector, compression prompt, and proactive-selection prompt
    all operate only on the supplied sample data; no facts are invented.
    """

    REPHRASE_QUESTION_PROMPT = """You are simulating a user in a support conversation.

Initial user question:
{initial_question}

Agent clarification question:
{question}

The user does NOT answer the agent's clarification question. Instead, write a
single short natural-language rephrasing of the initial user question only.
Do not answer the clarification, do not add any facts, do not ask a new
question, and do not mention these instructions. Output only the user reply.
"""

    COMPRESS_KNOWN_INFO_PROMPT = """You are simulating a user in a support conversation.

Initial user question:
{initial_question}

Agent clarification question:
{question}

The user knows this fact, which directly answers the clarification:
{known_fact}

Reply to the agent with one short, natural-language sentence. Preserve the
material information in the known fact, but compress its wording. Do not add
facts, explanations, or a second sentence. Output only the user reply.
"""

    PROACTIVE_KNOWN_INFO_PROMPT = """You are simulating a user in a support conversation.

Initial user question:
{initial_question}

Agent clarification question:
{question}

The user does not know the answer to the agent's clarification. To simulate a
user who volunteers a possibly useful detail, randomly select exactly one fact
from the numbered known facts below. Do not answer the clarification question.
Output only one allowed selection token, with no other text.

Known facts:
{known_info}
"""

    def __init__(
        self,
        client: OpenAIChatClient,
        config: RandomUserSimulatorConfig | None = None,
        *,
        model_mode: str = "qwen3_5",
        tokenizer_path: str | None = None,
    ):
        self.config = config or RandomUserSimulatorConfig()
        self._grounded_selector = UserSimulator(
            client,
            model_mode=model_mode,
            tokenizer_path=tokenizer_path,
        )

    def answer(self, sample: EvaluationSample, question: str) -> RandomUserSimulatorReply:
        """Return one sampled user reply for an Agent clarification question."""
        if self._sample(sample, question, "rephrase_question") < self.config.rephrase_question_probability:
            return RandomUserSimulatorReply(
                text=self._rephrase_question(sample, question),
                clarification_type="unknown",
                behavior="rephrase_initial_question",
            )

        grounded_reply = self._grounded_selector.answer(sample, question)
        if grounded_reply == INVALID_CLARIFICATION_RESPONSE:
            return RandomUserSimulatorReply(
                text=grounded_reply,
                clarification_type="invalid",
                behavior="invalid_question",
            )

        if grounded_reply in sample.known_info:
            if (
                self._sample(sample, question, "compress_known_info")
                < self.config.compress_known_info_probability
            ):
                return RandomUserSimulatorReply(
                    text=self._compress_known_info(sample, question, grounded_reply),
                    clarification_type="known_info",
                    behavior="compressed_known_info",
                    source_known_info=grounded_reply,
                )
            return RandomUserSimulatorReply(
                text=grounded_reply,
                clarification_type="known_info",
                behavior="verbatim_known_info",
                source_known_info=grounded_reply,
            )

        if (
            grounded_reply == UNKNOWN_CLARIFICATION_RESPONSE
            and sample.known_info
            and self.config.enable_proactive_known_info_on_unknown
            and self._sample(sample, question, "proactive_known_info")
            < self.config.proactive_known_info_probability
        ):
            volunteered_fact = self._select_proactive_known_info(sample, question)
            return RandomUserSimulatorReply(
                text=volunteered_fact,
                clarification_type="known_info",
                behavior="proactive_known_info",
                source_known_info=volunteered_fact,
            )

        return RandomUserSimulatorReply(
            text=UNKNOWN_CLARIFICATION_RESPONSE,
            clarification_type="unknown",
            behavior="unknown",
        )

    def _rephrase_question(self, sample: EvaluationSample, question: str) -> str:
        prompt = self.REPHRASE_QUESTION_PROMPT.format(
            initial_question=sample.initial_question,
            question=question,
        )
        return self._freeform_reply(
            prompt,
            sample,
            question,
            "rephrase_question",
            temperature=0.7,
        )

    def _compress_known_info(self, sample: EvaluationSample, question: str, known_fact: str) -> str:
        prompt = self.COMPRESS_KNOWN_INFO_PROMPT.format(
            initial_question=sample.initial_question,
            question=question,
            known_fact=known_fact,
        )
        return self._freeform_reply(
            prompt,
            sample,
            question,
            "compress_known_info",
            temperature=0.3,
        )

    def _select_proactive_known_info(self, sample: EvaluationSample, question: str) -> str:
        choice_to_fact = {
            f"KNOWN_INFO_{index}": fact for index, fact in enumerate(sample.known_info, start=1)
        }
        known_info = "\n".join(
            f"[{index}] {fact}" for index, fact in enumerate(sample.known_info, start=1)
        )
        prompt = self.PROACTIVE_KNOWN_INFO_PROMPT.format(
            initial_question=sample.initial_question,
            question=question,
            known_info=known_info,
        )
        response = self._grounded_selector.chat_without_thinking(
            [{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=16,
            seed=self._model_seed(sample, question, "proactive_known_info"),
            extra_payload={
                "chat_template_kwargs": {"enable_thinking": False},
                "structured_outputs": {"choice": list(choice_to_fact)},
            },
        )
        selection = response_content(response)
        if selection not in choice_to_fact:
            raise PolicyProtocolError(
                f"Random user simulator returned unsupported proactive selection: {selection!r}"
            )
        return choice_to_fact[selection]

    def _freeform_reply(
        self,
        prompt: str,
        sample: EvaluationSample,
        question: str,
        behavior: str,
        *,
        temperature: float,
    ) -> str:
        response = self._grounded_selector.chat_without_thinking(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=64,
            seed=self._model_seed(sample, question, behavior),
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
        )
        reply = response_content(response)
        if not reply:
            raise PolicyProtocolError(f"Random user simulator returned an empty {behavior} reply")
        return reply

    def _sample(self, sample: EvaluationSample, question: str, behavior: str) -> float:
        return self._random(sample, question, behavior).random()

    def _model_seed(self, sample: EvaluationSample, question: str, behavior: str) -> int | None:
        if self.config.seed is None:
            return None
        return self._random(sample, question, f"model:{behavior}").randrange(0, 2**31)

    def _random(self, sample: EvaluationSample, question: str, behavior: str) -> Random:
        seed_material = "\x1f".join(
            (
                str(self.config.seed),
                sample.sample_id,
                sample.initial_question,
                question,
                behavior,
            )
        ).encode("utf-8")
        return Random(int.from_bytes(sha256(seed_material).digest()[:16], "big"))

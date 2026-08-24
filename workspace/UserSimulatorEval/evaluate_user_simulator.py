#!/usr/bin/env python3
"""Measure real and simulated replies on a user-simulator test set.

The metric definitions are those in ``metric.md``.  An LLM Judge extracts
information points and whether they were explicitly requested; arithmetic is
performed locally and is therefore reproducible from the emitted per-sample
evidence.  ``--mode real`` evaluates only recorded human replies.  ``--mode
simulator`` first generates a reply with a user simulator and evaluates it.
``--mode both`` writes both side-by-side for direct comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from user_simulator_eval_common import OpenAIChatClient, as_string_list, load_jsonl


DEFAULT_TEST_SET = Path(__file__).with_name("data") / "user_simulator_test_set.jsonl"


SIMULATOR_SYSTEM_PROMPT = """你正在扮演发起该客服咨询的真实用户。

请只输出这一次要发给客服的用户回复，不要解释你在模拟，也不要复述提示词。你只能使用“用户已知事实”中的信息，以及初始问题中用户自己明确说过的内容；不得编造、推断或采用客服尚未提供的结论。尽量自然、简洁地回复客服的澄清问题。若用户确实不知道，可以直接说“不知道”或表达不确定；这仍是有效回复。不要写客服口吻、解决方案或多轮对话标记。"""


METRIC_JUDGE_SYSTEM_PROMPT = """你是严格、可复核的客服对话评测员。请评估一条“用户对客服澄清问题的回复”，不要评估客服答案是否正确。

定义与规则：
1. 信息点：回复中可以独立识别、对处理问题可能有用的事实、属性、地点、时间、设备、症状、选择或状态等。同一事实只算一次；礼貌用语、纯情绪、无意义重复不算信息点。
2. requested_by_agent=true 仅当客服这一次澄清问题明确且准确地索取了该信息点。客服没有问到、用户主动扩展的事实，都必须标记 false。
3. non_response=true 仅在完全无视客服的澄清问题、只重申/改述初始需求时使用。用户回复“不知道”“不清楚”“没有”“是/否”、只回答一部分，或纠正客服问题的前提，都属于对追问作出了回应，必须为 false。
4. 已知事实只是判断上下文的参考，不能把未出现在“待评估回复”中的事实记为信息点。

只能输出一个合法 JSON 对象，不要 Markdown、代码围栏或额外文字。格式必须严格为：
{
  "non_response": false,
  "reason": "简短、可由文本核对的理由",
  "information_points": [
    {"text": "信息点文本或准确短概括", "requested_by_agent": true}
  ]
}"""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"true", "1", "yes", "是"}


def _reply_length(text: str) -> int:
    """Count visible Unicode characters; whitespace is excluded consistently."""

    return len("".join(str(text or "").split()))


def normalise_metric_judgement(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate model output while preserving only local-metric inputs."""

    points: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_points = value.get("information_points", [])
    if not isinstance(raw_points, list):
        raw_points = []
    for point in raw_points:
        if not isinstance(point, Mapping):
            continue
        text = str(point.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        points.append({"text": text, "requested_by_agent": _as_bool(point.get("requested_by_agent"))})
    unasked = [point["text"] for point in points if not point["requested_by_agent"]]
    return {
        "non_response": _as_bool(value.get("non_response")),
        "reason": str(value.get("reason") or "").strip(),
        "information_points": points,
        "information_point_count": len(points),
        "unasked_information_points": unasked,
        "has_unasked_information": bool(unasked),
    }


def build_metric_input(sample: Mapping[str, Any], reply: str) -> dict[str, Any]:
    return {
        "initial_question": str(sample.get("initial_question") or ""),
        "known_info": as_string_list(sample.get("known_info")),
        "clarification_question": str(sample.get("clarification_question") or ""),
        "reply_to_evaluate": str(reply or ""),
    }


def judge_reply(sample: Mapping[str, Any], reply: str, judge: OpenAIChatClient) -> dict[str, Any]:
    """Use the no-thinking Judge to annotate one concrete reply."""

    response = judge.chat(
        [
            {"role": "system", "content": METRIC_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "请评估以下单条用户回复。\n\n" + json.dumps(build_metric_input(sample, reply), ensure_ascii=False, indent=2),
            },
        ],
        temperature=0.0,
        max_tokens=1024,
        json_object=True,
    )
    result = normalise_metric_judgement(response["json"])
    result.update(
        {
            "reply": str(reply or ""),
            "reply_length_characters": _reply_length(reply),
            "judge_raw_response": response["content"],
            "judge_request_mode": response["request_mode"],
        }
    )
    return result


def simulate_reply(
    sample: Mapping[str, Any],
    simulator: OpenAIChatClient,
    *,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    system_prompt: str,
) -> dict[str, Any]:
    """Ask a simulator for one natural-language reply with thinking disabled."""

    simulator_input = {
        "initial_question": str(sample.get("initial_question") or ""),
        "user_known_info": as_string_list(sample.get("known_info")),
        "agent_clarification_question": str(sample.get("clarification_question") or ""),
    }
    response = simulator.chat(
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "请基于以下数据生成这一次用户回复：\n\n" + json.dumps(simulator_input, ensure_ascii=False, indent=2),
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        json_object=False,
    )
    return {
        "reply": response["content"],
        "adapter": "natural_prompt",
        "request_mode": response["request_mode"],
        "request_body": response["request_body"],
    }


class ClarqSimulatorAdapter:
    """Run the actual grounded/random ClarQ simulator implementation.

    Keeping this adapter in the evaluator avoids silently measuring a generic
    prompt instead of the project user's simulator.  Imports are deliberately
    lazy so ``--mode real`` remains independent of the ClarQ eval package.
    """

    def __init__(
        self,
        *,
        kind: str,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float,
        max_retries: int,
        model_mode: str,
        tokenizer_path: str | None,
        random_seed: int | None,
        rephrase_probability: float,
        compress_known_probability: float,
        proactive_known_on_unknown: bool,
        proactive_known_probability: float,
    ):
        eval_directory = Path(__file__).resolve().parents[1] / "eval"
        if not eval_directory.is_dir():
            raise RuntimeError(f"找不到 ClarQ 评测实现目录：{eval_directory}")
        if str(eval_directory) not in sys.path:
            sys.path.insert(0, str(eval_directory))
        try:
            from clarq_eval.clients import INVALID_CLARIFICATION_RESPONSE
            from clarq_eval.clients import UNKNOWN_CLARIFICATION_RESPONSE
            from clarq_eval.clients import OpenAIChatClient as ClarqOpenAIChatClient
            from clarq_eval.clients import UserSimulator
            from clarq_eval.models import EvaluationSample
            from clarq_eval.random_user_simulator import RandomUserSimulator, RandomUserSimulatorConfig
        except ModuleNotFoundError as error:
            raise RuntimeError("无法导入 workspace/eval 的 ClarQ 用户模拟器实现") from error

        self._sample_type = EvaluationSample
        self._programmatic_invalid_reply = INVALID_CLARIFICATION_RESPONSE
        self._programmatic_unknown_reply = UNKNOWN_CLARIFICATION_RESPONSE
        client = ClarqOpenAIChatClient(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.kind = kind
        if kind == "clarq_grounded":
            self.simulator = UserSimulator(client, model_mode=model_mode, tokenizer_path=tokenizer_path)
        elif kind == "clarq_random":
            config = RandomUserSimulatorConfig(
                seed=random_seed,
                rephrase_question_probability=rephrase_probability,
                compress_known_info_probability=compress_known_probability,
                enable_proactive_known_info_on_unknown=proactive_known_on_unknown,
                proactive_known_info_probability=proactive_known_probability,
            )
            self.simulator = RandomUserSimulator(
                client,
                config,
                model_mode=model_mode,
                tokenizer_path=tokenizer_path,
            )
        else:
            raise ValueError(f"未知 ClarQ 用户模拟器类型：{kind}")

    def generate(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        source = sample.get("source")
        source_core_intent = source.get("source_core_intent") if isinstance(source, Mapping) else ""
        profile = self._sample_type(
            sample_id=str(sample.get("sample_id") or "user-simulator-eval"),
            domain="user_simulator_eval",
            target_case_id="",
            initial_question=str(sample["initial_question"]),
            core_intent=str(source_core_intent or sample["initial_question"]),
            known_info=tuple(as_string_list(sample.get("known_info"))),
        )
        raw_reply = self.simulator.answer(profile, str(sample["clarification_question"]))
        reply = str(getattr(raw_reply, "text", raw_reply)).strip()
        if not reply:
            raise RuntimeError("ClarQ 用户模拟器返回空回复")
        programmatic_response_kind: str | None = None
        if reply == self._programmatic_invalid_reply:
            programmatic_response_kind = "invalid_clarification"
        elif reply == self._programmatic_unknown_reply:
            programmatic_response_kind = "unknown"
        result: dict[str, Any] = {
            "reply": reply,
            "adapter": self.kind,
            # Both values are emitted by UserSimulator.answer() after a
            # constrained selector result. They are not LLM renderings and
            # must not skew natural-language reply-length comparisons.
            "programmatic_response_kind": programmatic_response_kind,
            "programmatic_invalid_clarification_response": programmatic_response_kind == "invalid_clarification",
            "programmatic_unknown_response": programmatic_response_kind == "unknown",
        }
        behavior = getattr(raw_reply, "behavior", None)
        source_fact = getattr(raw_reply, "source_known_info", None)
        clarification_type = getattr(raw_reply, "clarification_type", None)
        if isinstance(behavior, str):
            result["user_simulator_behavior"] = behavior
        if isinstance(source_fact, str):
            result["user_simulator_source_known_info"] = source_fact
        if isinstance(clarification_type, str):
            result["clarification_type"] = clarification_type
        return result


class NaturalPromptSimulatorAdapter:
    """Generate a reply from a standalone user-simulator service prompt."""

    def __init__(
        self,
        client: OpenAIChatClient,
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        system_prompt: str,
    ):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.system_prompt = system_prompt

    def generate(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        return simulate_reply(
            sample,
            self.client,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            system_prompt=self.system_prompt,
        )


def aggregate(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate every metric from normalised per-reply annotations locally."""

    total = len(scores)
    information_points = sum(int(score.get("information_point_count", 0)) for score in scores)
    leakage_count = sum(1 for score in scores if score.get("has_unasked_information"))
    non_response_count = sum(1 for score in scores if score.get("non_response"))
    length_scores = [score for score in scores if not score.get("exclude_from_average_reply_length")]
    total_length = sum(int(score.get("reply_length_characters", 0)) for score in length_scores)
    excluded_programmatic_invalid_count = sum(
        1
        for score in scores
        if score.get("reply_length_exclusion_reason") == "programmatic_invalid_clarification_response"
    )
    excluded_programmatic_unknown_count = sum(
        1
        for score in scores
        if score.get("reply_length_exclusion_reason") == "programmatic_unknown_response"
    )
    return {
        "evaluated_reply_count": total,
        "total_information_point_count": information_points,
        "information_efficiency_average_points_per_reply": information_points / total if total else 0.0,
        "answers_with_unasked_information_count": leakage_count,
        "information_leakage_rate": leakage_count / total if total else 0.0,
        "reply_length_included_reply_count": len(length_scores),
        "programmatic_invalid_clarification_reply_excluded_from_length_count": (
            excluded_programmatic_invalid_count
        ),
        "programmatic_unknown_reply_excluded_from_length_count": excluded_programmatic_unknown_count,
        "programmatic_reply_excluded_from_length_count": (
            excluded_programmatic_invalid_count + excluded_programmatic_unknown_count
        ),
        "total_reply_length_characters": total_length,
        "average_reply_length_characters": total_length / len(length_scores) if length_scores else 0.0,
        "non_response_count": non_response_count,
        "non_response_rate": non_response_count / total if total else 0.0,
    }


def _sample_identity(sample: Mapping[str, Any], position: int) -> str:
    return str(sample.get("sample_id") or f"row-{position + 1}").strip() or f"row-{position + 1}"


def validate_test_sample(sample: Mapping[str, Any], position: int) -> None:
    required = ("initial_question", "known_info", "clarification_question", "human_response")
    missing = [field for field in required if field not in sample]
    if missing:
        raise ValueError(f"样本 {_sample_identity(sample, position)} 缺少字段：{missing}")
    if not str(sample.get("initial_question") or "").strip():
        raise ValueError(f"样本 {_sample_identity(sample, position)} 的 initial_question 为空")
    if not str(sample.get("clarification_question") or "").strip():
        raise ValueError(f"样本 {_sample_identity(sample, position)} 的 clarification_question 为空")
    if not str(sample.get("human_response") or "").strip():
        raise ValueError(f"样本 {_sample_identity(sample, position)} 的 human_response 为空")
    if not isinstance(sample.get("known_info"), list):
        raise ValueError(f"样本 {_sample_identity(sample, position)} 的 known_info 必须是数组")


def load_system_prompt(path: str | None) -> str:
    if not path:
        return SIMULATOR_SYSTEM_PROMPT
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"模拟器 system prompt 文件为空：{path}")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 metric.md 评估真实用户回复或用户模拟器回复。")
    parser.add_argument("input", nargs="?", default=str(DEFAULT_TEST_SET), help="build_test_set.py 生成的 JSONL 测试集")
    parser.add_argument("--output", default=str(Path(__file__).with_name("outputs") / "user_simulator_evaluation.json"), help="评测输出 JSON")
    parser.add_argument("--mode", choices=("real", "simulator", "both"), default="both", help="real=真实回复；simulator=生成后评估；both=两者都做")
    parser.add_argument("--judge-url", default=os.getenv("USER_SIMULATOR_EVAL_JUDGE_URL", os.getenv("LLM_JUDGE_URL")), help="LLM Judge OpenAI-compatible 地址")
    parser.add_argument("--judge-model", default=os.getenv("USER_SIMULATOR_EVAL_JUDGE_MODEL", os.getenv("LLM_JUDGE_MODEL_NAME")), help="LLM Judge 模型")
    parser.add_argument("--judge-api-key", default=os.getenv("USER_SIMULATOR_EVAL_JUDGE_API_KEY", os.getenv("LLM_JUDGE_API_KEY", "")), help="LLM Judge API key")
    parser.add_argument("--simulator-url", default=os.getenv("USER_SIMULATOR_BASE_URL"), help="用户模拟器 OpenAI-compatible 地址；simulator/both 模式必填")
    parser.add_argument("--simulator-model", default=os.getenv("USER_SIMULATOR_MODEL"), help="用户模拟器模型名；simulator/both 模式必填")
    parser.add_argument("--simulator-api-key", default=os.getenv("USER_SIMULATOR_API_KEY", ""), help="用户模拟器 API key")
    parser.add_argument(
        "--simulator-adapter",
        choices=("clarq_random", "clarq_grounded", "natural_prompt"),
        default="clarq_random",
        help=(
            "被测模拟器实现：clarq_random/clarq_grounded 直接调用 workspace/eval 中的真实实现；"
            "natural_prompt 使用下方中文 prompt 直接调用服务"
        ),
    )
    parser.add_argument("--simulator-system-prompt-file", help="仅 natural_prompt：替换默认中文模拟器 system prompt 的文本文件")
    parser.add_argument("--simulator-temperature", type=float, default=0.7, help="用户模拟器生成温度")
    parser.add_argument("--simulator-max-tokens", type=int, default=128, help="用户模拟器最大生成 token")
    parser.add_argument("--seed", type=int, default=None, help="仅 natural_prompt：可选，传给支持 seed 的服务以便可复现")
    parser.add_argument(
        "--model-mode",
        choices=("qwen3", "qwen3_5"),
        default=os.getenv("USER_SIMULATOR_MODEL_MODE", "qwen3_5"),
        help="仅 ClarQ adapter：用户模拟器关闭思考的模型协议",
    )
    parser.add_argument(
        "--simulator-tokenizer-path",
        default=os.getenv("USER_SIMULATOR_TOKENIZER_PATH"),
        help="仅 ClarQ adapter 且 model-mode=qwen3：本地/HF tokenizer 路径",
    )
    parser.add_argument("--random-user-simulator-seed", type=int, default=42, help="仅 clarq_random：稳定随机采样种子")
    parser.add_argument("--random-user-rephrase-probability", type=float, default=0.16, help="仅 clarq_random：重述初始问题概率")
    parser.add_argument("--random-user-compress-known-probability", type=float, default=0.79, help="仅 clarq_random：压缩已知事实概率")
    parser.add_argument(
        "--random-user-proactive-known-on-unknown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅 clarq_random：用户不知道时能否主动透露一条已知事实",
    )
    parser.add_argument("--random-user-proactive-known-probability", type=float, default=0.47, help="仅 clarq_random：主动透露已知事实概率")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次 HTTP 请求超时秒数")
    parser.add_argument("--max-retries", type=int, default=2, help="每种请求格式的重试次数")
    parser.add_argument("--limit", type=int, help="最多评估多少条样本")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 --output")
    parser.add_argument("--fail-fast", action="store_true", help="单条失败即退出；默认记录错误并继续")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(f"error: 输出已存在：{output}；如需覆盖请加 --overwrite", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit <= 0:
        print("error: --limit 必须大于 0", file=sys.stderr)
        return 2
    probabilities = (
        args.random_user_rephrase_probability,
        args.random_user_compress_known_probability,
        args.random_user_proactive_known_probability,
    )
    if args.timeout <= 0 or args.simulator_max_tokens <= 0:
        print("error: timeout 和 simulator-max-tokens 必须大于 0", file=sys.stderr)
        return 2
    if not all(0.0 <= probability <= 1.0 for probability in probabilities):
        print("error: random 用户模拟器概率必须在 [0, 1] 内", file=sys.stderr)
        return 2
    if not args.judge_url or not args.judge_model:
        print("error: --judge-url 和 --judge-model 必填", file=sys.stderr)
        return 2
    if args.mode in {"simulator", "both"} and (not args.simulator_url or not args.simulator_model):
        print("error: simulator/both 模式需要 --simulator-url 和 --simulator-model", file=sys.stderr)
        return 2

    try:
        samples = load_jsonl(args.input)
        if args.limit is not None:
            samples = samples[: args.limit]
        for position, sample in enumerate(samples):
            validate_test_sample(sample, position)
        judge = OpenAIChatClient(args.judge_url, args.judge_model, args.judge_api_key, args.timeout, args.max_retries)
        simulator: ClarqSimulatorAdapter | NaturalPromptSimulatorAdapter | None = None
        if args.mode in {"simulator", "both"}:
            if args.simulator_adapter == "natural_prompt":
                simulator = NaturalPromptSimulatorAdapter(
                    OpenAIChatClient(
                        args.simulator_url,
                        args.simulator_model,
                        args.simulator_api_key,
                        args.timeout,
                        args.max_retries,
                    ),
                    temperature=args.simulator_temperature,
                    max_tokens=args.simulator_max_tokens,
                    seed=args.seed,
                    system_prompt=load_system_prompt(args.simulator_system_prompt_file),
                )
            else:
                simulator = ClarqSimulatorAdapter(
                    kind=args.simulator_adapter,
                    base_url=args.simulator_url,
                    model=args.simulator_model,
                    api_key=args.simulator_api_key,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    model_mode=args.model_mode,
                    tokenizer_path=args.simulator_tokenizer_path,
                    random_seed=args.random_user_simulator_seed,
                    rephrase_probability=args.random_user_rephrase_probability,
                    compress_known_probability=args.random_user_compress_known_probability,
                    proactive_known_on_unknown=args.random_user_proactive_known_on_unknown,
                    proactive_known_probability=args.random_user_proactive_known_probability,
                )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    real_scores: list[dict[str, Any]] = []
    simulator_scores: list[dict[str, Any]] = []
    for position, sample in enumerate(samples, 1):
        sample_id = _sample_identity(sample, position - 1)
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "initial_question": sample["initial_question"],
            "known_info": as_string_list(sample["known_info"]),
            "clarification_question": sample["clarification_question"],
            "human_response": sample["human_response"],
        }
        try:
            if args.mode in {"real", "both"}:
                record["real_user"] = judge_reply(sample, str(sample["human_response"]), judge)
                real_scores.append(record["real_user"])
            if simulator is not None:
                generated = simulator.generate(sample)
                simulator_evaluation = judge_reply(sample, generated["reply"], judge)
                programmatic_response_kind = generated.get("programmatic_response_kind")
                if programmatic_response_kind in {"invalid_clarification", "unknown"}:
                    simulator_evaluation["exclude_from_average_reply_length"] = True
                    simulator_evaluation["reply_length_exclusion_reason"] = f"programmatic_{programmatic_response_kind}_response"
                record["simulator"] = {
                    "generation": generated,
                    "evaluation": simulator_evaluation,
                }
                simulator_scores.append(record["simulator"]["evaluation"])
            records.append(record)
            print(f"[evaluate-user-simulator] {position}/{len(samples)} 完成 {sample_id}", file=sys.stderr)
        except Exception as error:  # noqa: BLE001 - preserve successful rows in long online runs
            errors.append({"sample_id": sample_id, "error_type": type(error).__name__, "error": str(error)})
            print(f"[evaluate-user-simulator] {position}/{len(samples)} 失败 {sample_id}：{type(error).__name__}: {error}", file=sys.stderr)
            if args.fail_fast:
                return 1

    metrics: dict[str, Any] = {}
    if args.mode in {"real", "both"}:
        metrics["real_user"] = aggregate(real_scores)
    if args.mode in {"simulator", "both"}:
        metrics["simulator"] = aggregate(simulator_scores)
    if args.mode == "both":
        # These deltas are descriptive only; individual metrics remain the
        # source of truth and no unsupported composite "human-likeness" score
        # is fabricated.
        metrics["simulator_minus_real"] = {
            key: metrics["simulator"][key] - metrics["real_user"][key]
            for key in (
                "information_efficiency_average_points_per_reply",
                "information_leakage_rate",
                "average_reply_length_characters",
                "non_response_rate",
            )
        }
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "mode": args.mode,
        "metric_definition": "workspace/UserSimulatorEval/metric.md",
        "thinking_disabled": True,
        "simulator_adapter": args.simulator_adapter if args.mode in {"simulator", "both"} else None,
        "simulator_model_mode": args.model_mode if args.mode in {"simulator", "both"} else None,
        "sample_count": len(samples),
        "completed_sample_count": len(records),
        "error_count": len(errors),
        "metrics": metrics,
        "records": records,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed_sample_count": len(records), "error_count": len(errors), "metrics": metrics}, ensure_ascii=False, indent=2))
    return 1 if errors and args.fail_fast else 0


if __name__ == "__main__":
    raise SystemExit(main())

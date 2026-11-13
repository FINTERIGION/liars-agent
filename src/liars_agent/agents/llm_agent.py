"""Qwen decision agent backed exclusively by real Tinker sampling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from liars_agent.agents.base import BaseAgent, public_opponent_profile, describe_action
from liars_agent.agents.prompt_builder import (SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT, build_user_prompt,
    build_bluff_user_prompt, validate_action_decision)
from liars_agent.contracts import require_contract
from liars_agent.agents.rule_bots import CardCounterBot
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.types import AgentDecisionRecord, CandidateAction

ModelMode = Literal["zero_shot", "lora_sft", "lora_rs_improved", "remote_api"]


@dataclass
class CostAndFallbackTracker:
    model_version: str
    total_calls: int = 0
    api_calls: int = 0
    fallback_count: int = 0
    belief_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    belief_generated_tokens_upper_bound: int = 0
    total_latency_ms: float = 0.0
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    price_per_1m_prompt_tokens: float = 0.30
    price_per_1m_completion_tokens: float = 0.807

    @property
    def illegal_output_rate(self) -> float:
        return self.fallback_count / max(1, self.total_calls)

    @property
    def estimated_cost_usd(self) -> float:
        return (self.total_prompt_tokens * self.price_per_1m_prompt_tokens +
                (self.total_completion_tokens + self.belief_generated_tokens_upper_bound) *
                self.price_per_1m_completion_tokens) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return {"model_version": self.model_version, "total_calls": self.total_calls,
                "api_calls": self.api_calls, "fallback_count": self.fallback_count, "belief_calls": self.belief_calls,
                "illegal_output_rate": self.illegal_output_rate,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "belief_generated_tokens_upper_bound": self.belief_generated_tokens_upper_bound,
                "avg_latency_ms": self.total_latency_ms / max(1, self.api_calls),
                "estimated_cost_usd": self.estimated_cost_usd}


class LLMAgent(BaseAgent):
    """An adapter path selects model weights; invalid model text uses CardCounter.

    Authentication, unavailable models, missing checkpoints and transport errors
    propagate to the caller. They never become a successful simulated game.
    """

    def __init__(self, name: str = "qwen35_9b_zero_shot", mode: ModelMode = "zero_shot",
                 model_version: str = "Qwen/Qwen3.5-9B", adapter_path: str | Path | None = None,
                 seed: int = 42, api_base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.2, *, backend: Any = None,
                 max_new_tokens: int = 256, top_p: float = 0.9) -> None:
        if mode not in ("zero_shot", "lora_sft", "lora_rs_improved", "remote_api"):
            raise ValueError(f"Unknown model mode: {mode}")
        if api_base_url is not None:
            raise ValueError("This agent uses the Tinker SDK; api_base_url is not a chat-completions endpoint.")
        super().__init__(name=name, style_tag=mode)
        self.mode = mode
        self.model_version = model_version
        self.adapter_path = adapter_path
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.adapter_metadata: dict[str, Any] = {}
        self.checkpoint: str | None = None
        if adapter_path is not None:
            path_text = str(adapter_path)
            if path_text.startswith("tinker://"):
                raise ValueError("A trained adapter requires contract-bearing metadata, not a bare checkpoint URI.")
            else:
                path = Path(adapter_path)
                if path.is_dir():
                    path = path / "adapter_metadata.json"
                self.adapter_metadata = json.loads(path.read_text(encoding="utf-8"))
                require_contract(self.adapter_metadata, "adapter metadata")
                if self.adapter_metadata.get("base_model") != model_version:
                    raise ValueError("Adapter metadata base_model does not match the configured model.")
                self.checkpoint = self.adapter_metadata.get("sampler_path")
            if not isinstance(self.checkpoint, str) or not self.checkpoint.startswith("tinker://") or "/sampler_weights/" not in self.checkpoint:
                raise ValueError("Adapter must contain a real Tinker sampler_path.")
        if mode in ("lora_sft", "lora_rs_improved") and not self.checkpoint:
            raise ValueError(f"{mode} requires a trained adapter checkpoint.")
        if mode == "zero_shot" and self.checkpoint:
            raise ValueError("zero_shot cannot load a trained checkpoint.")
        if backend is None:
            from liars_agent.training.tinker_backend import TinkerBackend
            backend = TinkerBackend(base_model=model_version, api_key=api_key)
        if backend.base_model != model_version:
            raise ValueError("Agent and backend models differ.")
        self.backend = backend
        self.tracker = CostAndFallbackTracker(model_version=f"{model_version}[{mode}]",
            price_per_1m_prompt_tokens=backend.rates["prefill"],
            price_per_1m_completion_tokens=backend.rates["sample"])
        self._fallback = CardCounterBot(name="card_counter_fallback", seed=seed)

    def _sampling_seed(self, obs: PlayerObservation) -> int:
        # Mode is intentionally absent: paired ablations retain the same RNG seed.
        value = f"{self.seed}:{obs.round_id}:{obs.step_id}:{obs.observer_seat}"
        return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") & 0x7FFFFFFF

    def _score_belief(self, obs: PlayerObservation):
        if obs.last_claim is None:
            return None
        score = self.backend.score_bluff(
            [{"role": "system", "content": BELIEF_SYSTEM_PROMPT},
             {"role": "user", "content": build_bluff_user_prompt(obs)}], checkpoint=self.checkpoint)
        self.tracker.belief_calls += 1
        self.tracker.api_calls += 2
        self.tracker.total_prompt_tokens += score.prompt_tokens
        self.tracker.belief_generated_tokens_upper_bound += score.belief_completion_tokens_upper_bound
        self.tracker.total_latency_ms += score.latency_ms
        return score

    def predict_bluff_prob(self, obs: PlayerObservation) -> float | None:
        """Score binary label tokens only, without generating or executing an action."""
        score = self._score_belief(obs)
        return None if score is None else score.probability

    def _decode(self, obs: PlayerObservation, result: Any, belief: Any = None,
                include_belief_usage: bool = True) -> tuple[CandidateAction, AgentDecisionRecord]:
        reason = ""
        parsed = None
        try:
            if result.stop_reason != "stop":
                raise ValueError("Model output reached its token limit.")
            parsed = validate_action_decision(obs, result.text)
        except ValueError as exc:
            reason = str(exc)
        source = "model" if belief is not None else ("not_applicable" if obs.last_claim is None else "not_requested")
        common = dict(belief_bluff_prob=None if belief is None else belief.probability,
            belief_source=source, opponent_profile=public_opponent_profile(obs),
            rationale_source="deterministic_action_summary", raw_output=result.text,
            latency_ms=result.latency_ms + (belief.latency_ms if belief is not None and include_belief_usage else 0),
            prompt_tokens=result.prompt_tokens + (belief.prompt_tokens if belief is not None and include_belief_usage else 0),
            completion_tokens=result.completion_tokens,
            belief_completion_tokens_upper_bound=(belief.belief_completion_tokens_upper_bound
                if belief is not None and include_belief_usage else 0))
        if not reason:
            action = next(a for a in obs.legal_actions if a.action_id == parsed["chosen_action_id"])
            return action, AgentDecisionRecord(rationale=describe_action(obs, action), **common)
        self.tracker.fallback_count += 1
        self.tracker.fallback_events.append({"round_id": obs.round_id, "step_id": obs.step_id,
            "seat": obs.observer_seat, "raw_output": result.text, "reason": reason})
        action, fallback = self._fallback.select_action(obs)
        return action, AgentDecisionRecord(was_fallback=True, fallback_reason=reason,
            fallback_record=fallback.to_dict(), rationale=describe_action(obs, action), **common)

    def sample_candidates(self, obs: PlayerObservation, num_samples: int = 8, *,
                          include_belief: bool = True) -> list[tuple[CandidateAction, AgentDecisionRecord]]:
        if not obs.legal_actions:
            raise ValueError("Cannot select an action without legal candidates.")
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        # One independent belief score is shared by this observation's candidates.
        belief = self._score_belief(obs) if include_belief else None
        results = self.backend.sample(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": build_user_prompt(obs)}],
            checkpoint=self.checkpoint, num_samples=num_samples, temperature=self.temperature,
            top_p=self.top_p, max_tokens=self.max_new_tokens, seed=self._sampling_seed(obs))
        self.tracker.api_calls += 1
        self.tracker.total_calls += len(results)
        self.tracker.total_prompt_tokens += sum(result.prompt_tokens for result in results)
        self.tracker.total_completion_tokens += sum(result.completion_tokens for result in results)
        self.tracker.total_latency_ms += sum(result.latency_ms for result in results)
        return [self._decode(obs, result, belief, index == 0)
                for index, result in enumerate(results)]

    def select_action(self, obs: PlayerObservation) -> tuple[CandidateAction, AgentDecisionRecord]:
        return self.sample_candidates(obs, num_samples=1)[0]

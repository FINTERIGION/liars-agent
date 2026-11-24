"""Online RL (Paired GRPO / PPO) training on Tinker with belief-calibration anchoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Sequence

from liars_agent.agents.base import BaseAgent, describe_action, public_opponent_profile
from liars_agent.agents.prompt_builder import action_prompt_for_method, validate_action_decision
from liars_agent.agents.rule_bots import CardCounterBot
from liars_agent.contracts import current_contract, require_contract
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.state import LiarsBarEngine, _derive_seed
from liars_agent.engine.types import AgentDecisionRecord, CandidateAction
from liars_agent.training.tinker_backend import ModelResponseError, SampleResult, TinkerBackend, resolve_future
from liars_agent.training.tinker_train import _tokenize_sft_datum, validate_training_sample

SUPPORTED_RL_LOSS_FNS = ("ppo", "importance_sampling", "cispo", "gspo", "stable_reinforce")


@dataclass(frozen=True)
class RLStepRecord:
    """One sampled decision turn from an online RL episode."""

    round_id: int
    step_id: int
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    completion_logprobs: tuple[float, ...]
    action_type: str
    was_fallback: bool
    challenge_succeeded: bool | None = None
    was_bluff_play: bool = False


@dataclass(frozen=True)
class RLEpisodeResult:
    """Complete episode trajectory and outcome statistics for one paired rollout."""

    game_id: str
    seat: int
    won: bool
    round_penalized_seat: dict[int, int]
    steps: tuple[RLStepRecord, ...]

    @property
    def num_decisions(self) -> int:
        return len(self.steps)

    @property
    def num_fallbacks(self) -> int:
        return sum(1 for s in self.steps if s.was_fallback)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def build_rl_datum(
    prompt_token_ids: Sequence[int],
    completion_token_ids: Sequence[int],
    completion_logprobs: Sequence[float],
    advantage: float,
    backend: TinkerBackend,
    max_seq_length: int,
) -> Any:
    """Construct a Tinker RL Datum (`target_tokens`, `logprobs`, `advantages`) with prompt masking."""
    prompt_ids = [int(t) for t in prompt_token_ids]
    comp_ids = [int(t) for t in completion_token_ids]
    comp_lps = [float(lp) for lp in completion_logprobs]
    if not prompt_ids:
        raise ValueError("RL datum requires non-empty prompt_token_ids.")
    if not comp_ids:
        raise ValueError("RL datum requires non-empty completion_token_ids.")
    if len(comp_ids) != len(comp_lps):
        raise ValueError("completion_token_ids and completion_logprobs must have the same length.")
    if any(not math.isfinite(lp) or lp > 1e-6 for lp in comp_lps):
        raise ValueError("completion_logprobs must be finite and nonpositive.")
    if not math.isfinite(advantage):
        raise ValueError("RL advantage must be finite.")
    full_tokens = prompt_ids + comp_ids
    if len(full_tokens) > max_seq_length:
        raise ValueError(
            f"RL rollout sequence has {len(full_tokens)} tokens, exceeding max_seq_length={max_seq_length}."
        )
    pad_len = len(prompt_ids) - 1
    target_tokens = full_tokens[1:]
    aligned_logprobs = [0.0] * pad_len + comp_lps
    aligned_advantages = [0.0] * pad_len + [float(advantage)] * len(comp_ids)
    return backend.sdk.Datum(
        model_input=backend.sdk.ModelInput.from_ints(full_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": aligned_logprobs,
            "advantages": aligned_advantages,
        },
    )


def compute_step_returns(
    episode: RLEpisodeResult,
    *,
    win_weight: float = 0.70,
    round_survival_weight: float = 0.30,
    fallback_penalty: float = -1.0,
) -> list[float]:
    """Compute per-decision scalar returns combining game win, round survival, and legality."""
    win_reward = 1.0 if episode.won else 0.0
    returns: list[float] = []
    for step in episode.steps:
        penalized_seat = episode.round_penalized_seat.get(step.round_id)
        survived_round = 1.0 if (penalized_seat is not None and penalized_seat != episode.seat) else 0.0
        r = win_weight * win_reward + round_survival_weight * survived_round
        if step.was_fallback:
            r += fallback_penalty
        returns.append(float(r))
    return returns


def compute_group_relative_advantages(
    episodes: Sequence[RLEpisodeResult],
    *,
    win_weight: float = 0.70,
    round_survival_weight: float = 0.30,
    fallback_penalty: float = -1.0,
    clip_advantage: float = 3.0,
) -> tuple[list[tuple[RLStepRecord, float]], dict[str, float]]:
    """GRPO group-relative normalization across paired rollouts sharing an env seed & lineup."""
    if not episodes:
        return [], {"mean_return": 0.0, "std_return": 0.0, "kept_steps": 0}
    per_ep_returns = [
        compute_step_returns(
            ep,
            win_weight=win_weight,
            round_survival_weight=round_survival_weight,
            fallback_penalty=fallback_penalty,
        )
        for ep in episodes
    ]
    # Group baseline uses episode-level mean return so longer games don't bias the baseline
    ep_scores = [
        (sum(rets) / len(rets)) if rets else (win_weight * float(ep.won))
        for ep, rets in zip(episodes, per_ep_returns)
    ]
    mean_r = sum(ep_scores) / len(ep_scores)
    var_r = sum((s - mean_r) ** 2 for s in ep_scores) / len(ep_scores)
    std_r = math.sqrt(var_r)

    # Also check step-level variance (e.g. if all episodes won/lost, round survival/fallback can still separate steps)
    all_step_returns = [r for rets in per_ep_returns for r in rets]
    if not all_step_returns:
        return [], {"mean_return": mean_r, "std_return": 0.0, "kept_steps": 0}
    step_mean = sum(all_step_returns) / len(all_step_returns)
    step_std = math.sqrt(sum((r - step_mean) ** 2 for r in all_step_returns) / len(all_step_returns))
    denom = max(std_r, step_std)
    if denom < 1e-6:
        return [], {"mean_return": mean_r, "std_return": 0.0, "kept_steps": 0}

    pairs: list[tuple[RLStepRecord, float]] = []
    for ep, rets in zip(episodes, per_ep_returns):
        for step, r in zip(ep.steps, rets):
            adv = max(-clip_advantage, min(clip_advantage, (r - step_mean) / denom))
            pairs.append((step, float(adv)))
    return pairs, {"mean_return": mean_r, "std_return": denom, "kept_steps": len(pairs)}


class _RLRolloutActor(BaseAgent):
    """Online sampling actor that records exact token IDs and sampling logprobs for RL."""

    def __init__(
        self,
        backend: TinkerBackend,
        sampling_client: Any,
        *,
        method_version: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        super().__init__(name="rl_policy_actor", style_tag="rl_policy")
        self.backend = backend
        self.sampling_client = sampling_client
        self.method_version = method_version
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.fallback = CardCounterBot(name="rl_fallback", seed=seed + 17)
        self.recorded_steps: list[RLStepRecord] = []

    def select_action(self, obs: PlayerObservation) -> tuple[CandidateAction, AgentDecisionRecord]:
        # Forced single-action states carry zero policy branching; execute legally without wasting tokens
        if len(obs.legal_actions) == 1:
            action = obs.legal_actions[0]
            return action, AgentDecisionRecord(
                belief_bluff_prob=None,
                belief_source="none",
                estimated_opponent_style="unknown",
                rationale=describe_action(obs, action),
                rationale_source="action_description",
                opponent_profile=public_opponent_profile(obs),
            )
        system, user = action_prompt_for_method(obs, self.method_version)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        step_seed = _derive_seed(self.seed, "rl_step", obs.round_id, obs.step_id, obs.observer_seat)
        sample: SampleResult = self.backend.sample(
            messages,
            sampling_client=self.sampling_client,
            num_samples=1,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            seed=step_seed,
        )[0]
        if sample.prompt_token_ids is None or sample.completion_token_ids is None or sample.completion_logprobs is None:
            raise ModelResponseError(
                "RL sampling requires prompt_token_ids, completion_token_ids, and completion_logprobs."
            )
        was_fallback = False
        try:
            parsed = validate_action_decision(obs, sample.text.strip())
            by_id = {a.action_id: a for a in obs.legal_actions}
            action = by_id[parsed["chosen_action_id"]]
        except ValueError:
            was_fallback = True
            action, _ = self.fallback.select_action(obs)

        is_bluff_play = (
            action.action_type == "PLAY"
            and not all(c == obs.target_rank or c == "JOKER" for c in action.cards)
        )
        self.recorded_steps.append(
            RLStepRecord(
                round_id=obs.round_id,
                step_id=obs.step_id,
                prompt_token_ids=sample.prompt_token_ids,
                completion_token_ids=sample.completion_token_ids,
                completion_logprobs=sample.completion_logprobs,
                action_type=action.action_type,
                was_fallback=was_fallback,
                was_bluff_play=is_bluff_play,
            )
        )
        return action, AgentDecisionRecord(
            belief_bluff_prob=None,
            belief_source="none",
            estimated_opponent_style="unknown",
            rationale=describe_action(obs, action),
            rationale_source="action_description",
            opponent_profile=public_opponent_profile(obs),
            was_fallback=was_fallback,
            raw_output=sample.text,
            prompt_tokens=sample.prompt_tokens,
            completion_tokens=sample.completion_tokens,
            latency_ms=sample.latency_ms,
        )


def collect_paired_rl_group(
    backend: TinkerBackend,
    sampling_client: Any,
    opponent_factories: Sequence[Callable[[int], BaseAgent]],
    *,
    env_seed: int,
    seat: int,
    group_size: int,
    group_seed: int,
    game_prefix: str,
    method_version: str = "adaptive_v4",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
) -> list[RLEpisodeResult]:
    """Run `group_size` paired full episodes sharing `(env_seed, seat, opponents)` for GRPO."""
    if len(opponent_factories) != 3:
        raise ValueError("Liar's Bar requires exactly 3 opponent factories.")
    if group_size < 2:
        raise ValueError("Paired GRPO requires group_size >= 2.")
    episodes: list[RLEpisodeResult] = []
    for sample_idx in range(group_size):
        actor = _RLRolloutActor(
            backend,
            sampling_client,
            method_version=method_version,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=_derive_seed(group_seed, "actor", sample_idx),
        )
        lineup: list[BaseAgent | None] = [None] * 4
        lineup[seat] = actor
        opp_idx = 0
        for s in range(4):
            if s != seat:
                lineup[s] = opponent_factories[opp_idx](_derive_seed(group_seed, "opp", s))
                opp_idx += 1
        engine = LiarsBarEngine(game_id=f"{game_prefix}_g{sample_idx}", game_seed=env_seed)
        round_penalized: dict[int, int] = {}
        while not engine.is_game_over():
            obs = engine.get_observation()
            turn = obs.observer_seat
            agent = lineup[turn]
            assert agent is not None
            act, rec = agent.select_action(obs)
            res = engine.step(act, rec)
            if res is not None:
                round_penalized[res.round_id] = res.penalized_seat
                if turn == seat and actor.recorded_steps and actor.recorded_steps[-1].action_type == "CHALLENGE":
                    last = actor.recorded_steps[-1]
                    actor.recorded_steps[-1] = RLStepRecord(
                        round_id=last.round_id,
                        step_id=last.step_id,
                        prompt_token_ids=last.prompt_token_ids,
                        completion_token_ids=last.completion_token_ids,
                        completion_logprobs=last.completion_logprobs,
                        action_type=last.action_type,
                        was_fallback=last.was_fallback,
                        challenge_succeeded=res.challenge_succeeded,
                        was_bluff_play=last.was_bluff_play,
                    )
        episodes.append(
            RLEpisodeResult(
                game_id=engine.game_id,
                seat=seat,
                won=(engine.winner_seat == seat),
                round_penalized_seat=round_penalized,
                steps=tuple(actor.recorded_steps),
            )
        )
    return episodes


def _refresh_inflight_sampler(client: Any, backend: TinkerBackend, step: int) -> Any:
    if hasattr(client, "save_weights_and_get_sampling_client"):
        return resolve_future(client.save_weights_and_get_sampling_client())
    sampler_path = resolve_future(client.save_weights_for_sampler(f"rl-step-{step:04d}")).path
    return backend.sampling_client(sampler_path, fresh=True)


def run_tinker_lora_rl(
    *,
    output_dir: str | Path,
    source_metadata: dict[str, Any],
    opponent_pool_sampler: Callable[[int], list[Callable[[int], BaseAgent]]],
    belief_train_path: str | Path | None = None,
    base_model: str = "Qwen/Qwen3.5-9B",
    lora_rank: int = 16,
    train_attn: bool = True,
    train_mlp: bool = True,
    train_unembed: bool = False,
    num_steps: int = 30,
    groups_per_step: int = 4,
    group_size: int = 4,
    learning_rate: float = 2.0e-5,
    loss_fn: str = "ppo",
    loss_fn_config: dict[str, Any] | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    max_seq_length: int = 16384,
    win_weight: float = 0.70,
    round_survival_weight: float = 0.30,
    fallback_penalty: float = -1.0,
    clip_advantage: float = 3.0,
    belief_aux_batch_size: int = 8,
    belief_aux_weight: float = 0.25,
    checkpoint_every_steps: int = 10,
    seed: int = 42,
    method_version: str = "adaptive_v4",
    budget_path: str | Path = "outputs/budget_usage.jsonl",
    budget_limit_usd: float = 5000.0,
    backend: TinkerBackend | None = None,
) -> dict[str, Any]:
    """Run paired GRPO/PPO online RL from an SFT state checkpoint on Tinker."""
    if loss_fn not in SUPPORTED_RL_LOSS_FNS:
        raise ValueError(f"Unsupported RL loss_fn: {loss_fn}; expected one of {SUPPORTED_RL_LOSS_FNS}.")
    if num_steps < 1 or groups_per_step < 1 or group_size < 2 or learning_rate <= 0:
        raise ValueError("RL requires num_steps>=1, groups_per_step>=1, group_size>=2, and learning_rate>0.")
    if checkpoint_every_steps < 1:
        raise ValueError("checkpoint_every_steps must be >= 1.")

    lora_config = {
        "r": lora_rank,
        "train_mlp": train_mlp,
        "train_attn": train_attn,
        "train_unembed": train_unembed,
    }
    require_contract(source_metadata, "RL source SFT metadata", method_version=method_version)
    source_state = source_metadata.get("state_path")
    if not isinstance(source_state, str) or not source_state.startswith("tinker://"):
        raise ValueError("RL requires a valid tinker:// source SFT state_path.")
    if source_metadata.get("base_model") != base_model:
        raise ValueError("RL source SFT base_model does not match requested base_model.")
    source_lora = source_metadata.get("lora_config")
    if source_lora != lora_config or any(type(source_lora[k]) is not type(v) for k, v in lora_config.items()):
        raise ValueError("RL source SFT LoRA configuration differs from requested rank/layers.")

    belief_rows: list[dict[str, Any]] = []
    if belief_aux_batch_size > 0 and belief_train_path is not None:
        for row in _iter_jsonl(Path(belief_train_path)):
            validate_training_sample(row, method_version=method_version)
            if row["task"] == "belief":
                belief_rows.append(row)
        if not belief_rows:
            raise ValueError("belief_train_path contains no valid belief rows.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "rl_progress.json"
    metrics_path = out / "rl_metrics.jsonl"

    start_step = 0
    resume_state = source_state
    resume_with_optimizer = False
    history: list[dict[str, Any]] = []
    if progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        require_contract(saved, "RL progress", method_version=method_version)
        if saved.get("source_state_path") != source_state or saved.get("loss_fn") != loss_fn:
            raise ValueError("Existing RL progress does not match current source_state_path or loss_fn.")
        start_step = int(saved["completed_steps"])
        resume_state = str(saved["state_path"])
        resume_with_optimizer = True
        if metrics_path.exists():
            history = list(_iter_jsonl(metrics_path))

    backend = backend or TinkerBackend(
        base_model=base_model,
        budget_path=budget_path,
        budget_limit_usd=budget_limit_usd,
    )
    backend.preflight()
    t0 = time.perf_counter()

    if resume_with_optimizer:
        client = resolve_future(backend.service.create_training_client_from_state_with_optimizer(resume_state))
    else:
        client = resolve_future(backend.service.create_training_client_from_state(resume_state))

    sampler_client = _refresh_inflight_sampler(client, backend, start_step)
    latest_state_path = resume_state
    latest_sampler_path: str | None = source_metadata.get("sampler_path")

    for step in range(start_step + 1, num_steps + 1):
        step_t0 = time.perf_counter()
        rl_datums: list[Any] = []
        step_episodes: list[RLEpisodeResult] = []
        skipped_groups = 0
        for g_idx in range(groups_per_step):
            group_seed = _derive_seed(seed, "rl_group", step, g_idx)
            env_seed = _derive_seed(group_seed, "env")
            seat = (step * groups_per_step + g_idx) % 4
            opp_factories = opponent_pool_sampler(group_seed)
            episodes = collect_paired_rl_group(
                backend,
                sampler_client,
                opp_factories,
                env_seed=env_seed,
                seat=seat,
                group_size=group_size,
                group_seed=group_seed,
                game_prefix=f"rl_s{step:04d}_g{g_idx:02d}",
                method_version=method_version,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            step_episodes.extend(episodes)
            pairs, g_stats = compute_group_relative_advantages(
                episodes,
                win_weight=win_weight,
                round_survival_weight=round_survival_weight,
                fallback_penalty=fallback_penalty,
                clip_advantage=clip_advantage,
            )
            if not pairs:
                skipped_groups += 1
                continue
            for rec, adv in pairs:
                if len(rec.prompt_token_ids) + len(rec.completion_token_ids) > max_seq_length:
                    continue
                rl_datums.append(
                    build_rl_datum(
                        rec.prompt_token_ids,
                        rec.completion_token_ids,
                        rec.completion_logprobs,
                        adv,
                        backend,
                        max_seq_length,
                    )
                )

        # Cosine LR decay across RL steps
        progress_frac = (step - 1) / max(1, num_steps - 1) if num_steps > 1 else 0.0
        step_lr = learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress_frac)))

        updated = False
        if rl_datums:
            use_belief_aux = bool(belief_rows) and belief_aux_batch_size > 0 and belief_aux_weight > 0
            backend.run_training_batch(
                client,
                rl_datums,
                learning_rate=step_lr,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
                step_optimizer=(not use_belief_aux),
            )
            if use_belief_aux:
                b_rng = random.Random(_derive_seed(seed, "rl_belief_aux", step))
                chosen_beliefs = [b_rng.choice(belief_rows) for _ in range(belief_aux_batch_size)]
                belief_datums = [
                    _tokenize_sft_datum(
                        r["system"],
                        r["prompt"],
                        "belief",
                        r["completion"],
                        r.get("sample_id", "?"),
                        backend,
                        max_seq_length,
                        sample_weight=belief_aux_weight,
                    )
                    for r in chosen_beliefs
                ]
                backend.run_training_batch(
                    client,
                    belief_datums,
                    learning_rate=step_lr,
                    loss_fn="cross_entropy",
                    step_optimizer=True,
                )
            updated = True
            sampler_client = _refresh_inflight_sampler(client, backend, step)

        total_eps = len(step_episodes)
        wins = sum(1 for e in step_episodes if e.won)
        all_steps = [s for e in step_episodes for s in e.steps]
        fallbacks = sum(1 for s in all_steps if s.was_fallback)
        challs = [s for s in all_steps if s.action_type == "CHALLENGE"]
        chall_hits = sum(1 for s in challs if s.challenge_succeeded)
        plays = [s for s in all_steps if s.action_type == "PLAY"]
        bluffs = sum(1 for s in plays if s.was_bluff_play)

        step_record = {
            "step": step,
            "learning_rate": step_lr,
            "episodes": total_eps,
            "win_rate": round(wins / max(1, total_eps), 4),
            "num_decisions": len(all_steps),
            "rl_datums": len(rl_datums),
            "skipped_groups": skipped_groups,
            "updated": updated,
            "fallback_rate": round(fallbacks / max(1, len(all_steps)), 4),
            "challenge_hit_rate": round(chall_hits / len(challs), 4) if challs else None,
            "bluff_rate": round(bluffs / len(plays), 4) if plays else None,
            "wall_time_sec": round(time.perf_counter() - step_t0, 2),
            "budget_spent_usd": round(backend.ledger.snapshot()["estimated_cost_usd"], 4),
        }
        history.append(step_record)
        _append_jsonl(metrics_path, step_record)
        print(
            f"[RL step {step:03d}/{num_steps}] win={step_record['win_rate']*100:5.1f}% "
            f"datums={len(rl_datums)} fb={step_record['fallback_rate']*100:4.1f}% "
            f"chal_hit={step_record['challenge_hit_rate']} spent=${step_record['budget_spent_usd']:.2f}",
            flush=True,
        )

        if step % checkpoint_every_steps == 0 or step == num_steps:
            tag = "rl-final" if step == num_steps else f"rl-step-{step:04d}"
            latest_state_path = resolve_future(client.save_state(tag)).path
            latest_sampler_path = resolve_future(client.save_weights_for_sampler(tag)).path
            _write_json(
                progress_path,
                {
                    "contract": current_contract(method_version),
                    "source_state_path": source_state,
                    "loss_fn": loss_fn,
                    "completed_steps": step,
                    "state_path": latest_state_path,
                    "sampler_path": latest_sampler_path,
                },
            )

    if latest_sampler_path is None:
        latest_sampler_path = resolve_future(client.save_weights_for_sampler("rl-final")).path

    metadata = {
        "contract": current_contract(method_version),
        "method_version": method_version,
        "adapter_name": f"qwen35_9b_liars_lora_rl_{method_version}",
        "base_model": base_model,
        "platform": "tinker",
        "backend": "tinker",
        "stage": "rl",
        "training_status": "complete",
        "formal_training_complete": True,
        "tasks": ["action", "belief"],
        "source_sft_state_path": source_state,
        "lora_config": lora_config,
        "sampler_path": latest_sampler_path,
        "state_path": latest_state_path,
        "rl_config": {
            "loss_fn": loss_fn,
            "loss_fn_config": loss_fn_config,
            "num_steps": num_steps,
            "groups_per_step": groups_per_step,
            "group_size": group_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "top_p": top_p,
            "win_weight": win_weight,
            "round_survival_weight": round_survival_weight,
            "fallback_penalty": fallback_penalty,
            "clip_advantage": clip_advantage,
            "belief_aux_batch_size": belief_aux_batch_size,
            "belief_aux_weight": belief_aux_weight,
            "seed": seed,
        },
        "summary": {
            "total_steps": num_steps,
            "mean_win_rate": round(sum(h["win_rate"] for h in history) / max(1, len(history)), 4),
            "final_5_win_rate": round(
                sum(h["win_rate"] for h in history[-5:]) / max(1, len(history[-5:])), 4
            ),
            "wall_time_sec": round(time.perf_counter() - t0, 2),
        },
        "cost_accounting": backend.ledger.snapshot(),
    }
    _write_json(out / "adapter_metadata.json", metadata)
    return metadata

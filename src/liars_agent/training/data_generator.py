"""Observation-only teachers and independently labelled action/belief tasks."""
from __future__ import annotations

import copy
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from liars_agent.agents.base import BaseAgent
from liars_agent.agents.prompt_builder import (
    SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT, build_user_prompt, build_bluff_user_prompt,
    validate_action_decision,
)
from liars_agent.agents.rule_bots import create_rule_agent
from liars_agent.contracts import current_contract
from liars_agent.engine.observation import PlayerObservation, apply_history_ablation
from liars_agent.engine.replay import GameReplayBundle, verify_replay_determinism
from liars_agent.engine.state import LiarsBarEngine, PendingPlayTruth, _derive_seed
from liars_agent.engine.types import CandidateAction, sort_cards

DEFAULT_TEACHER_CONFIG = {
    "actor": {"class": "CardCounterBot", "params": {"risk_aversion": 1.0, "posterior_challenge_cutoff": .64}},
    "opponents": [
        {"id": "honest_conservative", "class": "HonestConservativeBot", "params": {"challenge_threshold": .72}},
        {"id": "aggressive_bluffer", "class": "AggressiveBlufferBot", "params": {"bluff_rate": .65, "multi_card_bias": .55, "challenge_threshold": .80}},
        {"id": "card_counter", "class": "CardCounterBot", "params": {"risk_aversion": 1.0, "posterior_challenge_cutoff": .64}},
    ],
}


def sample_rollout_state(obs: PlayerObservation, seed: int) -> LiarsBarEngine:
    """Uniform compatible card slots, not a posterior over strategic choices."""
    rng = random.Random(seed)
    statuses = sorted(obs.players_public_status, key=lambda p: p.seat)
    if len(statuses) != 4 or [p.seat for p in statuses] != list(range(4)):
        raise ValueError("Rollouts require the four public seat statuses.")
    sim = LiarsBarEngine(game_id="rollout", game_seed=_derive_seed(seed, "future"), record_truth=False)
    sim.alive = [p.alive for p in statuses]
    sim.shots_taken = [p.shots_taken for p in statuses]
    sim.fatal_chamber_slot = [rng.randint(p.shots_taken + 1, p.max_chambers) if p.alive else p.shots_taken for p in statuses]
    deck = sim._build_full_deck()
    for card in obs.own_hand + obs.own_played_cards_this_round:
        deck.remove(card)
    rng.shuffle(deck)
    sim.hands = [[] for _ in statuses]
    sim.played_cards_this_round = [[] for _ in statuses]
    for p in statuses:
        if not p.alive:
            continue
        if p.seat == obs.observer_seat:
            if p.hand_count != len(obs.own_hand) or len(obs.own_known_cards_this_round) != sim.hand_size:
                raise ValueError("Own current-round card memory is inconsistent.")
            sim.hands[p.seat] = list(obs.own_hand)
            sim.played_cards_this_round[p.seat] = list(obs.own_played_cards_this_round)
        else:
            sim.hands[p.seat] = list(sort_cards([deck.pop() for _ in range(p.hand_count)]))
            sim.played_cards_this_round[p.seat] = [deck.pop() for _ in range(sim.hand_size - p.hand_count)]
    sim.round_id, sim.step_id, sim.target_rank = obs.round_id, obs.step_id, obs.target_rank
    sim.current_turn_seat = obs.observer_seat
    sim.current_round_history = list(obs.current_round_history)
    sim.past_public_actions = list(obs.past_public_actions)
    sim.past_rounds_history = list(obs.past_rounds_history)
    sim.pending_play = None
    if obs.last_claim:
        claim = obs.last_claim
        played = sim.played_cards_this_round[claim.claimer_seat]
        if len(played) < claim.claimed_count:
            raise ValueError("Last claim exceeds the claimer's played cards.")
        sim.pending_play = PendingPlayTruth(round_id=claim.round_id, step_id=claim.step_id,
            claimer_seat=claim.claimer_seat, claimed_rank=claim.claimed_rank,
            actual_cards=sort_cards(played[-claim.claimed_count:]))
    return sim


def _clone_simulation(state: LiarsBarEngine) -> LiarsBarEngine:
    sim = copy.copy(state)
    for name in ("alive", "shots_taken", "fatal_chamber_slot", "current_round_history", "past_rounds_history", "past_public_actions"):
        setattr(sim, name, list(getattr(state, name)))
    sim.hands = [list(hand) for hand in state.hands]
    sim.played_cards_this_round = [list(cards) for cards in state.played_cards_this_round]
    sim.step_truth_logs = []
    return sim


def _score_continuation(sim, actor_seat, action, agents, ablation_mode="full") -> float:
    resolution = sim.step(action)
    current_round_penalty = None if resolution is None else resolution.penalized_seat == actor_seat
    while not sim.is_game_over():
        seat = sim.current_turn_seat
        observation = sim.get_observation(seat)
        if seat == actor_seat:
            observation = apply_history_ablation(observation, ablation_mode)
        chosen, record = agents[seat].select_action(observation)
        resolution = sim.step(chosen, record)
        if current_round_penalty is None and resolution is not None:
            current_round_penalty = resolution.penalized_seat == actor_seat
    return .65 * (sim.winner_seat == actor_seat) + .35 * (not current_round_penalty)


def build_rollout_agents(teacher_config: dict, actor_seat: int, seed: int) -> list[BaseAgent]:
    """Fresh policy randomness per trajectory, independent of actual-game bots."""
    specs = teacher_config["opponents"]
    if len(specs) != 3 or {s["class"] for s in specs} != {"HonestConservativeBot", "AggressiveBlufferBot", "CardCounterBot"}:
        raise ValueError("Teacher requires one of each seen opponent class.")
    if teacher_config["actor"]["class"] != "CardCounterBot":
        raise ValueError("Teacher continuation actor must be CardCounterBot.")
    order = list(range(3))
    random.Random(_derive_seed(seed, "opponent_order")).shuffle(order)
    agents = []
    for seat in range(4):
        offset = (seat - actor_seat) % 4
        spec = teacher_config["actor"] if offset == 0 else specs[order[offset - 1]]
        agents.append(create_rule_agent(spec["class"], name=spec.get("id", "teacher_actor"),
            seed=_derive_seed(seed, "policy", seat), **spec.get("params", {})))
    return agents


def evaluate_observation_action_rewards(
    obs: PlayerObservation, teacher_config: dict | None = None, num_rollouts: int = 8,
    rollout_seed: int = 1000, candidate_actions: Sequence[CandidateAction] | None = None,
    *, start_index: int = 0,
) -> dict[str, list[float]]:
    """Paired rewards indexed by trajectory, with resumable deterministic seeds.

    A later call can evaluate indices ``start_index:start_index + num_rollouts``
    without repeating earlier trajectories. Every candidate uses the same hidden
    world and fresh copies of the same policy seeds at each trajectory index.
    """
    if obs.ablation_mode != "full":
        raise ValueError("Teacher training requires full observations; ablation is evaluation-only.")
    if type(num_rollouts) is not int or num_rollouts < 1:
        raise ValueError("num_rollouts must be a positive integer.")
    if type(start_index) is not int or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer.")
    config = DEFAULT_TEACHER_CONFIG if teacher_config is None else teacher_config
    if not isinstance(config, dict):
        raise TypeError("Teacher accepts a policy-pool configuration, never actual-game agent instances.")
    candidates = list(obs.legal_actions if candidate_actions is None else candidate_actions)
    legal = {a.action_id: a for a in obs.legal_actions}
    if not candidates or any(legal.get(a.action_id) != a for a in candidates) or len({a.action_id for a in candidates}) != len(candidates):
        raise ValueError("Candidates must be distinct legal actions from this observation.")
    rewards: dict[str, list[float]] = {a.action_id: [] for a in candidates}
    for index in range(start_index, start_index + num_rollouts):
        sample_seed = _derive_seed(rollout_seed, "trajectory", index)
        state = sample_rollout_state(obs, _derive_seed(sample_seed, "state"))
        rollout_agents = build_rollout_agents(config, obs.observer_seat, _derive_seed(sample_seed, "policies"))
        for action in candidates:
            rewards[action.action_id].append(_score_continuation(_clone_simulation(state), obs.observer_seat,
                action, copy.deepcopy(rollout_agents)))
    return rewards


def evaluate_observation_action_values(
    obs: PlayerObservation, teacher_config: dict | None = None, num_rollouts: int = 8,
    rollout_seed: int = 1000, candidate_actions: Sequence[CandidateAction] | None = None,
) -> dict[str, float]:
    """Compatibility API retaining the original ordered mean accumulation."""
    rewards = evaluate_observation_action_rewards(obs, teacher_config, num_rollouts,
                                                  rollout_seed, candidate_actions)
    return {action_id: sum(reward / num_rollouts for reward in values)
            for action_id, values in rewards.items()}


def make_action_sample(obs: PlayerObservation, action: CandidateAction, game_id: str, **metadata) -> dict:
    completion = json.dumps({"chosen_action_id": action.action_id}, ensure_ascii=False)
    validate_action_decision(obs, completion)
    return {"task": "action", "sample_id": f"{game_id}_r{obs.round_id}_s{obs.step_id}_action",
        "game_id": game_id, "round_id": obs.round_id, "step_id": obs.step_id,
        "actor_seat": obs.observer_seat, "ablation_mode": obs.ablation_mode,
        "observation": obs.to_dict(), "system": SYSTEM_PROMPT, "prompt": build_user_prompt(obs),
        "completion": completion, "contract": current_contract(), **metadata}


def _as_dict(replay):
    return replay.to_dict() if isinstance(replay, GameReplayBundle) else replay


def extract_belief_samples(replays, target_samples: int, seed: int = 42, actor_seat: int | None = None) -> list[dict]:
    """Uniform decisions within games, independent of action advantage and labels."""
    bundles = [_as_dict(r) for r in replays]
    rows = []
    if not bundles:
        return rows
    # First pass gives each complete game an equal quota. Fill shortages by
    # round-robin over remaining shuffled observations without label filtering.
    queues = []
    for bundle in bundles:
        points = [p for p in bundle["steps"] if p["pending_claim_was_bluff"] is not None
                  and (actor_seat is None or p["actor_seat"] == actor_seat)]
        random.Random(_derive_seed(seed, "belief_points", bundle["game_id"])).shuffle(points)
        queues.append((bundle["game_id"], points))
    while len(rows) < target_samples and any(points for _, points in queues):
        for game_id, points in queues:
            if not points or len(rows) == target_samples:
                continue
            point = points.pop()
            obs = PlayerObservation.from_dict(point["observation"])
            if obs.last_claim is None or obs.ablation_mode != "full":
                raise ValueError("Belief source must be a full observation with a pending claim.")
            label = point["pending_claim_was_bluff"]
            if not isinstance(label, bool):
                raise ValueError("Belief labels must be actual boolean engine truth.")
            rows.append({"task": "belief", "sample_id": f"{game_id}_r{obs.round_id}_s{obs.step_id}_belief",
                "game_id": game_id, "round_id": obs.round_id, "step_id": obs.step_id,
                "actor_seat": obs.observer_seat, "ablation_mode": "full", "observation": obs.to_dict(),
                "system": BELIEF_SYSTEM_PROMPT, "prompt": build_bluff_user_prompt(obs),
                "completion": str(int(label)), "label": int(label), "source": "engine_pending_claim_truth",
                "contract": current_contract()})
    return rows


def extract_action_samples(replays, target_samples, num_rollouts=8, margin=.05, seed=42, teacher_config=None):
    bundles = [_as_dict(r) for r in replays]
    samples = []
    for index, bundle in enumerate(bundles):
        quota = target_samples // len(bundles) + (index < target_samples % len(bundles))
        points = [p for p in bundle["steps"] if len(p["observation"]["legal_actions"]) > 1]
        random.Random(_derive_seed(seed, "action_points", bundle["game_id"])).shuffle(points)
        accepted = 0
        for point in points:
            if accepted >= quota:
                break
            obs = PlayerObservation.from_dict(point["observation"])
            values = evaluate_observation_action_values(obs, teacher_config, num_rollouts,
                _derive_seed(seed, "teacher", bundle["game_id"], obs.round_id, obs.step_id))
            best = max(obs.legal_actions, key=lambda a: (values[a.action_id], a.action_type, a.cards))
            advantage = values[best.action_id] - sum(values.values()) / len(values)
            if advantage < margin:
                continue
            samples.append(make_action_sample(obs, best, bundle["game_id"],
                rollout_best_value=values[best.action_id], rollout_advantage=advantage,
                candidate_count=len(values), teacher="pool_prior_counter_continuation", rollout_candidate_values=values))
            accepted += 1
    return samples


def generate_rule_replays(num_games: int, base_seed: int, opponent_factories=None, prefix="train"):
    if num_games < 1:
        raise ValueError("num_games must be positive.")
    factories = list(opponent_factories or [
        lambda seed, spec=spec: create_rule_agent(spec["class"], seed=seed, **spec["params"])
        for spec in DEFAULT_TEACHER_CONFIG["opponents"]])
    replays = []
    for index in range(num_games):
        game_id = f"{prefix}_{base_seed}_game_{index:05d}"  # private artifact identifier only
        environment_seed = _derive_seed(base_seed, "environment", index)
        lineup_rng = random.Random(_derive_seed(base_seed, "lineup", index))
        agents = [lineup_rng.choice(factories)(_derive_seed(base_seed, "policy", index, seat)) for seat in range(4)]
        engine = LiarsBarEngine(game_id=game_id, game_seed=environment_seed)
        while not engine.is_game_over():
            obs = engine.get_observation()
            action, record = agents[obs.observer_seat].select_action(obs)
            engine.step(action, record)
        replay = GameReplayBundle.from_engine(engine, [a.name for a in agents])
        verify_replay_determinism(replay, raise_on_error=True)
        replays.append(replay)
    return replays


def generate_mixed_opponent_sft_dataset(num_games=120, target_samples=1000, num_rollouts_per_action=8,
    min_advantage_margin=.05, history_ablation_ratio=0.0, base_seed=2026,
    output_jsonl_path: str | Path | None = None, opponent_factories=None, teacher_config=None):
    """Standalone action data helper; the pipeline partitions games before extraction."""
    if history_ablation_ratio != 0:
        raise ValueError("History augmentation is disabled; use evaluation-only ablations.")
    if target_samples < 0 or num_rollouts_per_action < 1:
        raise ValueError("Nonnegative target and positive rollout count required.")
    replays = generate_rule_replays(num_games, base_seed, opponent_factories)
    samples = extract_action_samples(replays, target_samples, num_rollouts_per_action,
        min_advantage_margin, base_seed, teacher_config)
    if output_jsonl_path:
        path = Path(output_jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in samples), encoding='utf-8')
    return samples, replays

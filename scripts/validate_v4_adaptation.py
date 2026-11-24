#!/usr/bin/env python3
"""Local zero-API validation for v4 cross-round online adaptation (data + posterior teacher)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Sequence

from liars_agent.agents.base import BaseAgent
from liars_agent.agents.classifier_bot import SmallClassifierBot
from liars_agent.agents.rule_bots import CardCounterBot, ChameleonAdaptiveBot, ParanoidHunterBot
from liars_agent.engine.observation import PlayerObservation, apply_history_ablation
from liars_agent.engine.replay import GameReplayBundle
from liars_agent.engine.state import LiarsBarEngine, _derive_seed
from liars_agent.engine.types import (
    AgentDecisionRecord,
    CandidateAction,
    ClaimEvent,
    PlayerPublicInfo,
    PublicEvent,
    RoundResolution,
)
from liars_agent.training.adaptive_opponents import (
    ArchetypeSpec,
    MidGameSwitcherBot,
    ParameterizedArchetypeBot,
    UNSEEN_MIRROR_MULTI_BLUFFER,
    UNSEEN_MIRROR_SINGLE_BLUFFER,
    V4_TRAIN_ARCHETYPES,
    render_count_conditioned_summary,
)
from liars_agent.training.adaptive_teacher import (
    CalibratedPosteriorBelief,
    CountHistoryLogisticClassifier,
    PosteriorAdaptiveBot,
    evaluate_posterior_teacher_values,
    infer_opponent_posteriors,
    posterior_bluff_probability,
)
from liars_agent.training.data_generator import evaluate_observation_action_values


def _brier(preds: list[float], labels: list[int]) -> float | None:
    if not preds:
        return None
    return sum((p - y) ** 2 for p, y in zip(preds, labels)) / len(preds)


def _logloss(preds: list[float], labels: list[int]) -> float | None:
    if not preds:
        return None
    eps = 1e-6
    total = 0.0
    for p, y in zip(preds, labels):
        pc = min(1.0 - eps, max(eps, p))
        total += -(y * math.log(pc) + (1 - y) * math.log(1.0 - pc))
    return total / len(preds)


def _roc_auc(preds: list[float], labels: list[int]) -> float | None:
    pos = [p for p, y in zip(preds, labels) if y == 1]
    neg = [p for p, y in zip(preds, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _eval_belief_metrics(rows: list[tuple[PlayerObservation, int]], predictor: Callable[[PlayerObservation], float]) -> dict[str, Any]:
    preds = [predictor(obs) for obs, _ in rows]
    labels = [y for _, y in rows]
    out: dict[str, Any] = {
        "n": len(rows),
        "bluff_rate": round(sum(labels) / max(1, len(labels)), 4),
        "log_loss": round(_logloss(preds, labels), 4) if rows else None,
        "brier": round(_brier(preds, labels), 4) if rows else None,
        "auc": round(_roc_auc(preds, labels), 4) if _roc_auc(preds, labels) is not None else None,
        "by_round_bucket": {},
    }
    buckets = [
        ("R1-2", lambda r: r <= 2),
        ("R3-5", lambda r: 3 <= r <= 5),
        ("R6+", lambda r: r >= 6),
    ]
    for bname, fn in buckets:
        sub_p = [p for (obs, _), p in zip(rows, preds) if fn(obs.round_id)]
        sub_y = [y for (obs, y) in rows if fn(obs.round_id)]
        auc = _roc_auc(sub_p, sub_y)
        out["by_round_bucket"][bname] = {
            "n": len(sub_y),
            "log_loss": round(_logloss(sub_p, sub_y), 4) if sub_y else None,
            "brier": round(_brier(sub_p, sub_y), 4) if sub_y else None,
            "auc": round(auc, 4) if auc is not None else None,
        }
    return out


def _generate_games_for_belief(
    num_games: int,
    base_seed: int,
    make_lineup: Callable[[int], list[BaseAgent]],
) -> tuple[list[dict[str, Any]], list[tuple[PlayerObservation, int]]]:
    replays: list[dict[str, Any]] = []
    belief_rows: list[tuple[PlayerObservation, int]] = []
    for idx in range(num_games):
        agents = make_lineup(idx)
        env_seed = _derive_seed(base_seed, "env", idx)
        engine = LiarsBarEngine(game_id=f"v4_{base_seed}_{idx:04d}", game_seed=env_seed)
        while not engine.is_game_over():
            obs = engine.get_observation()
            act, rec = agents[obs.observer_seat].select_action(obs)
            engine.step(act, rec)
        bundle = GameReplayBundle.from_engine(engine, [a.name for a in agents]).to_dict()
        replays.append(bundle)
        for step in bundle["steps"]:
            if step["pending_claim_was_bluff"] is None:
                continue
            obs = PlayerObservation.from_dict(step["observation"])
            if obs.last_claim is not None:
                belief_rows.append((obs, int(step["pending_claim_was_bluff"])))
    return replays, belief_rows


def _make_counterfactual_pair(
    own_hand: tuple[str, ...],
    target_rank: str,
    claimed_count: int,
    shots: int,
) -> tuple[PlayerObservation, PlayerObservation]:
    """Construct two observations identical in current state, opposite in past reveals for Seat 1."""
    # Build legal actions via a real engine state? Even cleaner: directly construct valid CandidateActions.
    actions: list[CandidateAction] = [CandidateAction(action_id="A0", action_type="CHALLENGE")]
    aid = 1
    seen_combos: set[tuple[str, ...]] = set()
    hand_list = list(own_hand)
    n = len(hand_list)
    for k in range(1, min(3, n) + 1):
        # Generate sorted unique combinations
        def _dfs(start: int, cur: list[str]) -> None:
            nonlocal aid
            if len(cur) == k:
                tup = tuple(cur)
                if tup not in seen_combos:
                    seen_combos.add(tup)
                    actions.append(CandidateAction(action_id=f"A{aid}", action_type="PLAY", cards=tup))
                    aid += 1
                return
            for i in range(start, n):
                cur.append(hand_list[i])
                _dfs(i + 1, cur)
                cur.pop()
        _dfs(0, [])

    statuses = (
        PlayerPublicInfo(seat=0, alive=True, hand_count=len(own_hand), shots_taken=shots),
        PlayerPublicInfo(seat=1, alive=True, hand_count=5 - claimed_count, shots_taken=1),
        PlayerPublicInfo(seat=2, alive=True, hand_count=5, shots_taken=1),
        PlayerPublicInfo(seat=3, alive=True, hand_count=5, shots_taken=1),
    )
    claim = ClaimEvent(
        round_id=6,
        step_id=2,
        claimer_seat=1,
        claimed_rank=target_rank,
        claimed_count=claimed_count,
        claimer_remaining_cards=5 - claimed_count,
    )
    cur_hist = (
        PublicEvent(
            round_id=6,
            step_id=1,
            actor_seat=1,
            action_type="PLAY",
            claimed_count=claimed_count,
            claimed_rank=target_rank,
            actor_remaining_cards=5 - claimed_count,
        ),
    )

    # History A: Seat 1 is a Single-Bluffer / Multi-Truth trapper (3 multi truths + 2 single bluffs)
    hist_a_res = (
        RoundResolution(1, "Q", 1, 2, 2, ("Q", "Q"), False, False, 2, 0, 1, False),
        RoundResolution(2, "K", 1, 3, 2, ("K", "JOKER"), False, False, 3, 0, 1, False),
        RoundResolution(3, "A", 1, 2, 3, ("A", "A", "A"), False, False, 2, 1, 2, False),
        RoundResolution(4, "Q", 1, 3, 1, ("K",), True, True, 1, 0, 1, False),
        RoundResolution(5, "A", 1, 0, 1, ("Q",), True, True, 1, 1, 2, False),
    )
    # History B: Seat 1 is a Multi-Bluffer / Single-Truth mirror (3 multi bluffs + 2 single truths)
    hist_b_res = (
        RoundResolution(1, "Q", 1, 2, 2, ("K", "A"), True, True, 1, 0, 1, False),
        RoundResolution(2, "K", 1, 3, 2, ("Q", "A"), True, True, 1, 1, 2, False),
        RoundResolution(3, "A", 1, 2, 3, ("Q", "K", "K"), True, True, 1, 2, 3, False),
        RoundResolution(4, "Q", 1, 3, 1, ("Q",), False, False, 3, 0, 1, False),
        RoundResolution(5, "A", 1, 0, 1, ("A",), False, False, 0, 0, 1, False),
    )

    past_actions = (
        PublicEvent(1, 1, 1, "PLAY", 2, "Q", 3),
        PublicEvent(2, 1, 1, "PLAY", 2, "K", 3),
        PublicEvent(3, 1, 1, "PLAY", 3, "A", 2),
        PublicEvent(4, 1, 1, "PLAY", 1, "Q", 4),
        PublicEvent(5, 1, 1, "PLAY", 1, "A", 4),
    )

    obs_a = PlayerObservation(
        round_id=6,
        step_id=2,
        observer_seat=0,
        target_rank=target_rank,
        own_hand=own_hand,
        own_played_cards_this_round=(),
        players_public_status=statuses,
        last_claim=claim,
        current_round_history=cur_hist,
        past_rounds_history=hist_a_res,
        legal_actions=tuple(actions),
        past_public_actions=past_actions,
        ablation_mode="full",
    )
    obs_b = PlayerObservation(
        round_id=6,
        step_id=2,
        observer_seat=0,
        target_rank=target_rank,
        own_hand=own_hand,
        own_played_cards_this_round=(),
        players_public_status=statuses,
        last_claim=claim,
        current_round_history=cur_hist,
        past_rounds_history=hist_b_res,
        legal_actions=tuple(actions),
        past_public_actions=past_actions,
        ablation_mode="full",
    )
    return obs_a, obs_b


class _TeacherPolicyBot(BaseAgent):
    """Wraps a rollout teacher directly as a playing agent for head-to-head teacher validation."""

    def __init__(self, name: str, evaluator: Callable[[PlayerObservation, int], dict[str, float]], seed: int = 42) -> None:
        super().__init__(name=name, style_tag=name)
        self.evaluator = evaluator
        self.seed = seed

    def select_action(self, obs: PlayerObservation) -> tuple[CandidateAction, AgentDecisionRecord]:
        if len(obs.legal_actions) == 1:
            act = obs.legal_actions[0]
            return act, AgentDecisionRecord(rationale="forced")
        vals = self.evaluator(obs, _derive_seed(self.seed, obs.round_id, obs.step_id, obs.observer_seat))
        best = max(obs.legal_actions, key=lambda a: (vals[a.action_id], a.action_type, a.cards))
        return best, AgentDecisionRecord(rationale=f"teacher_val={vals[best.action_id]:.3f}")


def _run_paired_seat_rotated_eval(
    num_blocks: int,
    base_seed: int,
    make_opponents: Callable[[int], list[Callable[[int], BaseAgent]]],
    candidate_factories: dict[str, tuple[Callable[[int], BaseAgent], str]],
) -> dict[str, dict[str, Any]]:
    """Seat-rotated paired evaluation across shared block seeds & opponent orders."""
    stats: dict[str, dict[str, Any]] = {
        name: {"wins": 0, "games": 0, "challenges": 0, "challenge_hits": 0, "bluffs": 0, "bluffs_caught": 0, "plays": 0}
        for name in candidate_factories
    }
    for block in range(num_blocks):
        opp_makers = make_opponents(block)
        perm_rng = random.Random(_derive_seed(base_seed, "perm", block))
        order = [0, 1, 2]
        perm_rng.shuffle(order)
        ordered_makers = [opp_makers[i] for i in order]
        for seat in range(4):
            env_seed = _derive_seed(base_seed, "block_env", block, seat)
            for name, (cand_maker, ablation) in candidate_factories.items():
                agents: list[BaseAgent] = []
                opp_i = 0
                for s in range(4):
                    if s == seat:
                        agents.append(cand_maker(_derive_seed(base_seed, "cand", block, seat)))
                    else:
                        agents.append(ordered_makers[opp_i](_derive_seed(base_seed, "opp", block, seat, s)))
                        opp_i += 1
                engine = LiarsBarEngine(game_id=f"eval_{name}_{block}_{seat}", game_seed=env_seed)
                while not engine.is_game_over():
                    turn = engine.current_turn_seat
                    obs = engine.get_observation(turn)
                    if turn == seat and ablation != "full":
                        obs = apply_history_ablation(obs, ablation)
                    act, rec = agents[turn].select_action(obs)
                    res = engine.step(act, rec)
                    if turn == seat:
                        if act.action_type == "CHALLENGE" and len(obs.legal_actions) > 1:
                            stats[name]["challenges"] += 1
                            if res is not None and res.challenge_succeeded:
                                stats[name]["challenge_hits"] += 1
                        elif act.action_type == "PLAY":
                            stats[name]["plays"] += 1
                            last_step = engine.step_truth_logs[-1]
                            is_bluff = not all(c == obs.target_rank or c == "JOKER" for c in act.cards)
                            if is_bluff:
                                stats[name]["bluffs"] += 1
                    elif res is not None and res.claimer_seat == seat and res.was_bluff:
                        stats[name]["bluffs_caught"] += 1

                stats[name]["games"] += 1
                if engine.winner_seat == seat:
                    stats[name]["wins"] += 1

    summary: dict[str, dict[str, Any]] = {}
    for name, s in stats.items():
        summary[name] = {
            "win_rate": round(s["wins"] / max(1, s["games"]), 4),
            "wins": s["wins"],
            "games": s["games"],
            "challenge_hit_rate": round(s["challenge_hits"] / s["challenges"], 4) if s["challenges"] else None,
            "challenges": s["challenges"],
            "bluff_rate": round(s["bluffs"] / max(1, s["plays"]), 4),
            "caught_when_bluffing_rate": round(s["bluffs_caught"] / s["bluffs"], 4) if s["bluffs"] else None,
        }
    return summary


def _run_teacher_block(args: tuple[int, int, list[float]]) -> dict[str, dict[str, int]]:
    block, base_seed, cal_weights = args
    cal = CalibratedPosteriorBelief()
    cal.weights = list(cal_weights)
    opp_makers = [
        lambda s: ParameterizedArchetypeBot(UNSEEN_MIRROR_SINGLE_BLUFFER, seed=s),
        lambda s: ParameterizedArchetypeBot(UNSEEN_MIRROR_MULTI_BLUFFER, seed=s),
        lambda s: CardCounterBot(seed=s),
    ]
    perm_rng = random.Random(_derive_seed(base_seed, "perm", block))
    order = [0, 1, 2]
    perm_rng.shuffle(order)
    ordered_makers = [opp_makers[i] for i in order]
    cand_factories: dict[str, tuple[Callable[[int], BaseAgent], str]] = {
        "v2_blind_teacher_policy": (
            lambda s: _TeacherPolicyBot(
                "v2_teacher",
                lambda obs, rseed: evaluate_observation_action_values(obs, num_rollouts=10, rollout_seed=rseed),
                seed=s,
            ),
            "full",
        ),
        "v4_posterior_teacher_no_history": (
            lambda s: _TeacherPolicyBot(
                "v4_teacher_no_hist",
                lambda obs, rseed: evaluate_posterior_teacher_values(
                    obs, num_rollouts=10, rollout_seed=rseed, belief_fn=cal.predict
                ),
                seed=s,
            ),
            "last_action_only",
        ),
        "v4_posterior_teacher_full_history": (
            lambda s: _TeacherPolicyBot(
                "v4_teacher_full",
                lambda obs, rseed: evaluate_posterior_teacher_values(
                    obs, num_rollouts=10, rollout_seed=rseed, belief_fn=cal.predict
                ),
                seed=s,
            ),
            "full",
        ),
    }
    stats = {
        name: {"wins": 0, "games": 0, "challenges": 0, "challenge_hits": 0, "bluffs": 0, "bluffs_caught": 0, "plays": 0}
        for name in cand_factories
    }
    for seat in range(4):
        env_seed = _derive_seed(base_seed, "block_env", block, seat)
        for name, (cand_maker, ablation) in cand_factories.items():
            agents: list[BaseAgent] = []
            opp_i = 0
            for s in range(4):
                if s == seat:
                    agents.append(cand_maker(_derive_seed(base_seed, "cand", block, seat)))
                else:
                    agents.append(ordered_makers[opp_i](_derive_seed(base_seed, "opp", block, seat, s)))
                    opp_i += 1
            engine = LiarsBarEngine(game_id=f"teval_{name}_{block}_{seat}", game_seed=env_seed)
            while not engine.is_game_over():
                turn = engine.current_turn_seat
                obs = engine.get_observation(turn)
                if turn == seat and ablation != "full":
                    obs = apply_history_ablation(obs, ablation)
                act, rec = agents[turn].select_action(obs)
                res = engine.step(act, rec)
                if turn == seat:
                    if act.action_type == "CHALLENGE" and len(obs.legal_actions) > 1:
                        stats[name]["challenges"] += 1
                        if res is not None and res.challenge_succeeded:
                            stats[name]["challenge_hits"] += 1
                    elif act.action_type == "PLAY":
                        stats[name]["plays"] += 1
                        if not all(c == obs.target_rank or c == "JOKER" for c in act.cards):
                            stats[name]["bluffs"] += 1
                elif res is not None and res.claimer_seat == seat and res.was_bluff:
                    stats[name]["bluffs_caught"] += 1
            stats[name]["games"] += 1
            if engine.winner_seat == seat:
                stats[name]["wins"] += 1
    return stats


def _run_parallel_teacher_eval(num_blocks: int, base_seed: int, cal_weights: list[float]) -> dict[str, dict[str, Any]]:
    tasks = [(b, base_seed, cal_weights) for b in range(num_blocks)]
    merged: dict[str, dict[str, int]] = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for part in pool.map(_run_teacher_block, tasks):
            for name, s in part.items():
                if name not in merged:
                    merged[name] = dict(s)
                else:
                    for k, v in s.items():
                        merged[name][k] += v
    summary: dict[str, dict[str, Any]] = {}
    for name, s in merged.items():
        summary[name] = {
            "win_rate": round(s["wins"] / max(1, s["games"]), 4),
            "wins": s["wins"],
            "games": s["games"],
            "challenge_hit_rate": round(s["challenge_hits"] / s["challenges"], 4) if s["challenges"] else None,
            "challenges": s["challenges"],
            "bluff_rate": round(s["bluffs"] / max(1, s["plays"]), 4),
            "caught_when_bluffing_rate": round(s["bluffs_caught"] / s["bluffs"], 4) if s["bluffs"] else None,
        }
    return summary


def main() -> None:
    t0 = time.time()
    print("=== v4 Cross-Round Online Adaptation Local Validation ===\n")

    # ------------------------------------------------------------------
    # 1. Belief Predictability & Round-Progression Curve
    # ------------------------------------------------------------------
    print("[1/4] Generating v4 diverse-pool train games & unseen probe sets...")
    def train_lineup(idx: int) -> list[BaseAgent]:
        rng = random.Random(_derive_seed(20261124, "train_lineup", idx))
        picks = [rng.choice(V4_TRAIN_ARCHETYPES) for _ in range(4)]
        return [ParameterizedArchetypeBot(spec, seed=_derive_seed(20261124, "tr", idx, s)) for s, spec in enumerate(picks)]

    def heldout_diverse_lineup(idx: int) -> list[BaseAgent]:
        rng = random.Random(_derive_seed(20261125, "test_lineup", idx))
        picks = [rng.choice(V4_TRAIN_ARCHETYPES) for _ in range(4)]
        return [ParameterizedArchetypeBot(spec, seed=_derive_seed(20261125, "te", idx, s)) for s, spec in enumerate(picks)]

    def mirror_tell_lineup(idx: int) -> list[BaseAgent]:
        return [
            PosteriorAdaptiveBot(seed=_derive_seed(20261126, "m0", idx)),
            ParameterizedArchetypeBot(UNSEEN_MIRROR_SINGLE_BLUFFER, seed=_derive_seed(20261126, "m1", idx)),
            ParameterizedArchetypeBot(UNSEEN_MIRROR_MULTI_BLUFFER, seed=_derive_seed(20261126, "m2", idx)),
            CardCounterBot(seed=_derive_seed(20261126, "m3", idx)),
        ]

    def switcher_lineup(idx: int) -> list[BaseAgent]:
        return [
            PosteriorAdaptiveBot(seed=_derive_seed(20261127, "s0", idx)),
            MidGameSwitcherBot(UNSEEN_MIRROR_MULTI_BLUFFER, UNSEEN_MIRROR_SINGLE_BLUFFER, switch_round=4, seed=_derive_seed(20261127, "s1", idx), name="switcher_m2s"),
            MidGameSwitcherBot(UNSEEN_MIRROR_SINGLE_BLUFFER, UNSEEN_MIRROR_MULTI_BLUFFER, switch_round=4, seed=_derive_seed(20261127, "s2", idx), name="switcher_s2m"),
            ChameleonAdaptiveBot(seed=_derive_seed(20261127, "s3", idx)),
        ]

    train_replays, train_belief = _generate_games_for_belief(140, 20261124, train_lineup)
    _, diverse_test_belief = _generate_games_for_belief(60, 20261125, heldout_diverse_lineup)
    _, mirror_test_belief = _generate_games_for_belief(80, 20261126, mirror_tell_lineup)
    _, switcher_test_belief = _generate_games_for_belief(80, 20261127, switcher_lineup)

    v3_flat_clf = SmallClassifierBot(name="v3_flat_classifier")
    v3_flat_clf.fit_from_replays(train_replays, learning_rate=0.05, epochs=15)

    v4_no_hist_clf = CountHistoryLogisticClassifier(include_history=False)
    v4_no_hist_clf.fit(train_belief, lr=0.06, epochs=25)

    v4_count_hist_clf = CountHistoryLogisticClassifier(include_history=True)
    v4_count_hist_clf.fit(train_belief, lr=0.06, epochs=25)

    v4_cal_post = CalibratedPosteriorBelief()
    v4_cal_post.fit(train_belief, lr=0.06, epochs=25)

    belief_predictors: dict[str, Callable[[PlayerObservation], float]] = {
        "v3_flat_classifier": lambda o: float(v3_flat_clf.predict_bluff_prob(o) or 0.5),
        "v4_no_history_clf": v4_no_hist_clf.predict,
        "v4_count_history_clf": v4_count_hist_clf.predict,
        "v4_exact_bayes_zero_fit": lambda o: float(posterior_bluff_probability(o) or 0.5),
        "v4_calibrated_posterior": v4_cal_post.predict,
    }

    belief_results: dict[str, Any] = {}
    for split_name, rows in (
        ("heldout_diverse_12", diverse_test_belief),
        ("unseen_mirror_tell", mirror_test_belief),
        ("unseen_midgame_switcher", switcher_test_belief),
    ):
        belief_results[split_name] = {
            mname: _eval_belief_metrics(rows, fn) for mname, fn in belief_predictors.items()
        }
        print(f"\n  Belief Split: {split_name} (n={len(rows)})")
        for mname, res in belief_results[split_name].items():
            b = res["by_round_bucket"]
            print(
                f"    {mname:22s} | LL={res['log_loss']:.4f} Brier={res['brier']:.4f} AUC={res['auc']:.4f} | "
                f"R1-2 LL={b['R1-2']['log_loss']} -> R3-5 LL={b['R3-5']['log_loss']} -> R6+ LL={b['R6+']['log_loss']}"
            )

    # ------------------------------------------------------------------
    # 2. Counterfactual History Sensitivity (v2 Teacher vs v4 Posterior Teacher)
    # ------------------------------------------------------------------
    print("\n[2/4] Running counterfactual history sensitivity test (v2 teacher vs v4 posterior teacher)...")
    scenarios = [
        # (own_hand, target_rank, claimed_count, shots)
        (("Q", "K", "K", "A", "A"), "Q", 2, 0),
        (("Q", "K", "K", "A", "A"), "Q", 2, 2),
        (("K", "K", "A", "A", "A"), "Q", 2, 1),
        (("Q", "Q", "K", "A", "A"), "Q", 2, 1),
        (("Q", "K", "K", "A", "A"), "Q", 3, 1),
        (("K", "K", "A", "A", "A"), "Q", 3, 2),
        (("Q", "K", "K", "A", "A"), "Q", 1, 0),
        (("K", "K", "A", "A", "A"), "Q", 1, 1),
        (("Q", "Q", "K", "A", "A"), "Q", 1, 2),
        (("K", "A", "A", "Q", "Q"), "K", 2, 1),
        (("A", "A", "Q", "Q", "Q"), "K", 2, 0),
        (("K", "A", "A", "Q", "Q"), "K", 1, 1),
    ]
    cf_details = []
    v2_abs_dq, v4_abs_dq = [], []
    v2_signed_dq, v4_signed_dq = [], []
    v2_flips, v4_flips = 0, 0
    for idx, (hand, trank, ccnt, shots) in enumerate(scenarios):
        obs_a, obs_b = _make_counterfactual_pair(hand, trank, ccnt, shots)
        seed = 5000 + idx
        v2_a = evaluate_observation_action_values(obs_a, num_rollouts=16, rollout_seed=seed)
        v2_b = evaluate_observation_action_values(obs_b, num_rollouts=16, rollout_seed=seed)
        v4_a = evaluate_posterior_teacher_values(obs_a, num_rollouts=16, rollout_seed=seed, belief_fn=v4_cal_post.predict)
        v4_b = evaluate_posterior_teacher_values(obs_b, num_rollouts=16, rollout_seed=seed, belief_fn=v4_cal_post.predict)

        # Expected direction of Q(CHALLENGE):
        # If claimed_count >= 2: History B (multi-bluffer) should have MUCH higher Q(CHALLENGE) than History A (multi-truth)
        # If claimed_count == 1: History A (single-bluffer) should have MUCH higher Q(CHALLENGE) than History B (single-truth)
        expected_sign = 1.0 if ccnt >= 2 else -1.0
        dq_v2 = v2_b["A0"] - v2_a["A0"]
        dq_v4 = v4_b["A0"] - v4_a["A0"]
        v2_abs_dq.append(abs(dq_v2))
        v4_abs_dq.append(abs(dq_v4))
        v2_signed_dq.append(expected_sign * dq_v2)
        v4_signed_dq.append(expected_sign * dq_v4)

        best_v2_a = max(v2_a, key=lambda k: v2_a[k])
        best_v2_b = max(v2_b, key=lambda k: v2_b[k])
        best_v4_a = max(v4_a, key=lambda k: v4_a[k])
        best_v4_b = max(v4_b, key=lambda k: v4_b[k])
        if best_v2_a != best_v2_b:
            v2_flips += 1
        if best_v4_a != best_v4_b:
            v4_flips += 1
        cf_details.append({
            "scenario_idx": idx,
            "claimed_count": ccnt,
            "shots": shots,
            "v2_q_ch_A": round(v2_a["A0"], 4),
            "v2_q_ch_B": round(v2_b["A0"], 4),
            "v4_q_ch_A": round(v4_a["A0"], 4),
            "v4_q_ch_B": round(v4_b["A0"], 4),
            "v2_flipped": best_v2_a != best_v2_b,
            "v4_flipped": best_v4_a != best_v4_b,
            "v4_p_bluff_A": round(posterior_bluff_probability(obs_a) or 0.0, 4),
            "v4_p_bluff_B": round(posterior_bluff_probability(obs_b) or 0.0, 4),
        })

    cf_summary = {
        "num_pairs": len(scenarios),
        "v2_teacher": {
            "mean_abs_dq_challenge": round(sum(v2_abs_dq) / len(v2_abs_dq), 4),
            "mean_aligned_dq_challenge": round(sum(v2_signed_dq) / len(v2_signed_dq), 4),
            "best_action_flip_rate": round(v2_flips / len(scenarios), 4),
            "flips": v2_flips,
        },
        "v4_posterior_teacher": {
            "mean_abs_dq_challenge": round(sum(v4_abs_dq) / len(v4_abs_dq), 4),
            "mean_aligned_dq_challenge": round(sum(v4_signed_dq) / len(v4_signed_dq), 4),
            "best_action_flip_rate": round(v4_flips / len(scenarios), 4),
            "flips": v4_flips,
        },
        "sample_prompt_summary": render_count_conditioned_summary(_make_counterfactual_pair(("Q", "K", "K", "A", "A"), "Q", 2, 1)[0]),
        "pairs": cf_details,
    }
    print(
        f"  v2_teacher:           mean_aligned_dQ={cf_summary['v2_teacher']['mean_aligned_dq_challenge']:+.4f}, "
        f"flip_rate={cf_summary['v2_teacher']['best_action_flip_rate']*100:.1f}% ({v2_flips}/{len(scenarios)})"
    )
    print(
        f"  v4_posterior_teacher: mean_aligned_dQ={cf_summary['v4_posterior_teacher']['mean_aligned_dq_challenge']:+.4f}, "
        f"flip_rate={cf_summary['v4_posterior_teacher']['best_action_flip_rate']*100:.1f}% ({v4_flips}/{len(scenarios)})"
    )

    # ------------------------------------------------------------------
    # 3. Paired Seat-Rotated Gameplay & History Ablation (45 blocks = 180 games each)
    # ------------------------------------------------------------------
    print("\n[3/4] Running paired seat-rotated gameplay & history ablation (45 blocks = 180 games/model)...")
    cand_factories: dict[str, tuple[Callable[[int], BaseAgent], str]] = {
        "rule_counter": (lambda s: CardCounterBot(seed=s), "full"),
        "v3_small_classifier": (lambda s: SmallClassifierBot(bluff_weights=v3_flat_clf.bluff_weights), "full"),
        "v4_posterior_no_history": (lambda s: PosteriorAdaptiveBot(seed=s, belief_fn=v4_cal_post.predict), "last_action_only"),
        "v4_posterior_full_history": (lambda s: PosteriorAdaptiveBot(seed=s, belief_fn=v4_cal_post.predict), "full"),
    }

    gameplay_mirror = _run_paired_seat_rotated_eval(
        num_blocks=45,
        base_seed=777001,
        make_opponents=lambda b: [
            lambda s: ParameterizedArchetypeBot(UNSEEN_MIRROR_SINGLE_BLUFFER, seed=s),
            lambda s: ParameterizedArchetypeBot(UNSEEN_MIRROR_MULTI_BLUFFER, seed=s),
            lambda s: CardCounterBot(seed=s),
        ],
        candidate_factories=cand_factories,
    )

    def _diverse_block_opps(block: int) -> list[Callable[[int], BaseAgent]]:
        rng = random.Random(_derive_seed(777002, "div_opps", block))
        picks = rng.sample(list(V4_TRAIN_ARCHETYPES), 3)
        return [lambda s, spec=spec: ParameterizedArchetypeBot(spec, seed=s) for spec in picks]

    gameplay_diverse = _run_paired_seat_rotated_eval(
        num_blocks=45,
        base_seed=777002,
        make_opponents=_diverse_block_opps,
        candidate_factories=cand_factories,
    )

    for pname, res in (("unseen_mirror_tell_pool", gameplay_mirror), ("diverse_13_pool", gameplay_diverse)):
        print(f"\n  Gameplay Pool: {pname} (180 games/model)")
        for mname, info in res.items():
            print(
                f"    {mname:26s} | WinRate={info['win_rate']*100:5.1f}% ({info['wins']:3d}/{info['games']}) | "
                f"ChallHit={str(round(info['challenge_hit_rate']*100,1))+'%' if info['challenge_hit_rate'] is not None else 'N/A':>6s} | "
                f"BluffRate={info['bluff_rate']*100:4.1f}% | Caught={str(round(info['caught_when_bluffing_rate']*100,1))+'%' if info['caught_when_bluffing_rate'] is not None else 'N/A':>5s}"
            )
        lift = (res["v4_posterior_full_history"]["win_rate"] - res["v4_posterior_no_history"]["win_rate"]) * 100
        print(f"    >>> History Ablation Lift (full - last_action_only): {lift:+.2f}%")

    # ------------------------------------------------------------------
    # 4. Direct Teacher-as-Policy Comparison on Unseen Mirror-Tell Pool (12 blocks = 48 games, parallelized)
    # ------------------------------------------------------------------
    print("\n[4/4] Running direct Teacher-as-Policy evaluation (12 blocks = 48 games, 10 rollouts/step, parallel)...")
    cal_weights = list(v4_cal_post.weights)
    teacher_eval = _run_parallel_teacher_eval(num_blocks=12, base_seed=888001, cal_weights=cal_weights)
    for mname, info in teacher_eval.items():
        print(
            f"    {mname:28s} | WinRate={info['win_rate']*100:5.1f}% ({info['wins']:2d}/{info['games']}) | "
            f"ChallHit={str(round(info['challenge_hit_rate']*100,1))+'%' if info['challenge_hit_rate'] is not None else 'N/A':>6s} | "
            f"BluffRate={info['bluff_rate']*100:4.1f}%"
        )

    out_path = Path("outputs/validation/v4_local_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "elapsed_sec": round(time.time() - t0, 2),
        "belief_validation": belief_results,
        "counterfactual_teacher_sensitivity": cf_summary,
        "gameplay_paired_evaluation": {
            "unseen_mirror_tell_pool": gameplay_mirror,
            "diverse_13_pool": gameplay_diverse,
        },
        "direct_teacher_policy_eval": teacher_eval,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved full validation report to {out_path} ({payload['elapsed_sec']}s).")


if __name__ == "__main__":
    main()

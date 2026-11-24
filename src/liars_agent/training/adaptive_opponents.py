"""Symmetric parameterized opponent pool and count-conditioned public profiles for v4 adaptation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Tuple

from liars_agent.agents.base import (
    BaseAgent,
    describe_action,
    estimate_heuristic_bluff_belief,
    public_opponent_profile,
    split_play_actions,
)
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.types import AgentDecisionRecord, CandidateAction, is_card_truthful


def _step_rng(base_seed: int, obs: PlayerObservation) -> random.Random:
    key = f"{base_seed}:{obs.round_id}:{obs.step_id}:{obs.observer_seat}"
    seed_int = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed_int)


@dataclass(frozen=True)
class ArchetypeSpec:
    """Declarative behavioral archetype for symmetric opponent pool and Bayesian inference."""

    archetype_id: str
    description: str
    # Probability of choosing a bluff play when holding >=1 truthful card, conditioned on count bucket
    single_bluff_when_has_truth: float
    multi_bluff_when_has_truth: float
    # Preference for multi-card (2~3) vs single-card (1) plays
    truth_multi_bias: float
    bluff_multi_bias: float
    # Challenge behavior
    challenge_threshold: float
    single_challenge_base: float
    multi_challenge_base: float
    # Risk modulation when shots_taken >= 2
    high_risk_bluff_multiplier: float = 1.0
    # Marginal likelihood priors used by the Bayesian posterior (observational likelihood model)
    obs_p_multi_play: float = 0.45
    obs_p_voluntary_challenge: float = 0.25
    obs_p_bluff_given_single: float = 0.45
    obs_p_bluff_given_multi: float = 0.45


# 13 diverse, symmetric, non-suicidal training archetypes covering opposing count-tell directions,
# count-specific challenge hunters, and risk sensitivities.
V4_TRAIN_ARCHETYPES: tuple[ArchetypeSpec, ...] = (
    ArchetypeSpec(
        archetype_id="single_tell_trapper",
        description="单张爱诈、2~3张全真（有真牌优先打2~3张，单张多为诈唬；偏抓多张）",
        single_bluff_when_has_truth=0.35,
        multi_bluff_when_has_truth=0.02,
        truth_multi_bias=0.92,
        bluff_multi_bias=0.04,
        challenge_threshold=0.60,
        single_challenge_base=0.14,
        multi_challenge_base=0.58,
        obs_p_multi_play=0.62,
        obs_p_voluntary_challenge=0.28,
        obs_p_bluff_given_single=0.86,
        obs_p_bluff_given_multi=0.05,
    ),
    ArchetypeSpec(
        archetype_id="multi_tell_bluffer",
        description="镜像风格：2~3张高频诈唬、1张极度诚实（偏抓单张）",
        single_bluff_when_has_truth=0.03,
        multi_bluff_when_has_truth=0.38,
        truth_multi_bias=0.08,
        bluff_multi_bias=0.92,
        challenge_threshold=0.60,
        single_challenge_base=0.58,
        multi_challenge_base=0.14,
        obs_p_multi_play=0.42,
        obs_p_voluntary_challenge=0.28,
        obs_p_bluff_given_single=0.08,
        obs_p_bluff_given_multi=0.86,
    ),
    ArchetypeSpec(
        archetype_id="ultra_honest_rock",
        description="极度诚实岩石：有真牌几乎必真，只在0真牌时被迫单张诈唬",
        single_bluff_when_has_truth=0.02,
        multi_bluff_when_has_truth=0.02,
        truth_multi_bias=0.55,
        bluff_multi_bias=0.05,
        challenge_threshold=0.70,
        single_challenge_base=0.12,
        multi_challenge_base=0.28,
        obs_p_multi_play=0.45,
        obs_p_voluntary_challenge=0.16,
        obs_p_bluff_given_single=0.22,
        obs_p_bluff_given_multi=0.05,
    ),
    ArchetypeSpec(
        archetype_id="chaotic_maniac",
        description="激进诈唬手：单张与多张都高频诈唬，极少主动质疑",
        single_bluff_when_has_truth=0.68,
        multi_bluff_when_has_truth=0.68,
        truth_multi_bias=0.60,
        bluff_multi_bias=0.60,
        challenge_threshold=0.82,
        single_challenge_base=0.08,
        multi_challenge_base=0.14,
        obs_p_multi_play=0.60,
        obs_p_voluntary_challenge=0.10,
        obs_p_bluff_given_single=0.78,
        obs_p_bluff_given_multi=0.78,
    ),
    ArchetypeSpec(
        archetype_id="paranoid_sheriff",
        description="多张警长：极高多张质疑倾向，对单张相对宽容，自己出牌稳健",
        single_bluff_when_has_truth=0.12,
        multi_bluff_when_has_truth=0.06,
        truth_multi_bias=0.45,
        bluff_multi_bias=0.12,
        challenge_threshold=0.52,
        single_challenge_base=0.16,
        multi_challenge_base=0.76,
        obs_p_multi_play=0.42,
        obs_p_voluntary_challenge=0.44,
        obs_p_bluff_given_single=0.28,
        obs_p_bluff_given_multi=0.10,
    ),
    ArchetypeSpec(
        archetype_id="single_hunter",
        description="单张猎手：专抓单张声称，放行多张，自己偏多张真牌",
        single_bluff_when_has_truth=0.14,
        multi_bluff_when_has_truth=0.06,
        truth_multi_bias=0.75,
        bluff_multi_bias=0.18,
        challenge_threshold=0.52,
        single_challenge_base=0.76,
        multi_challenge_base=0.14,
        obs_p_multi_play=0.60,
        obs_p_voluntary_challenge=0.42,
        obs_p_bluff_given_single=0.30,
        obs_p_bluff_given_multi=0.10,
    ),
    ArchetypeSpec(
        archetype_id="balanced_mixed",
        description="均衡混合型：单张与多张均保持中等真假混合，不易被单维度读牌",
        single_bluff_when_has_truth=0.28,
        multi_bluff_when_has_truth=0.26,
        truth_multi_bias=0.50,
        bluff_multi_bias=0.45,
        challenge_threshold=0.62,
        single_challenge_base=0.24,
        multi_challenge_base=0.32,
        obs_p_multi_play=0.48,
        obs_p_voluntary_challenge=0.26,
        obs_p_bluff_given_single=0.42,
        obs_p_bluff_given_multi=0.36,
    ),
    ArchetypeSpec(
        archetype_id="cautious_single_truth",
        description="慢节奏单张诚实型：偏好打 1 张真牌，极少多出，稳健防守",
        single_bluff_when_has_truth=0.05,
        multi_bluff_when_has_truth=0.25,
        truth_multi_bias=0.10,
        bluff_multi_bias=0.10,
        challenge_threshold=0.68,
        single_challenge_base=0.16,
        multi_challenge_base=0.34,
        obs_p_multi_play=0.14,
        obs_p_voluntary_challenge=0.20,
        obs_p_bluff_given_single=0.20,
        obs_p_bluff_given_multi=0.35,
    ),
    ArchetypeSpec(
        archetype_id="pressure_dumper",
        description="多张倾泻施压型：有牌就爱打2~3张（真假皆有），制造高张数压力",
        single_bluff_when_has_truth=0.18,
        multi_bluff_when_has_truth=0.48,
        truth_multi_bias=0.88,
        bluff_multi_bias=0.88,
        challenge_threshold=0.72,
        single_challenge_base=0.14,
        multi_challenge_base=0.24,
        obs_p_multi_play=0.84,
        obs_p_voluntary_challenge=0.18,
        obs_p_bluff_given_single=0.35,
        obs_p_bluff_given_multi=0.60,
    ),
    ArchetypeSpec(
        archetype_id="coward_under_fire",
        description="中枪变怂型：0~1枪高频诈唬，中2+枪后大幅收敛只打真牌",
        single_bluff_when_has_truth=0.52,
        multi_bluff_when_has_truth=0.42,
        truth_multi_bias=0.50,
        bluff_multi_bias=0.50,
        challenge_threshold=0.64,
        single_challenge_base=0.20,
        multi_challenge_base=0.32,
        high_risk_bluff_multiplier=0.12,
        obs_p_multi_play=0.50,
        obs_p_voluntary_challenge=0.24,
        obs_p_bluff_given_single=0.46,
        obs_p_bluff_given_multi=0.38,
    ),
    ArchetypeSpec(
        archetype_id="desperate_gambler",
        description="逆风赌徒型：低风险偏诚实，中2+枪后变得激进诈唬",
        single_bluff_when_has_truth=0.10,
        multi_bluff_when_has_truth=0.10,
        truth_multi_bias=0.45,
        bluff_multi_bias=0.65,
        challenge_threshold=0.56,
        single_challenge_base=0.28,
        multi_challenge_base=0.44,
        high_risk_bluff_multiplier=3.5,
        obs_p_multi_play=0.52,
        obs_p_voluntary_challenge=0.34,
        obs_p_bluff_given_single=0.38,
        obs_p_bluff_given_multi=0.42,
    ),
    ArchetypeSpec(
        archetype_id="mixed_card_trapper",
        description="单张诚实+多张混出陷阱型：1张多为真牌，2~3张常真假混出",
        single_bluff_when_has_truth=0.06,
        multi_bluff_when_has_truth=0.48,
        truth_multi_bias=0.20,
        bluff_multi_bias=0.84,
        challenge_threshold=0.62,
        single_challenge_base=0.42,
        multi_challenge_base=0.18,
        obs_p_multi_play=0.52,
        obs_p_voluntary_challenge=0.28,
        obs_p_bluff_given_single=0.14,
        obs_p_bluff_given_multi=0.70,
    ),
    ArchetypeSpec(
        archetype_id="card_counter_classic",
        description="经典记牌稳健型（对齐 v2/v3 CardCounter 行为分布）",
        single_bluff_when_has_truth=0.22,
        multi_bluff_when_has_truth=0.06,
        truth_multi_bias=0.28,
        bluff_multi_bias=0.08,
        challenge_threshold=0.64,
        single_challenge_base=0.18,
        multi_challenge_base=0.36,
        high_risk_bluff_multiplier=0.30,
        obs_p_multi_play=0.24,
        obs_p_voluntary_challenge=0.22,
        obs_p_bluff_given_single=0.38,
        obs_p_bluff_given_multi=0.14,
    ),
)

# Unseen probe archetypes (competitive, non-suicidal mirror tells + asymmetric bucket hunters)
UNSEEN_MIRROR_SINGLE_BLUFFER = ArchetypeSpec(
    archetype_id="unseen_mirror_single_bluffer",
    description="未见镜像竞技探针 A：2~3张几乎必真、1张几乎必诈，且专抓多张声称",
    single_bluff_when_has_truth=0.32,
    multi_bluff_when_has_truth=0.01,
    truth_multi_bias=0.96,
    bluff_multi_bias=0.02,
    challenge_threshold=0.58,
    single_challenge_base=0.10,
    multi_challenge_base=0.72,
    obs_p_multi_play=0.65,
    obs_p_voluntary_challenge=0.32,
    obs_p_bluff_given_single=0.92,
    obs_p_bluff_given_multi=0.03,
)

UNSEEN_MIRROR_MULTI_BLUFFER = ArchetypeSpec(
    archetype_id="unseen_mirror_multi_bluffer",
    description="未见镜像竞技探针 B：1张几乎必真、2~3张几乎必诈，且专抓单张声称",
    single_bluff_when_has_truth=0.01,
    multi_bluff_when_has_truth=0.32,
    truth_multi_bias=0.04,
    bluff_multi_bias=0.96,
    challenge_threshold=0.58,
    single_challenge_base=0.72,
    multi_challenge_base=0.10,
    obs_p_multi_play=0.38,
    obs_p_voluntary_challenge=0.32,
    obs_p_bluff_given_single=0.04,
    obs_p_bluff_given_multi=0.92,
)

ALL_V4_ARCHETYPES: tuple[ArchetypeSpec, ...] = (
    *V4_TRAIN_ARCHETYPES,
    UNSEEN_MIRROR_SINGLE_BLUFFER,
    UNSEEN_MIRROR_MULTI_BLUFFER,
)
ARCHETYPE_BY_ID: dict[str, ArchetypeSpec] = {spec.archetype_id: spec for spec in ALL_V4_ARCHETYPES}


def create_v4_opponent(class_name: str, name: str | None = None, seed: int = 42, **params: Any) -> BaseAgent:
    """Instantiate either a v4 parameterized/switcher opponent or a classic rule bot."""
    if class_name == "ParameterizedArchetypeBot":
        archetype_id = params.pop("archetype_id")
        if params:
            raise TypeError(f"Unexpected ParameterizedArchetypeBot params: {sorted(params)}")
        if archetype_id not in ARCHETYPE_BY_ID:
            raise ValueError(f"Unknown v4 archetype_id: {archetype_id}")
        return ParameterizedArchetypeBot(ARCHETYPE_BY_ID[archetype_id], seed=seed, name=name or archetype_id)
    if class_name == "MidGameSwitcherBot":
        early_id = params.pop("early_archetype_id")
        late_id = params.pop("late_archetype_id")
        switch_round = int(params.pop("switch_round", 4))
        if params:
            raise TypeError(f"Unexpected MidGameSwitcherBot params: {sorted(params)}")
        if early_id not in ARCHETYPE_BY_ID or late_id not in ARCHETYPE_BY_ID:
            raise ValueError(f"Unknown MidGameSwitcherBot archetype: {early_id} / {late_id}")
        return MidGameSwitcherBot(
            ARCHETYPE_BY_ID[early_id],
            ARCHETYPE_BY_ID[late_id],
            switch_round=switch_round,
            seed=seed,
            name=name or f"switcher_{early_id}_to_{late_id}",
        )
    from liars_agent.agents.rule_bots import create_rule_agent
    return create_rule_agent(class_name, name=name, seed=seed, **params)


class ParameterizedArchetypeBot(BaseAgent):
    """Deterministic bot driven by an ArchetypeSpec."""

    def __init__(self, spec: ArchetypeSpec, seed: int = 42, name: str | None = None) -> None:
        super().__init__(name=name or spec.archetype_id, style_tag=spec.archetype_id)
        self.spec = spec
        self.seed = seed

    def select_action(self, obs: PlayerObservation) -> Tuple[CandidateAction, AgentDecisionRecord]:
        rng = _step_rng(self.seed, obs)
        bluff_prob, _, _ = estimate_heuristic_bluff_belief(obs, use_history_adaptation=False)
        challenge_actions = [a for a in obs.legal_actions if a.action_type == "CHALLENGE"]
        truthful, pure_bluff, mixed_bluff = split_play_actions(obs)
        all_bluffs = pure_bluff + mixed_bluff
        other_active = sum(
            1 for p in obs.players_public_status if p.alive and p.hand_count > 0 and p.seat != obs.observer_seat
        )
        non_suicidal_bluffs = [
            a for a in all_bluffs
            if not (len(a.cards) == len(obs.own_hand) and other_active <= 1)
        ]

        if challenge_actions:
            if len(obs.legal_actions) == 1 or (not truthful and not non_suicidal_bluffs):
                return challenge_actions[0], self._record(obs, bluff_prob, challenge_actions[0])
            assert obs.last_claim is not None
            base_prob = (
                self.spec.single_challenge_base
                if obs.last_claim.claimed_count == 1
                else self.spec.multi_challenge_base
            )
            # Count-conditioned challenge sensitivity without reckless blind suicide:
            # hunters with high single/multi_challenge_base lower their threshold sharply on that bucket.
            eff_cut = self.spec.challenge_threshold - 0.42 * (base_prob - 0.25)
            if not truthful:
                eff_cut -= 0.18
            eff_cut = min(0.92, max(0.28, eff_cut))
            if bluff_prob >= eff_cut:
                return challenge_actions[0], self._record(obs, bluff_prob, challenge_actions[0])

        mult = self.spec.high_risk_bluff_multiplier if obs.own_shots_taken >= 2 else 1.0
        p_single_bluff = min(0.98, max(0.0, self.spec.single_bluff_when_has_truth * mult))
        p_multi_bluff = min(0.98, max(0.0, self.spec.multi_bluff_when_has_truth * mult))

        truth_single = [a for a in truthful if len(a.cards) == 1]
        truth_multi = [a for a in truthful if len(a.cards) >= 2]
        bluff_pool = non_suicidal_bluffs or all_bluffs
        bluff_single = [a for a in bluff_pool if len(a.cards) == 1]
        bluff_multi = [a for a in bluff_pool if len(a.cards) >= 2]

        # Decide whether to bluff or play truthful when both are available
        if truthful and bluff_pool:
            want_multi_if_truth = bool(truth_multi) and rng.random() < self.spec.truth_multi_bias
            want_multi_if_bluff = bool(bluff_multi) and rng.random() < self.spec.bluff_multi_bias
            eff_bluff_prob = p_multi_bluff if want_multi_if_bluff else p_single_bluff
            if rng.random() < eff_bluff_prob:
                pool = bluff_multi if (want_multi_if_bluff and bluff_multi) else (bluff_single or bluff_pool)
                chosen = rng.choice(pool)
                return chosen, self._record(obs, bluff_prob if obs.last_claim else None, chosen)
            pool = truth_multi if (want_multi_if_truth and truth_multi) else (truth_single or truthful)
            chosen = rng.choice(pool)
            return chosen, self._record(obs, bluff_prob if obs.last_claim else None, chosen)

        if truthful:
            pool = truth_multi if (truth_multi and rng.random() < self.spec.truth_multi_bias) else (truth_single or truthful)
            chosen = rng.choice(pool)
            return chosen, self._record(obs, bluff_prob if obs.last_claim else None, chosen)

        pool = bluff_multi if (bluff_multi and rng.random() < self.spec.bluff_multi_bias) else (bluff_single or bluff_pool)
        chosen = rng.choice(pool) if pool else obs.legal_actions[0]
        return chosen, self._record(obs, bluff_prob if obs.last_claim else None, chosen)

    def _record(
        self, obs: PlayerObservation, belief: float | None, action: CandidateAction
    ) -> AgentDecisionRecord:
        return AgentDecisionRecord(
            belief_bluff_prob=belief,
            belief_source="heuristic" if obs.last_claim else "none",
            estimated_opponent_style="unknown",
            rationale=describe_action(obs, action),
            rationale_source="action_description",
            opponent_profile=public_opponent_profile(obs),
        )


class MidGameSwitcherBot(BaseAgent):
    """Unseen probe bot that switches strategy mid-game at `switch_round`."""

    def __init__(
        self,
        early_spec: ArchetypeSpec,
        late_spec: ArchetypeSpec,
        switch_round: int = 4,
        seed: int = 999,
        name: str = "unseen_midgame_switcher",
    ) -> None:
        super().__init__(name=name, style_tag="midgame_switcher")
        self.early_bot = ParameterizedArchetypeBot(early_spec, seed=seed, name=name)
        self.late_bot = ParameterizedArchetypeBot(late_spec, seed=seed + 10007, name=name)
        self.switch_round = switch_round

    def select_action(self, obs: PlayerObservation) -> Tuple[CandidateAction, AgentDecisionRecord]:
        active = self.early_bot if obs.round_id < self.switch_round else self.late_bot
        return active.select_action(obs)


def extract_count_conditioned_profile(obs: PlayerObservation, recent_window: int = 3) -> dict[str, dict[str, Any]]:
    """Strictly public, count-conditioned per-seat history statistics (no hidden identities)."""
    events = (*obs.past_public_actions, *obs.current_round_history)
    # Track per-seat voluntary challenge opportunities & actions conditioned on preceding claim size
    by_seat_ch = {
        p.seat: {"s_opp": 0, "s_ch": 0, "m_opp": 0, "m_ch": 0}
        for p in obs.players_public_status
    }
    prev_round: int | None = None
    prev_claimed_count: int | None = None
    for ev in events:
        if ev.round_id != prev_round:
            prev_round = ev.round_id
            prev_claimed_count = None
        if ev.step_id > 1 and not ev.forced_challenge and prev_claimed_count is not None:
            bucket = by_seat_ch.get(ev.actor_seat)
            if bucket is not None:
                if prev_claimed_count == 1:
                    bucket["s_opp"] += 1
                    if ev.action_type == "CHALLENGE":
                        bucket["s_ch"] += 1
                else:
                    bucket["m_opp"] += 1
                    if ev.action_type == "CHALLENGE":
                        bucket["m_ch"] += 1
        prev_claimed_count = ev.claimed_count if ev.action_type == "PLAY" else None

    profile: dict[str, dict[str, Any]] = {}
    for player in obs.players_public_status:
        seat = player.seat
        own_events = [e for e in events if e.actor_seat == seat]
        plays = [e for e in own_events if e.action_type == "PLAY"]
        single_plays = sum(1 for e in plays if e.claimed_count == 1)
        multi_plays = sum(1 for e in plays if e.claimed_count >= 2)
        opps = sum(1 for e in own_events if e.step_id > 1 and not e.forced_challenge)
        challs = sum(1 for e in own_events if e.action_type == "CHALLENGE" and not e.forced_challenge)
        chall_hits = sum(
            1 for r in obs.past_rounds_history
            if r.challenger_seat == seat and not r.forced_challenge and r.challenge_succeeded
        )

        reveals = [r for r in obs.past_rounds_history if r.claimer_seat == seat]
        single_reveals = [r for r in reveals if r.claimed_count == 1]
        multi_reveals = [r for r in reveals if r.claimed_count >= 2]
        recent_reveals = reveals[-recent_window:]
        ch_b = by_seat_ch[seat]

        profile[str(seat)] = {
            "total_plays": len(plays),
            "single_plays": single_plays,
            "multi_plays": multi_plays,
            "multi_play_rate": multi_plays / len(plays) if plays else None,
            "voluntary_challenges": challs,
            "challenge_opportunities": opps,
            "voluntary_challenge_rate": challs / opps if opps else None,
            "single_challenges": ch_b["s_ch"],
            "single_challenge_opps": ch_b["s_opp"],
            "multi_challenges": ch_b["m_ch"],
            "multi_challenge_opps": ch_b["m_opp"],
            "challenge_hits": chall_hits,
            "reveals_total": len(reveals),
            "bluffs_total": sum(1 for r in reveals if r.was_bluff),
            "single_reveals": len(single_reveals),
            "single_bluffs": sum(1 for r in single_reveals if r.was_bluff),
            "multi_reveals": len(multi_reveals),
            "multi_bluffs": sum(1 for r in multi_reveals if r.was_bluff),
            "recent_reveals": len(recent_reveals),
            "recent_bluffs": sum(1 for r in recent_reveals if r.was_bluff),
        }
    return profile


def render_count_conditioned_summary(obs: PlayerObservation) -> str:
    """Compact deterministic Chinese summary suitable for v4 prompts (public info only)."""
    prof = extract_count_conditioned_profile(obs)
    lines = ["各座位公开历史统计（按声称张数分桶）:"]
    for seat_str in sorted(prof.keys(), key=int):
        s = prof[seat_str]
        if s["total_plays"] == 0 and s["voluntary_challenges"] == 0:
            lines.append(f"  - Seat {seat_str}: 尚无公开历史")
            continue
        s_rev = f"{s['single_bluffs']}诈/{s['single_reveals']}亮" if s["single_reveals"] else "无亮牌"
        m_rev = f"{s['multi_bluffs']}诈/{s['multi_reveals']}亮" if s["multi_reveals"] else "无亮牌"
        if s["challenge_opportunities"]:
            ch_txt = (
                f"{s['voluntary_challenges']}/{s['challenge_opportunities']}次"
                f"(对1张:{s['single_challenges']}/{s['single_challenge_opps']}, "
                f"对2~3张:{s['multi_challenges']}/{s['multi_challenge_opps']}, "
                f"命中{s['challenge_hits']})"
            )
        else:
            ch_txt = "0次"
        lines.append(
            f"  - Seat {seat_str}: 出牌(1张:{s['single_plays']}次, 2~3张:{s['multi_plays']}次) | "
            f"亮牌(1张:{s_rev}, 2~3张:{m_rev}) | 主动质疑:{ch_txt}"
        )
    return "\n".join(lines)

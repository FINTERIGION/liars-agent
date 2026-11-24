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


# 12 diverse, symmetric training archetypes covering opposing count-tell directions,
# challenge styles, and risk sensitivities.
V4_TRAIN_ARCHETYPES: tuple[ArchetypeSpec, ...] = (
    ArchetypeSpec(
        archetype_id="single_tell_trapper",
        description="单张爱诈、2~3张全真（类似当前 LoRA-SFT 牌风）",
        single_bluff_when_has_truth=0.80,
        multi_bluff_when_has_truth=0.04,
        truth_multi_bias=0.92,
        bluff_multi_bias=0.05,
        challenge_threshold=0.62,
        single_challenge_base=0.18,
        multi_challenge_base=0.35,
        obs_p_multi_play=0.65,
        obs_p_voluntary_challenge=0.22,
        obs_p_bluff_given_single=0.88,
        obs_p_bluff_given_multi=0.06,
    ),
    ArchetypeSpec(
        archetype_id="multi_tell_bluffer",
        description="镜像风格：2~3张高频诈唬、1张极度诚实",
        single_bluff_when_has_truth=0.05,
        multi_bluff_when_has_truth=0.82,
        truth_multi_bias=0.12,
        bluff_multi_bias=0.90,
        challenge_threshold=0.72,
        single_challenge_base=0.12,
        multi_challenge_base=0.28,
        obs_p_multi_play=0.60,
        obs_p_voluntary_challenge=0.18,
        obs_p_bluff_given_single=0.10,
        obs_p_bluff_given_multi=0.86,
    ),
    ArchetypeSpec(
        archetype_id="ultra_honest_rock",
        description="极度诚实岩石：有真牌几乎必真，只在0真牌时被迫单张诈唬",
        single_bluff_when_has_truth=0.03,
        multi_bluff_when_has_truth=0.02,
        truth_multi_bias=0.55,
        bluff_multi_bias=0.05,
        challenge_threshold=0.74,
        single_challenge_base=0.10,
        multi_challenge_base=0.25,
        obs_p_multi_play=0.45,
        obs_p_voluntary_challenge=0.15,
        obs_p_bluff_given_single=0.20,
        obs_p_bluff_given_multi=0.06,
    ),
    ArchetypeSpec(
        archetype_id="chaotic_maniac",
        description="全场疯狗：单张与多张都高频诈唬，极少主动质疑",
        single_bluff_when_has_truth=0.78,
        multi_bluff_when_has_truth=0.78,
        truth_multi_bias=0.65,
        bluff_multi_bias=0.65,
        challenge_threshold=0.84,
        single_challenge_base=0.08,
        multi_challenge_base=0.14,
        obs_p_multi_play=0.65,
        obs_p_voluntary_challenge=0.10,
        obs_p_bluff_given_single=0.82,
        obs_p_bluff_given_multi=0.82,
    ),
    ArchetypeSpec(
        archetype_id="paranoid_sheriff",
        description="多疑警长：极高质疑倾向（尤其针对多张），自己出牌偏稳",
        single_bluff_when_has_truth=0.18,
        multi_bluff_when_has_truth=0.10,
        truth_multi_bias=0.45,
        bluff_multi_bias=0.15,
        challenge_threshold=0.46,
        single_challenge_base=0.35,
        multi_challenge_base=0.72,
        obs_p_multi_play=0.40,
        obs_p_voluntary_challenge=0.55,
        obs_p_bluff_given_single=0.32,
        obs_p_bluff_given_multi=0.15,
    ),
    ArchetypeSpec(
        archetype_id="single_hunter",
        description="单张猎手：专抓单张声称，放行多张，自己偏多张真牌",
        single_bluff_when_has_truth=0.20,
        multi_bluff_when_has_truth=0.08,
        truth_multi_bias=0.75,
        bluff_multi_bias=0.20,
        challenge_threshold=0.60,
        single_challenge_base=0.62,
        multi_challenge_base=0.15,
        obs_p_multi_play=0.58,
        obs_p_voluntary_challenge=0.42,
        obs_p_bluff_given_single=0.35,
        obs_p_bluff_given_multi=0.12,
    ),
    ArchetypeSpec(
        archetype_id="balanced_mixed",
        description="均衡混合型：单张与多张均保持中等真假混合，不易被单维度读牌",
        single_bluff_when_has_truth=0.42,
        multi_bluff_when_has_truth=0.38,
        truth_multi_bias=0.50,
        bluff_multi_bias=0.45,
        challenge_threshold=0.62,
        single_challenge_base=0.22,
        multi_challenge_base=0.32,
        obs_p_multi_play=0.48,
        obs_p_voluntary_challenge=0.26,
        obs_p_bluff_given_single=0.48,
        obs_p_bluff_given_multi=0.42,
    ),
    ArchetypeSpec(
        archetype_id="cautious_single_truth",
        description="慢节奏单张诚实型：偏好打 1 张真牌，极少多出，很少冒险",
        single_bluff_when_has_truth=0.08,
        multi_bluff_when_has_truth=0.30,
        truth_multi_bias=0.10,
        bluff_multi_bias=0.10,
        challenge_threshold=0.70,
        single_challenge_base=0.14,
        multi_challenge_base=0.30,
        obs_p_multi_play=0.14,
        obs_p_voluntary_challenge=0.18,
        obs_p_bluff_given_single=0.22,
        obs_p_bluff_given_multi=0.35,
    ),
    ArchetypeSpec(
        archetype_id="pressure_dumper",
        description="多张倾泻施压型：有牌就爱打2~3张（真假皆有），制造高张数压力",
        single_bluff_when_has_truth=0.25,
        multi_bluff_when_has_truth=0.62,
        truth_multi_bias=0.88,
        bluff_multi_bias=0.88,
        challenge_threshold=0.76,
        single_challenge_base=0.12,
        multi_challenge_base=0.22,
        obs_p_multi_play=0.84,
        obs_p_voluntary_challenge=0.16,
        obs_p_bluff_given_single=0.40,
        obs_p_bluff_given_multi=0.65,
    ),
    ArchetypeSpec(
        archetype_id="coward_under_fire",
        description="中枪变怂型：0~1枪高频诈唬，中2+枪后大幅收敛只打真牌",
        single_bluff_when_has_truth=0.70,
        multi_bluff_when_has_truth=0.55,
        truth_multi_bias=0.50,
        bluff_multi_bias=0.50,
        challenge_threshold=0.66,
        single_challenge_base=0.20,
        multi_challenge_base=0.32,
        high_risk_bluff_multiplier=0.15,
        obs_p_multi_play=0.50,
        obs_p_voluntary_challenge=0.24,
        obs_p_bluff_given_single=0.52,
        obs_p_bluff_given_multi=0.42,
    ),
    ArchetypeSpec(
        archetype_id="desperate_gambler",
        description="逆风赌徒型：低风险偏诚实，中2+枪后变得激进诈唬",
        single_bluff_when_has_truth=0.15,
        multi_bluff_when_has_truth=0.15,
        truth_multi_bias=0.45,
        bluff_multi_bias=0.65,
        challenge_threshold=0.56,
        single_challenge_base=0.28,
        multi_challenge_base=0.42,
        high_risk_bluff_multiplier=3.5,
        obs_p_multi_play=0.52,
        obs_p_voluntary_challenge=0.34,
        obs_p_bluff_given_single=0.42,
        obs_p_bluff_given_multi=0.45,
    ),
    ArchetypeSpec(
        archetype_id="mixed_card_trapper",
        description="单张诚实+多张混出陷阱型：1张多为真牌，2~3张常真假混出",
        single_bluff_when_has_truth=0.12,
        multi_bluff_when_has_truth=0.68,
        truth_multi_bias=0.25,
        bluff_multi_bias=0.80,
        challenge_threshold=0.64,
        single_challenge_base=0.20,
        multi_challenge_base=0.38,
        obs_p_multi_play=0.58,
        obs_p_voluntary_challenge=0.27,
        obs_p_bluff_given_single=0.18,
        obs_p_bluff_given_multi=0.72,
    ),
    ArchetypeSpec(
        archetype_id="card_counter_classic",
        description="经典记牌稳健型（对齐 v2/v3 CardCounter 行为分布）",
        single_bluff_when_has_truth=0.30,
        multi_bluff_when_has_truth=0.12,
        truth_multi_bias=0.28,
        bluff_multi_bias=0.10,
        challenge_threshold=0.64,
        single_challenge_base=0.16,
        multi_challenge_base=0.34,
        high_risk_bluff_multiplier=0.35,
        obs_p_multi_play=0.24,
        obs_p_voluntary_challenge=0.20,
        obs_p_bluff_given_single=0.42,
        obs_p_bluff_given_multi=0.16,
    ),
)

# Unseen probe archetypes (not identical to training archetypes; extreme mirror tells)
UNSEEN_MIRROR_SINGLE_BLUFFER = ArchetypeSpec(
    archetype_id="unseen_mirror_single_bluffer",
    description="未见镜像探针 A：1张几乎必诈，2~3张几乎必真",
    single_bluff_when_has_truth=0.92,
    multi_bluff_when_has_truth=0.02,
    truth_multi_bias=0.95,
    bluff_multi_bias=0.03,
    challenge_threshold=0.66,
    single_challenge_base=0.15,
    multi_challenge_base=0.25,
    obs_p_multi_play=0.62,
    obs_p_voluntary_challenge=0.20,
    obs_p_bluff_given_single=0.94,
    obs_p_bluff_given_multi=0.04,
)

UNSEEN_MIRROR_MULTI_BLUFFER = ArchetypeSpec(
    archetype_id="unseen_mirror_multi_bluffer",
    description="未见镜像探针 B：2~3张几乎必诈，1张几乎必真",
    single_bluff_when_has_truth=0.02,
    multi_bluff_when_has_truth=0.92,
    truth_multi_bias=0.05,
    bluff_multi_bias=0.95,
    challenge_threshold=0.66,
    single_challenge_base=0.15,
    multi_challenge_base=0.25,
    obs_p_multi_play=0.62,
    obs_p_voluntary_challenge=0.20,
    obs_p_bluff_given_single=0.06,
    obs_p_bluff_given_multi=0.94,
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

        if challenge_actions:
            if len(obs.legal_actions) == 1:
                return challenge_actions[0], self._record(obs, bluff_prob, challenge_actions[0])
            assert obs.last_claim is not None
            base_prob = (
                self.spec.single_challenge_base
                if obs.last_claim.claimed_count == 1
                else self.spec.multi_challenge_base
            )
            if bluff_prob >= self.spec.challenge_threshold or (
                not truthful and rng.random() < min(0.85, base_prob + 0.15)
            ) or rng.random() < base_prob * 0.45:
                return challenge_actions[0], self._record(obs, bluff_prob, challenge_actions[0])

        mult = self.spec.high_risk_bluff_multiplier if obs.own_shots_taken >= 2 else 1.0
        p_single_bluff = min(0.98, max(0.0, self.spec.single_bluff_when_has_truth * mult))
        p_multi_bluff = min(0.98, max(0.0, self.spec.multi_bluff_when_has_truth * mult))

        truth_single = [a for a in truthful if len(a.cards) == 1]
        truth_multi = [a for a in truthful if len(a.cards) >= 2]
        bluff_single = [a for a in all_bluffs if len(a.cards) == 1]
        bluff_multi = [a for a in all_bluffs if len(a.cards) >= 2]

        # Decide whether to bluff or play truthful when both are available
        if truthful and all_bluffs:
            # Determine target count tendency and corresponding bluff probability
            want_multi_if_truth = bool(truth_multi) and rng.random() < self.spec.truth_multi_bias
            want_multi_if_bluff = bool(bluff_multi) and rng.random() < self.spec.bluff_multi_bias
            # Effective bluff chance depends on preferred bluff bucket
            eff_bluff_prob = p_multi_bluff if want_multi_if_bluff else p_single_bluff
            if rng.random() < eff_bluff_prob:
                pool = bluff_multi if (want_multi_if_bluff and bluff_multi) else (bluff_single or all_bluffs)
                chosen = rng.choice(pool)
                return chosen, self._record(obs, bluff_prob if obs.last_claim else None, chosen)
            pool = truth_multi if (want_multi_if_truth and truth_multi) else (truth_single or truthful)
            chosen = rng.choice(pool)
            return chosen, self._record(obs, bluff_prob if obs.last_claim else None, chosen)

        if truthful:
            pool = truth_multi if (truth_multi and rng.random() < self.spec.truth_multi_bias) else (truth_single or truthful)
            chosen = rng.choice(pool)
            return chosen, self._record(obs, bluff_prob if obs.last_claim else None, chosen)

        pool = bluff_multi if (bluff_multi and rng.random() < self.spec.bluff_multi_bias) else (bluff_single or all_bluffs)
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

        profile[str(seat)] = {
            "total_plays": len(plays),
            "single_plays": single_plays,
            "multi_plays": multi_plays,
            "multi_play_rate": multi_plays / len(plays) if plays else None,
            "voluntary_challenges": challs,
            "challenge_opportunities": opps,
            "voluntary_challenge_rate": challs / opps if opps else None,
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
        ch_txt = (
            f"{s['voluntary_challenges']}/{s['challenge_opportunities']}次(命中{s['challenge_hits']})"
            if s["challenge_opportunities"]
            else "0次"
        )
        lines.append(
            f"  - Seat {seat_str}: 出牌(1张:{s['single_plays']}次, 2~3张:{s['multi_plays']}次) | "
            f"亮牌(1张:{s_rev}, 2~3张:{m_rev}) | 主动质疑:{ch_txt}"
        )
    return "\n".join(lines)

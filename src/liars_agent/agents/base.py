"""智能体抽象接口、公开画像与规则启发式."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.types import (
    AgentDecisionRecord,
    CandidateAction,
    are_cards_truthful,
    is_card_truthful,
)


def public_opponent_profile(obs: PlayerObservation) -> dict:
    """Observed frequencies with denominators, never hidden strategy identities."""
    profile = {}
    events = (*obs.past_public_actions, *obs.current_round_history)
    for player in obs.players_public_status:
        seat = player.seat
        own_events = [e for e in events if e.actor_seat == seat]
        plays = [e for e in own_events if e.action_type == "PLAY"]
        opportunities = sum(e.step_id > 1 and not e.forced_challenge for e in own_events)
        challenges = sum(e.action_type == "CHALLENGE" and not e.forced_challenge for e in own_events)
        reveals = [r for r in obs.past_rounds_history if r.claimer_seat == seat]
        bluffs = sum(r.was_bluff for r in reveals)
        profile[str(seat)] = {
            "plays": len(plays), "cards_played": sum(e.claimed_count for e in plays),
            "mean_play_count": sum(e.claimed_count for e in plays) / len(plays) if plays else None,
            "voluntary_challenges": challenges, "challenge_opportunities": opportunities,
            "voluntary_challenge_rate": challenges / opportunities if opportunities else None,
            "reveals": len(reveals), "revealed_bluffs": bluffs,
            "revealed_bluff_rate": bluffs / len(reveals) if reveals else None,
        }
    return profile


def describe_action(obs: PlayerObservation, action: CandidateAction) -> str:
    """A factual action description, not a claim about a model's reasoning."""
    if action.action_type == "CHALLENGE":
        if len(obs.legal_actions) == 1:
            return "仅当前行动玩家还有牌，按规则必须质疑上一手。"
        return f"选择质疑 Seat {obs.last_claim.claimer_seat} 的上一手出牌。"
    truthful = are_cards_truthful(action.cards, obs.target_rank)
    return f"选择打出 {len(action.cards)} 张{'全真牌' if truthful else '含假牌的牌'}，剩余 {len(obs.own_hand) - len(action.cards)} 张。"


def estimate_heuristic_bluff_belief(
    obs: PlayerObservation, use_history_adaptation: bool = True,
) -> Tuple[float, str, str]:
    """Rule baseline score under hand-written assumptions; not a Bayesian posterior.

    The capacity prior concerns holding enough truthful cards. Revealed bluff
    frequencies describe selected challenged plays, not the player's full policy.
    No labels from this estimator are used to supervise model belief predictions.
    """
    if obs.last_claim is None:
        return 0.0, "unknown", "当前无待回应声称，诈唬概率不适用。"
    claim = obs.last_claim
    capacity = obs.hand_capacity_prior
    assert capacity is not None
    previous_cards = sum(e.claimed_count for e in obs.current_round_history
        if e.action_type == "PLAY" and e.actor_seat == claim.claimer_seat and e.step_id < claim.step_id)
    voluntary = {1: .36, 2: .72, 3: .86}[claim.claimed_count]
    voluntary += min(.22, .09 * previous_cards) + (.08 if claim.claimer_remaining_cards == 0 else 0)
    reveals = [r for r in obs.past_rounds_history if r.claimer_seat == claim.claimer_seat] if use_history_adaptation else []
    bluffs = sum(r.was_bluff for r in reveals)
    if reveals:
        selected_rate = (bluffs + 1) / (len(reveals) + 2)
        if selected_rate >= .58:
            voluntary += .14 * min(1, len(reveals) / 2)
        elif selected_rate <= .38:
            voluntary -= .16 * min(1, len(reveals) / 2)
    voluntary = min(.95, max(.05, voluntary))
    probability = 1 - capacity * (1 - voluntary)
    rationale = (f"均匀参考模型下手牌容量概率={capacity:.3f}；"
                 f"被揭示诈唬次数={bluffs}/{len(reveals)}；人工启发式估计={probability:.3f}。")
    return probability, "unknown", rationale


# Legacy Python import only; new prompts, samples and reports use the honest name.
estimate_bayesian_bluff_belief = estimate_heuristic_bluff_belief


def split_play_actions(
    obs: PlayerObservation,
) -> tuple[list[CandidateAction], list[CandidateAction], list[CandidateAction]]:
    """将当前合法 PLAY 动作拆分为：(全真牌动作, 纯假牌动作, 真假混出动作)."""
    truthful_plays: list[CandidateAction] = []
    pure_bluff_plays: list[CandidateAction] = []
    mixed_bluff_plays: list[CandidateAction] = []

    for act in obs.legal_actions:
        if act.action_type != "PLAY":
            continue
        if are_cards_truthful(act.cards, obs.target_rank):
            truthful_plays.append(act)
        elif any(is_card_truthful(c, obs.target_rank) for c in act.cards):
            mixed_bluff_plays.append(act)
        else:
            pure_bluff_plays.append(act)

    return truthful_plays, pure_bluff_plays, mixed_bluff_plays


class BaseAgent(ABC):
    """骗子酒馆决策智能体统一接口."""

    def __init__(self, name: str, style_tag: str = "generic") -> None:
        self.name = name
        self.style_tag = style_tag

    @abstractmethod
    def select_action(
        self, obs: PlayerObservation
    ) -> Tuple[CandidateAction, AgentDecisionRecord]:
        """根据局部可见状态选择合法候选动作，并返回信念估计与决策记录."""
        raise NotImplementedError

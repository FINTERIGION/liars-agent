"""严格信息隔离的玩家局部观测构造器、历史消融器与组合先验计算."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Optional

from liars_agent.contracts import RULES_VERSION, OBSERVATION_VERSION
from liars_agent.engine.types import (
    AblationMode,
    CandidateAction,
    ClaimEvent,
    PlayerPublicInfo,
    PublicEvent,
    RoundResolution,
    is_card_truthful,
    sort_cards,
)


def compute_hand_capacity_prior(
    known_own_cards: tuple[str, ...],
    target_rank: str,
    claimed_count: int,
    claimer_hand_size_before_play: int = 5,
    total_deck_size: int = 20,
    total_truth_cards: int = 8,
) -> float:
    """均匀剩余牌槽位模型中，上家手牌至少能凑齐 c 张真牌的概率。

    这是手牌容量参考量，不是当前出牌为真或诈唬的概率；它不包含
    对手如何选择牌或张数的策略似然，只使用本小局自己的已知牌。
    """
    if claimed_count <= 0:
        return 1.0
    own_truth = sum(1 for c in known_own_cards if is_card_truthful(c, target_rank))
    remaining_truth = max(0, total_truth_cards - own_truth)
    remaining_unknown = max(0, total_deck_size - len(known_own_cards))

    hand_size = min(remaining_unknown, claimer_hand_size_before_play)
    if claimed_count > remaining_truth or claimed_count > hand_size:
        return 0.0

    denom = math.comb(remaining_unknown, hand_size)
    if denom == 0:
        return 0.0

    non_truth_unknown = max(0, remaining_unknown - remaining_truth)
    favorable = 0
    for j in range(claimed_count, min(hand_size, remaining_truth) + 1):
        if hand_size - j <= non_truth_unknown:
            favorable += math.comb(remaining_truth, j) * math.comb(
                non_truth_unknown, hand_size - j
            )
    return min(1.0, max(0.0, favorable / denom))


@dataclass(frozen=True)
class PlayerObservation:
    """单个决策步下向智能体暴露的局部可见状态（严格不含他人手牌、未翻开牌或实弹位置）."""

    round_id: int
    step_id: int
    observer_seat: int
    target_rank: str
    own_hand: tuple[str, ...]
    own_played_cards_this_round: tuple[str, ...]
    players_public_status: tuple[PlayerPublicInfo, ...]
    last_claim: Optional[ClaimEvent]
    current_round_history: tuple[PublicEvent, ...]
    past_rounds_history: tuple[RoundResolution, ...]
    legal_actions: tuple[CandidateAction, ...]
    past_public_actions: tuple[PublicEvent, ...] = ()
    ablation_mode: AblationMode = "full"

    @property
    def own_known_cards_this_round(self) -> tuple[str, ...]:
        """观测者在本小局经手的所有牌（当前手牌 + 本小局自己已打出的牌）."""
        return sort_cards(self.own_hand + self.own_played_cards_this_round)

    @property
    def own_shots_taken(self) -> int:
        for p in self.players_public_status:
            if p.seat == self.observer_seat:
                return p.shots_taken
        return 0

    @property
    def own_next_shot_lethal_prob(self) -> float:
        for p in self.players_public_status:
            if p.seat == self.observer_seat:
                return p.next_shot_lethal_prob
        return 1.0 / 6.0

    @property
    def hand_capacity_prior(self) -> Optional[float]:
        """待回应声称对应的均匀牌槽位容量概率；没有声称时为 None。"""
        if self.last_claim is None:
            return None
        claimer_hand_before = self.last_claim.claimer_remaining_cards + self.last_claim.claimed_count
        return compute_hand_capacity_prior(
            known_own_cards=self.own_known_cards_this_round,
            target_rank=self.target_rank,
            claimed_count=self.last_claim.claimed_count,
            claimer_hand_size_before_play=claimer_hand_before,
        )

    @property
    def combinatorial_claim_truth_prior(self) -> Optional[float]:
        """Deprecated Python alias; use hand_capacity_prior for its precise meaning."""
        return self.hand_capacity_prior

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules_version": RULES_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "round_id": self.round_id,
            "step_id": self.step_id,
            "observer_seat": self.observer_seat,
            "target_rank": self.target_rank,
            "own_hand": list(self.own_hand),
            "own_played_cards_this_round": list(self.own_played_cards_this_round),
            "players_public_status": [p.to_dict() for p in self.players_public_status],
            "last_claim": self.last_claim.to_dict() if self.last_claim else None,
            "current_round_history": [e.to_dict() for e in self.current_round_history],
            "past_rounds_history": [r.to_dict() for r in self.past_rounds_history],
            "past_public_actions": [event.to_dict() for event in self.past_public_actions],
            "legal_actions": [a.to_dict() for a in self.legal_actions],
            "ablation_mode": self.ablation_mode,
            "hand_capacity_prior": (
                round(self.hand_capacity_prior, 4)
                if self.hand_capacity_prior is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerObservation:
        """Restore the public observation stored in samples and replay steps."""
        if data.get("rules_version") != RULES_VERSION or data.get("observation_version") != OBSERVATION_VERSION:
            raise ValueError("Observation version does not match the current rules/information contract.")
        return cls(
            round_id=int(data["round_id"]),
            step_id=int(data["step_id"]), observer_seat=int(data["observer_seat"]),
            target_rank=data["target_rank"], own_hand=sort_cards(data["own_hand"]),
            own_played_cards_this_round=sort_cards(data.get("own_played_cards_this_round", ())),
            players_public_status=tuple(PlayerPublicInfo(**{
                key: value for key, value in row.items() if key != "next_shot_lethal_prob"
            }) for row in data["players_public_status"]),
            last_claim=ClaimEvent(**data["last_claim"]) if data.get("last_claim") else None,
            current_round_history=tuple(PublicEvent(**row) for row in data.get("current_round_history", ())),
            past_rounds_history=tuple(RoundResolution(**{
                **row, "revealed_cards": tuple(row["revealed_cards"])
            }) for row in data.get("past_rounds_history", ())),
            legal_actions=tuple(CandidateAction.from_dict(row) for row in data["legal_actions"]),
            past_public_actions=tuple(PublicEvent(**row) for row in data["past_public_actions"]),
            ablation_mode=data.get("ablation_mode", "full"),
        )


def apply_history_ablation(
    obs: PlayerObservation,
    mode: AblationMode,
) -> PlayerObservation:
    """只删除公开历史维度，始终保留本小局自己已知的牌。"""
    if mode == "full":
        if obs.ablation_mode != "full":
            raise ValueError("Cannot restore full history from an already ablated observation.")
        return obs
    if mode == "current_round_only":
        return replace(obs, past_rounds_history=(), past_public_actions=(),
                       ablation_mode="current_round_only")
    if mode == "last_action_only":
        return replace(obs, current_round_history=obs.current_round_history[-1:],
                       past_rounds_history=(), past_public_actions=(),
                       ablation_mode="last_action_only")
    raise ValueError(f"Unsupported ablation mode: {mode}")


# Deprecated Python name retained for existing analysis notebooks. Serialized
# observations use only hand_capacity_prior.
compute_combinatorial_truth_prior = compute_hand_capacity_prior

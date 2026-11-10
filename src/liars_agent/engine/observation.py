"""严格信息隔离的玩家局部观测构造器、历史消融器与组合先验计算."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

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


def compute_combinatorial_truth_prior(
    known_own_cards: tuple[str, ...],
    target_rank: str,
    claimed_count: int,
    claimer_hand_size_before_play: int = 5,
    total_deck_size: int = 20,
    total_truth_cards: int = 8,
) -> float:
    """仅基于观测者自身已知牌（当前手牌+本小局已打出牌），计算上家在出牌前手牌中至少持有 `claimed_count` 张真牌的超几何先验概率.

    注意：此函数仅使用观测者自身合法可见的信息与上家公开手牌数，绝不读取牌桌上其他玩家的底牌。
    """
    if claimed_count <= 0:
        return 1.0
    own_truth = sum(1 for c in known_own_cards if is_card_truthful(c, target_rank))
    remaining_truth = max(0, total_truth_cards - own_truth)
    remaining_unknown = max(0, total_deck_size - len(known_own_cards))

    hand_size = max(claimed_count, min(remaining_unknown, claimer_hand_size_before_play))
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

    game_id: str
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
    def combinatorial_claim_truth_prior(self) -> Optional[float]:
        """若当前面临上家出牌，返回仅基于自身已知牌与上家手牌容量的超几何可真先验概率."""
        if self.last_claim is None:
            return None
        claimer_hand_before = self.last_claim.claimer_remaining_cards + self.last_claim.claimed_count
        return compute_combinatorial_truth_prior(
            known_own_cards=self.own_known_cards_this_round,
            target_rank=self.target_rank,
            claimed_count=self.last_claim.claimed_count,
            claimer_hand_size_before_play=claimer_hand_before,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
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
            "legal_actions": [a.to_dict() for a in self.legal_actions],
            "ablation_mode": self.ablation_mode,
            "combinatorial_claim_truth_prior": (
                round(self.combinatorial_claim_truth_prior, 4)
                if self.combinatorial_claim_truth_prior is not None
                else None
            ),
        }


def apply_history_ablation(
    obs: PlayerObservation,
    mode: AblationMode,
) -> PlayerObservation:
    """对玩家观测执行历史信息消融（用于第5~6周的历史信息利用对照实验）."""
    if mode == "full":
        return obs
    if mode == "current_round_only":
        return PlayerObservation(
            game_id=obs.game_id,
            round_id=obs.round_id,
            step_id=obs.step_id,
            observer_seat=obs.observer_seat,
            target_rank=obs.target_rank,
            own_hand=obs.own_hand,
            own_played_cards_this_round=obs.own_played_cards_this_round,
            players_public_status=obs.players_public_status,
            last_claim=obs.last_claim,
            current_round_history=obs.current_round_history,
            past_rounds_history=(),
            legal_actions=obs.legal_actions,
            ablation_mode="current_round_only",
        )
    if mode == "last_action_only":
        last_events = obs.current_round_history[-1:] if obs.current_round_history else ()
        return PlayerObservation(
            game_id=obs.game_id,
            round_id=obs.round_id,
            step_id=obs.step_id,
            observer_seat=obs.observer_seat,
            target_rank=obs.target_rank,
            own_hand=obs.own_hand,
            own_played_cards_this_round=(),
            players_public_status=obs.players_public_status,
            last_claim=obs.last_claim,
            current_round_history=last_events,
            past_rounds_history=(),
            legal_actions=obs.legal_actions,
            ablation_mode="last_action_only",
        )
    raise ValueError(f"Unsupported ablation mode: {mode}")

"""确定性骗子酒馆规则引擎核心数据结构与类型定义."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

CardRank = Literal["Q", "K", "A", "JOKER"]
TargetRank = Literal["Q", "K", "A"]
ActionType = Literal["PLAY", "CHALLENGE"]
AblationMode = Literal["full", "current_round_only", "last_action_only"]

STANDARD_DECK_COUNTS: dict[CardRank, int] = {
    "Q": 6,
    "K": 6,
    "A": 6,
    "JOKER": 2,
}
TARGET_RANKS: tuple[TargetRank, ...] = ("Q", "K", "A")
CARD_SORT_ORDER: dict[str, int] = {"Q": 0, "K": 1, "A": 2, "JOKER": 3}


def sort_cards(cards: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """按固定顺序 (Q, K, A, JOKER) 对手牌或出牌元组进行排序，消除同构顺序差异."""
    return tuple(sorted(cards, key=lambda c: CARD_SORT_ORDER.get(c, 99)))


def is_card_truthful(card: str, target_rank: str) -> bool:
    """判断单张卡牌在当前目标牌下是否为真牌（JOKER 视为任意目标牌真牌）."""
    return card == target_rank or card == "JOKER"


def are_cards_truthful(cards: tuple[str, ...], target_rank: str) -> bool:
    """判断一组打出的牌是否全部为真牌."""
    return all(is_card_truthful(c, target_rank) for c in cards)


@dataclass(frozen=True)
class CandidateAction:
    """候选合法动作（模型从离散合法候选中选择）."""

    action_id: str
    action_type: ActionType
    cards: tuple[str, ...] = ()

    @property
    def claim_count(self) -> int:
        return len(self.cards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "cards": list(self.cards),
            "claim_count": self.claim_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateAction:
        return cls(
            action_id=str(data["action_id"]),
            action_type=data["action_type"],
            cards=sort_cards(data.get("cards", ())),
        )


@dataclass(frozen=True)
class PlayerPublicInfo:
    """牌桌上单名玩家的公开状态（严格不含底牌与实弹位置）."""

    seat: int
    alive: bool
    hand_count: int
    shots_taken: int
    max_chambers: int = 6

    @property
    def next_shot_lethal_prob(self) -> float:
        """若该玩家下一次开枪，基于公开已开枪次数的条件致死率 1 / (6 - shots_taken)."""
        if not self.alive:
            return 0.0
        remaining = max(1, self.max_chambers - self.shots_taken)
        return 1.0 / remaining

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "alive": self.alive,
            "hand_count": self.hand_count,
            "shots_taken": self.shots_taken,
            "max_chambers": self.max_chambers,
            "next_shot_lethal_prob": round(self.next_shot_lethal_prob, 4),
        }


@dataclass(frozen=True)
class ClaimEvent:
    """最近一次出牌声称的公开信息（不含实际打出的牌面）."""

    round_id: int
    step_id: int
    claimer_seat: int
    claimed_rank: str
    claimed_count: int
    claimer_remaining_cards: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicEvent:
    """小局内的公开历史事件（出牌声称或质疑触发）."""

    round_id: int
    step_id: int
    actor_seat: int
    action_type: ActionType
    claimed_count: int = 0
    claimed_rank: str = ""
    actor_remaining_cards: int = 0
    forced_challenge: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoundResolution:
    """小局质疑与左轮结算的公开结果（此时翻开被质疑的牌，属于公开历史真值）."""

    round_id: int
    target_rank: str
    claimer_seat: int
    challenger_seat: int
    claimed_count: int
    revealed_cards: tuple[str, ...]
    was_bluff: bool
    challenge_succeeded: bool
    penalized_seat: int
    shots_taken_before: int
    shots_taken_after: int
    lethal_shot: bool
    forced_challenge: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["revealed_cards"] = list(self.revealed_cards)
        return data


@dataclass(frozen=True)
class AgentDecisionRecord:
    """智能体单步决策的附加记录（信念估计、模型理由、是否发生非法输出回退）."""

    belief_bluff_prob: Optional[float] = None
    estimated_opponent_style: str = "unknown"
    rationale: str = ""
    raw_output: str = ""
    was_fallback: bool = False
    fallback_reason: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

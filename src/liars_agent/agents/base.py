"""智能体统一抽象基类与贝叶斯对手信念估计工具."""

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


def estimate_bayesian_bluff_belief(
    obs: PlayerObservation,
    use_history_adaptation: bool = True,
) -> Tuple[float, str, str]:
    """根据当前可见信息（自身已知牌 + 公开历史）估计上家出牌为诈唬的后验概率.

    严格区分“观测事实”与“对手模型假设”：
    1. 组合数学能力先验 $P(\text{Capable})$：上家在出牌前手牌容量 $H$ 下，至少持有 $c$ 张真牌的超几何概率；
    2. 本小局真牌耗竭效应（Round Depletion）：随着本小局出牌轮次推进及上家此前已出牌张数增加，其手中剩余真牌被耗尽、被迫诈唬的概率上升；
    3. 跨小局对手画像更新（Opponent Adaptation）：根据 `past_rounds_history` 中该座位被翻牌时的真实诈唬率动态修正后验概率。

    返回：(belief_bluff_prob, estimated_opponent_style, rationale)
    """
    if obs.last_claim is None:
        return 0.0, "none", "当前为小局首出牌，无上家声称需要估计。"

    claim = obs.last_claim
    p_capable = obs.combinatorial_claim_truth_prior
    assert p_capable is not None

    if p_capable <= 0.0:
        return (
            1.0,
            "certain_bluffer",
            f"数学必然诈唬：我方已知真牌后，牌库剩余真牌不足 {claim.claimed_count} 张。",
        )

    claimer_seat = claim.claimer_seat
    reveals = 0
    bluffs = 0
    claimer_prior_cards_this_round = 0
    est_style = "balanced"

    # 统计本小局内上家在此前步骤已消耗的手牌数（若处于 last_action_only 消融则无此信息）
    for ev in obs.current_round_history[:-1]:
        if ev.actor_seat == claimer_seat and ev.action_type == "PLAY":
            claimer_prior_cards_this_round += ev.claimed_count

    if use_history_adaptation:
        for res in obs.past_rounds_history:
            if res.claimer_seat == claimer_seat:
                reveals += 1
                if res.was_bluff:
                    bluffs += 1

    # 基础诈唬率：结合声称张数（1张 ~38%，2张 ~72%，3张 ~88%）与本小局真牌耗竭修正
    if claim.claimed_count == 1:
        base_voluntary_bluff = 0.36
    elif claim.claimed_count == 2:
        base_voluntary_bluff = 0.72
    else:
        base_voluntary_bluff = 0.86

    # 若上家在本小局前面已经出过牌，其手中真牌耗竭概率显著上升
    depletion_bonus = min(0.22, 0.09 * claimer_prior_cards_this_round)
    empty_bonus = 0.08 if claim.claimer_remaining_cards == 0 else 0.0
    voluntary_bluff = base_voluntary_bluff + depletion_bonus + empty_bonus

    if use_history_adaptation and reveals >= 1:
        emp_bluff_rate = (bluffs + 1.0) / (reveals + 2.0)
        if emp_bluff_rate >= 0.58:
            est_style = "aggressive_bluffer"
            voluntary_bluff += 0.14 * min(1.0, reveals / 2.0)
        elif emp_bluff_rate <= 0.38:
            est_style = "honest_conservative"
            voluntary_bluff -= 0.16 * min(1.0, reveals / 2.0)
        else:
            est_style = "card_counter"

    voluntary_bluff = min(0.95, max(0.05, voluntary_bluff))
    prob_truth = p_capable * (1.0 - voluntary_bluff)
    post_bluff_prob = round(min(0.98, max(0.02, 1.0 - prob_truth)), 4)

    rationale = (
        f"上家(Seat{claimer_seat})持有>={claim.claimed_count}张真牌先验={p_capable:.2f}，"
        f"本局已消耗={claimer_prior_cards_this_round}张，跨局亮牌诈唬={bluffs}/{reveals}({est_style}) "
        f"-> 后验诈唬概率={post_bluff_prob:.2f}。"
    )
    return post_bluff_prob, est_style, rationale


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

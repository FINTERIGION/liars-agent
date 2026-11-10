"""小型分类器基线智能体 (Small Classifier Baseline).

提取 12 维局部观测统计特征，使用逻辑回归/线性分类器预测上家诈唬概率与最优动作类别，
用于计划书第 4 周要求的“规则策略、小型分类器与零样本/微调模型对照实验”。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Tuple

from liars_agent.agents.base import BaseAgent, split_play_actions
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.types import (
    AgentDecisionRecord,
    CandidateAction,
    is_card_truthful,
)


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def extract_observation_features(obs: PlayerObservation) -> list[float]:
    """从局部可见状态 `PlayerObservation` 提取 12 维数值特征向量."""
    hand_len = len(obs.own_hand)
    own_truth_count = sum(1 for c in obs.own_hand if is_card_truthful(c, obs.target_rank))
    own_truth_ratio = own_truth_count / max(1, hand_len)
    own_lethal_risk = obs.own_next_shot_lethal_prob

    if obs.last_claim is None:
        return [
            1.0,  # bias
            hand_len / 5.0,
            own_truth_ratio,
            own_lethal_risk,
            0.0,  # has_claim
            0.0,  # claimed_count_norm
            0.0,  # comb_bluff_prior
            0.0,  # claimer_empty_hand
            0.5,  # claimer_hist_bluff_rate
            0.0,  # claimer_hist_confidence
            obs.round_id / 6.0,
            len(obs.current_round_history) / 10.0,
        ]

    claim = obs.last_claim
    comb_truth = obs.combinatorial_claim_truth_prior or 0.5
    comb_bluff_prior = 1.0 - comb_truth
    claimer_empty = 1.0 if claim.claimer_remaining_cards == 0 else 0.0

    # 统计公开历史中上家的被亮牌诈唬率（若处于消融模式，past_rounds_history 为空则退化为先验 0.5）
    reveals = 0
    bluffs = 0
    for res in obs.past_rounds_history:
        if res.claimer_seat == claim.claimer_seat:
            reveals += 1
            if res.was_bluff:
                bluffs += 1
    hist_bluff_rate = (bluffs + 1.0) / (reveals + 2.0)
    hist_conf = min(1.0, reveals / 4.0)

    return [
        1.0,
        hand_len / 5.0,
        own_truth_ratio,
        own_lethal_risk,
        1.0,
        claim.claimed_count / 3.0,
        comb_bluff_prior,
        claimer_empty,
        hist_bluff_rate,
        hist_conf,
        obs.round_id / 6.0,
        len(obs.current_round_history) / 10.0,
    ]


class SmallClassifierBot(BaseAgent):
    """基于 12 维手工统计特征的小型分类器智能体（支持在线/离线 SGD 拟合与 JSON 权重导出）."""

    DEFAULT_BLUFF_WEIGHTS: list[float] = [
        -1.35,  # bias
        -0.10,  # hand_len
        0.45,   # own_truth_ratio
        -0.15,  # own_lethal_risk
        0.20,   # has_claim
        0.95,   # claimed_count_norm
        2.60,   # comb_bluff_prior
        0.55,   # claimer_empty_hand
        1.10,   # claimer_hist_bluff_rate
        0.40,   # claimer_hist_confidence
        0.08,   # round_id
        0.15,   # step_count
    ]

    def __init__(
        self,
        name: str = "small_classifier",
        bluff_weights: Optional[list[float]] = None,
        challenge_prob_cutoff: float = 0.57,
    ) -> None:
        super().__init__(name=name, style_tag="small_classifier")
        self.bluff_weights = list(bluff_weights or self.DEFAULT_BLUFF_WEIGHTS)
        self.challenge_prob_cutoff = challenge_prob_cutoff

    def predict_bluff_prob(self, obs: PlayerObservation) -> float:
        if obs.last_claim is None:
            return 0.0
        if (obs.combinatorial_claim_truth_prior or 0.0) <= 0.0:
            return 1.0
        feats = extract_observation_features(obs)
        logit = sum(w * x for w, x in zip(self.bluff_weights, feats))
        return round(_sigmoid(logit), 4)

    def fit_from_replays(
        self,
        replay_dicts: list[dict[str, Any]],
        learning_rate: float = 0.05,
        epochs: int = 15,
        l2_reg: float = 1e-3,
    ) -> dict[str, float]:
        """从历史对局真值日志中提取 (观测特征, 上家是否真实诈唬) 样本并训练逻辑回归分类器."""
        samples: list[tuple[list[float], float]] = []
        for bundle in replay_dicts:
            for step in bundle.get("steps", []):
                was_bluff = step.get("pending_claim_was_bluff")
                if was_bluff is None:
                    continue
                obs_dict = step["observation"]
                # 构造轻量级特征
                own_hand = tuple(obs_dict.get("own_hand", ()))
                target_rank = obs_dict.get("target_rank", "Q")
                hand_len = len(own_hand)
                own_truth = sum(1 for c in own_hand if is_card_truthful(c, target_rank))
                own_truth_ratio = own_truth / max(1, hand_len)
                last_claim = obs_dict.get("last_claim")
                if not last_claim:
                    continue
                comb_prior = obs_dict.get("combinatorial_claim_truth_prior")
                comb_bluff = 1.0 - (comb_prior if comb_prior is not None else 0.5)
                claimer_seat = last_claim["claimer_seat"]
                reveals = 0
                bluffs = 0
                for r in obs_dict.get("past_rounds_history", []):
                    if r["claimer_seat"] == claimer_seat:
                        reveals += 1
                        if r["was_bluff"]:
                            bluffs += 1
                hist_rate = (bluffs + 1.0) / (reveals + 2.0)
                feats = [
                    1.0,
                    hand_len / 5.0,
                    own_truth_ratio,
                    0.2,
                    1.0,
                    last_claim["claimed_count"] / 3.0,
                    comb_bluff,
                    1.0 if last_claim["claimer_remaining_cards"] == 0 else 0.0,
                    hist_rate,
                    min(1.0, reveals / 4.0),
                    obs_dict.get("round_id", 1) / 6.0,
                    len(obs_dict.get("current_round_history", [])) / 10.0,
                ]
                samples.append((feats, 1.0 if was_bluff else 0.0))

        if not samples:
            return {"num_samples": 0, "final_bce_loss": 0.0}

        weights = list(self.bluff_weights)
        final_loss = 0.0
        for _ in range(epochs):
            total_loss = 0.0
            for x, y in samples:
                pred = _sigmoid(sum(w * xi for w, xi in zip(weights, x)))
                err = pred - y
                total_loss += -(
                    y * math.log(max(1e-6, pred)) + (1.0 - y) * math.log(max(1e-6, 1.0 - pred))
                )
                for j in range(len(weights)):
                    grad = err * x[j] + l2_reg * weights[j]
                    weights[j] -= learning_rate * grad
            final_loss = total_loss / len(samples)

        self.bluff_weights = [round(w, 5) for w in weights]
        return {"num_samples": len(samples), "final_bce_loss": round(final_loss, 4)}

    def select_action(
        self, obs: PlayerObservation
    ) -> Tuple[CandidateAction, AgentDecisionRecord]:
        bluff_prob = self.predict_bluff_prob(obs)
        challenge_actions = [a for a in obs.legal_actions if a.action_type == "CHALLENGE"]
        truthful, pure_bluff, mixed_bluff = split_play_actions(obs)

        # 结合自身剩余枪膛致死率轻微修正分类器阈值
        cutoff = self.challenge_prob_cutoff + 0.10 * (obs.own_next_shot_lethal_prob - 1.0 / 6.0)
        if not truthful:
            cutoff -= 0.04

        if challenge_actions and (len(obs.legal_actions) == 1 or bluff_prob >= cutoff):
            return challenge_actions[0], AgentDecisionRecord(
                belief_bluff_prob=bluff_prob,
                estimated_opponent_style="classifier_scored",
                rationale=f"[SmallClassifier] 线性分类器预测诈唬概率={bluff_prob:.2f} >= 阈值={cutoff:.2f}，执行质疑。",
            )

        if truthful:
            # 分类器倾向稳健打出1~2张真牌
            chosen = truthful[0]
            return chosen, AgentDecisionRecord(
                belief_bluff_prob=bluff_prob if obs.last_claim else None,
                estimated_opponent_style="classifier_scored",
                rationale=f"[SmallClassifier] 预测上家诈唬率={bluff_prob:.2f} 低于阈值，稳健打出真牌。",
            )

        bluff_candidates = sorted(pure_bluff + mixed_bluff, key=lambda a: len(a.cards))
        chosen = bluff_candidates[0] if bluff_candidates else obs.legal_actions[0]
        return chosen, AgentDecisionRecord(
            belief_bluff_prob=bluff_prob if obs.last_claim else None,
            estimated_opponent_style="classifier_scored",
            rationale="[SmallClassifier] 无真牌可出，选择单张假牌降低被抓风险。",
        )

    def save_weights(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "bluff_weights": self.bluff_weights,
                    "challenge_prob_cutoff": self.challenge_prob_cutoff,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

"""5 类参数化规则策略智能体（支撑固定对手池训练与未见对手泛化评测）."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Tuple

from liars_agent.agents.base import (
    BaseAgent,
    estimate_bayesian_bluff_belief,
    split_play_actions,
)
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.types import AgentDecisionRecord, CandidateAction


def _step_rng(base_seed: int, obs: PlayerObservation) -> random.Random:
    """根据基础种子与当前观测步构造确定性随机数生成器，确保回放可复现."""
    key = f"{base_seed}:{obs.game_id}:{obs.round_id}:{obs.step_id}:{obs.observer_seat}"
    seed_int = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed_int)


class HonestConservativeBot(BaseAgent):
    """1. 诚实保守型（训练对手池）：有真牌必出真牌，无真牌只出1张假牌；仅在高把握时质疑."""

    def __init__(
        self,
        name: str = "honest_conservative",
        seed: int = 101,
        bluff_prob_when_forced: float = 0.15,
        challenge_threshold: float = 0.72,
    ) -> None:
        super().__init__(name=name, style_tag="honest_conservative")
        self.seed = seed
        self.bluff_prob_when_forced = bluff_prob_when_forced
        self.challenge_threshold = challenge_threshold

    def select_action(
        self, obs: PlayerObservation
    ) -> Tuple[CandidateAction, AgentDecisionRecord]:
        rng = _step_rng(self.seed, obs)
        bluff_prob, est_style, rationale = estimate_bayesian_bluff_belief(
            obs, use_history_adaptation=False
        )

        challenge_actions = [a for a in obs.legal_actions if a.action_type == "CHALLENGE"]
        truthful, pure_bluff, mixed_bluff = split_play_actions(obs)

        # 若仅剩 CHALLENGE 合法或上家诈唬概率极高，则执行质疑
        if challenge_actions and (
            len(obs.legal_actions) == 1 or bluff_prob >= self.challenge_threshold
        ):
            return challenge_actions[0], AgentDecisionRecord(
                belief_bluff_prob=bluff_prob,
                estimated_opponent_style=est_style,
                rationale=f"[HonestConservative] 诈唬概率 {bluff_prob:.2f} >= 阈值 {self.challenge_threshold:.2f}，选择质疑。",
            )

        # 优先打 1~2 张真牌
        if truthful:
            single_or_double = [a for a in truthful if len(a.cards) <= 2] or truthful
            chosen = single_or_double[0]
            return chosen, AgentDecisionRecord(
                belief_bluff_prob=bluff_prob if obs.last_claim else None,
                estimated_opponent_style=est_style,
                rationale="[HonestConservative] 手中有真牌，诚实打出真牌。",
            )

        # 无真牌时被迫诈唬：若有 CHALLENGE 且自身风险低，小概率直接质疑，否则出单张假牌
        if challenge_actions and bluff_prob >= 0.55 and rng.random() < 0.35:
            return challenge_actions[0], AgentDecisionRecord(
                belief_bluff_prob=bluff_prob,
                estimated_opponent_style=est_style,
                rationale="[HonestConservative] 无真牌可出且上家可疑，转为质疑。",
            )

        bluff_candidates = sorted(pure_bluff + mixed_bluff, key=lambda a: len(a.cards))
        chosen = bluff_candidates[0] if bluff_candidates else obs.legal_actions[-1]
        return chosen, AgentDecisionRecord(
            belief_bluff_prob=bluff_prob if obs.last_claim else None,
            estimated_opponent_style=est_style,
            rationale="[HonestConservative] 无真牌可出，最小化风险打出单张假牌。",
        )


class AggressiveBlufferBot(BaseAgent):
    """2. 激进诈唬型（训练对手池）：高频混出假牌或保留真牌先出假牌，倾向多张出牌快速清空手牌."""

    def __init__(
        self,
        name: str = "aggressive_bluffer",
        seed: int = 202,
        bluff_rate: float = 0.68,
        multi_card_bias: float = 0.60,
        challenge_threshold: float = 0.80,
    ) -> None:
        super().__init__(name=name, style_tag="aggressive_bluffer")
        self.seed = seed
        self.bluff_rate = bluff_rate
        self.multi_card_bias = multi_card_bias
        self.challenge_threshold = challenge_threshold

    def select_action(
        self, obs: PlayerObservation
    ) -> Tuple[CandidateAction, AgentDecisionRecord]:
        rng = _step_rng(self.seed, obs)
        bluff_prob, est_style, _ = estimate_bayesian_bluff_belief(
            obs, use_history_adaptation=False
        )
        challenge_actions = [a for a in obs.legal_actions if a.action_type == "CHALLENGE"]
        truthful, pure_bluff, mixed_bluff = split_play_actions(obs)

        if challenge_actions and (
            len(obs.legal_actions) == 1 or bluff_prob >= self.challenge_threshold
        ):
            return challenge_actions[0], AgentDecisionRecord(
                belief_bluff_prob=bluff_prob,
                estimated_opponent_style=est_style,
                rationale=f"[AggressiveBluffer] 仅在极高把握 ({bluff_prob:.2f}) 或强制时质疑。",
            )

        all_bluffs = pure_bluff + mixed_bluff
        if all_bluffs and (not truthful or rng.random() < self.bluff_rate):
            multi_bluffs = [a for a in all_bluffs if len(a.cards) >= 2]
            if multi_bluffs and rng.random() < self.multi_card_bias:
                chosen = rng.choice(multi_bluffs)
            else:
                chosen = rng.choice(all_bluffs)
            return chosen, AgentDecisionRecord(
                belief_bluff_prob=bluff_prob if obs.last_claim else None,
                estimated_opponent_style=est_style,
                rationale=f"[AggressiveBluffer] 主动施压诈唬，打出 {len(chosen.cards)} 张牌。",
            )

        if truthful:
            multi_truth = [a for a in truthful if len(a.cards) >= 2]
            chosen = multi_truth[-1] if multi_truth else truthful[0]
            return chosen, AgentDecisionRecord(
                belief_bluff_prob=bluff_prob if obs.last_claim else None,
                estimated_opponent_style=est_style,
                rationale="[AggressiveBluffer] 打出多张真牌快速倾泻手牌。",
            )

        return obs.legal_actions[0], AgentDecisionRecord(
            belief_bluff_prob=bluff_prob if obs.last_claim else None,
            estimated_opponent_style=est_style,
            rationale="[AggressiveBluffer] 默认回退动作。",
        )


class CardCounterBot(BaseAgent):
    """3. 记牌精算型（训练对手池）：基于超几何后验与自身左轮剩余枪膛风险计算期望效用."""

    def __init__(
        self,
        name: str = "card_counter",
        seed: int = 303,
        risk_aversion: float = 1.0,
        posterior_challenge_cutoff: float = 0.64,
    ) -> None:
        super().__init__(name=name, style_tag="card_counter")
        self.seed = seed
        self.risk_aversion = risk_aversion
        self.posterior_challenge_cutoff = posterior_challenge_cutoff

    def select_action(
        self, obs: PlayerObservation
    ) -> Tuple[CandidateAction, AgentDecisionRecord]:
        rng = _step_rng(self.seed, obs)
        bluff_prob, est_style, rationale = estimate_bayesian_bluff_belief(
            obs, use_history_adaptation=True
        )
        challenge_actions = [a for a in obs.legal_actions if a.action_type == "CHALLENGE"]
        truthful, pure_bluff, mixed_bluff = split_play_actions(obs)

        # 结合自身致死风险动态调整质疑阈值：自身已开枪越多，质疑越谨慎（除非自身无真牌可打）
        own_lethal = obs.own_next_shot_lethal_prob
        risk_adjustment = 0.12 * self.risk_aversion * (own_lethal - 1.0 / 6.0)
        hand_adjustment = -0.06 if not truthful else 0.03
        dynamic_cutoff = min(
            0.90, max(0.42, self.posterior_challenge_cutoff + risk_adjustment + hand_adjustment)
        )

        if challenge_actions and (
            len(obs.legal_actions) == 1 or bluff_prob >= dynamic_cutoff
        ):
            return challenge_actions[0], AgentDecisionRecord(
                belief_bluff_prob=bluff_prob,
                estimated_opponent_style=est_style,
                rationale=(
                    f"[CardCounter] {rationale} 动态质疑阈值={dynamic_cutoff:.2f}"
                    f"(自身致死风险={own_lethal:.2f})，执行质疑。"
                ),
            )

        # 出牌策略：若手握多张假牌且真牌充足，在安全时先伪装打出1张假牌保留真牌防守
        if pure_bluff and len(truthful) >= 1 and own_lethal <= 0.25 and rng.random() < 0.35:
            single_bluffs = [a for a in pure_bluff if len(a.cards) == 1]
            if single_bluffs:
                return single_bluffs[0], AgentDecisionRecord(
                    belief_bluff_prob=bluff_prob if obs.last_claim else None,
                    estimated_opponent_style=est_style,
                    rationale="[CardCounter] 低风险下先行试探打出1张假牌，保留真牌用于残局防守。",
                )

        if truthful:
            # 优先出1~2张真牌保持手牌结构
            chosen = truthful[0]
            return chosen, AgentDecisionRecord(
                belief_bluff_prob=bluff_prob if obs.last_claim else None,
                estimated_opponent_style=est_style,
                rationale="[CardCounter] 稳健打出真牌。",
            )

        bluff_candidates = sorted(pure_bluff + mixed_bluff, key=lambda a: len(a.cards))
        chosen = bluff_candidates[0] if bluff_candidates else obs.legal_actions[0]
        return chosen, AgentDecisionRecord(
            belief_bluff_prob=bluff_prob if obs.last_claim else None,
            estimated_opponent_style=est_style,
            rationale="[CardCounter] 手中无真牌，选择最低暴露风险的单张出牌。",
        )


class ParanoidHunterBot(BaseAgent):
    """4. 多疑猎手型（未见测试池 Unseen）：对 2~3 张多牌声称或残局声称极度敏感，高频抓诈唬."""

    def __init__(
        self,
        name: str = "paranoid_hunter",
        seed: int = 404,
        multi_claim_challenge_prob: float = 0.78,
        single_claim_challenge_prob: float = 0.38,
    ) -> None:
        super().__init__(name=name, style_tag="paranoid_hunter")
        self.seed = seed
        self.multi_claim_challenge_prob = multi_claim_challenge_prob
        self.single_claim_challenge_prob = single_claim_challenge_prob

    def select_action(
        self, obs: PlayerObservation
    ) -> Tuple[CandidateAction, AgentDecisionRecord]:
        rng = _step_rng(self.seed, obs)
        bluff_prob, est_style, _ = estimate_bayesian_bluff_belief(
            obs, use_history_adaptation=True
        )
        challenge_actions = [a for a in obs.legal_actions if a.action_type == "CHALLENGE"]
        truthful, pure_bluff, mixed_bluff = split_play_actions(obs)

        if challenge_actions and obs.last_claim is not None:
            if len(obs.legal_actions) == 1:
                return challenge_actions[0], AgentDecisionRecord(
                    belief_bluff_prob=bluff_prob,
                    estimated_opponent_style=est_style,
                    rationale="[ParanoidHunter] 强制质疑。",
                )
            claimed_n = obs.last_claim.claimed_count
            if claimed_n >= 2 and (
                bluff_prob >= 0.40 or rng.random() < self.multi_claim_challenge_prob
            ):
                return challenge_actions[0], AgentDecisionRecord(
                    belief_bluff_prob=bluff_prob,
                    estimated_opponent_style=est_style,
                    rationale=f"[ParanoidHunter] 针对上家多张声称({claimed_n}张)发起高压猎杀质疑。",
                )
            if claimed_n == 1 and (
                bluff_prob >= 0.52 or rng.random() < self.single_claim_challenge_prob
            ):
                return challenge_actions[0], AgentDecisionRecord(
                    belief_bluff_prob=bluff_prob,
                    estimated_opponent_style=est_style,
                    rationale="[ParanoidHunter] 主动突袭质疑上家单张声称。",
                )

        if truthful:
            return truthful[0], AgentDecisionRecord(
                belief_bluff_prob=bluff_prob if obs.last_claim else None,
                estimated_opponent_style=est_style,
                rationale="[ParanoidHunter] 自身出牌极度克制，优先打单张真牌。",
            )

        bluff_candidates = sorted(pure_bluff + mixed_bluff, key=lambda a: len(a.cards))
        chosen = bluff_candidates[0] if bluff_candidates else obs.legal_actions[0]
        return chosen, AgentDecisionRecord(
            belief_bluff_prob=bluff_prob if obs.last_claim else None,
            estimated_opponent_style=est_style,
            rationale="[ParanoidHunter] 被迫单张诈唬。",
        )


class ChameleonAdaptiveBot(BaseAgent):
    """5. 变色龙自适应型（未见测试池 Unseen）：根据小局阶段与自身受罚枪数在诚实与激进间切换."""

    def __init__(
        self,
        name: str = "chameleon_adaptive",
        seed: int = 505,
        high_risk_shots_threshold: int = 2,
        low_risk_bluff_rate: float = 0.72,
        high_risk_bluff_rate: float = 0.08,
    ) -> None:
        super().__init__(name=name, style_tag="chameleon_adaptive")
        self.seed = seed
        self.high_risk_shots_threshold = high_risk_shots_threshold
        self.low_risk_bluff_rate = low_risk_bluff_rate
        self.high_risk_bluff_rate = high_risk_bluff_rate

    def select_action(
        self, obs: PlayerObservation
    ) -> Tuple[CandidateAction, AgentDecisionRecord]:
        rng = _step_rng(self.seed, obs)
        bluff_prob, est_style, _ = estimate_bayesian_bluff_belief(
            obs, use_history_adaptation=True
        )
        challenge_actions = [a for a in obs.legal_actions if a.action_type == "CHALLENGE"]
        truthful, pure_bluff, mixed_bluff = split_play_actions(obs)

        high_risk_mode = (
            obs.own_shots_taken >= self.high_risk_shots_threshold or obs.round_id == 1
        )
        challenge_cutoff = 0.68 if high_risk_mode else 0.50

        if challenge_actions and (
            len(obs.legal_actions) == 1 or bluff_prob >= challenge_cutoff
        ):
            return challenge_actions[0], AgentDecisionRecord(
                belief_bluff_prob=bluff_prob,
                estimated_opponent_style=est_style,
                rationale=f"[ChameleonAdaptive] 风险模式={'保守' if high_risk_mode else '激进'}，触发质疑。",
            )

        target_bluff_rate = (
            self.high_risk_bluff_rate if high_risk_mode else self.low_risk_bluff_rate
        )
        all_bluffs = pure_bluff + mixed_bluff
        if all_bluffs and (not truthful or rng.random() < target_bluff_rate):
            chosen = (
                sorted(all_bluffs, key=lambda a: len(a.cards))[0]
                if high_risk_mode
                else rng.choice(all_bluffs)
            )
            return chosen, AgentDecisionRecord(
                belief_bluff_prob=bluff_prob if obs.last_claim else None,
                estimated_opponent_style=est_style,
                rationale=f"[ChameleonAdaptive] 动态策略切换（高风险={high_risk_mode}），执行诈唬出牌。",
            )

        if truthful:
            chosen = truthful[0] if high_risk_mode else truthful[-1]
            return chosen, AgentDecisionRecord(
                belief_bluff_prob=bluff_prob if obs.last_claim else None,
                estimated_opponent_style=est_style,
                rationale=f"[ChameleonAdaptive] 动态策略切换（高风险={high_risk_mode}），打出真牌。",
            )

        return obs.legal_actions[0], AgentDecisionRecord(
            belief_bluff_prob=bluff_prob if obs.last_claim else None,
            estimated_opponent_style=est_style,
            rationale="[ChameleonAdaptive] 回退动作。",
        )


RULE_BOT_REGISTRY: dict[str, type[BaseAgent]] = {
    "honest_conservative": HonestConservativeBot,
    "HonestConservativeBot": HonestConservativeBot,
    "aggressive_bluffer": AggressiveBlufferBot,
    "AggressiveBlufferBot": AggressiveBlufferBot,
    "card_counter": CardCounterBot,
    "CardCounterBot": CardCounterBot,
    "paranoid_hunter": ParanoidHunterBot,
    "ParanoidHunterBot": ParanoidHunterBot,
    "chameleon_adaptive": ChameleonAdaptiveBot,
    "ChameleonAdaptiveBot": ChameleonAdaptiveBot,
}


def create_rule_agent(
    agent_type: str,
    name: str | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> BaseAgent:
    """根据配置名称实例化规则对手智能体."""
    if agent_type not in RULE_BOT_REGISTRY:
        raise ValueError(
            f"Unknown rule agent type {agent_type!r}. Available: {list(RULE_BOT_REGISTRY)}"
        )
    cls = RULE_BOT_REGISTRY[agent_type]
    resolved_name = name or agent_type
    # 过滤掉目标类不支持的额外配置字段
    import inspect

    sig = inspect.signature(cls.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(name=resolved_name, seed=seed, **valid_kwargs)

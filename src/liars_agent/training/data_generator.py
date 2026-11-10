"""混合对手数据生成与基于多次续玩（Rollout）收益筛选的 SFT 数据合成器."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Optional

from liars_agent.agents.base import BaseAgent, estimate_bayesian_bluff_belief
from liars_agent.agents.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from liars_agent.agents.rule_bots import (
    AggressiveBlufferBot,
    CardCounterBot,
    HonestConservativeBot,
)
from liars_agent.engine.observation import apply_history_ablation
from liars_agent.engine.replay import GameReplayBundle
from liars_agent.engine.state import LiarsBarEngine, _derive_seed
from liars_agent.engine.types import CandidateAction


def evaluate_action_rollout_value(
    engine: LiarsBarEngine,
    actor_seat: int,
    candidate_action: CandidateAction,
    rollout_agents: list[BaseAgent],
    num_rollouts: int = 4,
    rollout_seed: int = 1000,
) -> float:
    """对当前状态下的某个候选动作开展多次蒙特卡洛续玩（Rollout），计算综合期望生存与整局胜率得分.

    评分公式：0.65 * 整局获胜率 + 0.35 * (1.0 - 下一小局开枪受罚率)
    """
    if num_rollouts <= 0:
        return 0.5

    total_score = 0.0
    initial_shots = engine.shots_taken[actor_seat]

    for r_idx in range(num_rollouts):
        sim = copy.deepcopy(engine)
        # 为续玩注入不同的后续小局洗牌随机性，避免固定后续发牌导致过拟合
        sim.game_seed = _derive_seed(engine.game_seed, "rollout", candidate_action.action_id, r_idx, rollout_seed)
        res = sim.step(candidate_action)

        # 检查是否在本步直接质疑失败或被抓开枪
        immediate_penalty = 1.0 if (sim.shots_taken[actor_seat] > initial_shots) else 0.0

        step_guard = 0
        while not sim.is_game_over() and step_guard < 120:
            step_guard += 1
            turn = sim.current_turn_seat
            obs = sim.get_observation(turn)
            act, rec = rollout_agents[turn].select_action(obs)
            sim.step(act, rec)
            if res is None and sim.shots_taken[actor_seat] > initial_shots:
                immediate_penalty = 1.0

        won_game = 1.0 if sim.winner_seat == actor_seat else 0.0
        total_score += 0.65 * won_game + 0.35 * (1.0 - immediate_penalty)

    return total_score / num_rollouts


def generate_mixed_opponent_sft_dataset(
    num_games: int = 120,
    target_samples: int = 1000,
    num_rollouts_per_action: int = 4,
    min_advantage_margin: float = 0.05,
    history_ablation_ratio: float = 0.20,
    base_seed: int = 2026,
    output_jsonl_path: Optional[str | Path] = None,
) -> tuple[list[dict[str, Any]], list[GameReplayBundle]]:
    """通过训练对手池混合博弈与多次续玩（Rollout）收益筛选生成高质量 SFT 指令微调样本.

    返回：(sft_samples, game_replay_bundles)
    """
    rng = random.Random(base_seed)
    sft_samples: list[dict[str, Any]] = []
    replay_bundles: list[GameReplayBundle] = []

    for g_idx in range(num_games):
        game_seed = _derive_seed(base_seed, "train_game", g_idx)
        game_id = f"train_game_{g_idx:05d}"

        # 从可见训练池随机采样 4 个座位的混合策略组合（确保至少包含 1 个精算型和不同激进度对手）
        pool_factories = [
            lambda s: HonestConservativeBot(
                name="honest_conservative",
                seed=s,
                challenge_threshold=rng.uniform(0.65, 0.78),
            ),
            lambda s: AggressiveBlufferBot(
                name="aggressive_bluffer",
                seed=s,
                bluff_rate=rng.uniform(0.55, 0.78),
            ),
            lambda s: CardCounterBot(
                name="card_counter",
                seed=s,
                posterior_challenge_cutoff=rng.uniform(0.50, 0.60),
            ),
        ]
        agents: list[BaseAgent] = [
            pool_factories[(g_idx + seat) % len(pool_factories)](_derive_seed(game_seed, seat))
            for seat in range(4)
        ]

        engine = LiarsBarEngine(game_id=game_id, game_seed=game_seed)

        while not engine.is_game_over():
            seat = engine.current_turn_seat
            obs = engine.get_observation(seat)
            legal_actions = obs.legal_actions

            # 若有多个合法候选动作且尚未收集满目标样本数，执行续玩评估选出最优示范动作
            if len(legal_actions) > 1 and len(sft_samples) < target_samples:
                action_values: dict[str, float] = {}
                for cand in legal_actions:
                    val = evaluate_action_rollout_value(
                        engine=engine,
                        actor_seat=seat,
                        candidate_action=cand,
                        rollout_agents=agents,
                        num_rollouts=num_rollouts_per_action,
                        rollout_seed=game_seed + engine.step_id,
                    )
                    action_values[cand.action_id] = val

                best_action = max(legal_actions, key=lambda a: action_values[a.action_id])
                mean_val = sum(action_values.values()) / len(action_values)
                advantage = action_values[best_action.action_id] - mean_val

                if advantage >= min_advantage_margin:
                    # 构造经过历史消融增强或完整历史的观测
                    ablation_mode = (
                        "current_round_only"
                        if rng.random() < history_ablation_ratio
                        else "full"
                    )
                    prompt_obs = apply_history_ablation(obs, ablation_mode)
                    bluff_prob, est_style, rationale = estimate_bayesian_bluff_belief(
                        prompt_obs,
                        use_history_adaptation=(ablation_mode == "full"),
                    )
                    target_completion = json.dumps(
                        {
                            "belief_bluff_prob": bluff_prob if prompt_obs.last_claim else 0.0,
                            "estimated_opponent_style": est_style,
                            "chosen_action_id": best_action.action_id,
                            "rationale": f"{rationale} 续玩期望收益优势=+{advantage:.2f}。",
                        },
                        ensure_ascii=False,
                    )
                    sft_samples.append(
                        {
                            "sample_id": f"{game_id}_r{obs.round_id}_s{obs.step_id}",
                            "game_id": game_id,
                            "round_id": obs.round_id,
                            "step_id": obs.step_id,
                            "actor_seat": seat,
                            "ablation_mode": ablation_mode,
                            "system": SYSTEM_PROMPT,
                            "prompt": build_user_prompt(prompt_obs),
                            "completion": target_completion,
                            "rollout_best_value": round(action_values[best_action.action_id], 4),
                            "rollout_advantage": round(advantage, 4),
                        }
                    )

            chosen_action, rec = agents[seat].select_action(obs)
            engine.step(chosen_action, rec)

        replay_bundles.append(
            GameReplayBundle.from_engine(
                engine, seat_agent_names=[a.name for a in agents]
            )
        )
        if len(sft_samples) >= target_samples:
            break

    if output_jsonl_path is not None:
        out = Path(output_jsonl_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for item in sft_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return sft_samples, replay_bundles

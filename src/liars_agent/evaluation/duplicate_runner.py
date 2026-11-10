"""座位轮换配对对局评测执行器 (Seat-Rotated Duplicate Match Runner).

以对局块（Block = 4 局）为单位评测被测智能体：
在每个 Block 内固定 `block_seed`（发牌序列与各座位实弹槽位完全固定），
将被测智能体依次轮换至座位 0, 1, 2, 3，从而消除发牌运气与先后手座位偏差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from liars_agent.agents.base import BaseAgent
from liars_agent.engine.observation import apply_history_ablation
from liars_agent.engine.replay import GameReplayBundle
from liars_agent.engine.state import LiarsBarEngine, _derive_seed
from liars_agent.engine.types import AblationMode


@dataclass
class DuplicateEvalResult:
    """单个被测智能体在 N 个座位轮换对局块上的完整评测结果."""

    agent_name: str
    pool_name: str
    ablation_mode: AblationMode
    num_blocks: int
    total_games: int
    overall_win_rate: float
    seat_win_rates: dict[int, float]
    block_win_rates: list[float]
    replays: list[GameReplayBundle]


class _AblatedAgentWrapper(BaseAgent):
    """在评测时对指定智能体注入历史消融模式的透明包装器."""

    def __init__(self, inner: BaseAgent, ablation_mode: AblationMode) -> None:
        super().__init__(name=inner.name, style_tag=inner.style_tag)
        self.inner = inner
        self.ablation_mode: AblationMode = ablation_mode

    def select_action(self, obs):
        ablated_obs = apply_history_ablation(obs, self.ablation_mode)
        return self.inner.select_action(ablated_obs)


def run_duplicate_evaluation(
    candidate_factory: Callable[[int], BaseAgent],
    opponent_factories: Sequence[Callable[[int], BaseAgent]],
    num_blocks: int = 30,
    base_seed: int = 8000,
    pool_name: str = "seen_pool",
    ablation_mode: AblationMode = "full",
) -> DuplicateEvalResult:
    """执行座位轮换配对评测.

    参数：
    - `candidate_factory`: 接收随机种子并返回被测智能体实例的工厂函数
    - `opponent_factories`: 长度为 3 的对手工厂列表（占据其余 3 个座位）
    - `num_blocks`: 对局块数量（每个块包含 4 局座位轮换对局，总对局数 = 4 * num_blocks）
    - `ablation_mode`: 被测智能体的历史信息消融模式 ('full' | 'current_round_only' | 'last_action_only')
    """
    if len(opponent_factories) != 3:
        raise ValueError("Duplicate evaluation requires exactly 3 opponent factories.")

    block_win_rates: list[float] = []
    seat_wins: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    replays: list[GameReplayBundle] = []
    candidate_name = "candidate"

    for b_idx in range(num_blocks):
        block_seed = _derive_seed(base_seed, "eval_block", b_idx)
        wins_in_block = 0

        for cand_seat in range(4):
            game_id = f"eval_{pool_name}_{ablation_mode}_b{b_idx:03d}_s{cand_seat}"
            cand_raw = candidate_factory(_derive_seed(block_seed, "cand", cand_seat))
            candidate_name = cand_raw.name
            cand_agent = _AblatedAgentWrapper(cand_raw, ablation_mode)

            # 将被测智能体放在 cand_seat，其余 3 个对手按固定相对顺序落座
            lineup: list[BaseAgent] = [None] * 4  # type: ignore[list-item]
            lineup[cand_seat] = cand_agent
            for offset in range(1, 4):
                opp_seat = (cand_seat + offset) % 4
                opp_factory = opponent_factories[offset - 1]
                lineup[opp_seat] = opp_factory(_derive_seed(block_seed, "opp", opp_seat))

            engine = LiarsBarEngine(game_id=game_id, game_seed=block_seed)
            while not engine.is_game_over():
                turn = engine.current_turn_seat
                obs = engine.get_observation(turn)
                act, rec = lineup[turn].select_action(obs)
                engine.step(act, rec)

            if engine.winner_seat == cand_seat:
                wins_in_block += 1
                seat_wins[cand_seat] += 1

            replays.append(
                GameReplayBundle.from_engine(
                    engine, seat_agent_names=[a.name for a in lineup]
                )
            )

        block_win_rates.append(wins_in_block / 4.0)

    total_games = num_blocks * 4
    overall_wr = sum(block_win_rates) / max(1, num_blocks)
    seat_wrs = {s: round(seat_wins[s] / max(1, num_blocks), 4) for s in range(4)}

    return DuplicateEvalResult(
        agent_name=candidate_name,
        pool_name=pool_name,
        ablation_mode=ablation_mode,
        num_blocks=num_blocks,
        total_games=total_games,
        overall_win_rate=round(overall_wr, 4),
        seat_win_rates=seat_wrs,
        block_win_rates=block_win_rates,
        replays=replays,
    )

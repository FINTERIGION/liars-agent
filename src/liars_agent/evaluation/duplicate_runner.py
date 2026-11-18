"""座位轮换配对对局评测执行器 (Seat-Rotated Duplicate Match Runner).

以对局块（Block = 4 局）为单位评测被测智能体：
在每个 Block 内固定 `block_seed`（发牌序列与各座位实弹槽位完全固定），
将被测智能体依次轮换至座位 0, 1, 2, 3，降低发牌与座位带来的比较噪声。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import random
from pathlib import Path
from typing import Any, Callable, Sequence

from liars_agent.agents.base import BaseAgent
from liars_agent.contracts import RULES_VERSION
from liars_agent.engine.observation import apply_history_ablation
from liars_agent.engine.replay import GameReplayBundle, save_replays_jsonl
from liars_agent.engine.state import LiarsBarEngine, _derive_seed
from liars_agent.engine.types import AblationMode
from liars_agent.evaluation.metrics import compute_block_bootstrap_ci


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
    block_records: list[dict[str, Any]] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    decision_metrics: dict[str, Any] = field(default_factory=dict)
    win_rate_ci: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_replays: bool = False) -> dict[str, Any]:
        result = {
            "agent_name": self.agent_name,
            "pool_name": self.pool_name,
            "ablation_mode": self.ablation_mode,
            "num_blocks": self.num_blocks,
            "total_games": self.total_games,
            "overall_win_rate": self.overall_win_rate,
            "seat_win_rates": self.seat_win_rates,
            "block_win_rates": self.block_win_rates,
            "block_records": self.block_records,
            "model_metadata": self.model_metadata,
            "decision_metrics": self.decision_metrics,
            "win_rate_ci": self.win_rate_ci,
        }
        if include_replays:
            result["replays"] = [replay.to_dict() for replay in self.replays]
        return result


def run_duplicate_evaluation(
    candidate_factory: Callable[[int], BaseAgent],
    opponent_factories: Sequence[Callable[[int], BaseAgent]],
    num_blocks: int = 30,
    base_seed: int = 8000,
    pool_name: str = "seen_pool",
    ablation_mode: AblationMode = "full",
    model_metadata: dict[str, Any] | None = None,
    opponent_specs: list[dict[str, Any]] | None = None,
    num_resamples: int = 10000,
    bootstrap_seed: int = 42,
    cache_dir: str | Path | None = None,
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
    if num_blocks < 1:
        raise ValueError("num_blocks must be positive.")
    if num_resamples < 1:
        raise ValueError("num_resamples must be positive.")
    if opponent_specs is not None and len(opponent_specs) != 3:
        raise ValueError("opponent_specs must describe exactly three opponents.")
    if opponent_specs is None:
        # Built-in bots expose their complete scalar configuration as attributes.
        # Custom or service-backed opponents should supply explicit config specs.
        opponent_specs = []
        for factory in opponent_factories:
            agent = factory(_derive_seed(base_seed, "opponent_metadata"))
            params = {key: value for key, value in vars(agent).items()
                      if key not in {"seed", "name", "style_tag"}
                      and isinstance(value, (str, int, float, bool, type(None)))}
            opponent_specs.append({"class": type(agent).__name__, "name": agent.name, "params": params})
    opponent_specs = deepcopy(opponent_specs)

    block_win_rates: list[float] = []
    seat_wins: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    replays: list[GameReplayBundle] = []
    candidate_name = "candidate"
    block_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    metadata = dict(model_metadata or {})
    block_cache_root = Path(cache_dir) if cache_dir is not None else None
    if block_cache_root is not None:
        block_cache_root.mkdir(parents=True, exist_ok=True)

    for b_idx in range(num_blocks):
        block_seed = _derive_seed(base_seed, "eval_block", b_idx)
        wins_in_block = 0
        block_id = f"eval_{base_seed}_b{b_idx:04d}"
        games: list[dict[str, Any]] = []
        opponent_order = [0, 1, 2]
        random.Random(_derive_seed(block_seed, "opponent_order")).shuffle(opponent_order)

        if block_cache_root is not None:
            block_json_path = block_cache_root / f"block_{b_idx:04d}.json"
            block_replay_path = block_cache_root / f"block_{b_idx:04d}.replays.jsonl"
            if block_json_path.exists() and block_replay_path.exists():
                import json
                from liars_agent.engine.replay import verify_replay_determinism
                cached_block = json.loads(block_json_path.read_text(encoding="utf-8"))
                record = cached_block.get("block_record", {})
                if (cached_block.get("ablation_mode") == ablation_mode
                        and cached_block.get("model_metadata") == metadata
                        and record.get("block_id") == block_id
                        and record.get("block_seed") == block_seed
                        and record.get("opponent_specs") == opponent_specs
                        and record.get("opponent_order") == opponent_order
                        and record.get("rules_version") == RULES_VERSION):
                    cached_replays = [
                        GameReplayBundle.from_dict(json.loads(line))
                        for line in block_replay_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    if len(cached_replays) == 4 and all(verify_replay_determinism(r) for r in cached_replays):
                        candidate_name = cached_block["candidate_name"]
                        block_win_rates.append(float(record["win_rate"]))
                        block_records.append(record)
                        for game in record["games"]:
                            if game["won"]:
                                seat_wins[int(game["candidate_seat"])] += 1
                        candidate_records.extend(cached_block.get("candidate_records", []))
                        replays.extend(cached_replays)
                        continue

        block_replays: list[GameReplayBundle] = []
        block_candidate_records: list[dict[str, Any]] = []

        for cand_seat in range(4):
            # Experiment labels must not perturb agents' random streams.
            game_id = f"{block_id}_s{cand_seat}"
            cand_agent = candidate_factory(_derive_seed(block_seed, "cand", cand_seat))
            candidate_name = cand_agent.name
            if not metadata:
                metadata = {"agent_class": type(cand_agent).__name__}

            # 每块随机一次相对排列；四座位轮换以及各模型使用同一排列。
            lineup: list[BaseAgent] = [None] * 4  # type: ignore[list-item]
            lineup[cand_seat] = cand_agent
            for offset in range(1, 4):
                opp_seat = (cand_seat + offset) % 4
                opp_factory = opponent_factories[opponent_order[offset - 1]]
                lineup[opp_seat] = opp_factory(_derive_seed(block_seed, "opp", opp_seat))

            engine = LiarsBarEngine(game_id=game_id, game_seed=block_seed)
            while not engine.is_game_over():
                turn = engine.current_turn_seat
                obs = engine.get_observation(turn)
                if turn == cand_seat:
                    obs = apply_history_ablation(obs, ablation_mode)
                act, rec = lineup[turn].select_action(obs)
                engine.step(act, rec)
                if turn == cand_seat:
                    # The replay should record what the agent actually received.
                    engine.step_truth_logs[-1].observation = obs.to_dict()
                    rec_dict = rec.to_dict()
                    candidate_records.append(rec_dict)
                    block_candidate_records.append(rec_dict)

            if engine.winner_seat == cand_seat:
                wins_in_block += 1
                seat_wins[cand_seat] += 1

            games.append({
                "game_id": game_id,
                "candidate_seat": cand_seat,
                "winner_seat": engine.winner_seat,
                "won": engine.winner_seat == cand_seat,
            })

            bundle = GameReplayBundle.from_engine(
                engine, seat_agent_names=[a.name for a in lineup]
            )
            replays.append(bundle)
            block_replays.append(bundle)

        block_win_rates.append(wins_in_block / 4.0)
        block_record = {
            "block_id": block_id,
            "block_seed": block_seed,
            "win_rate": wins_in_block / 4.0,
            "opponent_specs": deepcopy(opponent_specs),
            "opponent_order": opponent_order,
            "rules_version": RULES_VERSION,
            "games": games,
        }
        block_records.append(block_record)
        if block_cache_root is not None:
            import json
            save_replays_jsonl(block_replays, block_cache_root / f"block_{b_idx:04d}.replays.jsonl")
            tmp_path = block_cache_root / f"block_{b_idx:04d}.json.tmp"
            tmp_path.write_text(
                json.dumps(
                    {
                        "candidate_name": candidate_name,
                        "ablation_mode": ablation_mode,
                        "model_metadata": metadata,
                        "block_record": block_record,
                        "candidate_records": block_candidate_records,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            tmp_path.replace(block_cache_root / f"block_{b_idx:04d}.json")

    total_games = num_blocks * 4
    overall_wr = sum(block_win_rates) / num_blocks
    seat_wrs = {s: seat_wins[s] / num_blocks for s in range(4)}
    total_decisions = len(candidate_records)
    fallback_count = sum(bool(r.get("was_fallback")) for r in candidate_records)
    decision_metrics = {
        "total_decisions": total_decisions,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / total_decisions if total_decisions else None,
        "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in candidate_records),
        "completion_tokens": sum(r.get("completion_tokens", 0) for r in candidate_records),
        "belief_completion_tokens_upper_bound": sum(r.get("belief_completion_tokens_upper_bound", 0) for r in candidate_records),
        "mean_latency_ms": sum(r.get("latency_ms", 0.0) for r in candidate_records) / total_decisions if total_decisions else None,
    }

    return DuplicateEvalResult(
        agent_name=candidate_name,
        pool_name=pool_name,
        ablation_mode=ablation_mode,
        num_blocks=num_blocks,
        total_games=total_games,
        overall_win_rate=overall_wr,
        seat_win_rates=seat_wrs,
        block_win_rates=block_win_rates,
        replays=replays,
        block_records=block_records,
        model_metadata=metadata,
        decision_metrics=decision_metrics,
        win_rate_ci=compute_block_bootstrap_ci(block_win_rates, num_resamples, bootstrap_seed),
    )

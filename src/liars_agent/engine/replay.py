"""对局录制、JSONL 序列化与确定性回放校验器."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.types import AgentDecisionRecord, CandidateAction


@dataclass
class GameReplayBundle:
    """单局完整回放数据包（包含种子、座位智能体配置、每步决策与结局摘要）."""

    game_id: str
    game_seed: int
    seat_agent_names: list[str]
    winner_seat: int
    total_rounds: int
    total_steps: int
    final_shots_taken: list[int]
    fatal_chamber_slots: list[int]
    steps: list[dict[str, Any]]
    rounds_summary: list[dict[str, Any]]

    def compute_trajectory_hash(self) -> str:
        """计算对局轨迹的确定性 SHA256 指纹（用于验证回放 100% 一致性）."""
        canonical = {
            "game_seed": self.game_seed,
            "winner_seat": self.winner_seat,
            "total_rounds": self.total_rounds,
            "total_steps": self.total_steps,
            "final_shots_taken": self.final_shots_taken,
            "actions": [
                {
                    "round_id": s["round_id"],
                    "step_id": s["step_id"],
                    "actor_seat": s["actor_seat"],
                    "chosen_action": s["chosen_action"],
                }
                for s in self.steps
            ],
        }
        raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["trajectory_hash"] = self.compute_trajectory_hash()
        return data

    @classmethod
    def from_engine(
        cls,
        engine: LiarsBarEngine,
        seat_agent_names: Optional[list[str]] = None,
    ) -> GameReplayBundle:
        if not engine.is_game_over() or engine.winner_seat is None:
            raise ValueError("Cannot build GameReplayBundle from an unfinished game.")
        names = seat_agent_names or [f"Seat_{i}" for i in range(engine.num_players)]
        return cls(
            game_id=engine.game_id,
            game_seed=engine.game_seed,
            seat_agent_names=list(names),
            winner_seat=engine.winner_seat,
            total_rounds=engine.round_id,
            total_steps=len(engine.step_truth_logs),
            final_shots_taken=list(engine.shots_taken),
            fatal_chamber_slots=list(engine.fatal_chamber_slot),
            steps=[asdict(s) for s in engine.step_truth_logs],
            rounds_summary=[r.to_dict() for r in engine.past_rounds_history],
        )


def verify_replay_determinism(bundle: GameReplayBundle | dict[str, Any]) -> bool:
    """使用回放包中的 `game_seed` 与记录的动作序列重新在引擎中推演，验证每步状态与结局 100% 一致."""
    data = bundle.to_dict() if isinstance(bundle, GameReplayBundle) else bundle
    engine = LiarsBarEngine(
        game_id=data["game_id"],
        game_seed=int(data["game_seed"]),
    )

    for step_entry in data["steps"]:
        expected_seat = step_entry["actor_seat"]
        if engine.current_turn_seat != expected_seat:
            return False
        obs = engine.get_observation(expected_seat)
        if obs.own_hand != tuple(step_entry["observation"]["own_hand"]):
            return False
        action = CandidateAction.from_dict(step_entry["chosen_action"])
        rec = AgentDecisionRecord(
            belief_bluff_prob=step_entry["decision_record"].get("belief_bluff_prob"),
            estimated_opponent_style=step_entry["decision_record"].get(
                "estimated_opponent_style", "unknown"
            ),
            rationale=step_entry["decision_record"].get("rationale", ""),
        )
        engine.step(action, decision_record=rec)

    if not engine.is_game_over():
        return False
    replayed = GameReplayBundle.from_engine(engine, data.get("seat_agent_names"))
    expected_hash = data.get("trajectory_hash") or GameReplayBundle(**{
        k: v for k, v in data.items() if k != "trajectory_hash"
    }).compute_trajectory_hash()
    return replayed.compute_trajectory_hash() == expected_hash


def save_replays_jsonl(bundles: list[GameReplayBundle], path: str | Path) -> None:
    """将多局回放保存为 JSONL 文件."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for b in bundles:
            f.write(json.dumps(b.to_dict(), ensure_ascii=False) + "\n")


def load_replays_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """从 JSONL 文件加载回放列表."""
    in_path = Path(path)
    records: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

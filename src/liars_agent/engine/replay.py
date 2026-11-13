"""Recorded-game fingerprints and deterministic engine-state verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from liars_agent.contracts import RULES_VERSION, OBSERVATION_VERSION
from liars_agent.engine.observation import apply_history_ablation
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.types import AgentDecisionRecord, CandidateAction


def _normalize(value: Any) -> Any:
    # JSONL turns tuples into arrays and integer object keys into strings.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _payload_hash(data: dict[str, Any]) -> str:
    payload = {key: value for key, value in data.items() if key != "trajectory_hash"}
    raw = json.dumps(_normalize(payload), sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class GameReplayBundle:
    """A complete recording, including private truth only for post-game use."""

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
    rules_version: str = RULES_VERSION
    observation_version: str = OBSERVATION_VERSION

    def compute_trajectory_hash(self) -> str:
        """Fingerprint the complete recording, including recorded decision metadata.

        This detects changes to the saved data. The engine verifier separately
        reconstructs deterministic game fields; it does not regenerate LLM text,
        probability predictions, or inference timings.
        """
        return _payload_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["trajectory_hash"] = _payload_hash(data)
        return data

    @classmethod
    def from_engine(cls, engine: LiarsBarEngine,
                    seat_agent_names: Optional[list[str]] = None) -> GameReplayBundle:
        if not engine.is_game_over() or engine.winner_seat is None:
            raise ValueError("Cannot build GameReplayBundle from an unfinished game.")
        if not engine.record_truth:
            raise ValueError("Cannot build a complete replay when truth recording is disabled.")
        names = seat_agent_names or [f"Seat_{i}" for i in range(engine.num_players)]
        return cls(game_id=engine.game_id, game_seed=engine.game_seed,
            seat_agent_names=list(names), winner_seat=engine.winner_seat,
            total_rounds=engine.round_id, total_steps=len(engine.step_truth_logs),
            final_shots_taken=list(engine.shots_taken),
            fatal_chamber_slots=list(engine.fatal_chamber_slot),
            steps=[asdict(step) for step in engine.step_truth_logs],
            rounds_summary=[resolution.to_dict() for resolution in engine.past_rounds_history])


class ReplayVerificationError(ValueError):
    """First mismatching deterministic field, or an invalid recording checksum."""

    def __init__(self, field: str, reason: str, step_index: int | None = None) -> None:
        self.field = field
        self.reason = reason
        self.step_index = step_index
        super().__init__(f"Replay verification failed at {field}: {reason}")


def _first_difference(actual: Any, expected: Any, path: str) -> tuple[str, str] | None:
    if type(actual) is not type(expected):
        return path, f"expected {type(expected).__name__}, found {type(actual).__name__}"
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            missing = sorted(expected.keys() - actual.keys())
            extra = sorted(actual.keys() - expected.keys())
            return path, f"missing keys {missing}; extra keys {extra}"
        for key in expected:
            mismatch = _first_difference(actual[key], expected[key], f"{path}.{key}" if path else key)
            if mismatch:
                return mismatch
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            return path, f"expected {len(expected)} entries, found {len(actual)}"
        for index, (left, right) in enumerate(zip(actual, expected)):
            mismatch = _first_difference(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
    elif actual != expected:
        return path, f"expected {expected!r}, found {actual!r}"
    return None


def _expect_equal(actual: Any, expected: Any, field: str, step_index: int | None = None) -> None:
    mismatch = _first_difference(_normalize(actual), _normalize(expected), field)
    if mismatch:
        raise ReplayVerificationError(*mismatch, step_index=step_index)


def verify_replay_determinism(bundle: GameReplayBundle | dict[str, Any], *,
                              raise_on_error: bool = False) -> bool:
    """Replay every action and verify all reconstructable state and truth fields.

    Saved observations are reconstructed with their recorded public-history
    ablation. Decision metadata is preserved, not independently regenerated;
    the recording's checksum protects it from unnoticed edits. Set
    ``raise_on_error=True`` to receive the first failing step and field.
    """
    step_index = None
    context_field = "recording"
    try:
        data = _normalize(bundle.to_dict() if isinstance(bundle, GameReplayBundle) else bundle)
        _expect_equal(data.get("rules_version"), RULES_VERSION, "rules_version")
        _expect_equal(data.get("observation_version"), OBSERVATION_VERSION, "observation_version")
        engine = LiarsBarEngine(game_id=data["game_id"], game_seed=data["game_seed"])
        for step_index, step_entry in enumerate(data["steps"]):
            context_field = f"steps[{step_index}]"
            if engine.is_game_over():
                raise ReplayVerificationError(context_field, "recorded action after game end", step_index)
            _expect_equal(step_entry["actor_seat"], engine.current_turn_seat,
                          f"{context_field}.actor_seat", step_index)
            mode = step_entry["observation"]["ablation_mode"]
            observation = apply_history_ablation(engine.get_observation(), mode)
            _expect_equal(step_entry["observation"], observation.to_dict(),
                          f"{context_field}.observation", step_index)
            action = CandidateAction.from_dict(step_entry["chosen_action"])
            record = AgentDecisionRecord(**step_entry["decision_record"])
            engine.step(action, decision_record=record)
            # The evaluation runner logs the actual ablated input instead of the
            # engine's full observation. Reproduce that exact public projection.
            engine.step_truth_logs[-1].observation = observation.to_dict()
            _expect_equal(step_entry, asdict(engine.step_truth_logs[-1]), context_field, step_index)
        step_index = None
        context_field = "winner_seat"
        if not engine.is_game_over():
            raise ReplayVerificationError(context_field, "recording ends before the game is finished")
        reconstructed = GameReplayBundle.from_engine(engine, data["seat_agent_names"]).to_dict()
        _expect_equal({k: v for k, v in data.items() if k not in ("steps", "trajectory_hash")},
                      {k: v for k, v in reconstructed.items() if k not in ("steps", "trajectory_hash")},
                      "recording")
        _expect_equal(data.get("trajectory_hash"), _payload_hash(data), "trajectory_hash")
        _expect_equal(data["trajectory_hash"], reconstructed["trajectory_hash"], "trajectory_hash")
        return True
    except ReplayVerificationError:
        if raise_on_error:
            raise
        return False
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as exc:
        if raise_on_error:
            raise ReplayVerificationError(context_field, str(exc), step_index) from exc
        return False


def save_replays_jsonl(bundles: list[GameReplayBundle], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for bundle in bundles:
            stream.write(json.dumps(bundle.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")


def load_replays_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

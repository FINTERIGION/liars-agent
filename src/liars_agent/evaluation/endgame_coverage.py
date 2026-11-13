"""An independent, deliberately balanced endgame check set, not training data.

Games start from the normal engine and advance only through legal actions. The
coverage policy sees its own observation. Hidden labels are used only to select
the completed games' evaluation examples, and are kept separate from inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from liars_agent.agents.base import describe_action
from liars_agent.agents.prompt_builder import (
    SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT, build_user_prompt, build_bluff_user_prompt,
)
from liars_agent.contracts import current_contract, require_contract
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.replay import GameReplayBundle, verify_replay_determinism
from liars_agent.engine.state import LiarsBarEngine, _derive_seed
from liars_agent.engine.types import AgentDecisionRecord, is_card_truthful

SCENARIOS = ("forced_challenge", "empty_claimer_free", "one_card_free")
UNREACHABLE_REASON = (
    "With two survivors, an empty last claimer leaves only the current player "
    "holding cards. Rule 5 therefore requires CHALLENGE and forbids PLAY."
)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_game_identity(source_run: str | Path) -> tuple[set[str], set[int]]:
    """Read source identities without rerunning the main dataset's game checks."""
    source = Path(source_run)
    require_contract(json.loads((source / "config.json").read_text()), "source run")
    partition = json.loads((source / "dataset/game_splits.json").read_text())
    if "game_splits" in partition:
        partition = partition["game_splits"]
    ids = {game for split in ("train", "val", "test") for game in partition[split]}
    recorded_ids, seeds = set(), set()
    paths = sorted((source / "dataset/batches").glob("replays_*.jsonl"))
    if not paths:
        raise ValueError("Source replay files are required to check environment-seed independence.")
    for path in paths:
        for replay in _read_jsonl(path):
            recorded_ids.add(replay["game_id"])
            seeds.add(replay["game_seed"])
    if not ids <= recorded_ids:
        raise ValueError("Source split IDs are missing from the source replay files.")
    return ids | recorded_ids, seeds


def classify_endgame_observation(observation: dict[str, Any]) -> list[str]:
    """Classify only public conditions, independently of the hidden bluff label."""
    legal = observation["legal_actions"]
    claim = observation["last_claim"]
    if claim is None:
        return []
    can_play = any(action["action_type"] == "PLAY" for action in legal)
    can_challenge = any(action["action_type"] == "CHALLENGE" for action in legal)
    tags = []
    if len(legal) == 1 and can_challenge:
        tags.append("forced_challenge")
    if claim["claimer_remaining_cards"] == 0 and can_play and can_challenge:
        tags.append("empty_claimer_free")
    if len(observation["own_hand"]) == 1 and can_play and can_challenge:
        tags.append("one_card_free")
    return tags


def _cell(tag: str, alive: int, label: int) -> str:
    return f"{tag}/alive_{alive}/label_{label}"


def _reachable_cells() -> list[str]:
    return [_cell(tag, alive, label) for tag in SCENARIOS for alive in (2, 3, 4)
            for label in (0, 1) if not (tag == "empty_claimer_free" and alive == 2)]


def _coverage_action(obs: PlayerObservation, seed: int):
    """Keep playing towards empty hands; alternate keeping true/false last cards."""
    plays = [action for action in obs.legal_actions if action.action_type == "PLAY"]
    if not plays:
        return obs.legal_actions[0]
    rng = random.Random(_derive_seed(seed, "coverage_policy", obs.round_id, obs.step_id, obs.observer_seat))
    size = len(obs.own_hand)
    play_count = rng.choice((2, 3)) if size == 5 else max(1, min(3, size - 1))
    candidates = [action for action in plays if len(action.cards) == play_count]
    keep_truth = (obs.round_id + obs.observer_seat + seed) % 2 == 0
    def preference(action):
        true_cards = sum(is_card_truthful(card, obs.target_rank) for card in action.cards)
        return len(action.cards) - true_cards if keep_truth else true_cards
    best = max(map(preference, candidates))
    return rng.choice([action for action in candidates if preference(action) == best])


def generate_coverage_replay(game_id: str, game_seed: int, policy_seed: int) -> dict:
    """Play a complete legal coverage trajectory with explicit independent seeds."""
    engine = LiarsBarEngine(game_id=game_id, game_seed=game_seed)
    while not engine.is_game_over():
        obs = engine.get_observation()
        action = _coverage_action(obs, policy_seed)
        engine.step(action, AgentDecisionRecord(
            rationale=describe_action(obs, action), rationale_source="action_description",
            belief_source="not_requested",
        ))
    return GameReplayBundle.from_engine(engine, ["endgame_coverage_policy"] * 4).to_dict()


def _public_sample(sample_id: str, observation: dict, tags: list[str]) -> dict:
    obs = PlayerObservation.from_dict(observation)
    return {"sample_id": sample_id, "observation": observation,
            "legal_actions": observation["legal_actions"], "scenario_tags": tags,
            "alive_players": sum(player["alive"] for player in observation["players_public_status"]),
            "action_system": SYSTEM_PROMPT, "action_prompt": build_user_prompt(obs),
            "belief_system": BELIEF_SYSTEM_PROMPT if obs.last_claim else None,
            "belief_prompt": build_bluff_user_prompt(obs) if obs.last_claim else None}


def build_endgame_coverage(
    source_run: str | Path, output_dir: str | Path, *, seed: int = 2026111301,
    samples_per_cell: int = 10, max_games: int = 500, no_claim_per_alive: int = 5,
) -> dict[str, Any]:
    """Build a bounded, independently seeded check set; never append to training.

    A game contributes at most two examples to each labelled cell, giving at
    least five distinct source games when samples_per_cell=10. Multi-tag rows
    are stored once and may make a cell exceed its minimum requested quota.
    """
    source, output = Path(source_run).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents:
        raise ValueError("Coverage output must be outside the source run.")
    if samples_per_cell < 1 or max_games < 1 or no_claim_per_alive < 0:
        raise ValueError("Positive cell/game counts and nonnegative no-claim count are required.")
    if (output / "private/generation_metadata.json").exists():
        raise FileExistsError("Coverage output already exists; use a new output directory.")
    source_ids, source_seeds = source_game_identity(source)
    counts = Counter({cell: 0 for cell in _reachable_cells()})
    no_claim_counts = Counter({alive: 0 for alive in (2, 3, 4)})
    observations, no_claim, labels, locations, replays = [], [], [], [], []
    namespace = hashlib.sha256(f"endgame_coverage_v1:{seed}".encode()).hexdigest()[:16]
    used_seeds = set()
    games_attempted = 0
    for game_index in range(max_games):
        if (all(count >= samples_per_cell for count in counts.values())
                and all(count >= no_claim_per_alive for count in no_claim_counts.values())):
            break
        game_seed = _derive_seed(seed, "independent_endgame_environment_v1", game_index)
        game_id = f"endgame_{namespace}_{game_index:05d}"
        if game_seed in source_seeds or game_seed in used_seeds or game_id in source_ids:
            raise ValueError("Generated environment identity overlaps another source game.")
        used_seeds.add(game_seed)
        replay = generate_coverage_replay(game_id, game_seed,
            _derive_seed(seed, "independent_endgame_policy_v1", game_index))
        games_attempted += 1
        per_game_counts, no_claim_seen = Counter(), set()
        selected = 0
        # Deterministic shuffle prevents always choosing neighbouring early states.
        step_indices = list(range(len(replay["steps"])))
        random.Random(_derive_seed(seed, "coverage_selection_v1", game_index)).shuffle(step_indices)
        for step_index in step_indices:
            step = replay["steps"][step_index]
            observation = step["observation"]
            alive = sum(player["alive"] for player in observation["players_public_status"])
            truth = step["pending_claim_was_bluff"]
            tags = classify_endgame_observation(observation)
            if truth is None:
                if (observation["last_claim"] is not None or alive in no_claim_seen
                        or no_claim_counts[alive] >= no_claim_per_alive):
                    continue
                tags = ["no_pending_claim"]
                no_claim_seen.add(alive)
                no_claim_counts[alive] += 1
            else:
                cells = [_cell(tag, alive, int(truth)) for tag in tags]
                if (not cells or not any(counts[cell] < samples_per_cell for cell in cells)
                        or any(per_game_counts[cell] >= 2 for cell in cells)):
                    continue
                for cell in cells:
                    counts[cell] += 1
                    per_game_counts[cell] += 1
            sample_id = f"endgame_sample_{len(labels):05d}"
            row = _public_sample(sample_id, observation, tags)
            (no_claim if truth is None else observations).append(row)
            labels.append({"sample_id": sample_id, "label": None if truth is None else int(truth),
                           "source": "engine_pending_claim_truth" if truth is not None else "not_applicable"})
            locations.append({"sample_id": sample_id, "game_id": game_id, "step_index": step_index})
            selected += 1
        if selected:
            replays.append(replay)

    _write_jsonl(output / "observations.jsonl", observations)
    _write_jsonl(output / "no_pending_claim.jsonl", no_claim)
    _write_jsonl(output / "labels.jsonl", labels)
    _write_jsonl(output / "private/sample_sources.jsonl", locations)
    _write_jsonl(output / "private/replays.jsonl", replays)
    _write_json(output / "private/generation_metadata.json", {
        "contract": current_contract(), "source_run": str(source), "seed": seed,
        "samples_per_cell": samples_per_cell, "max_games": max_games,
        "no_claim_per_alive": no_claim_per_alive, "games_generated": games_attempted,
        "selection": "purpose-built coverage policy; post-hoc truth-balanced selection; max two per cell per game",
        "intended_use": "independent diagnostic check set only; not training or natural-distribution evaluation",
    })
    report = _validate_endgame_coverage(output, (source_ids, source_seeds))
    _write_json(output / "coverage_report.json", report)
    return report


def validate_endgame_coverage(output_dir: str | Path, source_run: str | Path) -> dict[str, Any]:
    """Check stored inputs, labels, rules, complete replays and source independence."""
    return _validate_endgame_coverage(Path(output_dir), source_game_identity(source_run))


def _validate_endgame_coverage(output: Path, source_identity) -> dict[str, Any]:
    metadata = json.loads((output / "private/generation_metadata.json").read_text())
    require_contract(metadata, "endgame coverage")
    def indexed(path, key):
        result = {}
        for record in _read_jsonl(output / path):
            if record[key] in result:
                raise ValueError(f"Duplicate {key} in {path}.")
            result[record[key]] = record
        return result
    rows = indexed("observations.jsonl", "sample_id")
    no_claim = indexed("no_pending_claim.jsonl", "sample_id")
    if rows.keys() & no_claim.keys():
        raise ValueError("A sample occurs in both labelled and no-claim sets.")
    rows.update(no_claim)
    labels = indexed("labels.jsonl", "sample_id")
    locations = indexed("private/sample_sources.jsonl", "sample_id")
    if rows.keys() != labels.keys() or rows.keys() != locations.keys():
        raise ValueError("Input, label and replay-reference sample IDs differ.")
    replays = indexed("private/replays.jsonl", "game_id")
    source_ids, source_seeds = source_identity
    replay_seeds = [replay["game_seed"] for replay in replays.values()]
    if replays.keys() & source_ids or set(replay_seeds) & source_seeds:
        raise ValueError("Coverage source games overlap the main dataset by ID or environment seed.")
    if len(set(replay_seeds)) != len(replay_seeds):
        raise ValueError("Coverage games share an environment seed.")
    for replay in replays.values():
        verify_replay_determinism(replay, raise_on_error=True)
    counts = Counter({cell: 0 for cell in _reachable_cells()})
    per_cell_games = defaultdict(Counter)
    no_claim_counts = Counter({alive: 0 for alive in (2, 3, 4)})
    unique_steps = set()
    for sample_id, row in rows.items():
        location = locations[sample_id]
        source_key = (location["game_id"], location["step_index"])
        if source_key in unique_steps:
            raise ValueError("One source observation has multiple sample IDs.")
        unique_steps.add(source_key)
        step = replays[location["game_id"]]["steps"][location["step_index"]]
        observation = step["observation"]
        truth = step["pending_claim_was_bluff"]
        expected_label = None if truth is None else int(truth)
        if labels[sample_id]["label"] != expected_label:
            raise ValueError(f"Sample {sample_id} label differs from the source replay truth.")
        tags = ["no_pending_claim"] if truth is None else classify_endgame_observation(observation)
        if row != _public_sample(sample_id, observation, tags):
            raise ValueError(f"Sample {sample_id} input, prompt or classification differs from its source observation.")
        alive = row["alive_players"]
        legal = observation["legal_actions"]
        current = observation["observer_seat"]
        others_with_cards = [p for p in observation["players_public_status"]
                             if p["alive"] and p["seat"] != current and p["hand_count"] > 0]
        should_force = observation["last_claim"] is not None and not others_with_cards
        is_forced = len(legal) == 1 and legal[0]["action_type"] == "CHALLENGE"
        if is_forced != should_force:
            raise ValueError("Forced-challenge legal actions contradict the public hand counts.")
        if truth is None:
            if sample_id not in no_claim or observation["last_claim"] is not None:
                raise ValueError("No-claim sample has a pending claim or was mixed into labelled inputs.")
            no_claim_counts[alive] += 1
        else:
            if sample_id in no_claim or not tags:
                raise ValueError("Labelled sample is not an eligible endgame observation.")
            for tag in tags:
                cell = _cell(tag, alive, int(truth))
                if cell not in counts:
                    raise ValueError("An observation claims a rule-impossible coverage cell.")
                if tag != "forced_challenge" and not any(action["action_type"] == "PLAY" for action in legal):
                    raise ValueError("A free-action sample has no legal PLAY.")
                counts[cell] += 1
                per_cell_games[cell][location["game_id"]] += 1
    if any(max(games.values(), default=0) > 2 for games in per_cell_games.values()):
        raise ValueError("A source game contributes more than two observations to a coverage cell.")
    requested = metadata["samples_per_cell"]
    matrix = [{"scenario": tag, "alive_players": alive, "label": label,
               "status": "unreachable" if tag == "empty_claimer_free" and alive == 2 else "covered" if counts[_cell(tag, alive, label)] >= requested else "shortfall",
               "samples": counts[_cell(tag, alive, label)],
               "source_games": len(per_cell_games[_cell(tag, alive, label)]),
               **({"reason": UNREACHABLE_REASON} if tag == "empty_claimer_free" and alive == 2 else {})}
              for tag in SCENARIOS for alive in (2, 3, 4) for label in (0, 1)]
    complete = all(count >= requested for count in counts.values()) and all(
        count >= metadata["no_claim_per_alive"] for count in no_claim_counts.values())
    return {"contract": current_contract(), "status": "complete" if complete else "shortfall",
            "purpose": "independent_endgame_coverage_only", "natural_distribution": False,
            "samples_per_reachable_cell_requested": requested,
            "num_labelled_observations": len(rows) - len(no_claim),
            "num_no_claim_observations": len(no_claim), "no_claim_by_alive_players": dict(no_claim_counts),
            "games_generated": metadata["games_generated"], "source_games_saved": len(replays),
            "all_saved_replays_verified": True, "all_samples_match_source_observation_and_truth": True,
            "main_source_games_checked": len(source_ids), "main_source_seeds_checked": len(source_seeds),
            "overlapping_game_ids": 0, "overlapping_environment_seeds": 0,
            "max_samples_per_cell_per_game": 2, "coverage_matrix": matrix,
            "unreachable_explanation": UNREACHABLE_REASON,
            "model_input_files": ["observations.jsonl", "no_pending_claim.jsonl"],
            "evaluator_only_files": ["labels.jsonl", "private/replays.jsonl", "private/sample_sources.jsonl", "private/generation_metadata.json"],
            "interpretation": "Truth-balanced diagnostic strata; no aggregate win rate or natural-distribution accuracy is estimated."}

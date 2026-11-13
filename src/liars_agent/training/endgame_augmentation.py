"""Replace ordinary training actions with independently generated endgame actions.

The truth-balanced coverage quotas apply only to action imitation. Belief data,
validation/test data, and the independent diagnostic set are copied unchanged.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

from liars_agent.agents.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from liars_agent.contracts import current_contract, require_contract
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.replay import verify_replay_determinism
from liars_agent.engine.state import _derive_seed
from liars_agent.evaluation.endgame_coverage import (
    _cell, _reachable_cells, classify_endgame_observation, generate_coverage_replay,
)
from liars_agent.experiment import read_json, read_jsonl, teacher_config, write_json, write_jsonl
from liars_agent.training.action_refinement import _summarize
from liars_agent.training.action_supervision import (
    make_action_supervision, mean_action_weight, normalize_action_policy,
    resolve_action_supervision,
)
from liars_agent.training.data_generator import evaluate_observation_action_rewards, make_action_sample
from liars_agent.training.dataset_quality import file_hash, input_key
from liars_agent.training.prepare_supervision import validate_prepared_supervision_run
from liars_agent.training.tinker_train import validate_training_sample

ENDGAME_AUGMENTATION_VERSION = "training_endgame_coverage_v1"
ENDGAME_TEACHER_VERSION = "fixed_count_endgame_teacher_v1"
SPLITS = ("train", "val", "test")


def _tree_hashes(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): file_hash(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def _source_bindings(source: Path) -> dict[str, str]:
    paths = ["config.json", "dataset/data_summary.json", "dataset/game_splits.json",
             "dataset/sft_all_samples.jsonl", "models/small_classifier_weights.json",
             "quality/action_supervision_summary.json"]
    paths += [f"dataset/{prefix}{split}.jsonl" for split in SPLITS
              for prefix in ("", "action_", "belief_")]
    return {p: file_hash(source / p) for p in paths}


def _forbidden_sources(source: Path, check_dir: Path) -> tuple[set, set, set]:
    """Include all original source games, not only games retained in training."""
    summary = read_json(source / "dataset/data_summary.json")
    ids, seeds, inputs = set(), set(), set()
    for batch in summary["batches"]:
        for replay in read_jsonl(Path(batch["replays_path"])):
            ids.add(replay["game_id"]); seeds.add(replay["game_seed"])
    partition = read_json(source / "dataset/game_splits.json")
    if not {gid for values in partition.values() for gid in values} <= ids:
        raise ValueError("Source partition games are absent from their original replays.")
    for replay in read_jsonl(check_dir / "private/replays.jsonl"):
        ids.add(replay["game_id"]); seeds.add(replay["game_seed"])
    for split in SPLITS:
        for row in read_jsonl(source / f"dataset/action_{split}.jsonl"):
            inputs.add(input_key(row))
    for filename in ("observations.jsonl", "no_pending_claim.jsonl"):
        for row in read_jsonl(check_dir / filename):
            inputs.add(input_key({"system": row["action_system"], "prompt": row["action_prompt"]}))
    return ids, seeds, inputs


def _game_identity(seed: int, index: int) -> tuple[str, int, int]:
    namespace = hashlib.sha256(f"{ENDGAME_AUGMENTATION_VERSION}:{seed}".encode()).hexdigest()[:16]
    return (f"training_endgame_{namespace}_{index:05d}",
            _derive_seed(seed, "training_endgame_environment_v1", index),
            _derive_seed(seed, "training_endgame_policy_v1", index))


def _candidate(replay: dict, game_index: int, step_index: int) -> dict:
    step = replay["steps"][step_index]
    obs = PlayerObservation.from_dict(step["observation"])
    label = step["pending_claim_was_bluff"]
    tags = classify_endgame_observation(step["observation"])
    if type(label) is not bool or not tags:
        raise ValueError("Endgame training sources need a pending claim and a reachable scenario.")
    row = make_action_sample(obs, obs.legal_actions[0], replay["game_id"])
    return {"sample_id": row["sample_id"], "game_id": replay["game_id"],
        "game_index": game_index, "step_index": step_index, "observation": obs.to_dict(),
        "scenario_tags": tags, "label": int(label), "input_key": input_key(row)}


def _coverage_matrix(candidates: list[dict]) -> list[dict]:
    counts, games = Counter(), defaultdict(Counter)
    for row in candidates:
        alive = sum(p["alive"] for p in row["observation"]["players_public_status"])
        for tag in row["scenario_tags"]:
            cell = _cell(tag, alive, row["label"])
            if cell not in _reachable_cells():
                raise ValueError("Training source claims a rule-impossible endgame cell.")
            counts[cell] += 1; games[cell][row["game_id"]] += 1
    return [{"cell": cell, "samples": counts[cell], "source_games": len(games[cell]),
             "max_samples_per_game": max(games[cell].values(), default=0)}
            for cell in _reachable_cells()]


def _validate_generation(output: Path, settings: dict, forbidden: tuple) -> list[dict]:
    directory = output / "augmentation/private"
    metadata = read_json(directory / "generation_metadata.json")
    require_contract(metadata, "training endgame generation")
    if metadata["settings"] != settings:
        raise ValueError("Endgame generation belongs to another source or configuration.")
    candidates = list(read_jsonl(directory / "candidates.jsonl"))
    replays = list(read_jsonl(directory / "replays.jsonl"))
    by_game = {r["game_id"]: r for r in replays}
    if len(by_game) != len(replays):
        raise ValueError("Duplicate supplemental game identity.")
    forbidden_ids, forbidden_seeds, forbidden_inputs = forbidden
    seeds = [r["game_seed"] for r in replays]
    if set(by_game) & forbidden_ids or set(seeds) & forbidden_seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Supplemental games overlap original or independent-check identities.")
    for replay in replays:
        verify_replay_determinism(replay, raise_on_error=True)
    used_steps, seen_inputs = set(), set()
    for row in candidates:
        gid, environment_seed, _ = _game_identity(settings["seed"], row["game_index"])
        if gid != row["game_id"] or by_game[gid]["game_seed"] != environment_seed:
            raise ValueError("Supplemental environment does not match its independent seed schedule.")
        if row != _candidate(by_game[gid], row["game_index"], row["step_index"]):
            raise ValueError("Supplemental input/label differs from its complete replay.")
        key = (gid, row["step_index"])
        if key in used_steps or row["input_key"] in seen_inputs | forbidden_inputs:
            raise ValueError("Duplicate or held-out supplemental observation.")
        used_steps.add(key); seen_inputs.add(row["input_key"])
    if {r["game_id"] for r in candidates} != set(by_game):
        raise ValueError("Supplemental replay list differs from selected source games.")
    matrix = _coverage_matrix(candidates)
    if any(cell["samples"] < settings["samples_per_cell"]
           or cell["source_games"] < settings["min_games_per_cell"]
           or cell["max_samples_per_game"] > 2 for cell in matrix):
        raise ValueError("Training endgame coverage has a cell or independent-game shortfall.")
    if metadata["coverage_matrix"] != matrix or metadata["num_samples"] != len(candidates):
        raise ValueError("Training endgame coverage metadata differs from saved observations.")
    if metadata["games_generated"] != max(row["game_index"] for row in candidates) + 1:
        raise ValueError("Generated game count differs from the coverage stopping point.")
    return candidates


def _generate_candidates(output: Path, settings: dict, forbidden: tuple) -> list[dict]:
    directory = output / "augmentation/private"
    if (directory / "generation_metadata.json").exists():
        return _validate_generation(output, settings, forbidden)
    ids, forbidden_seeds, forbidden_inputs = forbidden
    counts, cell_games = Counter(), defaultdict(set)
    candidates, replays, seen_inputs, used_seeds = [], [], set(), set()
    attempted = 0
    for index in range(settings["max_games"]):
        if all(counts[cell] >= settings["samples_per_cell"]
               and len(cell_games[cell]) >= settings["min_games_per_cell"] for cell in _reachable_cells()):
            break
        gid, environment_seed, policy_seed = _game_identity(settings["seed"], index)
        if gid in ids or environment_seed in forbidden_seeds | used_seeds:
            raise ValueError("Training endgame seed schedule overlaps an existing source.")
        used_seeds.add(environment_seed)
        replay = generate_coverage_replay(gid, environment_seed, policy_seed)
        attempted += 1
        indices = list(range(len(replay["steps"])))
        random.Random(_derive_seed(settings["seed"], "training_endgame_selection_v1", index)).shuffle(indices)
        local = Counter(); selected = 0
        for step_index in indices:
            step = replay["steps"][step_index]
            if not classify_endgame_observation(step["observation"]):
                continue
            row = _candidate(replay, index, step_index)
            if row["input_key"] in seen_inputs | forbidden_inputs:
                continue
            alive = sum(p["alive"] for p in row["observation"]["players_public_status"])
            cells = [_cell(tag, alive, row["label"]) for tag in row["scenario_tags"]]
            needed = any(counts[c] < settings["samples_per_cell"] or (
                len(cell_games[c]) < settings["min_games_per_cell"] and gid not in cell_games[c]) for c in cells)
            if not needed or any(local[c] >= 2 for c in cells):
                continue
            candidates.append(row); seen_inputs.add(row["input_key"]); selected += 1
            for cell in cells:
                counts[cell] += 1; local[cell] += 1; cell_games[cell].add(gid)
        if selected:
            replays.append(replay)
    matrix = _coverage_matrix(candidates)
    write_jsonl(directory / "candidates.jsonl", candidates)
    write_jsonl(directory / "replays.jsonl", replays)
    write_json(directory / "generation_metadata.json", {"contract": current_contract(),
        "settings": settings, "games_generated": attempted, "num_samples": len(candidates),
        "coverage_matrix": matrix})
    return _validate_generation(output, settings, forbidden)


def _assemble_teacher_report(obs: PlayerObservation, key: str, seed: int, teacher: dict,
                             rewards: dict[str, list[float]], count: int) -> dict:
    if set(rewards) != {a.action_id for a in obs.legal_actions} or any(
        len(values) != count or any(type(x) not in (int, float) or x not in (0., .35, .65, 1.) for x in values)
        for values in rewards.values()):
        raise ValueError("Endgame teacher rewards do not cover every candidate and trajectory.")
    return {"contract": current_contract(), "version": ENDGAME_TEACHER_VERSION,
        "input_key": key, "seed": seed, "teacher_config": teacher,
        "rollouts_per_candidate": count, "total_rollouts": count * len(rewards),
        "candidate_rewards": rewards, "stages": [_summarize(obs, rewards, n) for n in (8, 32, count)],
        "final": _summarize(obs, rewards, count),
        "selection": "coverage retained independently of best-minus-mean advantage"}


def _score_candidate(job: tuple) -> tuple[str, dict | None]:
    row, settings, teacher, directory = job
    obs = PlayerObservation.from_dict(row["observation"])
    if len(obs.legal_actions) == 1:
        return row["input_key"], None
    seed = _derive_seed(settings["seed"], "training_endgame_teacher_v1", row["input_key"])
    path = Path(directory) / f"{row['input_key']}.json"
    if path.exists():
        stored = read_json(path)
        expected = _assemble_teacher_report(obs, row["input_key"], seed, teacher,
            stored["candidate_rewards"], settings["num_rollouts"])
        if stored != expected:
            raise ValueError("Cached endgame teacher report differs from its scores or method.")
        return row["input_key"], stored
    rewards = evaluate_observation_action_rewards(obs, teacher, settings["num_rollouts"], seed)
    report = _assemble_teacher_report(obs, row["input_key"], seed, teacher, rewards, settings["num_rollouts"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    write_json(temporary, report); temporary.replace(path)
    return row["input_key"], report


def _training_row(candidate: dict, report: dict | None, policy: dict) -> dict:
    obs = PlayerObservation.from_dict(candidate["observation"])
    if len(obs.legal_actions) == 1:
        if report is not None or obs.legal_actions[0].action_type != "CHALLENGE":
            raise ValueError("Only a forced legal challenge can bypass teacher scoring.")
        chosen, values, source = obs.legal_actions[0], None, "single_legal_action_rule"
    else:
        if report is None:
            raise ValueError("Free endgame actions require recorded teacher rewards.")
        chosen = next(a for a in obs.legal_actions if a.action_id == report["final"]["chosen_action_id"])
        values, source = report["final"]["candidate_means"], ENDGAME_TEACHER_VERSION
    row = make_action_sample(obs, chosen, candidate["game_id"], teacher=source)
    row["endgame_augmentation"] = {"version": ENDGAME_AUGMENTATION_VERSION,
        "input_key": candidate["input_key"], "game_index": candidate["game_index"],
        "step_index": candidate["step_index"], "scenario_tags": candidate["scenario_tags"],
        "pending_claim_was_bluff": bool(candidate["label"]),
        "rollouts_per_candidate": 0 if report is None else report["rollouts_per_candidate"]}
    if values is not None:
        row["rollout_candidate_values"] = values
        row["rollout_advantage"] = report["final"]["best_minus_mean_advantage"]
        row["rollout_best_value"] = max(values.values())
    row["action_supervision"] = make_action_supervision(row, values, policy, source=source)
    validate_training_sample(row, action_supervision_policy=policy)
    return row


def _replacement_rows(rows: list[dict], count: int, seed: int) -> list[dict]:
    eligible = [r for r in rows if r["task"] == "action"
                and len(r["observation"]["own_hand"]) >= 3
                and not classify_endgame_observation(r["observation"])]
    # Private IDs affect selection only, never model inputs or teacher seeds.
    eligible.sort(key=lambda r: (_derive_seed(seed, "replace_ordinary_training_action_v1", r["sample_id"]), r["sample_id"]))
    if len(eligible) < count:
        raise ValueError("Not enough ordinary training actions to replace while preserving all existing endgames.")
    return eligible[:count]


def _supervision_summary(rows_by_split: dict, policy: dict) -> dict:
    train = [r for r in rows_by_split["train"] if r["task"] == "action"]
    mean = mean_action_weight(train, policy)
    return {"contract": current_contract(), "version": policy["version"], "policy": policy,
        "samples_per_split": {split: {
            "groups": dict(Counter(resolve_action_supervision(r, policy)["group"] for r in rows if r["task"] == "action")),
            "target_set_sizes": dict(Counter(str(len(resolve_action_supervision(r, policy)["target_action_ids"]))
                                             for r in rows if r["task"] == "action"))}
            for split, rows in rows_by_split.items()},
        "training_action_raw_weight_mean": mean, "training_action_weight_scale": 1 / mean,
        "training_action_effective_total_weight": len(train),
        "training_action_answer_sequences": sum(len(resolve_action_supervision(r, policy)["target_action_ids"]) for r in train),
        "belief_weight": 1.0, "model_api_calls": 0,
        "interpretation": "Independent action-only coverage replacement; all accepted answers share state weight; belief distribution unchanged."}


def _augmentation_summary(candidates: list[dict], reports: dict, source_summary: dict, generation: dict) -> dict:
    return {"version": ENDGAME_AUGMENTATION_VERSION,
        "num_replacements": len(candidates), "new_training_games": len({r['game_id'] for r in candidates}),
        "supplemental_games_generated": generation["games_generated"],
        "original_source_games": source_summary["games_generated"], "coverage_matrix": _coverage_matrix(candidates),
        "free_action_rows": sum(len(r["observation"]["legal_actions"]) > 1 for r in candidates),
        "forced_action_rows": sum(len(r["observation"]["legal_actions"]) == 1 for r in candidates),
        "total_teacher_candidate_continuations": sum(r["total_rollouts"] for r in reports.values() if r is not None),
        "source_preserved": True, "belief_unchanged": True, "validation_test_unchanged": True,
        "independent_check_unchanged": True, "model_api_calls": 0}


def prepare_endgame_augmented_run(source_run: str | Path, output: str | Path, *,
    seed: int = 2026111407, samples_per_cell: int = 20, min_games_per_cell: int = 10,
    num_rollouts: int = 64, max_games: int = 500, workers: int = 8,
) -> dict:
    source, output = Path(source_run).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Augmentation needs a separate sibling run; preserve its source.")
    if any(type(v) is not int or v < 1 for v in (samples_per_cell, min_games_per_cell, max_games, workers)):
        raise ValueError("Coverage quotas, game cap and worker count must be positive integers.")
    if type(seed) is not int or type(num_rollouts) is not int or num_rollouts < 64:
        raise ValueError("Use an integer seed and at least 64 paired teacher trajectories.")
    validate_prepared_supervision_run(source, check_sources=True)
    source_config = read_json(source / "config.json")
    policy = normalize_action_policy(source_config["training"]["action_supervision"])
    check_dir = source / source_config.get("evaluation", {}).get("endgame_coverage_dir", "endgame_coverage")
    if not (check_dir / "private/replays.jsonl").exists():
        raise ValueError("The independent endgame check set is required for source separation.")
    settings = {"version": ENDGAME_AUGMENTATION_VERSION, "source_run": str(source),
        "source_files_sha256": _source_bindings(source),
        "independent_check_dir": str(check_dir), "independent_check_files_sha256": _tree_hashes(check_dir),
        "seed": seed, "samples_per_cell": samples_per_cell, "min_games_per_cell": min_games_per_cell,
        "num_rollouts": num_rollouts, "max_games": max_games,
        "selection": "truth-balanced endgame actions only; no new belief rows; preserve ordinary-state count by replacement"}
    config = copy.deepcopy(source_config)
    config["dataset"].pop("action_refinement", None)
    config["dataset"].pop("supervision_preparation", None)
    config["dataset"]["endgame_augmentation"] = settings
    config["evaluation"]["endgame_coverage_dir"] = "endgame_coverage"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and read_json(output / "config.json") != config:
        raise ValueError("Augmentation output belongs to a different source or configuration.")
    if any((output / f"models/{name}").exists() for name in ("sft", "rs", "sft_trial")):
        raise ValueError("Do not re-prepare a run that already contains model checkpoints.")
    write_json(output / "config.json", config)
    write_json(output / "run_status.json", {"stages": {"data": "augmenting_endgames"}, "real_experiments": "pending"})
    forbidden = _forbidden_sources(source, check_dir)
    candidates = _generate_candidates(output, settings, forbidden)
    source_rows = {split: list(read_jsonl(source / f"dataset/{split}.jsonl")) for split in SPLITS}
    removed = _replacement_rows(source_rows["train"], len(candidates), seed)
    teacher = teacher_config(config)
    jobs = [(row, settings, teacher, str(output / "augmentation/candidate_reports")) for row in candidates]
    reports = {}
    print(f"Scoring {sum(len(r['observation']['legal_actions']) > 1 for r in candidates)} free endgames; "
          f"{len(candidates)} action replacements, {num_rollouts} trajectories per candidate", flush=True)
    if workers == 1:
        for key, report in map(_score_candidate, jobs): reports[key] = report
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, (key, report) in enumerate(pool.map(_score_candidate, jobs)):
                reports[key] = report
                if (index + 1) % 20 == 0:
                    print(f"Endgame labels ready: {index + 1}/{len(candidates)}", flush=True)
    added = [_training_row(row, reports[row["input_key"]], policy) for row in candidates]
    replacements = {old["sample_id"]: new for old, new in zip(removed, added)}
    rows_by_split = {**source_rows, "train": [replacements.get(row["sample_id"], row) for row in source_rows["train"]]}
    write_jsonl(output / "dataset/train.jsonl", rows_by_split["train"])
    write_jsonl(output / "dataset/action_train.jsonl", (r for r in rows_by_split["train"] if r["task"] == "action"))
    for split in SPLITS:
        for prefix in (("belief_",) if split == "train" else ("", "action_", "belief_")):
            target = output / f"dataset/{prefix}{split}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / f"dataset/{prefix}{split}.jsonl", target)
    write_jsonl(output / "dataset/sft_all_samples.jsonl", (r for split in SPLITS for r in rows_by_split[split]))
    partition = read_json(source / "dataset/game_splits.json")
    partition["train"] += sorted({row["game_id"] for row in candidates})
    write_json(output / "dataset/game_splits.json", partition)
    (output / "models").mkdir(exist_ok=True)
    shutil.copyfile(source / "models/small_classifier_weights.json", output / "models/small_classifier_weights.json")
    shutil.copytree(check_dir, output / "endgame_coverage", dirs_exist_ok=True)
    manifest = [{"removed_sample_id": old["sample_id"], "added_sample_id": new["sample_id"],
                 "removed_input_key": input_key(old), "added_input_key": input_key(new)}
                for old, new in zip(removed, added)]
    write_jsonl(output / "augmentation/replacements.jsonl", manifest)
    write_jsonl(output / "augmentation/added_actions.jsonl", added)
    write_jsonl(output / "augmentation/removed_actions.jsonl", removed)
    supervision = _supervision_summary(rows_by_split, policy)
    source_summary = read_json(source / "dataset/data_summary.json")
    summary = copy.deepcopy(source_summary)
    summary.pop("quality_refinement", None)
    summary["action_supervision_summary"] = supervision
    generation = read_json(output / "augmentation/private/generation_metadata.json")
    summary["endgame_augmentation"] = _augmentation_summary(candidates, reports, source_summary, generation)
    summary["games_generated"] = source_summary["games_generated"] + generation["games_generated"]
    summary["games_with_saved_replays"] = sum(len(ids) for ids in partition.values())
    summary["games_per_split"] = {s: len(ids) for s, ids in partition.items()}
    write_json(output / "dataset/data_summary.json", summary)
    write_json(output / "quality/action_supervision_summary.json", supervision)
    validation = validate_endgame_augmented_run(output, check_sources=True)
    write_json(output / "augmentation/verification_report.json", validation)
    write_json(output / "run_status.json", {"stages": {"data": "complete"}, "real_experiments": "pending"})
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    return validation


def validate_endgame_augmented_run(output: str | Path, *, check_sources=False) -> dict:
    output = Path(output).resolve()
    config = read_json(output / "config.json")
    require_contract(config, "training endgame augmentation")
    settings = config["dataset"]["endgame_augmentation"]
    if settings["version"] != ENDGAME_AUGMENTATION_VERSION:
        raise ValueError("Incompatible training endgame augmentation version.")
    if (type(settings["num_rollouts"]) is not int or settings["num_rollouts"] < 64
            or any(type(settings[k]) is not int or settings[k] < 1
                   for k in ("samples_per_cell", "min_games_per_cell", "max_games"))):
        raise ValueError("Endgame augmentation requires positive coverage quotas and at least 64 trajectories.")
    policy = normalize_action_policy(config["training"]["action_supervision"])
    source = Path(settings["source_run"])
    validate_prepared_supervision_run(source, check_sources=check_sources)
    if _source_bindings(source) != settings["source_files_sha256"]:
        raise ValueError("Prepared source changed after endgame augmentation.")
    check_dir = Path(settings["independent_check_dir"])
    if (_tree_hashes(check_dir) != settings["independent_check_files_sha256"]
            or _tree_hashes(output / "endgame_coverage") != settings["independent_check_files_sha256"]):
        raise ValueError("Independent endgame check set changed during augmentation.")
    candidates = _validate_generation(output, settings, _forbidden_sources(source, check_dir))
    teacher = teacher_config(config)
    expected_added, reports = [], {}
    for row in candidates:
        obs = PlayerObservation.from_dict(row["observation"])
        report = None
        if len(obs.legal_actions) > 1:
            path = output / f"augmentation/candidate_reports/{row['input_key']}.json"
            report = read_json(path)
            seed = _derive_seed(settings["seed"], "training_endgame_teacher_v1", row["input_key"])
            if report != _assemble_teacher_report(obs, row["input_key"], seed, teacher,
                    report["candidate_rewards"], settings["num_rollouts"]):
                raise ValueError("Endgame teacher report disagrees with saved candidate rewards.")
        reports[row["input_key"]] = report
        expected_added.append(_training_row(row, report, policy))
    added = list(read_jsonl(output / "augmentation/added_actions.jsonl"))
    if added != expected_added:
        raise ValueError("Supplemental action labels differ from source observations or teacher scores.")
    source_rows = {s: list(read_jsonl(source / f"dataset/{s}.jsonl")) for s in SPLITS}
    removed = _replacement_rows(source_rows["train"], len(added), settings["seed"])
    if list(read_jsonl(output / "augmentation/removed_actions.jsonl")) != removed:
        raise ValueError("Removed rows differ from deterministic ordinary-action selection.")
    manifest = [{"removed_sample_id": old["sample_id"], "added_sample_id": new["sample_id"],
                 "removed_input_key": input_key(old), "added_input_key": input_key(new)}
                for old, new in zip(removed, added)]
    if list(read_jsonl(output / "augmentation/replacements.jsonl")) != manifest:
        raise ValueError("Endgame replacement manifest differs from actual training rows.")
    replacements = {old["sample_id"]: new for old, new in zip(removed, added)}
    expected_train = [replacements.get(row["sample_id"], row) for row in source_rows["train"]]
    rows_by_split = {s: list(read_jsonl(output / f"dataset/{s}.jsonl")) for s in SPLITS}
    if rows_by_split["train"] != expected_train:
        raise ValueError("Augmented training data differs from the declared action-only replacements.")
    for split in SPLITS:
        for prefix in (("belief_",) if split == "train" else ("", "action_", "belief_")):
            relative = f"dataset/{prefix}{split}.jsonl"
            if file_hash(source / relative) != file_hash(output / relative):
                raise ValueError(f"Belief or held-out data changed: {relative}")
    if list(read_jsonl(output / "dataset/action_train.jsonl")) != [r for r in expected_train if r["task"] == "action"]:
        raise ValueError("Combined training rows differ from the action task file.")
    merged = [r for split in SPLITS for r in rows_by_split[split]]
    if list(read_jsonl(output / "dataset/sft_all_samples.jsonl")) != merged:
        raise ValueError("Merged augmented dataset differs from its partitions.")
    expected_partition = read_json(source / "dataset/game_splits.json")
    expected_partition["train"] += sorted({r["game_id"] for r in added})
    if read_json(output / "dataset/game_splits.json") != expected_partition:
        raise ValueError("Augmentation changed held-out partitions or omitted new training games.")
    if len({r["sample_id"] for r in merged}) != len(merged):
        raise ValueError("Augmented data repeats a sample identity.")
    supervision = _supervision_summary(rows_by_split, policy)
    summary = read_json(output / "dataset/data_summary.json")
    if (read_json(output / "quality/action_supervision_summary.json") != supervision
            or summary["action_supervision_summary"] != supervision):
        raise ValueError("Augmented supervision counts or weights do not match training rows.")
    source_summary = read_json(source / "dataset/data_summary.json")
    generation = read_json(output / "augmentation/private/generation_metadata.json")
    extra = _augmentation_summary(candidates, reports, source_summary, generation)
    matrix = _coverage_matrix(candidates)
    if summary["endgame_augmentation"] != extra:
        raise ValueError("Augmentation summary differs from verified coverage and replacements.")
    counts = {s: dict(Counter(r["task"] for r in rows)) for s, rows in rows_by_split.items()}
    half = config["training"]["batch_size"] // 2
    return {"status": "passed", "contract": current_contract(), "version": ENDGAME_AUGMENTATION_VERSION,
        "source_run": str(source), "num_samples": len(merged), "task_samples_per_split": counts,
        "num_replacements": len(added), "new_training_games": len({r["game_id"] for r in added}),
        "supplemental_games_generated": generation["games_generated"],
        "free_action_rows": extra["free_action_rows"], "forced_action_rows": extra["forced_action_rows"],
        "total_teacher_candidate_continuations": extra["total_teacher_candidate_continuations"],
        "coverage_matrix": matrix, "source_preserved": True, "all_saved_replays_verified": True,
        "all_teacher_reports_recomputed": True, "original_endgame_training_rows_preserved": True,
        "belief_data_unchanged": True, "validation_test_unchanged": True, "independent_check_unchanged": True,
        "overlapping_game_ids": 0, "overlapping_environment_seeds": 0, "overlapping_action_inputs": 0,
        "training_action_raw_weight_mean": supervision["training_action_raw_weight_mean"],
        "training_optimizer_steps": math.ceil(max(counts["train"].values()) / half) * config["training"]["num_epochs"],
        "model_api_calls": 0, "interpretation": "Action-only truth-balanced coverage; no inference or model performance was evaluated."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026111407)
    parser.add_argument("--samples-per-cell", type=int, default=20)
    parser.add_argument("--min-games-per-cell", type=int, default=10)
    parser.add_argument("--num-rollouts", type=int, default=64)
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_endgame_augmented_run(args.output_dir, check_sources=True), ensure_ascii=False, indent=2))
    elif args.source_run is None:
        parser.error("--source-run is required unless --validate-only is used")
    else:
        prepare_endgame_augmented_run(args.source_run, args.output_dir, seed=args.seed,
            samples_per_cell=args.samples_per_cell, min_games_per_cell=args.min_games_per_cell,
            num_rollouts=args.num_rollouts, max_games=args.max_games, workers=args.workers)

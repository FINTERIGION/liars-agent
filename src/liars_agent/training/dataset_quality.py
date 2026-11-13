"""Local adaptive relabelling of an existing, whole-game-partitioned dataset."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from liars_agent.agents.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from liars_agent.contracts import current_contract, require_contract
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.state import _derive_seed
from liars_agent.experiment import read_json, read_jsonl, teacher_config, write_json, write_jsonl
from liars_agent.training.action_refinement import (
    ACTION_REFINEMENT_VERSION, refine_action_observation, validate_action_refinement_report,
)
from liars_agent.training.data_generator import make_action_sample

SPLITS = ("train", "val", "test")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_key(row: dict) -> str:
    """Identical model inputs use the same independent teacher random stream."""
    return hashlib.sha256((row["system"] + "\n" + row["prompt"]).encode()).hexdigest()


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    write_json(temporary, data)
    temporary.replace(path)


def _evaluate_job(job: tuple) -> tuple[str, dict]:
    row, teacher, seed, force, directory, reference = job
    key = input_key(row)
    obs = PlayerObservation.from_dict(row["observation"])
    path = Path(directory) / f"{key}.json"
    rollout_seed = _derive_seed(seed, "quality_observation", key)
    if path.exists():
        saved = read_json(path)
        if (saved["input_key"] != key or saved["teacher_config"] != teacher
                or saved["report"]["seed"] != rollout_seed
                or saved["report"]["force_refine"] != force
                or saved["report"]["reference_action_id"] != reference):
            raise ValueError("Refinement cache belongs to different inputs/settings.")
        report = saved["report"]
        validate_action_refinement_report(obs, report)
    else:
        report = refine_action_observation(obs, teacher, rollout_seed, force_refine=force,
                                           reference_action_id=reference)
        validate_action_refinement_report(obs, report)
        _atomic_json(path, {"contract": current_contract(), "input_key": key,
            "teacher_config": teacher, "report": report})
    return key, report


def _evaluate(rows, teacher, seed, counts, directory, workers, *, use_original_labels=True):
    jobs = [(row, teacher, seed, counts[input_key(row)] > 1, str(directory),
             json.loads(row["completion"])["chosen_action_id"]
             if use_original_labels and counts[input_key(row)] == 1 else None) for row in rows]
    reports = {}
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(_evaluate_job, jobs, chunksize=1) if pool else map(_evaluate_job, jobs)
        for key, report in results:
            reports[key] = report
            if len(reports) % 100 == 0 or len(reports) == len(jobs):
                stages = Counter(r["rollouts_per_candidate"] for r in reports.values())
                print(f"Reviewed {len(reports)}/{len(jobs)} distinct inputs; rollouts={dict(stages)}", flush=True)
    finally:
        if pool:
            pool.shutdown()
    return reports


def _updated_row(row, report, origin="original"):
    result = copy.deepcopy(row)
    result.update(completion=json.dumps({"chosen_action_id": report["chosen_action_id"]}, ensure_ascii=False),
        rollout_best_value=max(report["final"]["candidate_means"].values()),
        rollout_advantage=report["final"]["best_minus_mean_advantage"],
        candidate_count=len(report["final"]["candidate_means"]), teacher="pool_prior_counter_continuation")
    result["action_refinement"] = {"version": ACTION_REFINEMENT_VERSION, "input_key": input_key(row),
        "origin": origin, "original_completion": row.get("completion") if origin == "original" else None,
        "rollouts_per_candidate": report["rollouts_per_candidate"], "ambiguous": report["ambiguous"],
        "best_vs_runnerup_gap": report["final"]["best_vs_runnerup_gap"]}
    return result


def _replacement_candidates(summary, owner, excluded, seed, per_game=8):
    """Reserve observations evenly across source games, without label filtering."""
    queues = {s: [] for s in SPLITS}
    for batch in summary["batches"]:
        for game in read_jsonl(Path(batch["replays_path"])):
            points = [p for p in game["steps"] if len(p["observation"]["legal_actions"]) > 1]
            random.Random(_derive_seed(seed, "replacement", game["game_id"])).shuffle(points)
            reserved = []
            for point in points:
                obs = PlayerObservation.from_dict(point["observation"])
                row = make_action_sample(obs, obs.legal_actions[0], game["game_id"])
                key = input_key(row)
                if key not in excluded:
                    reserved.append(row)
                    excluded.add(key)
                if len(reserved) >= per_game:
                    break
            queues[owner[game["game_id"]]].append(reserved)
    # Each pass takes at most one point from each source game.
    return {split: [q[index] for index in range(per_game) for q in games if len(q) > index]
            for split, games in queues.items()}


def improve_dataset(source_run: Path, output: Path, workers=8, seed=20261114) -> dict:
    source_run, output = source_run.resolve(), output.resolve()
    if source_run == output or source_run in output.parents:
        raise ValueError("Use a separate sibling run directory; preserve the source dataset.")
    if workers < 1:
        raise ValueError("workers must be positive")
    source_config = read_json(source_run / "config.json")
    source_summary = read_json(source_run / "dataset/data_summary.json")
    require_contract(source_config); require_contract(source_summary)
    partition = read_json(source_run / "dataset/game_splits.json")
    owner = {}
    for split, games in partition.items():
        for game in games:
            if game in owner:
                raise ValueError("Source game crosses dataset partitions.")
            owner[game] = split
    files = [source_run / "config.json", source_run / "dataset/data_summary.json",
             source_run / "dataset/game_splits.json", source_run / "models/small_classifier_weights.json"]
    files += [source_run / f"dataset/{s}.jsonl" for s in SPLITS]
    files += [source_run / f"dataset/belief_{s}.jsonl" for s in SPLITS]
    files += [Path(b["replays_path"]) for b in source_summary["batches"]]
    binding = {str(path): file_hash(path) for path in files}
    settings = {"version": ACTION_REFINEMENT_VERSION, "seed": seed,
        "source_run": str(source_run), "source_sha256": binding,
        "rollout_schedule": [8, 32, 64], "gap_threshold": .05,
        "minimum_best_minus_mean": source_config["dataset"]["min_advantage_margin"],
        "duplicate_inputs": "shared_report_forced_refinement",
        "instability_signals": ["top_gap_or_tie", "original_label_outside_screen_best", "duplicate_input"],
        "unresolved_close_actions": "keep_deterministic_best_and_mark_ambiguous"}
    config = copy.deepcopy(source_config)
    config["dataset"]["action_refinement"] = settings
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and read_json(output / "config.json") != config:
        raise ValueError("Output run belongs to a different refinement configuration.")
    write_json(output / "config.json", config)
    write_json(output / "run_status.json", {"stages": {"data": "running"}, "real_experiments": "pending"})
    from liars_agent.training.tinker_train import validate_training_sample
    original = {}; beliefs = {}; groups = {}; counts = Counter()
    for split in SPLITS:
        rows = list(read_jsonl(source_run / f"dataset/{split}.jsonl"))
        original[split] = [r for r in rows if r["task"] == "action"]
        beliefs[split] = [r for r in rows if r["task"] == "belief"]
        for row in rows:
            validate_training_sample(row)
            if owner.get(row["game_id"]) != split:
                raise ValueError("Source sample is not in its whole-game partition.")
        if beliefs[split] != list(read_jsonl(source_run / f"dataset/belief_{split}.jsonl")):
            raise ValueError("Source belief task files do not match combined files.")
        for row in original[split]:
            key = input_key(row); counts[key] += 1
            if key in groups and groups[key]["observation"] != row["observation"]:
                raise ValueError("Identical action prompts hide differing observations.")
            groups.setdefault(key, row)
    print(f"Refining {sum(counts.values())} action rows / {len(groups)} distinct inputs", flush=True)
    directory = output / "quality/candidate_reports"
    teacher = teacher_config(source_config)
    reports = _evaluate(list(groups.values()), teacher, seed, counts, directory, workers)
    accepted = {s: [] for s in SPLITS}; rejected = []; margin = settings["minimum_best_minus_mean"]
    for split in SPLITS:
        for row in original[split]:
            report = reports[input_key(row)]
            updated = _updated_row(row, report)
            if updated["rollout_advantage"] >= margin:
                accepted[split].append(updated)
            else:
                rejected.append({"split": split, "reason": "refined_advantage_below_margin", "sample": updated})
    deficits = {s: len(original[s]) - len(accepted[s]) for s in SPLITS}
    replacement_ids = []; replacement_reviewed = 0
    print(f"Low-advantage rows to replace within original partitions: {deficits}", flush=True)
    if any(deficits.values()):
        candidates = _replacement_candidates(source_summary, owner, set(groups), seed)
        for split in SPLITS:
            cursor = 0
            while deficits[split]:
                batch = candidates[split][cursor:cursor + deficits[split]]
                if not batch:
                    raise ValueError(f"Not enough qualifying unused {split} observations; increase reserve or source games.")
                cursor += len(batch); replacement_reviewed += len(batch)
                extra = _evaluate(batch, teacher, seed, Counter({input_key(r): 1 for r in batch}), directory, workers,
                                  use_original_labels=False)
                reports.update(extra)
                for row in batch:
                    updated = _updated_row(row, extra[input_key(row)], "replacement")
                    if updated["rollout_advantage"] >= margin:
                        accepted[split].append(updated); replacement_ids.append(updated["sample_id"])
                        deficits[split] -= 1
                    else:
                        rejected.append({"split": split, "reason": "replacement_advantage_below_margin", "sample": updated})
    all_rows = []; changes = Counter(); rollout_counts = Counter(); ambiguous = 0
    for split in SPLITS:
        for row in accepted[split]:
            meta = row["action_refinement"]
            rollout_counts[meta["rollouts_per_candidate"]] += 1
            ambiguous += int(meta["ambiguous"])
            if meta["origin"] == "original" and row["completion"] != meta["original_completion"]:
                changes[split] += 1
        combined = accepted[split] + beliefs[split]
        all_rows.extend(combined)
        write_jsonl(output / f"dataset/action_{split}.jsonl", accepted[split])
        shutil.copyfile(source_run / f"dataset/belief_{split}.jsonl", output / f"dataset/belief_{split}.jsonl")
        write_jsonl(output / f"dataset/{split}.jsonl", combined)
    shutil.copyfile(source_run / "dataset/game_splits.json", output / "dataset/game_splits.json")
    write_jsonl(output / "dataset/sft_all_samples.jsonl", all_rows)
    (output / "models").mkdir(exist_ok=True)
    shutil.copyfile(source_run / "models/small_classifier_weights.json", output / "models/small_classifier_weights.json")
    quality = {"version": ACTION_REFINEMENT_VERSION, "model_api_calls": 0,
        "original_action_rows": sum(counts.values()), "distinct_original_inputs": len(groups),
        "duplicate_input_groups": sum(n > 1 for n in counts.values()),
        "changed_retained_labels_by_split": dict(changes), "replacements": len(replacement_ids),
        "replacement_candidates_reviewed": replacement_reviewed, "rejected_candidates": len(rejected),
        "final_rollouts_per_candidate": dict(rollout_counts), "remaining_ambiguous_rows": ambiguous,
        "source_preserved": True, "belief_samples_unchanged": True, "game_partition_unchanged": True,
        "replacement_sample_ids": replacement_ids,
        "interpretation": "Monte Carlo teacher preferences; ambiguous does not imply wrong, and gap is not a confidence bound."}
    write_jsonl(output / "quality/rejected_candidates.jsonl", rejected)
    write_json(output / "quality/summary.json", quality)
    summary = copy.deepcopy(source_summary)
    summary.update(quality_refinement=quality, samples_per_split={s:len(accepted[s])+len(beliefs[s]) for s in SPLITS},
        task_samples_per_split={s:{"action":len(accepted[s]),"belief":len(beliefs[s])} for s in SPLITS})
    write_json(output / "dataset/data_summary.json", summary)
    validate_refined_dataset(output, check_sources=True)
    write_json(output / "run_status.json", {"stages":{"data":"complete"}, "real_experiments":"pending"})
    print(json.dumps(quality, ensure_ascii=False, indent=2), flush=True)
    return quality


def validate_refined_dataset(output: Path, *, check_sources=False) -> dict:
    """Recompute saved reward summaries and verify every refined target and split."""
    config = read_json(output / "config.json")
    settings = config["dataset"]["action_refinement"]
    if settings["version"] != ACTION_REFINEMENT_VERSION:
        raise ValueError("Stale action refinement version")
    source = Path(settings["source_run"])
    if check_sources:
        for path, expected in settings["source_sha256"].items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f"Source changed during refinement: {path}")
    if file_hash(output / "dataset/game_splits.json") != settings["source_sha256"][str(source / "dataset/game_splits.json")]:
        raise ValueError("Refinement altered game partitions.")
    partition = read_json(output / "dataset/game_splits.json")
    owner = {gid:sp for sp,ids in partition.items() for gid in ids}
    if len(owner) != sum(map(len, partition.values())):
        raise ValueError("A game crosses partitions.")
    teacher = teacher_config(config); cache = {}; ids = set(); completions = {}; total = 0
    from liars_agent.training.tinker_train import validate_training_sample
    for split in SPLITS:
        bp = output / f"dataset/belief_{split}.jsonl"
        if file_hash(bp) != settings["source_sha256"][str(source / f"dataset/belief_{split}.jsonl")]:
            raise ValueError("Refinement altered belief data.")
        rows = list(read_jsonl(output / f"dataset/{split}.jsonl"))
        actions = list(read_jsonl(output / f"dataset/action_{split}.jsonl"))
        if [r for r in rows if r["task"] == "action"] != actions:
            raise ValueError("Combined action file differs from task file.")
        if [r for r in rows if r["task"] == "belief"] != list(read_jsonl(bp)):
            raise ValueError("Combined belief file differs from task file.")
        for row in rows:
            obs = validate_training_sample(row)
            if row["sample_id"] in ids or owner.get(row["game_id"]) != split:
                raise ValueError("Duplicate sample or game split mismatch.")
            ids.add(row["sample_id"])
            if row["task"] != "action":
                continue
            key = input_key(row)
            if key not in cache:
                saved = read_json(output / f"quality/candidate_reports/{key}.json")
                require_contract(saved)
                if saved["input_key"] != key or saved["teacher_config"] != teacher:
                    raise ValueError("Teacher or input differs from refinement report.")
                report = saved["report"]
                if report["seed"] != _derive_seed(settings["seed"], "quality_observation", key):
                    raise ValueError("Refinement random stream does not match settings.")
                validate_action_refinement_report(obs, report)
                cache[key] = report
            report = cache[key]
            if json.loads(row["completion"])["chosen_action_id"] != report["chosen_action_id"]:
                raise ValueError("Training action differs from refined choice.")
            if row["action_refinement"]["version"] != ACTION_REFINEMENT_VERSION or row["action_refinement"]["input_key"] != key:
                raise ValueError("Row is not bound to its refinement report.")
            meta = row["action_refinement"]
            for field, expected in (("rollouts_per_candidate", report["rollouts_per_candidate"]),
                                    ("ambiguous", report["ambiguous"]),
                                    ("best_vs_runnerup_gap", report["final"]["best_vs_runnerup_gap"])):
                if meta[field] != expected:
                    raise ValueError(f"Refined row has incorrect {field} metadata.")
            for field, expected in (("rollout_advantage", report["final"]["best_minus_mean_advantage"]),
                                    ("rollout_best_value", max(report["final"]["candidate_means"].values()))):
                if abs(row[field] - expected) > 1e-12:
                    raise ValueError(f"Refined row has incorrect {field}.")
            if row["rollout_advantage"] < settings["minimum_best_minus_mean"]:
                raise ValueError("Low-advantage sample entered training data.")
            if key in completions and completions[key] != row["completion"]:
                raise ValueError("Identical action inputs received conflicting targets.")
            completions[key] = row["completion"]; total += 1
    if total != config["dataset"]["total_target_samples"]:
        raise ValueError("Refinement did not retain the requested action count.")
    return {"action_rows_verified": total, "distinct_action_inputs": len(cache), "conflicting_action_inputs": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    improve_dataset(args.source_run, args.output_dir, args.workers, args.seed)


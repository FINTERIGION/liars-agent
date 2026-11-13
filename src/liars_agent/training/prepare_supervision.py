"""Materialize uncertainty-aware targets in a new local run, without training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from liars_agent.contracts import current_contract, require_contract
from liars_agent.experiment import read_json, read_jsonl, write_json, write_jsonl
from liars_agent.training.action_supervision import (
    ACTION_SUPERVISION_VERSION, make_action_supervision, normalize_action_policy,
    resolve_action_supervision, mean_action_weight,
)
from liars_agent.training.dataset_quality import file_hash, input_key, validate_refined_dataset

SPLITS = ("train", "val", "test")


def report_tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.json")):
        digest.update(f"{path.name}:{file_hash(path)}\n".encode())
    return digest.hexdigest()


def prepare_supervision_run(source_run: Path, output: Path, policy: dict | None = None) -> dict:
    source_run, output = source_run.resolve(), output.resolve()
    if source_run == output or source_run in output.parents:
        raise ValueError("Prepare in a separate run directory and preserve the refined source.")
    policy = normalize_action_policy(policy)
    validate_refined_dataset(source_run, check_sources=True)
    source_config = read_json(source_run / "config.json")
    files = ["config.json", "dataset/data_summary.json", "dataset/game_splits.json",
             "models/small_classifier_weights.json", "quality/summary.json"]
    files += [f"dataset/{s}.jsonl" for s in SPLITS]
    files += [f"dataset/belief_{s}.jsonl" for s in SPLITS]
    binding = {name:file_hash(source_run / name) for name in files}
    report_hash = report_tree_hash(source_run / "quality/candidate_reports")
    preparation = {"version": ACTION_SUPERVISION_VERSION, "source_run": str(source_run),
                   "source_files_sha256": binding, "candidate_reports_sha256": report_hash}
    config = copy.deepcopy(source_config)
    config["training"]["action_supervision"] = policy
    config["dataset"]["supervision_preparation"] = preparation
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and read_json(output / "config.json") != config:
        raise ValueError("This run belongs to a different supervision preparation.")
    if any((output / f"models/{s}").exists() for s in ("sft", "rs")):
        raise ValueError("Do not re-prepare a run that already has model checkpoints.")
    write_json(output / "config.json", config)
    write_json(output / "run_status.json", {"stages":{"data":"preparing_supervision"},"real_experiments":"pending"})
    all_rows = []; by_split = {}; train_actions = []
    for split in SPLITS:
        rows = []; counts = Counter(); target_sizes = Counter()
        for old in read_jsonl(source_run / f"dataset/{split}.jsonl"):
            row = copy.deepcopy(old)
            if row["task"] == "action":
                report = read_json(source_run / f"quality/candidate_reports/{input_key(row)}.json")["report"]
                row["action_supervision"] = make_action_supervision(row, report["final"]["candidate_means"], policy,
                                                                   source=report["version"])
                descriptor = row["action_supervision"]
                counts[descriptor["group"]] += 1
                target_sizes[len(descriptor["target_action_ids"])] += 1
                if split == "train":
                    train_actions.append(row)
            rows.append(row)
        write_jsonl(output / f"dataset/{split}.jsonl", rows)
        write_jsonl(output / f"dataset/action_{split}.jsonl", (r for r in rows if r["task"] == "action"))
        shutil.copyfile(source_run / f"dataset/belief_{split}.jsonl", output / f"dataset/belief_{split}.jsonl")
        all_rows.extend(rows)
        by_split[split] = {"groups": dict(counts), "target_set_sizes": dict(target_sizes)}
    write_jsonl(output / "dataset/sft_all_samples.jsonl", all_rows)
    shutil.copyfile(source_run / "dataset/game_splits.json", output / "dataset/game_splits.json")
    (output / "models").mkdir(exist_ok=True)
    shutil.copyfile(source_run / "models/small_classifier_weights.json", output / "models/small_classifier_weights.json")
    for name in ("candidate_reports",):
        shutil.copytree(source_run / f"quality/{name}", output / f"quality/{name}", dirs_exist_ok=True)
    for name in ("rejected_candidates.jsonl", "summary.json"):
        shutil.copyfile(source_run / f"quality/{name}", output / f"quality/{name}")
    if (source_run / "endgame_coverage").exists():
        shutil.copytree(source_run / "endgame_coverage", output / "endgame_coverage", dirs_exist_ok=True)
    weight_mean = mean_action_weight(train_actions, policy)
    result = {"contract":current_contract(), "version": ACTION_SUPERVISION_VERSION, "policy":policy,
        "source_run": str(source_run), "samples_per_split":by_split,
        "training_action_raw_weight_mean":weight_mean,
        "training_action_weight_scale":1 / weight_mean,
        "training_action_effective_total_weight":len(train_actions),
        "belief_weight":1.0, "canonical_labels_unchanged":True, "belief_data_unchanged":True,
        "game_partition_unchanged":True, "model_api_calls":0,
        "interpretation":"Ties are empirical reward-mean ties, not proven equivalence. Small-gap weight is a configured heuristic. Epoch cycling optimizes mean target NLL; checkpoint set NLL scores total acceptable-answer mass."}
    summary = read_json(source_run / "dataset/data_summary.json")
    summary["action_supervision_summary"] = result
    write_json(output / "dataset/data_summary.json", summary)
    write_json(output / "quality/action_supervision_summary.json", result)
    validate_prepared_supervision_run(output, check_sources=True)
    write_json(output / "run_status.json", {"stages":{"data":"complete"},"real_experiments":"pending"})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def validate_prepared_supervision_run(output: Path, *, check_sources=False) -> dict:
    config = read_json(output / "config.json")
    policy = normalize_action_policy(config["training"]["action_supervision"])
    preparation = config["dataset"]["supervision_preparation"]
    if preparation["version"] != ACTION_SUPERVISION_VERSION:
        raise ValueError("Stale supervision preparation version.")
    source = Path(preparation["source_run"])
    if check_sources:
        for relative, expected in preparation["source_files_sha256"].items():
            if file_hash(source / relative) != expected:
                raise ValueError(f"Source changed during supervision preparation: {relative}")
        if report_tree_hash(source / "quality/candidate_reports") != preparation["candidate_reports_sha256"]:
            raise ValueError("Source candidate reports changed.")
    if report_tree_hash(output / "quality/candidate_reports") != preparation["candidate_reports_sha256"]:
        raise ValueError("Copied candidate reports differ from the refined source.")
    validate_refined_dataset(output)
    train = []; actual = {}; all_ids = set(); all_rows = {}
    for split in SPLITS:
        original = {r["sample_id"]:r for r in read_jsonl(source / f"dataset/{split}.jsonl")}
        rows = list(read_jsonl(output / f"dataset/{split}.jsonl"))
        if len(rows) != len(original) or {r["sample_id"] for r in rows} != set(original):
            raise ValueError("Supervision preparation changed sample identity or counts.")
        counts = Counter(); sizes = Counter()
        for row in rows:
            sid = row["sample_id"]
            if sid in all_ids:
                raise ValueError("Duplicated prepared sample ID.")
            all_ids.add(sid); all_rows[sid] = row
            stripped = {k:v for k,v in row.items() if k != "action_supervision"}
            old_stripped = {k:v for k,v in original[sid].items() if k != "action_supervision"}
            if stripped != old_stripped:
                raise ValueError("Preparation changed canonical supervision or observations.")
            if row["task"] == "action":
                descriptor = resolve_action_supervision(row, policy)
                report = read_json(output / f"quality/candidate_reports/{input_key(row)}.json")["report"]
                expected = make_action_supervision(row, report["final"]["candidate_means"], policy, source=report["version"])
                if descriptor != expected:
                    raise ValueError("Action supervision differs from its complete refinement report.")
                counts[descriptor["group"]] += 1; sizes[len(descriptor["target_action_ids"])] += 1
                if split == "train":train.append(row)
            elif "action_supervision" in row:
                raise ValueError("Belief samples must not inherit action weights or targets.")
        actual[split] = {"groups":dict(counts),"target_set_sizes":{str(k):v for k,v in sizes.items()}}
    merged = list(read_jsonl(output / "dataset/sft_all_samples.jsonl"))
    if len(merged) != len(all_rows) or {r["sample_id"]:r for r in merged} != all_rows:
        raise ValueError("Prepared merged file differs from the partitions.")
    summary = read_json(output / "quality/action_supervision_summary.json")
    require_contract(summary)
    if summary != read_json(output / "dataset/data_summary.json")["action_supervision_summary"]:
        raise ValueError("Supervision summary copies differ.")
    if summary["policy"] != policy or summary["samples_per_split"] != actual:
        raise ValueError("Supervision summary differs from actual sample groups.")
    mean = mean_action_weight(train, policy)
    if (abs(summary["training_action_raw_weight_mean"] - mean) > 1e-12
            or abs(summary["training_action_weight_scale"] - 1 / mean) > 1e-12
            or summary["training_action_effective_total_weight"] != len(train)
            or summary["belief_weight"] != 1.0):
        raise ValueError("Training weight normalization differs from training samples.")
    return {"status":"passed", "version":ACTION_SUPERVISION_VERSION, "num_samples":len(all_rows),
            "samples_per_split":actual,"training_action_raw_weight_mean":mean,"model_api_calls":0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--positive-gap-weight",type=float,default=.5)
    args=parser.parse_args()
    prepare_supervision_run(args.source_run,args.output_dir,normalize_action_policy({"positive_gap_weight":args.positive_gap_weight}))

"""Independently recheck refined data, source games and endgame probes locally.

Only the final verification report is written. No training backend or model
service is constructed; tokenizer files must already exist in the local cache.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

from liars_agent.contracts import require_contract
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.replay import verify_replay_determinism
from liars_agent.engine.state import _derive_seed
from liars_agent.evaluation.endgame_coverage import validate_endgame_coverage
from liars_agent.experiment import read_json, read_jsonl, teacher_config, write_json
from liars_agent.training.action_refinement import validate_action_refinement_report
from liars_agent.training.data_generator import evaluate_observation_action_rewards
from liars_agent.training.dataset_quality import file_hash, input_key, validate_refined_dataset
from liars_agent.training.tinker_train import build_sft_datum, validate_training_sample

SPLITS = ("train", "val", "test")
AUDIT_SEED = 20261115


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sample_source_key(row):
    obs = row["observation"]
    return row["game_id"], obs["round_id"], obs["step_id"], obs["observer_seat"]


def unique_row_hashes(path):
    result = {}
    for row in read_jsonl(path):
        key = row["sample_id"]
        require(key not in result, f"Duplicate sample_id in {path}: {key}")
        result[key] = fingerprint(row)
    return result


def verify_dataset_quality(run_dir: Path, coverage_dir: Path | None = None,
                           *, skip_tokenizer=False) -> dict:
    started = time.monotonic()
    run_dir = run_dir.resolve()
    coverage_dir = (coverage_dir or run_dir / "endgame_coverage").resolve()
    config = read_json(run_dir / "config.json")
    summary = read_json(run_dir / "dataset/data_summary.json")
    quality = read_json(run_dir / "quality/summary.json")
    require_contract(config); require_contract(summary)
    require(summary["shortfall"] == 0, "Dataset has an unresolved sample shortfall.")
    require(summary["quality_refinement"] == quality, "Quality summary copies differ.")
    settings = config["dataset"]["action_refinement"]
    source_run = Path(settings["source_run"])
    refinement_check = validate_refined_dataset(run_dir, check_sources=True)
    print("All retained targets and stored reward summaries verified; source hashes unchanged", flush=True)

    partition = read_json(run_dir / "dataset/game_splits.json")
    owner = {}
    for split, games in partition.items():
        for game_id in games:
            require(game_id not in owner, f"Game crosses partitions: {game_id}")
            owner[game_id] = split
    sources = {}; verified_games = set()
    seen_names = {spec["id"] for spec in config["opponents"]["seen_pool"]["agents"]}
    for batch in summary["batches"]:
        require_contract(batch)
        for replay in read_jsonl(Path(batch["replays_path"])):
            verify_replay_determinism(replay, raise_on_error=True)
            gid = replay["game_id"]
            require(gid in owner and gid not in verified_games, f"Invalid source game: {gid}")
            require(set(replay["seat_agent_names"]) <= seen_names, f"Unseen source strategy: {gid}")
            verified_games.add(gid)
            for step in replay["steps"]:
                obs = step["observation"]
                key = gid, obs["round_id"], obs["step_id"], obs["observer_seat"]
                require(key not in sources, f"Duplicate source decision: {key}")
                sources[key] = fingerprint(obs), step["pending_claim_was_bluff"]
        print(f"Reconstructed {len(verified_games)} complete source games", flush=True)
    require(verified_games == set(owner), "Partition and replay game identities differ.")

    originals = {}; original_counts = Counter(); original_input_counts = Counter()
    for split in SPLITS:
        for row in read_jsonl(source_run / f"dataset/{split}.jsonl"):
            if row["task"] == "action":
                require(row["sample_id"] not in originals, "Source action IDs are not unique.")
                originals[row["sample_id"]] = (row["completion"], input_key(row), split,
                                                fingerprint(row["observation"]))
                original_counts[split] += 1
                original_input_counts[input_key(row)] += 1

    backend = None
    if not skip_tokenizer:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import tinker
        from transformers import AutoTokenizer
        backend = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(
            config["base_model"], local_files_only=True), sdk=tinker)

    checked_reports = set(); audit_pool = defaultdict(dict); teacher = teacher_config(config)

    def check_action_report(row):
        key = input_key(row)
        path = run_dir / f"quality/candidate_reports/{key}.json"
        saved = read_json(path); report = saved["report"]
        require_contract(saved)
        require(saved["input_key"] == key and saved["teacher_config"] == teacher,
                f"Teacher report context differs for {row['sample_id']}.")
        require(report["seed"] == _derive_seed(settings["seed"], "quality_observation", key),
                f"Unexpected rollout seed for {row['sample_id']}.")
        obs = PlayerObservation.from_dict(row["observation"])
        if key not in checked_reports:
            validate_action_refinement_report(obs, report)
            checked_reports.add(key)
            audit_pool[report["rollouts_per_candidate"]][key] = (row["observation"], row["sample_id"])
        meta = row["action_refinement"]
        require(meta["input_key"] == key and meta["version"] == settings["version"],
                "Row and report version/input key differ.")
        for field in ("rollouts_per_candidate", "ambiguous"):
            require(meta[field] == report[field], f"Row has incorrect {field} metadata.")
        require(meta["best_vs_runnerup_gap"] == report["final"]["best_vs_runnerup_gap"],
                "Row has an incorrect runner-up gap.")
        require(json.loads(row["completion"])["chosen_action_id"] == report["chosen_action_id"],
                "Row target differs from its report.")
        require(abs(row["rollout_advantage"] - report["final"]["best_minus_mean_advantage"]) <= 1e-12,
                "Row advantage differs from stored rewards.")
        original = originals.get(row["sample_id"])
        if meta["origin"] == "original":
            require(original is not None, "Retained action has no original source row.")
            require((meta["original_completion"], key, owner[row["game_id"]], fingerprint(row["observation"])) == original,
                    "Retained original action provenance changed.")
            expected_force = original_input_counts[key] > 1
            expected_reference = None if expected_force else json.loads(original[0])["chosen_action_id"]
        else:
            require(meta["origin"] == "replacement" and original is None,
                    "Replacement reuses an original sample ID or has an unknown origin.")
            require(meta["original_completion"] is None, "Replacement has a fabricated original target.")
            expected_force, expected_reference = False, None
        require(report["force_refine"] == expected_force and report["reference_action_id"] == expected_reference,
                "Refinement did not use the expected original-label/duplicate-input screening rule.")
        return report

    def check_source(row, split):
        obs = validate_training_sample(row).to_dict()
        require(owner.get(row["game_id"]) == split, "Sample crossed its original game partition.")
        require(obs["ablation_mode"] == "full", "Non-full observation entered training data.")
        require(not ({"game_id", "game_seed", "ground_truth_hands", "fatal_chamber_slots"} & obs.keys()),
                "Private execution/truth field entered player observation.")
        require(row["game_id"] not in row["prompt"], "Game identifier entered model input.")
        require(row["round_id"] == obs["round_id"] and row["step_id"] == obs["step_id"]
                and row["actor_seat"] == obs["observer_seat"], "Sample position metadata differs from observation.")
        recorded = sources.get(sample_source_key(row))
        require(recorded is not None and fingerprint(obs) == recorded[0],
                f"Sample differs from replay observation: {row['sample_id']}")
        if row["task"] == "belief":
            require(recorded[1] is not None and row["label"] == int(recorded[1]),
                    f"Belief truth differs from engine: {row['sample_id']}")
        return obs

    ids = set(); all_hashes = {}; totals = Counter(); task_counts = {}; label_counts = {}
    token_stats = {}; longest = None; changed = Counter(); rollout_counts = Counter()
    replacements = []; ambiguous = 0; min_advantage = float("inf"); action_types = Counter()
    accepted_action_ids = set(); accepted_originals = set()
    for split in SPLITS:
        counts = Counter(); labels = Counter(); lengths = {t: [] for t in ("action", "belief")}
        task_hashes = {t: {} for t in ("action", "belief")}
        for row in read_jsonl(run_dir / f"dataset/{split}.jsonl"):
            sid = row["sample_id"]; task = row["task"]
            require(sid not in ids, f"Duplicate sample ID: {sid}")
            ids.add(sid); obs = check_source(row, split)
            counts[task] += 1; totals[task] += 1
            if task == "belief":
                labels[str(row["label"])] += 1
            else:
                report = check_action_report(row); meta = row["action_refinement"]
                accepted_action_ids.add(sid)
                require(row["rollout_advantage"] >= settings["minimum_best_minus_mean"],
                        "Low-advantage action entered training data.")
                min_advantage = min(min_advantage, row["rollout_advantage"])
                rollout_counts[str(report["rollouts_per_candidate"])] += 1
                ambiguous += int(report["ambiguous"])
                if meta["origin"] == "original":
                    accepted_originals.add(sid)
                    if row["completion"] != originals[sid][0]:
                        changed[split] += 1
                else:
                    replacements.append(sid)
                action = next(a for a in obs["legal_actions"] if a["action_id"] == report["chosen_action_id"])
                action_types[action["action_type"]] += 1
            if backend is not None:
                datum = build_sft_datum(row, backend, config["training"]["max_seq_length"])
                length = len(datum.model_input.to_ints()) + 1
                lengths[task].append(length)
                if longest is None or length > longest["tokens"]:
                    longest = {"sample_id": sid, "tokens": length, "split": split, "task": task}
            task_hashes[task][sid] = all_hashes[sid] = fingerprint(row)
        for task in ("action", "belief"):
            require(unique_row_hashes(run_dir / f"dataset/{task}_{split}.jsonl") == task_hashes[task],
                    f"Dedicated {task}/{split} file does not match combined data.")
        require(dict(counts) == summary["task_samples_per_split"][split], "Split/task summary counts differ.")
        require(sum(counts.values()) == summary["samples_per_split"][split], "Split summary count differs.")
        require(counts["action"] == original_counts[split], "Refinement changed action allocation across splits.")
        task_counts[split] = dict(counts); label_counts[split] = dict(labels)
        token_stats[split] = {task: {"count": len(values), "min": min(values), "max": max(values),
            "mean": sum(values) / len(values), "total": sum(values)} for task, values in lengths.items() if values}
        print(f"Verified {split}: {dict(counts)}; tokenizer={'skipped' if skip_tokenizer else 'passed'}", flush=True)
    require(totals["action"] == config["dataset"]["total_target_samples"] == summary["actual_samples"],
            "Action sample target/count differs.")
    require(totals["belief"] == config["dataset"]["belief_target_samples"] == summary["actual_belief_samples"],
            "Belief sample target/count differs.")
    require(unique_row_hashes(run_dir / "dataset/sft_all_samples.jsonl") == all_hashes,
            "Merged sample file differs from partition files.")

    rejected_ids = set(); rejected_originals = set(); rejected_replacements = 0
    for entry in read_jsonl(run_dir / "quality/rejected_candidates.jsonl"):
        row = entry["sample"]; sid = row["sample_id"]
        require(sid not in rejected_ids and sid not in accepted_action_ids, "Rejected action was duplicated or accepted.")
        rejected_ids.add(sid); check_source(row, entry["split"]); check_action_report(row)
        require(row["rollout_advantage"] < settings["minimum_best_minus_mean"], "Rejection reason contradicts refined advantage.")
        if row["action_refinement"]["origin"] == "original":
            rejected_originals.add(sid)
        else:
            rejected_replacements += 1
    require(accepted_originals | rejected_originals == set(originals), "Some original actions were lost without an audit record.")
    require({p.stem for p in (run_dir / "quality/candidate_reports").glob("*.json")} == checked_reports,
            "Candidate report files do not match reviewed/accepted/rejected inputs.")
    require(Counter(quality["changed_retained_labels_by_split"]) == changed, "Changed-label summary differs.")
    require(quality["original_action_rows"] == len(originals)
            and quality["distinct_original_inputs"] == len(original_input_counts), "Original input summary differs.")
    require(quality["duplicate_input_groups"] == sum(n > 1 for n in original_input_counts.values()),
            "Duplicate input summary differs.")
    require(quality["replacements"] == len(replacements) and set(quality["replacement_sample_ids"]) == set(replacements),
            "Replacement summary differs.")
    require(quality["replacement_candidates_reviewed"] == len(replacements) + rejected_replacements,
            "Replacement review count differs.")
    require(quality["rejected_candidates"] == len(rejected_ids), "Rejected count differs.")
    require(quality["final_rollouts_per_candidate"] == dict(rollout_counts)
            and quality["remaining_ambiguous_rows"] == ambiguous, "Effort/ambiguity summary differs.")
    classifier_path = run_dir / "models/small_classifier_weights.json"
    require_contract(read_json(classifier_path))
    require(file_hash(classifier_path) == settings["source_sha256"][str(source_run / "models/small_classifier_weights.json")],
            "Classifier changed despite unchanged belief training data.")
    require(summary["classifier_fit"]["num_samples"] == task_counts["train"]["belief"], "Classifier fit count differs.")

    regenerated = []
    for count in (8, 32, 64):
        keys = sorted(audit_pool[count])
        chosen = random.Random(_derive_seed(AUDIT_SEED, count)).sample(keys, min(4, len(keys)))
        for key in chosen:
            observation, sid = audit_pool[count][key]
            saved = read_json(run_dir / f"quality/candidate_reports/{key}.json")
            report = saved["report"]
            rewards = evaluate_observation_action_rewards(PlayerObservation.from_dict(observation),
                saved["teacher_config"], count, report["seed"])
            require(rewards == report["final"]["candidate_rewards"], f"Seed-based reward regeneration differs: {sid}")
            regenerated.append({"sample_id": sid, "input_key": key, "rollouts_per_candidate": count,
                                "candidate_count": len(rewards), "all_rewards_match": True})
        print(f"Reproduced stored rewards for {len(chosen)} inputs with {count} trajectories per candidate", flush=True)

    # Refined runs retain the original game partition and refer to its replay
    # files instead of copying the large replay corpus into the new run.
    coverage = validate_endgame_coverage(coverage_dir, source_run)
    require(coverage["status"] == "complete", "Endgame coverage has an unresolved shortfall.")
    saved_coverage = read_json(coverage_dir / "coverage_report.json")
    require(fingerprint(coverage) == fingerprint(saved_coverage), "Stored coverage summary differs from independent check.")
    probe_ids = {r["sample_id"] for name in ("observations.jsonl", "no_pending_claim.jsonl")
                 for r in read_jsonl(coverage_dir / name)}
    require(not probe_ids & ids, "Coverage probes were mixed into main dataset.")
    checks = ["source_file_hashes_unchanged", "whole_game_partition_unchanged", "all_source_replays_reconstructed",
              "seen_source_strategies_only", "all_sample_observations_match_source", "belief_labels_match_engine_truth",
              "full_observations_without_execution_ids", "all_sample_protocols_valid", "unique_ids_and_exact_file_consistency",
              "all_accepted_and_rejected_reward_summaries", "original_and_replacement_provenance", "quality_summary_recomputed",
              "selected_rewards_regenerated_from_seed", "classifier_weights_and_training_beliefs_unchanged",
              "independent_endgame_coverage_replays_labels_and_rules"]
    if not skip_tokenizer:
        checks.append("real_cached_tokenizer_and_training_masks_all_samples")
    try:
        code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parents[1]).strip()
        code_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
            cwd=Path(__file__).resolve().parents[1]).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        code_commit, code_dirty = None, None
    report = {"status": "partial_tokenizer_skipped" if skip_tokenizer else "passed",
        "contract": config["contract"], "refinement_version": settings["version"],
        "run_dir": str(run_dir), "source_run": str(source_run), "code_commit": code_commit,
        "code_worktree_has_changes": code_dirty, "verified_complete_games": len(verified_games),
        "unique_samples": len(ids), "samples_by_task": dict(totals), "samples_by_split_and_task": task_counts,
        "games_by_split": {key:len(value) for key,value in partition.items()},
        "belief_labels_by_split": label_counts, "action_types": dict(action_types),
        "minimum_action_advantage": min_advantage, "changed_retained_labels_by_split": dict(changed),
        "replacements": len(replacements), "rejected_candidates": len(rejected_ids),
        "final_rollouts_per_candidate": dict(rollout_counts), "remaining_ambiguous_rows": ambiguous,
        "refinement_arithmetic_check": refinement_check, "all_candidate_reports_verified": len(checked_reports),
        "seed_regeneration_audit": {"selection_seed": AUDIT_SEED, "num_inputs": len(regenerated), "results": regenerated},
        "tokenizer_checked": not skip_tokenizer, "token_counts_including_final_target": token_stats,
        "longest_sample": longest, "max_seq_length": config["training"]["max_seq_length"],
        "source_hashes_unchanged": True, "source_files_hashed": len(settings["source_sha256"]),
        "endgame_coverage": coverage, "checks": checks, "model_api_calls": 0,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "limitations": "Checks establish provenance, arithmetic and reproducibility; no model training or win-rate improvement was evaluated."}
    report_path = run_dir / "quality" / ("verification_report_partial.json" if skip_tokenizer else "verification_report.json")
    write_json(report_path, report)
    print(f"Verification {report['status']}: {report_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--coverage-dir", type=Path)
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="For local tests only; emits a partial report, never a full pass.")
    args = parser.parse_args()
    verify_dataset_quality(args.run_dir, args.coverage_dir, skip_tokenizer=args.skip_tokenizer)


if __name__ == "__main__":
    main()

"""Independently rescore a fixed random sample of retained original train labels.

This local diagnostic writes only an audit report. It neither selects training
rows nor changes labels, and its results must not feed back into refinement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from liars_agent.agents.prompt_builder import validate_action_decision
from liars_agent.contracts import current_contract, require_contract
from liars_agent.engine.state import _derive_seed
from liars_agent.experiment import read_json, teacher_config
from liars_agent.training.data_generator import evaluate_observation_action_rewards
from liars_agent.training.tinker_train import validate_training_sample

AUDIT_SEED = 20261116
ROLLOUTS_PER_CANDIDATE = 128
TIE_TOLERANCE = 1e-12
AUDIT_VERSION = "retained_original_train_independent_128_v1"


def _sample_original_rows(path: Path, count: int) -> tuple[list[dict], dict]:
    """Uniform reservoir sampling uses row eligibility only, never label outcomes."""
    rng = random.Random(AUDIT_SEED)
    selected: list[dict] = []
    digest = hashlib.sha256()
    total_rows = eligible_rows = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "action":
                raise ValueError("action_train.jsonl contains a non-action sample.")
            total_rows += 1
            if row.get("action_refinement", {}).get("origin") != "original":
                continue
            eligible_rows += 1
            if len(selected) < count:
                selected.append(row)
            else:
                index = rng.randrange(eligible_rows)
                if index < count:
                    selected[index] = row
    if eligible_rows < count:
        raise ValueError(f"Requested {count} samples, but only {eligible_rows} retained original train rows exist.")
    # Ordering changes execution order only; the random membership is fixed above.
    selected.sort(key=lambda row: row["sample_id"])
    if len({row["sample_id"] for row in selected}) != count:
        raise ValueError("Selected training sample IDs are not unique.")
    return selected, {
        "source_sha256": digest.hexdigest(),
        "total_action_train_rows": total_rows,
        "eligible_original_train_rows": eligible_rows,
        "excluded_nonoriginal_rows": total_rows - eligible_rows,
    }


def _audit_row(job: tuple[dict, dict]) -> dict[str, Any]:
    row, teacher = job
    obs = validate_training_sample(row)
    if obs.ablation_mode != "full":
        raise ValueError("Independent teacher audit requires full observations.")
    original_completion = row["action_refinement"]["original_completion"]
    old_id = validate_action_decision(obs, original_completion)["chosen_action_id"]
    new_id = validate_action_decision(obs, row["completion"])["chosen_action_id"]
    seed = _derive_seed(AUDIT_SEED, "independent_label_audit", row["sample_id"])
    rewards = evaluate_observation_action_rewards(obs, teacher, ROLLOUTS_PER_CANDIDATE, seed)
    means = {action_id: sum(values) / ROLLOUTS_PER_CANDIDATE for action_id, values in rewards.items()}
    best_value = max(means.values())
    best_set = sorted(action_id for action_id, value in means.items()
                      if abs(best_value - value) <= TIE_TOLERANCE)
    return {
        "sample_id": row["sample_id"],
        "game_id": row["game_id"],
        "split": "train",
        "origin": "original",
        "input_key": row["action_refinement"]["input_key"],
        "seed": seed,
        "rollouts_per_candidate": ROLLOUTS_PER_CANDIDATE,
        "candidate_count": len(means),
        "total_rollouts": len(means) * ROLLOUTS_PER_CANDIDATE,
        "candidate_means": means,
        "candidate_rewards": rewards,
        "independent_best_action_ids": best_set,
        "independent_best_estimated_reward": best_value,
        "original_completion": original_completion,
        "refined_completion": row["completion"],
        "old_action_id": old_id,
        "new_action_id": new_id,
        "label_changed": old_id != new_id,
        "old_estimated_reward": means[old_id],
        "new_estimated_reward": means[new_id],
        "old_in_estimated_best_set": old_id in best_set,
        "new_in_estimated_best_set": new_id in best_set,
        "old_estimated_regret": best_value - means[old_id],
        "new_estimated_regret": best_value - means[new_id],
        "new_minus_old_estimated_reward": means[new_id] - means[old_id],
    }


def audit_refined_labels(run_dir: str | Path, samples: int = 60, workers: int = 4) -> dict[str, Any]:
    if samples < 1 or workers < 1:
        raise ValueError("samples and workers must be positive.")
    started = time.perf_counter()
    run_dir = Path(run_dir).resolve()
    status = read_json(run_dir / "run_status.json")
    if status.get("stages", {}).get("data") != "complete":
        raise ValueError("Wait until the refined dataset data stage is complete before auditing final labels.")
    config_path = run_dir / "config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    require_contract(config, "independent audit configuration")
    teacher = teacher_config(config)
    source_path = run_dir / "dataset/action_train.jsonl"
    rows, population = _sample_original_rows(source_path, samples)
    jobs = [(row, teacher) for row in rows]
    records = []
    pool = ProcessPoolExecutor(max_workers=min(workers, samples)) if workers > 1 else None
    try:
        results = pool.map(_audit_row, jobs, chunksize=1) if pool else map(_audit_row, jobs)
        for result in results:
            records.append(result)
            if len(records) % 5 == 0 or len(records) == samples:
                print(f"Independent label audit: {len(records)}/{samples}", flush=True)
    finally:
        if pool:
            pool.shutdown()
    gains = [row["new_minus_old_estimated_reward"] for row in records]
    summary = {
        "sample_count": len(records),
        "label_changed_count": sum(row["label_changed"] for row in records),
        "old_best_set_hit_count": sum(row["old_in_estimated_best_set"] for row in records),
        "new_best_set_hit_count": sum(row["new_in_estimated_best_set"] for row in records),
        "old_mean_estimated_reward": statistics.mean(row["old_estimated_reward"] for row in records),
        "new_mean_estimated_reward": statistics.mean(row["new_estimated_reward"] for row in records),
        "old_mean_estimated_regret": statistics.mean(row["old_estimated_regret"] for row in records),
        "new_mean_estimated_regret": statistics.mean(row["new_estimated_regret"] for row in records),
        "mean_new_minus_old_estimated_reward": statistics.mean(gains),
        "mean_absolute_reward_change": statistics.mean(abs(value) for value in gains),
        "max_absolute_reward_change": max(abs(value) for value in gains),
        "minimum_reward_change": min(gains),
        "maximum_reward_change": max(gains),
        "positive_reward_change_count": sum(value > TIE_TOLERANCE for value in gains),
        "tied_reward_count": sum(abs(value) <= TIE_TOLERANCE for value in gains),
        "negative_reward_change_count": sum(value < -TIE_TOLERANCE for value in gains),
        "total_rollouts": sum(row["total_rollouts"] for row in records),
    }
    report = {
        "version": AUDIT_VERSION,
        "contract": current_contract(),
        "status": "complete",
        "run_dir": str(run_dir),
        "source_path": str(source_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "population": population,
        "sampling": {"seed": AUDIT_SEED, "requested_samples": samples,
                     "method": "uniform_reservoir_without_replacement_over_retained_original_train_rows",
                     "selection_uses_label_results": False},
        "teacher_config": teacher,
        "scoring": {
            "rollouts_per_candidate": ROLLOUTS_PER_CANDIDATE,
            "seed_derivation": "_derive_seed(AUDIT_SEED, 'independent_label_audit', sample_id)",
            "paired_across_all_legal_candidates": True,
            "reward": "0.65 * final_game_win + 0.35 * no_penalty_in_current_round",
            "tie_tolerance": TIE_TOLERANCE,
        },
        "scope": "Fixed random sample of retained original training rows only; replacement rows, validation and test sets are excluded.",
        "interpretation": "Finite-sample continuation-teacher stability diagnostic; estimated rewards are not real-model win rates or proof of optimality. No p-values or significance claims are computed.",
        "feedback_policy": "Audit only; results must not be used to select rows, relabel data or tune refinement settings.",
        "api_calls": 0,
        "summary": summary,
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    target = run_dir / "quality/independent_label_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"report": str(target), "api_calls": 0, "summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    audit_refined_labels(args.run_dir, args.samples, args.workers)


if __name__ == "__main__":
    main()

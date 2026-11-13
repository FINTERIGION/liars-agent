"""Reuse a verified five-step token estimate and add only small closure probes.

No dataset generation, rollout, model forward, service preflight or optimizer
step runs here. The historical body estimate is usable only for the exact seven
files recorded by the existing offline acceptance report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path

from liars_agent import experiment
from liars_agent.contracts import require_contract
from liars_agent.training.action_supervision import normalize_action_policy
from liars_agent.training import tinker_backend
from liars_agent.training.tinker_train import next_logical_batch_indices, training_diagnostic_plan

DEFAULT_VERIFIED_REPORT = experiment.ROOT / "docs/validation/training_readiness_v3.json"
VERIFIED_FILES = {
    "config.json", "dataset/train.jsonl", "dataset/val.jsonl", "dataset/test.jsonl",
    "dataset/game_splits.json", "augmentation/replacements.jsonl", "augmentation/added_actions.jsonl",
}
BODY_FIELDS = {
    "balanced_training_tokens", "full_validation_tokens", "diagnostic_tokens", "validation_tokens",
    "training_optimizer_steps", "training_logical_state_visits", "training_expanded_sequences",
    "training_plan", "sft_train_and_validation_usd", "rates_per_million_tokens",
    "pricing_date", "pricing_source", "endgame_evaluation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def refresh_trial_estimate(run_dir: str | Path,
                           verified_report: str | Path = DEFAULT_VERIFIED_REPORT) -> dict:
    """Refresh closure costs after local provenance checks, without full tokenization."""
    run_dir, verified_report = Path(run_dir).resolve(), Path(verified_report).resolve()
    verified = experiment.read_json(verified_report)
    hashes = verified.get("dataset_files_sha256", {})
    _require(set(hashes) == VERIFIED_FILES, "Verified report must bind the seven supported run files.")
    for relative, expected in hashes.items():
        _require(_sha256(run_dir / relative) == expected,
                 f"Verified dataset/config hash differs: {relative}; do not reuse old token totals.")

    config = experiment.load_config(run_dir / "config.json")
    require_contract(config, "verified run configuration")
    policy = normalize_action_policy(config["training"]["action_supervision"])
    estimate_path = run_dir / "trial_cost_estimate.json"
    previous = experiment.read_json(estimate_path)
    require_contract(previous, "existing trial estimate")
    baseline = verified.get("five_step_trial_estimate", {})
    _require(set(baseline) >= BODY_FIELDS, "Verified report lacks the complete five-step body estimate.")
    for key, value in baseline.items():
        _require(key in previous and previous[key] == value,
                 f"Existing trial estimate differs from the verified baseline: {key}.")
    _require(previous.get("model") == config["base_model"], "Trial estimate base model differs from configuration.")
    _require(previous.get("action_supervision_policy") == policy, "Existing trial estimate uses a different action objective.")
    supervision = verified.get("supervision", {})
    require_contract(supervision, "verified supervision report")
    tokenizer_check = verified.get("tokenizer_check", {})
    _require(supervision.get("policy") == policy and tokenizer_check.get("policy") == policy,
             "Verified token totals belong to a different action objective.")
    _require(tokenizer_check.get("status") == "passed"
             and tokenizer_check.get("all_action_targets_stop_at_first_assistant_eos") is True
             and tokenizer_check.get("belief_unit_weight_verified_for_all_rows") is True,
             "The existing report does not certify token boundaries and task masks.")
    _require(tokenizer_check.get("max_seq_length") == config["training"]["max_seq_length"],
             "Verified tokenizer limit differs from the current configuration.")
    _require(baseline["rates_per_million_tokens"] == tinker_backend.DEFAULT_RATES
             and baseline["pricing_source"] == tinker_backend.PRICING_SOURCE
             and baseline["pricing_date"] == tinker_backend.PRICING_DATE,
             "Verified estimate uses different backend rates or a different pricing snapshot.")

    splits = {name: list(experiment.read_jsonl(run_dir / f"dataset/{name}.jsonl"))
              for name in ("train", "val", "test")}
    counts = {name: dict(Counter(row["task"] for row in rows)) for name, rows in splits.items()}
    _require(all(set(tasks) == {"action", "belief"} for tasks in counts.values()),
             "The verified run must contain both tasks in each partition.")
    augmentation = verified.get("augmentation_verification", {})
    require_contract(augmentation, "verified dataset report")
    _require(augmentation.get("status") == "passed" and augmentation.get("task_samples_per_split") == counts,
             "Actual task counts differ from the verified dataset.")
    ds = config["dataset"]
    _require(sum(values["action"] for values in counts.values()) == ds["total_target_samples"]
             and sum(values["belief"] for values in counts.values()) == ds["belief_target_samples"],
             "Actual task totals differ from the fixed dataset configuration.")
    training = config["training"]
    plan = training_diagnostic_plan(counts["train"], training["num_epochs"], training["batch_size"],
        training.get("diagnostic_steps", [50, 100]), max_train_steps=5)
    _require(plan["planned_steps"] == 5 and baseline["training_plan"] == plan
             and baseline["training_optimizer_steps"] == 5,
             "Only the verified five-step training/diagnostic schedule can reuse this estimate.")
    expected_visits = {task: 5 * (training["batch_size"] // 2) for task in ("action", "belief")}
    _require(baseline["training_logical_state_visits"] == expected_visits,
             "Verified body state visits differ from the current logical batch plan.")
    actions = [row for row in splits["train"] if row["task"] == "action"]
    seed = training.get("seed", config["seed"])
    expanded_actions = sum(len(actions[index]["action_supervision"]["target_action_ids"])
        for step in range(5) for index in next_logical_batch_indices(
            counts["train"], training["batch_size"], seed, step)["action"])
    _require(baseline["training_expanded_sequences"] == {
        "action": expanded_actions, "belief": expected_visits["belief"]},
        "Verified expanded-answer counts differ from the actual five-step state schedule.")
    _require(baseline["validation_tokens"] == baseline["full_validation_tokens"] + baseline["diagnostic_tokens"],
             "Verified validation token subtotals are inconsistent.")
    body_tokens = baseline["balanced_training_tokens"] + baseline["validation_tokens"]
    _require(math.isclose(baseline["sft_train_and_validation_usd"],
        body_tokens * tinker_backend.DEFAULT_RATES["train"] / 1_000_000, rel_tol=1e-12, abs_tol=1e-12),
        "Verified body cost does not match its token count and rate.")

    # All provenance/config checks precede even this local constructor. The
    # estimator below encodes only fixed probes and one continuation batch.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    backend = experiment.make_backend(config, run_dir)
    _require(backend.rates == baseline["rates_per_million_tokens"], "Constructed backend rates differ from the verified estimate.")
    closure = experiment.estimate_trial_closure(config, backend, splits, max_train_steps=5)
    _require(type(closure.get("total_usd_upper_bound")) in (int, float)
             and math.isfinite(closure["total_usd_upper_bound"]) and closure["total_usd_upper_bound"] >= 0,
             "Closure estimate must contain a finite nonnegative cost.")
    result = copy.deepcopy(previous)
    result["trial_verification"] = closure
    result["trial_total_usd_upper_bound"] = baseline["sft_train_and_validation_usd"] + closure["total_usd_upper_bound"]
    result["verified_baseline_reuse"] = {
        "verified_report": str(verified_report), "verified_report_sha256": _sha256(verified_report),
        "matched_file_hashes": sorted(hashes), "matched_baseline_fields": sorted(baseline),
        "baseline_body_tokens_reused": body_tokens, "body_optimizer_steps": 5,
        "full_dataset_retokenized": False, "tokenized_scope": "four validation probes and one logical continuation batch; sampler label candidates included",
        "model_api_calls": 0, "note": "Historical token accounting reused only after exact dataset/config and pricing checks."}
    result["scope"] = "Verified five-step body estimate plus locally tokenized update/sampler/state/optimizer checks; excludes checkpoint storage and separate endgame evaluation."
    temporary = estimate_path.with_suffix(".json.tmp")
    experiment.write_json(temporary, result)
    temporary.replace(estimate_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verified-report", type=Path, default=DEFAULT_VERIFIED_REPORT)
    args = parser.parse_args()
    result = refresh_trial_estimate(args.run_dir, args.verified_report)
    print(json.dumps({key: result[key] for key in (
        "sft_train_and_validation_usd", "trial_verification", "trial_total_usd_upper_bound", "verified_baseline_reuse")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

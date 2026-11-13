"""Offline acceptance check for prepared action supervision and token accounting.

Uses a cached real tokenizer and SDK data types only. No service client,
preflight, model forward, optimizer step or paid API request is constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

from liars_agent.contracts import current_contract
from liars_agent.experiment import read_json, read_jsonl, write_json
from liars_agent.training.action_supervision import (
    action_target_completions, mean_action_weight, normalize_action_policy,
    resolve_action_supervision,
)
from liars_agent.training.action_validation import candidate_completion
from liars_agent.training.prepare_supervision import validate_prepared_supervision_run
from liars_agent.training.tinker_train import (build_sft_datum, validate_training_sample,
    select_diagnostic_rows, training_diagnostic_plan)

SPLITS = ("train", "val", "test")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def as_list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def check_action_supervision(run_dir: Path, *, cost_estimate: Path | None = None) -> dict:
    started = time.monotonic()
    run_dir = run_dir.resolve()
    config = read_json(run_dir / "config.json")
    if config["dataset"].get("endgame_augmentation"):
        from liars_agent.training.endgame_augmentation import validate_endgame_augmented_run
        source_check = validate_endgame_augmented_run(run_dir, check_sources=True)
    else:
        source_check = validate_prepared_supervision_run(run_dir, check_sources=True)
    print("Prepared schemas, stored targets and unchanged sources verified", flush=True)
    config = read_json(run_dir / "config.json")
    training = config["training"]
    policy = normalize_action_policy(training["action_supervision"])
    seed = training.get("seed", config["seed"])
    epochs = training["num_epochs"]
    max_seq_length = training["max_seq_length"]
    batch_size = training["batch_size"]
    require(type(epochs) is int and epochs > 0, "num_epochs must be a positive integer.")
    require(type(batch_size) is int and batch_size >= 2 and batch_size % 2 == 0,
            "Balanced training requires a positive even batch size.")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    import tinker
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], local_files_only=True)
    backend = SimpleNamespace(tokenizer=tokenizer, sdk=tinker)
    grouped = {split: {task: [] for task in ("action", "belief")} for split in SPLITS}
    for split in SPLITS:
        for row in read_jsonl(run_dir / f"dataset/{split}.jsonl"):
            grouped[split][row["task"]].append(row)
    require(all(grouped["train"][task] for task in ("action", "belief")), "Both training tasks require rows.")
    train_mean = mean_action_weight(grouped["train"]["action"], policy)
    prompt_lengths, datum_cache = {}, {}
    datum_builds = 0
    max_input_tokens = 0
    longest = None

    def check_datum(row, completion=None, weight=1.0, *, fresh=False):
        nonlocal datum_builds, max_input_tokens, longest
        answer = row["completion"] if completion is None else completion
        key = row["sample_id"], answer, weight
        if not fresh and key in datum_cache:
            return datum_cache[key]
        datum = build_sft_datum(row, backend, max_seq_length, completion_override=completion,
                                sample_weight=weight, action_supervision_policy=policy)
        datum_builds += 1
        input_tokens = datum.model_input.to_ints()
        targets = as_list(datum.loss_fn_inputs["target_tokens"])
        weights = as_list(datum.loss_fn_inputs["weights"])
        if row["sample_id"] not in prompt_lengths:
            messages = [{"role": "system", "content": row["system"]},
                        {"role": "user", "content": row["prompt"]}]
            prefix = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            prompt_lengths[row["sample_id"]] = len(tokenizer.encode(prefix, add_special_tokens=False))
        prefix_length = prompt_lengths[row["sample_id"]]
        answer_tokens = len(input_tokens) + 1 - prefix_length
        require(len(input_tokens) == len(targets) == len(weights), "Model inputs, targets and masks differ in length.")
        require(0 < len(input_tokens) + 1 <= max_seq_length, "A complete answer exceeds max_seq_length.")
        require(answer_tokens > 0, "A supervised answer has no tokens.")
        require(all(value == 0 for value in weights[:prefix_length - 1]), "Prompt tokens have nonzero loss weight.")
        require(len(weights[prefix_length - 1:]) == answer_tokens, "Answer mask starts at the wrong token boundary.")
        expected = weight / answer_tokens
        require(all(math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-8)
                    for value in weights[prefix_length - 1:]), "Answer token weights differ from the per-state weight.")
        require(math.isclose(math.fsum(weights), weight, rel_tol=1e-6, abs_tol=1e-8),
                "Total supervised weight differs from the requested state weight.")
        if row["task"] == "belief":
            label_tokens = tokenizer.encode(row["completion"], add_special_tokens=False)
            require(weight == 1.0 and answer_tokens == len(label_tokens) == 1,
                    "Belief supervision must be one token with weight one.")
            require(targets[-1] == label_tokens[0] and weights[-1] == 1.0,
                    "Belief target or unit loss weight changed.")
        else:
            supervised_tokens = targets[prefix_length - 1:]
            require(supervised_tokens[-1] == tokenizer.eos_token_id,
                    "Action answer does not end at assistant EOS.")
            require(tokenizer.eos_token_id not in supervised_tokens[:-1],
                    "Action target contains tokens after its first EOS.")
        if len(input_tokens) > max_input_tokens:
            max_input_tokens = len(input_tokens)
            longest = {"sample_id": row["sample_id"], "task": row["task"],
                       "input_tokens": len(input_tokens), "complete_sequence_tokens": len(input_tokens) + 1}
        result = {"input_tokens": len(input_tokens), "answer_tokens": answer_tokens,
                  "supervised_weight": math.fsum(weights)}
        datum_cache[key] = result
        return result

    summary_by_split = {}
    action_expanded_lengths, action_target_counts = [], []
    canonical_lengths = {split: {task: [] for task in ("action", "belief")} for split in SPLITS}
    effective_train_weights = []
    measured_train_weights = []
    all_target_variants = 0
    belief_action_metadata_checks = 0
    belief_weight_rejection_checks = 0
    checked_rows = 0
    for split in SPLITS:
        groups, target_sizes = Counter(), Counter()
        action_target_variants = 0
        for row in grouped[split]["action"]:
            validate_training_sample(row, action_supervision_policy=policy)
            descriptor = resolve_action_supervision(row, policy)
            targets = descriptor["target_action_ids"]
            groups[descriptor["group"]] += 1
            target_sizes[len(targets)] += 1
            weight = descriptor["loss_weight"] / train_mean if split == "train" else 1.0
            answer_weight = weight / len(targets)
            if split == "train":
                effective_train_weights.append(weight)
            canonical_lengths[split]["action"].append(check_datum(row, weight=answer_weight)["input_tokens"])
            completions = action_target_completions(row, policy)
            selected = [json.loads(completion)["chosen_action_id"] for completion in completions]
            require(len(selected) == len(targets) and set(selected) == set(targets),
                    "State expansion must contain every acceptable answer exactly once.")
            require(action_target_completions(row, policy) == completions,
                    "State answer expansion is not deterministic.")
            target_data = []
            for completion in completions:
                target_data.append(check_datum(row, completion, answer_weight))
                action_target_variants += 1
                all_target_variants += 1
            state_weight = math.fsum(item["supervised_weight"] for item in target_data)
            require(math.isclose(state_weight, weight, rel_tol=1e-6, abs_tol=1e-8),
                    "Expanded answers do not preserve their logical state's total loss weight.")
            if split == "train":
                measured_train_weights.append(state_weight)
                action_expanded_lengths.append(sum(item["input_tokens"] for item in target_data))
                action_target_counts.append(len(target_data))
            checked_rows += 1
            if checked_rows % 1000 == 0:
                print(f"Checked {checked_rows} action rows and {all_target_variants} acceptable target variants", flush=True)
        for row in grouped[split]["belief"]:
            require("action_supervision" not in row, "A belief row inherited action supervision fields.")
            validate_training_sample(row, action_supervision_policy=policy)
            canonical_lengths[split]["belief"].append(check_datum(row)["input_tokens"])
        if grouped[split]["belief"]:
            # A representative mutation per partition checks that action metadata
            # cannot alter belief masks, in addition to checking every real row.
            belief = grouped[split]["belief"][0]
            injected = {**belief, "action_supervision": {"loss_weight": .001,
                        "target_action_ids": ["not-a-belief-target"]}}
            require(check_datum(injected, fresh=True) == check_datum(belief),
                    "Injected action metadata changed belief tokenization or masks.")
            belief_action_metadata_checks += 1
            try:
                build_sft_datum(belief, backend, max_seq_length, sample_weight=.5,
                                action_supervision_policy=policy)
            except ValueError:
                belief_weight_rejection_checks += 1
            else:
                raise ValueError("The builder accepted an action-style weight on a belief row.")
        summary_by_split[split] = {
            "num_action_rows": len(grouped[split]["action"]), "num_belief_rows": len(grouped[split]["belief"]),
            "groups": dict(groups), "target_set_sizes": dict(sorted(target_sizes.items())),
            "acceptable_target_variants_checked": action_target_variants,
            "each_state_contains_all_acceptable_targets": True,
        }
        print(f"{split}: all target masks, per-state weight sums and belief masks checked", flush=True)
    require(math.isclose(math.fsum(effective_train_weights), len(effective_train_weights), rel_tol=1e-12),
            "Normalized action weights do not total the number of training states.")
    require(math.isclose(math.fsum(measured_train_weights), len(effective_train_weights), rel_tol=1e-6),
            "Expanded answer masks do not total the original action-state weight budget.")

    validation_action_tokens = 0
    validation_candidates = 0
    for index, row in enumerate(grouped["val"]["action"]):
        for action in row["observation"]["legal_actions"]:
            completion = candidate_completion(row, action["action_id"])
            validation_action_tokens += check_datum(row, completion)["input_tokens"]
            validation_candidates += 1
        if (index + 1) % 100 == 0:
            print(f"Tokenized all legal validation candidates for {index + 1} states", flush=True)
    half = batch_size // 2
    steps = math.ceil(max(len(grouped["train"][task]) for task in ("action", "belief")) / half)
    epoch_stats = []
    for epoch in range(epochs):
        per_task = {}
        for task_index, task in enumerate(("action", "belief")):
            lengths = action_expanded_lengths if task == "action" else canonical_lengths["train"]["belief"]
            order = list(range(len(lengths)))
            random.Random(seed + 2 * epoch + task_index).shuffle(order)
            visits = [order[index % len(order)] for index in range(steps * half)]
            require(len(set(visits)) == len(lengths), "An epoch omitted a logical training state.")
            expanded_count = sum(action_target_counts[index] for index in visits) if task == "action" else len(visits)
            per_task[task] = {"logical_state_visits": len(visits), "expanded_sequences": expanded_count,
                              "input_tokens": sum(lengths[index] for index in visits),
                              "cycled_extra_visits": len(visits) - len(set(visits))}
        epoch_stats.append({"epoch": epoch + 1, "optimizer_steps_if_run": steps,
                            "all_training_states_and_their_targets_covered": True,
                            "tasks": per_task, "input_tokens": sum(item["input_tokens"] for item in per_task.values())})
    validation_belief_tokens = sum(canonical_lengths["val"]["belief"])
    plan = training_diagnostic_plan({task: len(rows) for task, rows in grouped["train"].items()},
        epochs, batch_size, training.get("diagnostic_steps", [50, 100]))
    diagnostics = select_diagnostic_rows(
        {split: [row for task in tasks.values() for row in task] for split, tasks in grouped.items()},
        seed, training.get("diagnostic_samples_per_task", 64))
    diagnostic_action_tokens = sum(check_datum(row, candidate_completion(row, action["action_id"]))["input_tokens"]
        for row in diagnostics["action"] for action in row["observation"]["legal_actions"])
    diagnostic_belief_tokens = sum(check_datum(row)["input_tokens"] for row in diagnostics["belief"])
    diagnostic_tokens = (diagnostic_action_tokens + diagnostic_belief_tokens) * len(plan["diagnostic_steps"])
    full_validation_tokens = (validation_action_tokens + validation_belief_tokens) * epochs
    accounting = {
        "balanced_training_tokens": sum(epoch["input_tokens"] for epoch in epoch_stats),
        "training_optimizer_steps_per_epoch": steps,
        "training_optimizer_steps": steps * epochs,
        "training_logical_state_visits": {task: sum(epoch["tasks"][task]["logical_state_visits"] for epoch in epoch_stats)
                                         for task in ("action", "belief")},
        "training_expanded_sequences": {task: sum(epoch["tasks"][task]["expanded_sequences"] for epoch in epoch_stats)
                                       for task in ("action", "belief")},
        "validation_action_candidate_sequences_per_epoch": validation_candidates,
        "validation_action_tokens_per_epoch": validation_action_tokens,
        "validation_belief_tokens_per_epoch": validation_belief_tokens,
        "full_validation_tokens": full_validation_tokens,
        "diagnostic_tokens": diagnostic_tokens,
        "diagnostic_action_tokens_per_check": diagnostic_action_tokens,
        "diagnostic_belief_tokens_per_check": diagnostic_belief_tokens,
        "validation_tokens": full_validation_tokens + diagnostic_tokens,
        "diagnostic_sample_ids": {task: [row["sample_id"] for row in rows] for task, rows in diagnostics.items()},
        "training_plan": plan,
        "tokens_by_split_and_task": {split: {task: sum(values) for task, values in tasks.items()}
                                      for split, tasks in canonical_lengths.items()},
    }
    estimate_path = cost_estimate or run_dir / "cost_estimate.json"
    comparison = {"status": "not_present", "path": str(estimate_path),
                  "note": "Token totals below are available for independent comparison with experiment.estimate_cost."}
    if estimate_path.exists():
        estimate = read_json(estimate_path)
        require(estimate.get("action_supervision_policy") == policy, "Cost estimate uses a different supervision policy.")
        for key, value in accounting.items():
            require(estimate.get(key) == value, f"Cost-estimate token accounting differs at {key}.")
        comparison = {"status": "matched", "path": str(estimate_path), "fields": list(accounting)}
    report = {
        "status": "passed", "scope": "local schema, source identity, tokenizer, all-answer masks, logical-state expansion and token accounting only",
        "contract": current_contract(), "run_dir": str(run_dir), "source_check": source_check,
        "model": config["base_model"], "tokenizer": type(tokenizer).__name__, "tokenizer_local_files_only": True,
        "hf_offline": True, "model_api_calls": 0, "model_forwards": 0, "optimizer_steps": 0,
        "training_completed": False, "model_performance_evaluated": False,
        "policy": policy, "training_seed": seed, "configured_epochs": epochs,
        "samples_per_split": summary_by_split, "train_action_raw_weight_mean": train_mean,
        "train_action_effective_total_weight": math.fsum(effective_train_weights),
        "train_expanded_answer_measured_total_weight": math.fsum(measured_train_weights),
        "answer_weight_check": "train: raw_state_weight / train_mean / target_count; validation/test target-mask audit: 1 / target_count; actual validation forwards: unit weight per legal candidate",
        "belief_unit_weight_verified_for_all_rows": True,
        "all_action_targets_stop_at_first_assistant_eos": True,
        "belief_injected_action_metadata_checks": belief_action_metadata_checks,
        "belief_weight_override_rejections": belief_weight_rejection_checks,
        "answer_expansion_deterministic": True,
        "every_training_state_has_all_targets_each_epoch": True,
        "optimizer_step_basis": "unchanged logical-state batches; answers expand inside each original batch",
        "all_acceptable_target_variants_checked": all_target_variants,
        "datum_builds_including_local_checks": datum_builds,
        "max_input_tokens_across_checked_variants": max_input_tokens,
        "max_complete_sequence_tokens": max_input_tokens + 1, "max_seq_length": max_seq_length,
        "longest_checked_variant": longest, "training_epoch_tokens_if_run": epoch_stats,
        "token_accounting": accounting, "cost_estimate_comparison": comparison,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json(run_dir / "quality/action_supervision_check.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cost-estimate", type=Path)
    args = parser.parse_args()
    result = check_action_supervision(args.run_dir, cost_estimate=args.cost_estimate)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

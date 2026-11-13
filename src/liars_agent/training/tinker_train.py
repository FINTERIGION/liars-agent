"""Balanced action imitation and binary belief learning on one Tinker LoRA."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from liars_agent.agents.prompt_builder import (SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT,
    build_user_prompt, build_bluff_user_prompt, validate_action_decision)
from liars_agent.contracts import current_contract, require_contract
from liars_agent.engine.observation import PlayerObservation
from liars_agent.training.action_supervision import (
    mean_action_weight, normalize_action_policy, require_training_objective,
    resolve_action_supervision, select_action_completion,
)
from liars_agent.training.dataset_splitter import split_dataset_by_game_id
from liars_agent.training.tinker_backend import TinkerBackend, resolve_future


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_training_sample(sample: dict[str, Any], *,
                             action_supervision_policy: dict | None = None) -> PlayerObservation:
    """Reject stale contracts or malformed supervision before any model API call."""
    if sample.get("historical_example"):
        raise ValueError("Historical examples cannot be used for training; regenerate data with current rules.")
    require_contract(sample, "training sample")
    obs = PlayerObservation.from_dict(sample["observation"])
    task = sample.get("task")
    if task == "action":
        validate_action_decision(obs, sample["completion"])
        if action_supervision_policy is not None:
            resolve_action_supervision(sample, action_supervision_policy)
        elif "action_supervision" in sample:
            stored = sample["action_supervision"]
            stored_policy = stored.get("policy") if isinstance(stored, dict) else None
            resolve_action_supervision(sample, stored_policy)
        expected_system, expected_prompt = SYSTEM_PROMPT, build_user_prompt(obs)
    elif task == "belief":
        if obs.last_claim is None or sample.get("completion") not in ("0", "1"):
            raise ValueError("Belief supervision requires a pending claim and the exact label 0 or 1.")
        if type(sample.get("label")) is not int or sample["label"] != int(sample["completion"]):
            raise ValueError("Belief label must be an integer matching its completion.")
        expected_system, expected_prompt = BELIEF_SYSTEM_PROMPT, build_bluff_user_prompt(obs)
    else:
        raise ValueError("Training sample task must be action or belief.")
    if sample.get("system") != expected_system or sample.get("prompt") != expected_prompt:
        raise ValueError("Training sample prompt does not match its versioned observation and task.")
    return obs


def build_sft_datum(sample: dict[str, Any], backend: TinkerBackend, max_seq_length: int, *,
                    completion_override: str | None = None, sample_weight: float = 1.0,
                    action_supervision_policy: dict | None = None) -> Any:
    """Weighted mean action-token NLL, or one unweighted binary belief label.

    Legal action overrides serve both epoch target rotation and validation of all
    candidates. They do not mutate the canonical sample or its supervision set.
    Default arguments retain canonical tokenization for read-only data checks.
    """
    obs = validate_training_sample(sample, action_supervision_policy=action_supervision_policy)
    if type(sample_weight) not in (int, float) or not math.isfinite(sample_weight) or sample_weight <= 0:
        raise ValueError("sample_weight must be positive and finite.")
    if sample["task"] == "belief" and (completion_override is not None or sample_weight != 1.0):
        raise ValueError("Belief targets must remain single-label and unweighted.")
    completion = sample["completion"] if completion_override is None else completion_override
    if completion_override is not None:
        validate_action_decision(obs, completion)
    messages = [{"role": "system", "content": sample["system"]},
                {"role": "user", "content": sample["prompt"]}]
    tokenizer = backend.tokenizer
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if sample["task"] == "belief":
        labels = [tokenizer.encode(label, add_special_tokens=False) for label in ("0", "1")]
        if any(len(tokens) != 1 for tokens in labels) or labels[0] == labels[1]:
            raise ValueError("Binary belief labels 0 and 1 must be distinct single tokens.")
        completion_tokens = labels[int(sample["completion"])]
        tokens = prompt_tokens + completion_tokens
    else:
        full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": completion}],
                    tokenize=False, add_generation_prompt=False, enable_thinking=False)
        if not full.startswith(prompt):
            raise ValueError("Training chat template does not preserve the inference prompt prefix.")
        tokens = tokenizer.encode(full, add_special_tokens=False)
        if tokens[:len(prompt_tokens)] != prompt_tokens:
            raise ValueError("Training token boundary differs from the inference prompt.")
        completion_tokens = tokens[len(prompt_tokens):]
    if not completion_tokens:
        raise ValueError("SFT completion has no tokens.")
    if len(tokens) > max_seq_length:
        raise ValueError(f"Sample {sample.get('sample_id', '?')} has {len(tokens)} tokens, "
                         f"exceeding max_seq_length={max_seq_length}; no JSON was truncated.")
    weights = [0.0] * len(prompt_tokens) + [sample_weight / len(completion_tokens)] * len(completion_tokens)
    return backend.sdk.Datum(model_input=backend.sdk.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]})


def _list(value: Any) -> list[Any]:
    return value.tolist() if hasattr(value, "tolist") else list(value)


def batch_loss(result: Any, data: list[Any]) -> tuple[float, float]:
    """Return weighted NLL sum and the effective sample-weight sum."""
    if len(result.loss_fn_outputs) != len(data):
        raise RuntimeError("Tinker loss output count differs from batch size.")
    total, count = 0.0, 0.0
    for output, datum in zip(result.loss_fn_outputs, data):
        logprobs = _list(output["logprobs"])
        weights = _list(datum.loss_fn_inputs["weights"])
        if len(logprobs) != len(weights):
            raise RuntimeError("Tinker logprob lengths do not align with completion masks.")
        for logprob, weight in zip(logprobs, weights):
            total -= float(logprob) * float(weight)
            count += float(weight)
    if count <= 0 or not math.isfinite(total):
        raise RuntimeError("Training returned invalid supervised-token loss.")
    return total, count


def run_tinker_lora_sft(
    samples: list[dict[str, Any]], output_dir: str | Path,
    base_model: str = "Qwen/Qwen3.5-9B", lora_rank: int = 16,
    learning_rate: float = 2e-4, num_epochs: int = 3, batch_size: int = 32, seed: int = 42,
    *, backend: TinkerBackend | None = None,
    splits: dict[str, list[dict[str, Any]]] | None = None,
    max_seq_length: int = 16384, warmup_ratio: float = .05,
    resume_from: str | None = None, resume_metadata: dict[str, Any] | None = None,
    resume: bool = False, stage: str = "sft", checkpoint_ttl_seconds: int | None = None,
    train_attn: bool = True, train_mlp: bool = True, train_unembed: bool = False,
    action_supervision_policy: dict | None = None,
) -> dict[str, Any]:
    """Use equal task counts per batch; cycle the shorter task within each epoch.

    Exact empirical ties rotate their targets deterministically by epoch. Action
    weights are normalized by their training-set mean; belief retains weight one.
    Only unweighted action sequence-set validation NLL chooses the checkpoint.
    """
    if num_epochs < 1 or batch_size < 2 or batch_size % 2 or lora_rank < 1 or learning_rate <= 0:
        raise ValueError("Use positive epochs/rank/rate and an even batch_size of at least 2.")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1).")
    policy = normalize_action_policy(action_supervision_policy)
    # Validate every provided row, including test rows, before constructing an API backend.
    all_rows = samples + [row for rows in (splits or {}).values() for row in rows]
    for row in all_rows:
        validate_training_sample(row, action_supervision_policy=policy)
    if resume_from is not None:
        require_contract(resume_metadata or {}, "source checkpoint metadata")
        require_training_objective(resume_metadata or {}, policy)
        if resume_metadata.get("state_path") != resume_from or resume_metadata.get("base_model") != base_model:
            raise ValueError("Source checkpoint metadata does not match its state path/model.")
    splits = splits if splits is not None else split_dataset_by_game_id(samples, train_ratio=.8, val_ratio=.1, seed=seed)
    grouped = {name: {task: [row for row in rows if row["task"] == task] for task in ("action", "belief")}
               for name, rows in splits.items()}
    if any(not grouped.get(name, {}).get(task) for name in ("train", "val") for task in ("action", "belief")):
        raise ValueError("Training and validation must each contain both action and belief samples.")
    games = {key: {row["game_id"] for row in rows} for key, rows in splits.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if games.get(left, set()) & games.get(right, set()):
            raise ValueError(f"Game-level leakage between {left} and {right}.")
    action_weight_mean = mean_action_weight(grouped["train"]["action"], policy)
    fingerprint = hashlib.sha256(json.dumps(splits, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    out = Path(output_dir)
    progress_path = out / "training_progress.json"
    config = {"base_model": base_model, "lora_rank": lora_rank, "learning_rate": learning_rate,
        "batch_size": batch_size, "seed": seed, "max_seq_length": max_seq_length,
        "num_epochs": num_epochs, "warmup_ratio": warmup_ratio, "stage": stage,
        "train_attn": train_attn, "train_mlp": train_mlp, "train_unembed": train_unembed,
        "checkpoint_ttl_seconds": checkpoint_ttl_seconds,
        "task_balance": "equal_per_batch_cycle_shorter",
        "action_supervision_policy": policy,
        "train_action_weight_mean": action_weight_mean,
        "checkpoint_selection": "val_action_set_nll",
        "loss_metric_aliases": {"val_action_loss": "val_action_set_nll", "val_loss": "val_action_set_nll"}}
    progress: dict[str, Any] = {}
    if resume:
        if not progress_path.exists():
            raise ValueError("No completed-epoch checkpoint exists in this output directory.")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        require_contract(progress, "training progress")
        require_training_objective(progress, policy)
        if progress["dataset_sha256"] != fingerprint:
            raise ValueError("Cannot resume with a changed dataset.")
        if progress.get("config") != config:
            raise ValueError("Cannot resume with changed training configuration.")
    elif progress_path.exists():
        raise ValueError("Training output already has progress; use resume=True or a new output directory.")
    backend = backend or TinkerBackend(base_model=base_model)
    if backend.base_model != base_model:
        raise ValueError("Training model and backend model differ.")

    def action_data_for_epoch(epoch: int) -> list[Any]:
        return [build_sft_datum(row, backend, max_seq_length,
                    completion_override=select_action_completion(row, seed, epoch, policy),
                    sample_weight=resolve_action_supervision(row, policy)["loss_weight"] / action_weight_mean,
                    action_supervision_policy=policy)
                for row in grouped["train"]["action"]]

    start_epoch = int(progress.get("next_epoch", 0))
    train_data = {"action": action_data_for_epoch(start_epoch),
                  "belief": [build_sft_datum(row, backend, max_seq_length,
                                action_supervision_policy=policy) for row in grouped["train"]["belief"]]}
    val_beliefs = [build_sft_datum(row, backend, max_seq_length,
                    action_supervision_policy=policy) for row in grouped["val"]["belief"]]
    # Check every answer we may actually use before starting a paid session.
    # Later-epoch ties and noncanonical validation candidates can tokenize to
    # different lengths; checking only the initial target would miss them.
    for row in grouped["train"]["action"]:
        checked = {select_action_completion(row, seed, start_epoch, policy)}
        for epoch in range(start_epoch + 1, num_epochs):
            completion = select_action_completion(row, seed, epoch, policy)
            if completion not in checked:
                build_sft_datum(row, backend, max_seq_length, completion_override=completion,
                                action_supervision_policy=policy)
                checked.add(completion)
    # Lazy import lets the validator share the datum builder without a cycle.
    from liars_agent.training.action_validation import candidate_completion, evaluate_action_validation
    for row in grouped["val"]["action"]:
        obs = PlayerObservation.from_dict(row["observation"])
        for action in obs.legal_actions:
            build_sft_datum(row, backend, max_seq_length,
                            completion_override=candidate_completion(row, action.action_id),
                            action_supervision_policy=policy)
    backend.preflight()
    if resume:
        client = resolve_future(backend.service.create_training_client_from_state_with_optimizer(progress["resume_state_path"]))
    elif resume_from:
        client = resolve_future(backend.service.create_training_client_from_state(resume_from))
    else:
        client = resolve_future(backend.service.create_lora_training_client(base_model=base_model,
                    rank=lora_rank, seed=seed, train_mlp=train_mlp, train_attn=train_attn, train_unembed=train_unembed))
    out.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        with (out / f"{split_name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    half = batch_size // 2
    steps_per_epoch = math.ceil(max(len(rows) for rows in train_data.values()) / half)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    loss_curve = progress.get("loss_curve", [])
    best = progress.get("best", {})
    start_time = time.perf_counter()
    for epoch in range(start_epoch, num_epochs):
        if epoch != start_epoch:
            train_data["action"] = action_data_for_epoch(epoch)
        orders = {}
        for index, task in enumerate(("action", "belief")):
            orders[task] = list(range(len(train_data[task])))
            random.Random(seed + 2 * epoch + index).shuffle(orders[task])
        totals = {task: [0.0, 0.0] for task in ("action", "belief")}
        for local_step in range(steps_per_epoch):
            batch = [train_data[task][orders[task][(local_step * half + offset) % len(orders[task])]]
                     for task in ("action", "belief") for offset in range(half)]
            step = epoch * steps_per_epoch + local_step + 1
            result = backend.run_training_batch(client, batch, learning_rate=learning_rate * min(1, step / max(1, warmup_steps)))
            for index, task in enumerate(("action", "belief")):
                start, stop = index * half, (index + 1) * half
                nll, count = batch_loss(SimpleNamespace(loss_fn_outputs=result.loss_fn_outputs[start:stop]), batch[start:stop])
                totals[task][0] += nll
                totals[task][1] += count
        entry = {"epoch": epoch + 1, "step": (epoch + 1) * steps_per_epoch}
        for task in ("action", "belief"):
            entry[f"train_{task}_loss"] = totals[task][0] / totals[task][1]
        action_metrics = evaluate_action_validation(grouped["val"]["action"], backend, client,
                                                    batch_size, max_seq_length, policy)
        entry["action_validation"] = action_metrics
        entry["val_action_set_nll"] = action_metrics["action_set_nll"]
        entry["val_action_canonical_token_nll"] = action_metrics["canonical_mean_token_nll"]
        entry["val_action_loss"] = entry["val_action_set_nll"]
        nll_total = weight_total = 0.0
        for offset in range(0, len(val_beliefs), batch_size):
            batch = val_beliefs[offset:offset + batch_size]
            nll, count = batch_loss(backend.run_training_batch(client, batch, learning_rate=None), batch)
            nll_total += nll
            weight_total += count
        entry["val_belief_loss"] = nll_total / weight_total
        entry["train_loss"] = (entry["train_action_loss"] + entry["train_belief_loss"]) / 2
        entry["val_loss"] = entry["val_action_set_nll"]
        loss_curve.append(entry)
        state = resolve_future(client.save_state(name=f"{stage}-epoch-{epoch + 1}", ttl_seconds=checkpoint_ttl_seconds))
        if not best or entry["val_action_set_nll"] < best["val_loss"]:
            sampler = resolve_future(client.save_weights_for_sampler(name=f"{stage}-best-epoch-{epoch + 1}", ttl_seconds=checkpoint_ttl_seconds))
            best = {"epoch": epoch + 1, "val_loss": entry["val_action_set_nll"],
                    "metric": "val_action_set_nll",
                    "sampler_path": sampler.path, "state_path": state.path}
        _write_json(progress_path, {"contract": current_contract(), "dataset_sha256": fingerprint,
            "action_supervision_policy": policy,
            "config": config, "next_epoch": epoch + 1, "resume_state_path": state.path,
            "best": best, "loss_curve": loss_curve})
        _write_json(out / "loss_curve.json", loss_curve)
    metadata = {"contract": current_contract(), "adapter_name": f"qwen35_9b_liars_lora_{stage}",
        "base_model": base_model, "platform": "tinker", "backend": "tinker", "stage": stage,
        "action_supervision_policy": policy,
        "tasks": ["action", "belief"], "sampler_path": best["sampler_path"], "state_path": best["state_path"],
        "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
        "dataset_sha256": fingerprint, "source_state_path": resume_from,
        "lora_config": {"r": lora_rank, "train_mlp": train_mlp, "train_attn": train_attn, "train_unembed": train_unembed},
        "dataset_summary": {name: {task: len(rows) for task, rows in tasks.items()} for name, tasks in grouped.items()},
        "training_metrics": {**config, "total_steps": total_steps,
            "final_train_loss": loss_curve[-1]["train_loss"], "final_val_loss": loss_curve[-1]["val_loss"],
            "loss_curve": loss_curve},
        "cost_accounting": {**backend.ledger.snapshot(), "wall_time_sec": time.perf_counter() - start_time,
                            "rates_per_million_tokens": backend.rates},
        "checkpoint_ttl_seconds": checkpoint_ttl_seconds}
    _write_json(out / "adapter_metadata.json", metadata)
    return metadata

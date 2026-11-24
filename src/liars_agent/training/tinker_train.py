"""Balanced action imitation and binary belief learning on one Tinker LoRA."""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
import uuid
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from liars_agent.agents.prompt_builder import (SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT,
    build_user_prompt, build_bluff_user_prompt, validate_action_decision)
from liars_agent.contracts import current_contract, require_contract
from liars_agent.engine.observation import PlayerObservation
from liars_agent.training.action_supervision import (
    action_target_completions, mean_action_weight, normalize_action_policy,
    require_training_objective, resolve_action_supervision,
)
from liars_agent.training.dataset_splitter import split_dataset_by_game_id
from liars_agent.training.tinker_backend import TinkerBackend, resolve_future


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _record_trial_failure(function):
    """Keep persisted trial evidence accurate for train, save and verification errors."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        output = kwargs.get("output_dir", args[1] if len(args) > 1 else None)
        path = Path(output) / "trial_verification.json" if output is not None else None
        prior_attempt = (json.loads(path.read_text(encoding="utf-8")).get("attempt_id")
                         if kwargs.get("max_train_steps") is not None and path is not None and path.exists() else None)
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            if kwargs.get("max_train_steps") is not None and output is not None:
                if path.exists():
                    report = json.loads(path.read_text(encoding="utf-8"))
                    if report.get("attempt_id") == prior_attempt:
                        raise  # Input rejection must preserve any previous attempt's evidence.
                    report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
                    if hasattr(exc, "partial_report"):
                        report["failed_check"] = exc.partial_report
                    _write_json(path, report)
            raise
    return wrapped


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


def _tokenize_sft_datum(system: str, prompt_text: str, task: str, completion: str, sample_id: Any,
                        backend: TinkerBackend, max_seq_length: int, *, sample_weight: float = 1.0) -> Any:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt_text}]
    tokenizer = backend.tokenizer
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if task == "belief":
        labels = [tokenizer.encode(label, add_special_tokens=False) for label in ("0", "1")]
        if any(len(tokens) != 1 for tokens in labels) or labels[0] == labels[1]:
            raise ValueError("Binary belief labels 0 and 1 must be distinct single tokens.")
        completion_tokens = labels[int(completion)]
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
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if type(eos_id) is not int or eos_id not in completion_tokens:
            raise ValueError("Action chat template must contain an assistant EOS token.")
        # Generation ends at this EOS. Template separators after it are not part
        # of the generated answer and must not influence train or validation NLL.
        completion_tokens = completion_tokens[:completion_tokens.index(eos_id) + 1]
        tokens = prompt_tokens + completion_tokens
    if not completion_tokens:
        raise ValueError("SFT completion has no tokens.")
    if len(tokens) > max_seq_length:
        raise ValueError(f"Sample {sample_id} has {len(tokens)} tokens, "
                         f"exceeding max_seq_length={max_seq_length}; no JSON was truncated.")
    weights = [0.0] * len(prompt_tokens) + [sample_weight / len(completion_tokens)] * len(completion_tokens)
    return backend.sdk.Datum(model_input=backend.sdk.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]})


def build_sft_datum(sample: dict[str, Any], backend: TinkerBackend, max_seq_length: int, *,
                    completion_override: str | None = None, sample_weight: float = 1.0,
                    action_supervision_policy: dict | None = None) -> Any:
    """Weighted mean action-token NLL, or one unweighted binary belief label.

    Legal action overrides serve both all-answer supervision and validation of all
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
    return _tokenize_sft_datum(sample["system"], sample["prompt"], sample["task"], completion,
                               sample.get("sample_id", "?"), backend, max_seq_length, sample_weight=sample_weight)


def splits_sha256(splits: dict[str, list[dict[str, Any]]]) -> str:
    """Stream-hash splits identically to json.dumps(splits, sort_keys=True, ensure_ascii=False)."""
    digest = hashlib.sha256()
    digest.update(b"{")
    for split_index, (split_name, rows) in enumerate(sorted(splits.items())):
        if split_index:
            digest.update(b", ")
        digest.update(json.dumps(split_name, ensure_ascii=False).encode())
        digest.update(b": [")
        for row_index, row in enumerate(rows):
            if row_index:
                digest.update(b", ")
            digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"]")
    digest.update(b"}")
    return digest.hexdigest()


def split_paths_sha256(split_paths: dict[str, Path]) -> str:
    """Stream-hash JSONL split files identically to splits_sha256 without loading all splits into RAM."""
    digest = hashlib.sha256()
    digest.update(b"{")
    for split_index, (split_name, path) in enumerate(sorted(split_paths.items())):
        if split_index:
            digest.update(b", ")
        digest.update(json.dumps(split_name, ensure_ascii=False).encode())
        digest.update(b": [")
        with Path(path).open("r", encoding="utf-8") as stream:
            first = True
            for line in stream:
                if not line.strip():
                    continue
                if not first:
                    digest.update(b", ")
                first = False
                row = json.loads(line)
                digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"]")
    digest.update(b"}")
    return digest.hexdigest()


def _iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _slim_action_record(row: dict[str, Any], policy: dict) -> tuple[str, str, Any, list[str], float]:
    completions = action_target_completions(row, policy)
    weight = resolve_action_supervision(row, policy)["loss_weight"]
    return (row["system"], row["prompt"], row.get("sample_id", "?"), completions, weight)


def _slim_belief_record(row: dict[str, Any]) -> tuple[str, str, Any, str]:
    return (row["system"], row["prompt"], row.get("sample_id", "?"), row["completion"])


def _build_train_action_data(record: tuple[str, str, Any, list[str], float], action_weight_mean: float,
                             backend: TinkerBackend, max_seq_length: int) -> list[Any]:
    system, prompt_text, sample_id, completions, raw_weight = record
    answer_weight = raw_weight / action_weight_mean / len(completions)
    return [_tokenize_sft_datum(system, prompt_text, "action", completion, sample_id,
                                backend, max_seq_length, sample_weight=answer_weight)
            for completion in completions]


def _build_train_belief_datum(record: tuple[str, str, Any, str],
                              backend: TinkerBackend, max_seq_length: int) -> Any:
    system, prompt_text, sample_id, completion = record
    return _tokenize_sft_datum(system, prompt_text, "belief", completion, sample_id, backend, max_seq_length)


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


def select_diagnostic_rows(splits: dict[str, list[dict]], seed: int,
                           samples_per_task: int = 64) -> dict[str, list[dict]]:
    """One deterministic validation subset, shared by diagnostics and cost estimates."""
    if type(samples_per_task) is not int or samples_per_task < 1:
        raise ValueError("diagnostic_samples_per_task must be a positive integer.")
    selected = {}
    for task in ("action", "belief"):
        rows = sorted((row for row in splits.get("val", []) if row["task"] == task),
                      key=lambda row: row["sample_id"])
        task_seed = int.from_bytes(hashlib.sha256(f"{seed}:validation_diagnostic:{task}".encode()).digest(), "big")
        indices = sorted(random.Random(task_seed).sample(range(len(rows)), min(len(rows), samples_per_task)))
        selected[task] = [rows[index] for index in indices]
    return selected


def training_diagnostic_plan(train_counts: dict[str, int], num_epochs: int, batch_size: int,
                             diagnostic_steps: list[int] | tuple[int, ...] = (50, 100),
                             max_train_steps: int | None = None) -> dict:
    """Count optimizer steps by original states, never by expanded answers."""
    if (type(num_epochs) is not int or num_epochs < 1 or type(batch_size) is not int
            or batch_size < 2 or batch_size % 2
            or any(type(train_counts.get(task)) is not int or train_counts[task] < 1
                   for task in ("action", "belief"))):
        raise ValueError("Use positive task counts/epochs and an even logical batch_size.")
    if (not isinstance(diagnostic_steps, (list, tuple))
            or any(type(step) is not int or step < 0 for step in diagnostic_steps)):
        raise ValueError("diagnostic_steps must contain nonnegative integer optimizer steps.")
    if max_train_steps is not None and (type(max_train_steps) is not int or max_train_steps < 1):
        raise ValueError("max_train_steps must be a positive integer or None.")
    per_epoch = math.ceil(max(train_counts.values()) / (batch_size // 2))
    total = per_epoch * num_epochs
    planned = total if max_train_steps is None else min(total, max_train_steps)
    steps = {0, *(step for step in diagnostic_steps if step <= planned)}
    if max_train_steps is not None:
        steps.add(planned)
    return {"total_steps": total, "steps_per_epoch": per_epoch, "planned_steps": planned,
            "full_validation_epochs": planned // per_epoch,
            "diagnostic_steps": sorted(steps), "is_trial": max_train_steps is not None}


def next_logical_batch_indices(train_counts: dict[str, int], batch_size: int,
                                seed: int, absolute_step: int) -> dict[str, list[int]]:
    """Zero-based batch position in the existing epoch/shuffle/cycling schedule."""
    half = batch_size // 2
    per_epoch = math.ceil(max(train_counts.values()) / half)
    epoch, local_step = divmod(absolute_step, per_epoch)
    selected = {}
    for task_index, task in enumerate(("action", "belief")):
        order = list(range(train_counts[task]))
        random.Random(seed + 2 * epoch + task_index).shuffle(order)
        selected[task] = [order[(local_step * half + offset) % len(order)] for offset in range(half)]
    return selected


@_record_trial_failure
def run_tinker_lora_sft(
    samples: list[dict[str, Any]], output_dir: str | Path,
    base_model: str = "Qwen/Qwen3.5-9B", lora_rank: int = 16,
    learning_rate: float = 2e-4, num_epochs: int = 3, batch_size: int = 32, seed: int = 42,
    *, backend: TinkerBackend | None = None,
    splits: dict[str, list[dict[str, Any]]] | None = None,
    split_paths: dict[str, Path] | None = None,
    max_seq_length: int = 16384, warmup_ratio: float = .05,
    resume_from: str | None = None, resume_metadata: dict[str, Any] | None = None,
    resume: bool = False, stage: str = "sft", checkpoint_ttl_seconds: int | None = None,
    checkpoint_every_steps: int | None = None,
    train_attn: bool = True, train_mlp: bool = True, train_unembed: bool = False,
    action_supervision_policy: dict | None = None,
    diagnostic_steps: list[int] | tuple[int, ...] = (50, 100),
    diagnostic_samples_per_task: int = 64, max_train_steps: int | None = None,
    trial_verification_config: dict | None = None, trial_sampling_config: dict | None = None,
) -> dict[str, Any]:
    """Use equal state counts per batch; cycle the shorter task within each epoch.

    Every accepted action answer shares its state's effective weight equally.
    State weights are normalized by their training-set mean; belief retains one.
    Expanding action answers does not change optimizer steps or the LR schedule.
    Only unweighted action sequence-set validation NLL chooses the checkpoint.
    """
    if (num_epochs < 1 or batch_size < 2 or batch_size % 2 or type(lora_rank) is not int
            or lora_rank < 1 or learning_rate <= 0
            or any(type(flag) is not bool for flag in (train_attn, train_mlp, train_unembed))):
        raise ValueError("Use positive epochs/rank/rate and an even batch_size of at least 2.")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1).")
    if checkpoint_every_steps is not None and (type(checkpoint_every_steps) is not int or checkpoint_every_steps < 1):
        raise ValueError("checkpoint_every_steps must be a positive integer or None.")
    policy = normalize_action_policy(action_supervision_policy)
    lora_config = {"r": lora_rank, "train_mlp": train_mlp,
                   "train_attn": train_attn, "train_unembed": train_unembed}
    # Validate every provided row, including test rows, before constructing an API backend.
    if resume_from is not None:
        require_contract(resume_metadata or {}, "source checkpoint metadata")
        require_training_objective(resume_metadata or {}, policy)
        if resume_metadata.get("state_path") != resume_from or resume_metadata.get("base_model") != base_model:
            raise ValueError("Source checkpoint metadata does not match its state path/model.")
        source_lora = resume_metadata.get("lora_config")
        if (source_lora != lora_config
                or any(type(source_lora[key]) is not type(value) for key, value in lora_config.items())):
            raise ValueError("Source checkpoint LoRA configuration differs from the requested rank/layers.")
    train_records: dict[str, list[Any]] = {"action": [], "belief": []}
    val_grouped: dict[str, list[dict[str, Any]]] = {"action": [], "belief": []}
    all_val_rows: list[dict[str, Any]] = []
    dataset_summary: dict[str, dict[str, int]] = {}
    if split_paths is not None:
        games: dict[str, set[Any]] = {}
        action_weight_sum = 0.0
        for split_name, path in split_paths.items():
            games[split_name] = set()
            dataset_summary[split_name] = {"action": 0, "belief": 0}
            for row in _iter_jsonl(path):
                validate_training_sample(row, action_supervision_policy=policy)
                games[split_name].add(row["game_id"])
                task = row["task"]
                dataset_summary[split_name][task] += 1
                if split_name == "train":
                    if task == "action":
                        record = _slim_action_record(row, policy)
                        train_records["action"].append(record)
                        action_weight_sum += record[4]
                    else:
                        train_records["belief"].append(_slim_belief_record(row))
                elif split_name == "val":
                    val_grouped[task].append(row)
                    all_val_rows.append(row)
        if any(not dataset_summary.get(name, {}).get(task) for name in ("train", "val") for task in ("action", "belief")):
            raise ValueError("Training and validation must each contain both action and belief samples.")
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if games.get(left, set()) & games.get(right, set()):
                raise ValueError(f"Game-level leakage between {left} and {right}.")
        action_weight_mean = action_weight_sum / len(train_records["action"])
        grouped_train_for_check = None
        diagnostic_rows = select_diagnostic_rows({"val": all_val_rows}, seed, diagnostic_samples_per_task)
        fingerprint = split_paths_sha256(split_paths)
    else:
        all_rows = samples + [row for rows in (splits or {}).values() for row in rows]
        for row in all_rows:
            validate_training_sample(row, action_supervision_policy=policy)
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
        for row in grouped["train"]["action"]:
            train_records["action"].append(_slim_action_record(row, policy))
        for row in grouped["train"]["belief"]:
            train_records["belief"].append(_slim_belief_record(row))
        val_grouped = grouped["val"]
        all_val_rows = list(splits.get("val", []))
        dataset_summary = {name: {task: len(rows) for task, rows in tasks.items()} for name, tasks in grouped.items()}
        grouped_train_for_check = grouped["train"]
        diagnostic_rows = select_diagnostic_rows(splits, seed, diagnostic_samples_per_task)
        fingerprint = splits_sha256(splits)
    plan = training_diagnostic_plan({task: len(rows) for task, rows in train_records.items()},
                                   num_epochs, batch_size, diagnostic_steps, max_train_steps)
    from liars_agent.training.trial_verification import normalize_trial_verification_config
    verification_config = normalize_trial_verification_config(trial_verification_config)
    sampling_config = {"temperature": .35, "top_p": .9, "max_new_tokens": 256,
                       **(trial_sampling_config or {})}
    out = Path(output_dir)
    progress_path = out / "training_progress.json"
    if plan["is_trial"] and resume:
        raise ValueError("Step-limited trials do not resume; use a fresh trial output directory.")
    if (out / "trial_metadata.json").exists() or (plan["is_trial"] and (out / "adapter_metadata.json").exists()):
        raise ValueError("Trials and formal adapters require separate, unused output directories.")
    config = {"base_model": base_model, "lora_rank": lora_rank, "learning_rate": learning_rate,
        "batch_size": batch_size, "seed": seed, "max_seq_length": max_seq_length,
        "num_epochs": num_epochs, "warmup_ratio": warmup_ratio, "stage": stage,
        "train_attn": train_attn, "train_mlp": train_mlp, "train_unembed": train_unembed,
        "checkpoint_ttl_seconds": checkpoint_ttl_seconds,
        "diagnostic_steps": list(diagnostic_steps), "diagnostic_samples_per_task": diagnostic_samples_per_task,
        "max_train_steps": max_train_steps, "diagnostic_plan": plan,
        "trial_verification_config": verification_config if plan["is_trial"] else None,
        "trial_sampling_config": sampling_config if plan["is_trial"] else None,
        "task_balance": "equal_states_per_batch_cycle_shorter",
        "logical_batch_task_counts": {"action": batch_size // 2, "belief": batch_size // 2},
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
        saved_config = dict(progress.get("config") or {})
        saved_config.pop("checkpoint_every_steps", None)
        if saved_config != config:
            raise ValueError("Cannot resume with changed training configuration.")
    elif progress_path.exists():
        raise ValueError("Training output already has progress; use resume=True or a new output directory.")
    backend = backend or TinkerBackend(base_model=base_model)
    if backend.base_model != base_model:
        raise ValueError("Training model and backend model differ.")

    start_epoch = int(progress.get("next_epoch", 0))
    start_local_step = int(progress.get("next_local_step", 0))
    partial_state = progress.get("partial_epoch_state") if start_local_step > 0 else None
    # Keep original states as the sampling unit. All answers of a selected state
    # enter the same optimizer step and collectively retain that state's weight.
    # Pre-validate tokenization and max_seq_length across all train/val/candidate
    # sequences without retaining all Datum objects simultaneously in RAM.
    train_iter = _iter_jsonl(split_paths["train"]) if split_paths is not None else (
        row for task in ("action", "belief") for row in grouped_train_for_check[task]
    )
    for row in train_iter:
        if row["task"] == "action":
            completions = action_target_completions(row, policy)
            answer_weight = (resolve_action_supervision(row, policy)["loss_weight"]
                             / action_weight_mean / len(completions))
            for completion in completions:
                build_sft_datum(row, backend, max_seq_length,
                                completion_override=completion, sample_weight=answer_weight,
                                action_supervision_policy=policy)
        else:
            build_sft_datum(row, backend, max_seq_length, action_supervision_policy=policy)
    val_belief_records = []
    for row in val_grouped["belief"]:
        build_sft_datum(row, backend, max_seq_length, action_supervision_policy=policy)
        val_belief_records.append(_slim_belief_record(row))
    diagnostic_belief_records = []
    for row in diagnostic_rows["belief"]:
        build_sft_datum(row, backend, max_seq_length, action_supervision_policy=policy)
        diagnostic_belief_records.append(_slim_belief_record(row))
    # Every training answer is validated above. Also check all noncanonical
    # validation candidates before starting a paid session.
    # Lazy import lets the validator share the datum builder without a cycle.
    from liars_agent.training.action_validation import candidate_completion, evaluate_action_validation
    for row in val_grouped["action"]:
        obs = PlayerObservation.from_dict(row["observation"])
        for action in obs.legal_actions:
            build_sft_datum(row, backend, max_seq_length,
                            completion_override=candidate_completion(row, action.action_id),
                            action_supervision_policy=policy)
    trial_probes = probe_data = None
    if plan["is_trial"]:
        from liars_agent.training.trial_verification import select_trial_probe_rows, build_trial_probe_data
        from liars_agent.training.trial_sampling import estimate_saved_sampler_verification
        trial_probes = select_trial_probe_rows({"val": all_val_rows}, seed, verification_config["probe_per_task"])
        probe_data = build_trial_probe_data(trial_probes, backend, max_seq_length, policy)
        # Validate both label templates and output limits before creating a paid session.
        estimate_saved_sampler_verification(backend, trial_probes, max_seq_length=max_seq_length,
                                            max_new_tokens=sampling_config["max_new_tokens"])
    backend.preflight()
    if resume:
        client = resolve_future(backend.service.create_training_client_from_state_with_optimizer(progress["resume_state_path"]))
    elif resume_from:
        client = resolve_future(backend.service.create_training_client_from_state(resume_from))
    else:
        client = resolve_future(backend.service.create_lora_training_client(base_model=base_model,
                    rank=lora_rank, seed=seed, train_mlp=train_mlp, train_attn=train_attn, train_unembed=train_unembed))
    out.mkdir(parents=True, exist_ok=True)
    if split_paths is not None:
        for split_name, path in split_paths.items():
            shutil.copyfile(Path(path), out / f"{split_name}.jsonl")
    else:
        for split_name, rows in splits.items():
            with (out / f"{split_name}.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    half = batch_size // 2
    steps_per_epoch = plan["steps_per_epoch"]
    total_steps = plan["total_steps"]
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    loss_curve = progress.get("loss_curve", [])
    best = progress.get("best", {})
    diagnostics = progress.get("diagnostics", [])
    step_checkpoints = list(progress.get("step_checkpoints", []))
    subset = {"seed": seed, "source_split": "val", "samples_per_task": diagnostic_samples_per_task,
              "sample_ids": {task: [row["sample_id"] for row in rows] for task, rows in diagnostic_rows.items()}}
    completed_steps = start_epoch * steps_per_epoch + start_local_step
    logical_visits = {task: sum(entry["training_logical_state_visits"][task] for entry in loss_curve)
                      + (int(partial_state["logical_visits"][task]) if partial_state else 0)
                      for task in ("action", "belief")}
    expanded_sequences = {task: sum(entry["training_expanded_sequences"][task] for entry in loss_curve)
                          + (int(partial_state["expanded_sequences"][task]) if partial_state else 0)
                          for task in ("action", "belief")}
    start_time = time.perf_counter()
    closure = {"contract": current_contract(), "version": verification_config["version"],
        "attempt_id": uuid.uuid4().hex,
        "status": "running", "config": verification_config, "sampling_config": sampling_config,
        "dataset_sha256": fingerprint, "base_model": base_model, "lora_config": lora_config,
        "source_split": "val", "model_api_calls_are_real_when_executed": True}
    closure_path = out / "trial_verification.json"
    if plan["is_trial"]:
        from liars_agent.training.trial_verification import score_trial_probe
        closure["probe_ids"] = {task: [row["sample_id"] for row in rows] for task, rows in trial_probes.items()}
        _write_json(closure_path, closure)
        try:
            closure["before_update"] = score_trial_probe(client, backend, probe_data)
            closure["baseline_control"] = score_trial_probe(client, backend, probe_data)
            _write_json(closure_path, closure)
        except Exception as exc:
            closure.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            _write_json(closure_path, closure)
            raise

    def belief_validation(records: list[tuple[str, str, Any, str]]) -> float:
        nll_total = weight_total = 0.0
        for offset in range(0, len(records), batch_size):
            batch = [_build_train_belief_datum(record, backend, max_seq_length)
                     for record in records[offset:offset + batch_size]]
            nll, count = batch_loss(backend.run_training_batch(client, batch, learning_rate=None), batch)
            nll_total += nll
            weight_total += count
        return nll_total / weight_total

    def diagnose(step: int) -> None:
        if step not in plan["diagnostic_steps"] or any(row["step"] == step for row in diagnostics):
            return
        action = evaluate_action_validation(diagnostic_rows["action"], backend, client,
                                             batch_size, max_seq_length, policy)
        diagnostics.append({"step": step, "action_validation": action,
            "val_action_set_nll": action["action_set_nll"],
            "val_belief_loss": belief_validation(diagnostic_belief_records),
            "used_for_checkpoint_selection": False})
        _write_json(out / "training_diagnostics.json", {"subset": subset, "records": diagnostics})

    if start_epoch == 0 and start_local_step == 0:
        diagnose(0)
    if plan["is_trial"]:
        closure["phase"] = "training_and_diagnostics"
        _write_json(closure_path, closure)
    partial_epoch = None
    for epoch in range(start_epoch, num_epochs):
        orders = {}
        for index, task in enumerate(("action", "belief")):
            orders[task] = list(range(len(train_records[task])))
            random.Random(seed + 2 * epoch + index).shuffle(orders[task])
        if epoch == start_epoch and partial_state is not None:
            totals = {task: [float(partial_state["totals"][task][0]), float(partial_state["totals"][task][1])]
                      for task in ("action", "belief")}
            sequence_counts = {task: int(partial_state["sequence_counts"][task]) for task in ("action", "belief")}
            local_start = start_local_step
        else:
            totals = {task: [0.0, 0.0] for task in ("action", "belief")}
            sequence_counts = {task: 0 for task in ("action", "belief")}
            local_start = 0
        for local_step in range(local_start, steps_per_epoch):
            selected_action = [train_records["action"][orders["action"][(local_step * half + offset) % len(orders["action"])]]
                               for offset in range(half)]
            selected_belief = [train_records["belief"][orders["belief"][(local_step * half + offset) % len(orders["belief"])]]
                               for offset in range(half)]
            task_batches = {
                "action": [datum for record in selected_action
                           for datum in _build_train_action_data(record, action_weight_mean, backend, max_seq_length)],
                "belief": [_build_train_belief_datum(record, backend, max_seq_length)
                           for record in selected_belief],
            }
            batch = task_batches["action"] + task_batches["belief"]
            step = epoch * steps_per_epoch + local_step + 1
            result = backend.run_training_batch(client, batch, learning_rate=learning_rate * min(1, step / max(1, warmup_steps)))
            if len(result.loss_fn_outputs) != len(batch):
                raise RuntimeError("Tinker loss output count differs from expanded batch size.")
            start = 0
            for task in ("action", "belief"):
                stop = start + len(task_batches[task])
                nll, count = batch_loss(SimpleNamespace(loss_fn_outputs=result.loss_fn_outputs[start:stop]), task_batches[task])
                totals[task][0] += nll
                totals[task][1] += count
                sequence_counts[task] += len(task_batches[task])
                logical_visits[task] += half
                expanded_sequences[task] += len(task_batches[task])
                start = stop
            completed_steps = step
            diagnose(step)
            if (not plan["is_trial"] and checkpoint_every_steps is not None
                    and step % checkpoint_every_steps == 0 and local_step + 1 < steps_per_epoch):
                step_state = resolve_future(client.save_state(
                    name=f"{stage}-step-{step}", ttl_seconds=checkpoint_ttl_seconds))
                step_checkpoints.append({"epoch": epoch + 1, "step": step,
                    "local_step": local_step + 1, "state_path": step_state.path})
                _write_json(progress_path, {"contract": current_contract(), "dataset_sha256": fingerprint,
                    "action_supervision_policy": policy, "config": config,
                    "next_epoch": epoch, "next_local_step": local_step + 1,
                    "resume_state_path": step_state.path,
                    "partial_epoch_state": {
                        "totals": totals,
                        "sequence_counts": sequence_counts,
                        "logical_visits": {task: (local_step + 1) * half for task in ("action", "belief")},
                        "expanded_sequences": dict(sequence_counts),
                    },
                    "step_checkpoints": step_checkpoints,
                    "best": best, "loss_curve": loss_curve, "diagnostics": diagnostics})
            if step == plan["planned_steps"]:
                break
        if local_step + 1 < steps_per_epoch:
            partial_epoch = {"epoch": epoch + 1, "completed_steps_in_epoch": local_step + 1,
                **{f"train_{task}_loss": totals[task][0] / totals[task][1] for task in ("action", "belief")}}
            break
        entry = {"epoch": epoch + 1, "step": (epoch + 1) * steps_per_epoch,
            "training_logical_state_visits": {task: steps_per_epoch * half for task in ("action", "belief")},
            "training_expanded_sequences": sequence_counts,
            "training_optimizer_steps": steps_per_epoch}
        for task in ("action", "belief"):
            entry[f"train_{task}_loss"] = totals[task][0] / totals[task][1]
        action_metrics = evaluate_action_validation(val_grouped["action"], backend, client,
                                                    batch_size, max_seq_length, policy)
        entry["action_validation"] = action_metrics
        entry["val_action_set_nll"] = action_metrics["action_set_nll"]
        entry["val_action_canonical_token_nll"] = action_metrics["canonical_mean_token_nll"]
        entry["val_action_loss"] = entry["val_action_set_nll"]
        entry["val_belief_loss"] = belief_validation(val_belief_records)
        entry["train_loss"] = (entry["train_action_loss"] + entry["train_belief_loss"]) / 2
        entry["val_loss"] = entry["val_action_set_nll"]
        loss_curve.append(entry)
        if plan["is_trial"]:
            closure["phase"] = "save_epoch_state"
            _write_json(closure_path, closure)
        state = resolve_future(client.save_state(name=f"{stage}-epoch-{epoch + 1}", ttl_seconds=checkpoint_ttl_seconds))
        if plan["is_trial"]:
            closure.setdefault("epoch_checkpoints", []).append({"epoch": epoch + 1, "state_path": state.path})
            _write_json(closure_path, closure)
        if not best or entry["val_action_set_nll"] < best["val_loss"]:
            if plan["is_trial"]:
                closure["phase"] = "save_epoch_sampler"
                _write_json(closure_path, closure)
            sampler = resolve_future(client.save_weights_for_sampler(name=f"{stage}-best-epoch-{epoch + 1}", ttl_seconds=checkpoint_ttl_seconds))
            if plan["is_trial"]:
                closure["epoch_checkpoints"][-1]["sampler_path"] = sampler.path
                _write_json(closure_path, closure)
            best = {"epoch": epoch + 1, "val_loss": entry["val_action_set_nll"],
                    "metric": "val_action_set_nll",
                    "sampler_path": sampler.path, "state_path": state.path}
        if not plan["is_trial"]:
            progress_payload = {"contract": current_contract(), "dataset_sha256": fingerprint,
                "action_supervision_policy": policy,
                "config": config, "next_epoch": epoch + 1, "resume_state_path": state.path,
                "best": best, "loss_curve": loss_curve, "diagnostics": diagnostics}
            if step_checkpoints:
                progress_payload["step_checkpoints"] = step_checkpoints
            _write_json(progress_path, progress_payload)
        _write_json(out / "loss_curve.json", loss_curve)
        if completed_steps == plan["planned_steps"]:
            break
    trial_paths = None
    if plan["is_trial"]:
        closure.update(phase="save_state", completed_training_steps=completed_steps, checkpoints={})
        _write_json(closure_path, closure)
        state = resolve_future(client.save_state(name=f"{stage}-trial-step-{completed_steps}", ttl_seconds=checkpoint_ttl_seconds))
        closure["checkpoints"]["state_path"] = state.path
        closure["phase"] = "save_sampler"
        _write_json(closure_path, closure)
        sampler = resolve_future(client.save_weights_for_sampler(name=f"{stage}-trial-step-{completed_steps}", ttl_seconds=checkpoint_ttl_seconds))
        trial_paths = {"state_path": state.path, "sampler_path": sampler.path}
        closure.update(checkpoints=trial_paths, phase="verify_saved_sampler")
        _write_json(closure_path, closure)
        try:
            from liars_agent.training.trial_verification import verify_trial_reload_and_optimizer
            from liars_agent.training.trial_sampling import verify_saved_sampler
            closure["saved_snapshot"] = score_trial_probe(client, backend, probe_data)
            # Sampling must be compared with the saved state before the three
            # diagnostic continuation branches mutate their in-memory clients.
            closure["sampling"] = verify_saved_sampler(backend, client, sampler.path, trial_probes,
                max_seq_length=max_seq_length, seed=seed,
                temperature=sampling_config["temperature"], top_p=sampling_config["top_p"],
                max_new_tokens=sampling_config["max_new_tokens"],
                logprob_atol=verification_config["logprob_atol"], logprob_rtol=verification_config["logprob_rtol"])
            _write_json(closure_path, closure)
            if closure["sampling"]["status"] != "passed":
                raise RuntimeError("Saved sampler action/belief verification did not pass; see trial_verification.json.")
            indices = next_logical_batch_indices({task: len(rows) for task, rows in train_records.items()},
                                                  batch_size, seed, completed_steps)
            verification_batch = [datum for index in indices["action"]
                                  for datum in _build_train_action_data(train_records["action"][index], action_weight_mean, backend, max_seq_length)]
            verification_batch += [_build_train_belief_datum(train_records["belief"][index], backend, max_seq_length)
                                   for index in indices["belief"]]
            next_learning_rate = learning_rate * min(1, (completed_steps + 1) / max(1, warmup_steps))
            closure["phase"] = "verify_state_and_optimizer"
            _write_json(closure_path, closure)
            closure["training"] = verify_trial_reload_and_optimizer(
                backend, client, state.path, probe_data, closure["before_update"],
                closure["baseline_control"], closure["saved_snapshot"], verification_batch,
                next_learning_rate, config=verification_config)
            closure["status"] = "passed" if closure["training"]["status"] == "passed" else "not_passed"
            closure["phase"] = "complete"
            _write_json(closure_path, closure)
            if closure["status"] != "passed":
                raise RuntimeError("Trial update/state/optimizer verification did not pass; see trial_verification.json.")
        except Exception as exc:
            closure.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            if hasattr(exc, "partial_report"):
                closure["failed_check"] = exc.partial_report
            _write_json(closure_path, closure)
            raise
    metadata = {"contract": current_contract(), "adapter_name": f"qwen35_9b_liars_lora_{stage}",
        "base_model": base_model, "platform": "tinker", "backend": "tinker", "stage": stage,
        "action_supervision_policy": policy,
        "training_status": "trial_complete" if plan["is_trial"] else "complete",
        "max_train_steps": max_train_steps,
        "formal_training_complete": not plan["is_trial"],
        "tasks": ["action", "belief"],
        **(trial_paths or {"sampler_path": best["sampler_path"], "state_path": best["state_path"]}),
        "best_epoch": None if plan["is_trial"] else best.get("epoch"),
        "best_val_loss": None if plan["is_trial"] else best.get("val_loss"),
        "full_validation_best": best or None,
        "checkpoint_role": "trial_final_step" if plan["is_trial"] else "full_validation_best",
        "early_diagnostics": {"subset": subset, "records": diagnostics},
        "trial_verification": closure if plan["is_trial"] else None,
        "trial_verification_optimizer_steps": (closure["training"]["plan"]["diagnostic_optimizer_steps"]
                                              if plan["is_trial"] else 0),
        "dataset_sha256": fingerprint, "source_state_path": resume_from,
        "lora_config": lora_config,
        "dataset_summary": dataset_summary,
        "training_metrics": {**config, "total_steps": total_steps, "completed_steps": completed_steps,
            "training_optimizer_steps": completed_steps,
            "training_optimizer_steps_per_epoch": steps_per_epoch,
            "training_logical_state_visits": logical_visits, "training_expanded_sequences": expanded_sequences,
            "full_validation_epochs": len(loss_curve), "partial_epoch_metrics": partial_epoch,
            "final_train_loss": ((partial_epoch["train_action_loss"] + partial_epoch["train_belief_loss"]) / 2
                if partial_epoch else loss_curve[-1]["train_loss"]),
            "final_val_loss": loss_curve[-1]["val_loss"] if loss_curve else None,
            "loss_curve": loss_curve},
        "cost_accounting": {**backend.ledger.snapshot(), "wall_time_sec": time.perf_counter() - start_time,
                            "rates_per_million_tokens": backend.rates},
        "checkpoint_ttl_seconds": checkpoint_ttl_seconds}
    _write_json(out / ("trial_metadata.json" if plan["is_trial"] else "adapter_metadata.json"), metadata)
    return metadata

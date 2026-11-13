"""Bounded, local-planned checks for a live trial and its saved training state.

This module never saves weights or selects a formal checkpoint. Its three extra
optimizer updates affect only the live diagnostic branches, after the trial
checkpoint has been saved. Finite probes diagnose restoration; they do not
prove equality of every parameter or improvement in playing strength.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from liars_agent.training.tinker_backend import resolve_future

TRIAL_VERIFICATION_VERSION = "trial_training_sampling_closure_v2"
DEFAULT_TRIAL_VERIFICATION_CONFIG = {
    "version": TRIAL_VERIFICATION_VERSION,
    "probe_per_task": 2,
    "logprob_atol": 1e-4,
    "logprob_rtol": 1e-4,
    "min_update_delta": 1e-6,
    "jitter_multiplier": 3.0,
}


class TrialVerificationError(RuntimeError):
    """An unsuccessful check or service error, with JSON-serializable evidence."""

    def __init__(self, message: str, partial_report: dict):
        super().__init__(message)
        self.partial_report = _json_evidence(partial_report)


def _json_evidence(value: Any) -> Any:
    """Keep even invalid numeric responses persistable as failure evidence."""
    if isinstance(value, dict):
        return {key: _json_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_evidence(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return copy.deepcopy(value)


def normalize_trial_verification_config(config: dict | None = None) -> dict:
    if config is not None and (not isinstance(config, dict)
                              or set(config) - DEFAULT_TRIAL_VERIFICATION_CONFIG.keys()):
        raise ValueError("Unknown trial verification configuration fields.")
    result = {**DEFAULT_TRIAL_VERIFICATION_CONFIG, **(config or {})}
    if result["version"] != TRIAL_VERIFICATION_VERSION:
        raise ValueError("Incompatible trial verification version.")
    if type(result["probe_per_task"]) is not int or result["probe_per_task"] < 1:
        raise ValueError("probe_per_task must be a positive integer.")
    for key in ("logprob_atol", "logprob_rtol", "min_update_delta", "jitter_multiplier"):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
        result[key] = float(value)
    if result["min_update_delta"] <= 0 or result["jitter_multiplier"] < 1:
        raise ValueError("Use a positive update floor and a jitter multiplier of at least one.")
    return result


def select_trial_probe_rows(splits: dict[str, list[dict]], seed: int,
                           samples_per_task: int = 2) -> dict[str, list[dict]]:
    """Select fixed validation rows; training, test and endgame sets are unused."""
    from liars_agent.training.tinker_train import select_diagnostic_rows

    rows = select_diagnostic_rows(splits, seed, samples_per_task)
    if any(not rows[task] for task in ("action", "belief")):
        raise ValueError("Trial probes require both tasks in the validation split.")
    return rows


class _ProbeData(list):
    """Task sidecar for fixed probes; SDK Datum fields remain unchanged."""

    def __init__(self, data: list[Any], tasks: list[str]):
        super().__init__(data)
        self.tasks = tasks


def build_trial_probe_data(rows: dict[str, list[dict]], backend: Any,
                           max_seq_length: int, policy: dict | None = None) -> list[Any]:
    """Canonical complete action answers and one true binary label per belief."""
    from liars_agent.training.action_supervision import normalize_action_policy
    from liars_agent.training.tinker_train import build_sft_datum

    policy = normalize_action_policy(policy)
    if any(not rows.get(task) for task in ("action", "belief")):
        raise ValueError("Trial probes require both validation tasks.")
    if any(row["task"] != task for task, items in rows.items() for row in items):
        raise ValueError("Trial probe rows do not match their task group.")
    tasks = [task for task in ("action", "belief") for _ in rows[task]]
    return _ProbeData([build_sft_datum(row, backend, max_seq_length, action_supervision_policy=policy)
                      for task in ("action", "belief") for row in rows[task]], tasks)


def _list(value: Any) -> list:
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _probe_layout(data: list[Any]) -> dict:
    if not data:
        raise ValueError("Trial probe data must not be empty.")
    tasks = getattr(data, "tasks", ["unspecified"] * len(data))
    if len(tasks) != len(data):
        raise ValueError("Trial probe task sidecar length differs from its data.")
    encoded, sequences = [], []
    for task, datum in zip(tasks, data):
        tokens = datum.model_input.to_ints()
        targets, weights = (_list(datum.loss_fn_inputs[key]) for key in ("target_tokens", "weights"))
        if len(tokens) != len(targets) or len(tokens) != len(weights):
            raise ValueError("Probe input, target and mask lengths differ.")
        if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Probe masks must be finite and nonnegative.")
        positions = [index for index, weight in enumerate(weights) if weight > 0]
        if not positions:
            raise ValueError("A trial probe has no supervised tokens.")
        sequences.append({"task": task, "input_tokens": len(tokens), "positions": positions,
                          "target_tokens": [targets[index] for index in positions]})
        encoded.append({"task": task, "input_tokens": tokens, "target_tokens": targets, "positions": positions})
    return {"probe_sha256": hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest(),
            "num_sequences": len(data), "input_tokens": sum(item["input_tokens"] for item in sequences),
            "num_supervised_tokens": sum(len(item["positions"]) for item in sequences), "sequences": sequences}


def _snapshot(result: Any, data: list[Any]) -> dict:
    snapshot = _probe_layout(data)
    outputs = getattr(result, "loss_fn_outputs", None)
    if outputs is None or len(outputs) != len(data):
        raise ValueError("Trial forward output count differs from its input sequences.")
    for sequence, output in zip(snapshot["sequences"], outputs):
        scores = _list(output["logprobs"])
        if len(scores) != sequence["input_tokens"]:
            raise ValueError("Trial forward logprobs do not align with input tokens.")
        supervised = [scores[index] for index in sequence["positions"]]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x > 0 for x in supervised):
            raise ValueError("Trial forward requires finite nonpositive supervised-token logprobs.")
        sequence["logprobs"] = [float(x) for x in supervised]
    return snapshot


def score_trial_probe(client: Any, backend: Any, data: list[Any]) -> dict:
    """One forward call, with failed/invalid responses surfaced, never retried here."""
    report = {"version": TRIAL_VERIFICATION_VERSION, "status": "running", "stage": "probe_forward"}
    try:
        report["probe_layout"] = _probe_layout(data)
        return _snapshot(backend.run_training_batch(client, data, learning_rate=None), data)
    except Exception as exc:
        report.update(status="not_passed", error={"type": type(exc).__name__, "message": str(exc)})
        raise TrialVerificationError("Trial probe scoring failed.", report) from exc


def trial_verification_plan(probe_data: list[Any], next_batch: list[Any]) -> dict:
    """Includes before/control/saved probes provided by the caller and this helper.

    Sampler verification is planned separately. Checkpoint save/load have no
    token count here; persistent storage is not included in token cost estimates.
    """
    probe_tokens = _probe_layout(probe_data)["input_tokens"]
    batch_tokens = _probe_layout(next_batch)["input_tokens"]
    return {"version": TRIAL_VERIFICATION_VERSION,
        "probe_forward_calls": 7, "probe_forward_tokens": 7 * probe_tokens,
        "diagnostic_training_batches": 3, "diagnostic_training_tokens": 3 * batch_tokens,
        "diagnostic_optimizer_steps": 3, "state_reload_calls": 2,
        "total_processed_tokens": 7 * probe_tokens + 3 * batch_tokens,
        "sdk_calls": {"forward": 7, "forward_backward": 3, "optim_step": 3,
                      "create_training_client_from_state_with_optimizer": 1,
                      "create_training_client_from_state": 1},
        "scope": "Three caller probes, one reloaded probe, three branch probes and updates; includes weight-only negative control; excludes sampler checks and checkpoint storage."}


def _values(snapshot: dict, layout: dict) -> list[float]:
    if not isinstance(snapshot, dict) or set(snapshot) != set(layout):
        raise ValueError("Missing or malformed trial snapshot.")
    for key in ("probe_sha256", "num_sequences", "input_tokens", "num_supervised_tokens"):
        if snapshot[key] != layout[key]:
            raise ValueError("Trial snapshots refer to different fixed probes.")
    if len(snapshot["sequences"]) != layout["num_sequences"]:
        raise ValueError("Trial snapshot sequence count is inconsistent.")
    values = []
    for actual, expected in zip(snapshot["sequences"], layout["sequences"]):
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("Trial snapshot target IDs or supervised positions differ.")
        scores = actual.get("logprobs")
        if not isinstance(scores, list) or len(scores) != len(expected["positions"]):
            raise ValueError("Trial snapshot has missing supervised-token scores.")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x > 0 for x in scores):
            raise ValueError("Trial snapshot has invalid supervised-token scores.")
        values.extend(scores)
    return values


def _difference(left: list[float], right: list[float], config: dict) -> dict:
    deltas = [abs(a - b) for a, b in zip(left, right)]
    return {"max_abs_delta": max(deltas), "mean_abs_delta": sum(deltas) / len(deltas),
        "matched": all(delta <= config["logprob_atol"] + config["logprob_rtol"] * abs(a)
                       for delta, a in zip(deltas, left))}


def _observable_change(before: list[float], control: list[float], saved: list[float], config: dict) -> dict:
    jitter = _difference(before, control, config)
    delta = _difference(before, saved, config)
    threshold = max(config["min_update_delta"], config["jitter_multiplier"] * jitter["max_abs_delta"])
    return {"passed": delta["max_abs_delta"] > threshold and not delta["matched"],
            "baseline_jitter": jitter, "update_delta": delta,
            "required_max_abs_delta_exclusive": threshold,
            "exceeds_reload_tolerance": not delta["matched"]}


def verify_trial_reload_and_optimizer(backend: Any, original_client: Any, state_path: str,
        probe_data: list[Any], before_update_snapshot: dict, baseline_control_snapshot: dict,
        saved_snapshot: dict, next_batch: list[Any], next_learning_rate: float,
        config: dict | None = None) -> dict:
    """Check update, saved weights and optimizer restoration with fixed probes.

    The caller must capture before/control on the same unchanged client, then
    train and capture saved_snapshot immediately before saving state_path. Any
    sampler check must finish before this function's three diagnostic updates.
    The original branch remains the live client; it is never reinitialized.
    """
    config = normalize_trial_verification_config(config)
    report = {"version": TRIAL_VERIFICATION_VERSION, "config": config, "status": "running",
        "state_path": state_path, "next_learning_rate": next_learning_rate,
        "checks": {}, "snapshots": {"before": before_update_snapshot, "baseline_control": baseline_control_snapshot,
                                    "saved": saved_snapshot}, "operations": [],
        "diagnostic_optimizer_steps": {"original": 0, "restored": 0, "weight_only": 0},
        "interpretation": "Finite validation probes must distinguish updated weights and restored optimizer state from unupdated/weight-only controls; this is not loss improvement, parameter equality proof or playing strength."}

    def operation(name, function):
        event = {"name": name, "status": "started"}
        report["operations"].append(event)
        try:
            value = function()
            event["status"] = "succeeded"
            return value
        except Exception:
            event["status"] = "failed"
            raise

    try:
        layout = _probe_layout(probe_data)
        report["plan"] = trial_verification_plan(probe_data, next_batch)
        if not isinstance(state_path, str) or not state_path.startswith("tinker://") or "/weights/" not in state_path:
            raise ValueError("Trial state verification requires a Tinker training-state path.")
        if type(next_learning_rate) not in (int, float) or not math.isfinite(next_learning_rate) or next_learning_rate <= 0:
            raise ValueError("The diagnostic next learning rate must be finite and positive.")
        before, control, saved = (_values(s, layout) for s in
                                  (before_update_snapshot, baseline_control_snapshot, saved_snapshot))
        if {item["task"] for item in layout["sequences"]} != {"action", "belief"}:
            raise ValueError("Trial verification requires explicit action and belief probe task sidecars.")
        belief_indices, offset = [], 0
        for item in layout["sequences"]:
            end = offset + len(item["positions"])
            if item["task"] == "belief":
                belief_indices.extend(range(offset, end))
            offset = end
        observable = _observable_change(before, control, saved, config)
        belief_change = _observable_change(*[[values[i] for i in belief_indices]
                                             for values in (before, control, saved)], config)
        observable["belief"] = belief_change
        observable["passed"] = observable["passed"] and belief_change["passed"]
        observable["criterion"] = "All probes and belief probes each exceed the noise/floor threshold and reload matching tolerance."
        report["checks"]["observable_update"] = observable
        if not observable["passed"]:
            raise ValueError("This trial did not verify an observable update distinguishable from baseline on all/belief probes above noise and reload tolerance.")
        restored = operation("load_state_with_optimizer", lambda: resolve_future(
            backend.service.create_training_client_from_state_with_optimizer(state_path)))
        loaded_snapshot = operation("reloaded_probe", lambda: score_trial_probe(restored, backend, probe_data))
        report["snapshots"]["reloaded"] = loaded_snapshot
        loaded = _values(loaded_snapshot, layout)
        comparison = _difference(saved, loaded, config)
        report["checks"]["state_reload"] = {"passed": comparison["matched"], **comparison}
        if not comparison["matched"]:
            raise ValueError("Reloaded trial weights do not match the saved probe scores.")
        def update_and_probe(name, client):
            def diagnostic_update():
                result = backend.run_training_batch(client, next_batch, learning_rate=next_learning_rate)
                # A returned update has happened even if its score response is invalid.
                report["diagnostic_optimizer_steps"][name] = 1
                return _snapshot(result, next_batch)

            operation(f"{name}_diagnostic_update", diagnostic_update)
            report["snapshots"][f"{name}_next"] = operation(f"{name}_next_probe",
                lambda client=client: score_trial_probe(client, backend, probe_data))

        for name, client in (("original", original_client), ("restored", restored)):
            update_and_probe(name, client)
        original_next = _values(report["snapshots"]["original_next"], layout)
        restored_next = _values(report["snapshots"]["restored_next"], layout)
        comparison = _difference(original_next, restored_next, config)
        report["checks"]["optimizer_restore"] = {"passed": comparison["matched"], **comparison,
            "same_next_batch": True, "same_next_learning_rate": True,
            "live_branch": "original client retained across the trial"}
        if not comparison["matched"]:
            raise ValueError("Optimizer-restored and live clients disagree after the same next update.")
        weight_only = operation("load_state_without_optimizer", lambda: resolve_future(
            backend.service.create_training_client_from_state(state_path)))
        update_and_probe("weight_only", weight_only)
        negative_values = _values(report["snapshots"]["weight_only_next"], layout)
        negative = _difference(original_next, negative_values, config)
        noise_floor = observable["required_max_abs_delta_exclusive"]
        distinguishable = not negative["matched"] and negative["max_abs_delta"] > noise_floor
        report["checks"]["optimizer_negative_control"] = {
            "passed": distinguishable, **negative, "required_max_abs_delta_exclusive": noise_floor,
            "same_saved_state": True, "same_next_batch": True, "same_next_learning_rate": True,
            "criterion": "Live continuation differs from weight-only restoration beyond both matching tolerance and measured baseline noise floor."}
        if not distinguishable:
            raise ValueError("This trial did not distinguish optimizer restoration from the weight-only negative control.")
        report["status"] = "passed"
        return report
    except Exception as exc:
        report.update(status="not_passed", error={"type": type(exc).__name__, "message": str(exc)})
        if isinstance(exc, TrialVerificationError):
            report["failed_operation_evidence"] = exc.partial_report
        raise TrialVerificationError(str(exc), report) from exc

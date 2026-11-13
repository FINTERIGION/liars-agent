"""Verify a saved trial sampler against its still-unmodified training client.

These are protocol and round-trip checks on fixed validation observations, not
policy-quality measurements. No rule fallback or test-set selection is used.
"""
from __future__ import annotations

import math
from typing import Any

from liars_agent.agents.prompt_builder import validate_action_decision
from liars_agent.training.tinker_backend import ModelResponseError


class TrialSamplingServiceError(RuntimeError):
    """A remote/budget failure, with completed checks available for persistence."""

    def __init__(self, message: str, partial_report: dict):
        super().__init__(message)
        self.partial_report = partial_report


def _messages(row: dict) -> list[dict[str, str]]:
    return [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["prompt"]}]


def _prepare(backend, probes_by_task: dict, max_seq_length: int, max_new_tokens: int):
    # Lazy import avoids a cycle when the trainer invokes this verification.
    from liars_agent.training.tinker_train import build_sft_datum, validate_training_sample
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("Trial sampling max_new_tokens must be a positive integer.")
    actions, beliefs, data = [], [], []
    for task in ("action", "belief"):
        rows = probes_by_task.get(task, [])
        if not rows:
            raise ValueError("Sampler verification requires validation probes for both tasks.")
        for row in rows:
            if row.get("task") != task:
                raise ValueError("Trial probe task does not match its group.")
            obs = validate_training_sample(row)
            prefix = backend.encode_messages(_messages(row), generation=True)
            if task == "action":
                if len(prefix) + max_new_tokens > max_seq_length:
                    raise ValueError("Trial action prompt plus generation allowance exceeds max_seq_length.")
                actions.append((row, obs, len(prefix)))
            else:
                # Counterfactual label copies score both vocabulary tokens; the
                # original observed label and dataset row remain unchanged.
                for label in (0, 1):
                    datum = build_sft_datum({**row, "completion": str(label), "label": label},
                                            backend, max_seq_length)
                    if datum.model_input.to_ints() != prefix:
                        raise ValueError("Training and sampler belief prompt tokens differ.")
                    data.append(datum)
                beliefs.append((row, obs, len(prefix)))
    return actions, beliefs, data


def estimate_saved_sampler_verification(
    backend, probes_by_task: dict, *, max_seq_length: int, max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Offline count of every extra forward and uncached sampling request."""
    actions, beliefs, data = _prepare(backend, probes_by_task, max_seq_length, max_new_tokens)
    forward = sum(len(datum.model_input.to_ints()) for datum in data)
    action_prefill = sum(length for _, _, length in actions)
    belief_prefill = sum(2 * (length + 1) for _, _, length in beliefs)
    action_generation = len(actions) * max_new_tokens
    hidden_generation = 2 * len(beliefs)
    forward_cost = backend.estimate_training_cost(forward)
    sample_cost = ((action_prefill + belief_prefill) * backend.rates["prefill"]
                   + (action_generation + hidden_generation) * backend.rates["sample"]) / 1_000_000
    return {"training_forward_sequences": len(data), "training_forward_tokens": forward,
            "action_sample_requests": len(actions), "belief_label_score_requests": 2 * len(beliefs),
            "sampling_prefill_tokens": action_prefill + belief_prefill,
            "action_generated_tokens_upper_bound": action_generation,
            "belief_generated_tokens_upper_bound": hidden_generation,
            "sampling_generated_tokens_upper_bound": action_generation + hidden_generation,
            "training_forward_usd": forward_cost, "sampling_usd_upper_bound": sample_cost,
            "total_usd_upper_bound": forward_cost + sample_cost,
            "basis": "two training label forwards per belief probe, one native action per action probe, "
                     "two full-prefix single-label sampler scores per belief probe; no cache discount"}


def _label_logprobs(result, data: list) -> list[float]:
    outputs = getattr(result, "loss_fn_outputs", None)
    if outputs is None or len(outputs) != len(data):
        raise ValueError("Training belief forward output count differs from label candidates.")
    values = []
    for output, datum in zip(outputs, data):
        if "logprobs" not in output:
            raise ValueError("Training belief forward omitted per-token logprobs.")
        raw = output["logprobs"]
        logprobs = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        weights = datum.loss_fn_inputs["weights"]
        weights = weights.tolist() if hasattr(weights, "tolist") else list(weights)
        if len(logprobs) != len(weights) or [i for i, weight in enumerate(weights) if weight > 0] != [len(weights) - 1]:
            raise ValueError("Training belief logprobs do not match the single-label mask.")
        value = logprobs[-1]
        if type(value) not in (int, float) or not math.isfinite(value) or value > 0:
            raise ValueError("Training belief label logprob must be finite and nonpositive.")
        values.append(float(value))
    if any(sum(math.exp(value) for value in values[index:index + 2]) > 1 + 1e-6
           for index in range(0, len(values), 2)):
        raise ValueError("Distinct training belief label probabilities sum to more than one.")
    return values


def verify_saved_sampler(
    backend, training_client, sampler_path: str, probes_by_task: dict, *,
    max_seq_length: int, temperature: float = .2, top_p: float = .9,
    max_new_tokens: int = 256, seed: int = 42,
    logprob_atol: float = 1e-4, logprob_rtol: float = 1e-4,
) -> dict[str, Any]:
    """Reload and check a saved sampler before any further diagnostic updates.

    Protocol failures return not_passed. Service/budget failures raise
    TrialSamplingServiceError with partial_report and retain the original cause.
    Tolerances are explicit and are never automatically relaxed.
    """
    for tolerance in (logprob_atol, logprob_rtol):
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Sampler logprob tolerances must be finite and nonnegative.")
    if not isinstance(sampler_path, str) or not sampler_path.startswith("tinker://") or "/sampler_weights/" not in sampler_path:
        raise ValueError("Trial verification requires the saved sampler checkpoint path.")
    actions, beliefs, data = _prepare(backend, probes_by_task, max_seq_length, max_new_tokens)
    before = backend.ledger.snapshot()
    report = {"status": "not_passed", "sampler_path": sampler_path, "probe_source": "val",
              "probe_ids": {task: [row["sample_id"] for row in probes_by_task[task]] for task in ("action", "belief")},
              "tolerances": {"logprob_atol": logprob_atol, "logprob_rtol": logprob_rtol},
              "sampling": {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens, "seed": seed},
              "fresh_sampler_created": False, "action_checks": [], "belief_checks": [], "errors": [],
              "interpretation": "Native JSON legality and individual 0/1 token logprob agreement only; "
                                "no fallback, action-quality accuracy or test-set model selection."}

    def finish():
        after = backend.ledger.snapshot()
        report["usage"] = {key: after[key] - before[key] for key in (
            "estimated_cost_usd", "reserved_unknown_usd", "pending_requests", "training_tokens",
            "prompt_tokens", "completion_tokens", "completion_tokens_upper_bound")}
        return report

    def call(stage, function):
        try:
            return function()
        except ModelResponseError:
            raise
        except Exception as exc:
            report["errors"].append({"stage": stage, "kind": "service_or_budget_failure",
                                     "type": type(exc).__name__, "message": str(exc)})
            raise TrialSamplingServiceError(f"Trial sampler verification failed during {stage}: {exc}", finish()) from exc

    call("fresh_sampler_load", lambda: backend.sampling_client(sampler_path, fresh=True))
    report["fresh_sampler_created"] = True
    result = call("training_label_forward", lambda: backend.run_training_batch(training_client, data, learning_rate=None))
    try:
        expected = _label_logprobs(result, data)
    except (ValueError, TypeError, KeyError) as exc:
        report["errors"].append({"stage": "training_label_forward", "kind": "invalid_response", "message": str(exc)})
        return finish()
    for index, (row, obs, _) in enumerate(actions):
        entry = {"sample_id": row["sample_id"], "passed": False,
                 "forced_challenge": len(obs.legal_actions) == 1 and obs.legal_actions[0].action_type == "CHALLENGE",
                 "belief_requested": False}
        try:
            result = call("native_action_sample", lambda: backend.sample(
                _messages(row), checkpoint=sampler_path, num_samples=1, temperature=temperature,
                top_p=top_p, max_tokens=max_new_tokens, seed=seed + index))[0]
            entry.update(raw_output=result.text, stop_reason=result.stop_reason)
            if result.stop_reason != "stop":
                raise ValueError("Native action generation did not terminate normally.")
            parsed = validate_action_decision(obs, result.text)
            entry.update(passed=True, chosen_action_id=parsed["chosen_action_id"])
        except (ValueError, ModelResponseError) as exc:
            entry["error"] = str(exc)
        report["action_checks"].append(entry)
    for index, (row, obs, _) in enumerate(beliefs):
        training_logprobs = expected[2 * index:2 * index + 2]
        entry = {"sample_id": row["sample_id"], "passed": False,
                 "training_logprobs": dict(zip(("0", "1"), training_logprobs))}
        try:
            score = call("sampler_belief_score", lambda: backend.score_bluff(_messages(row), checkpoint=sampler_path))
            values = (score.logprob_zero, score.logprob_one)
            if any(type(value) not in (int, float) or not math.isfinite(value) or value > 0 for value in values):
                raise ValueError("Sampler belief label logprobs must be finite and nonpositive.")
            if sum(math.exp(value) for value in values) > 1 + 1e-6:
                raise ValueError("Distinct sampler belief label probabilities sum to more than one.")
            entry.update(sampler_logprobs=dict(zip(("0", "1"), values)),
                         sampler_probability=score.probability,
                         absolute_errors={label: abs(actual - wanted)
                                          for label, actual, wanted in zip(("0", "1"), values, training_logprobs)})
            entry["passed"] = all(math.isclose(actual, wanted, abs_tol=logprob_atol, rel_tol=logprob_rtol)
                                   for actual, wanted in zip(values, training_logprobs))
            if not entry["passed"]:
                entry["error"] = "Sampler and training raw label logprobs differ beyond configured tolerance."
        except (ValueError, ModelResponseError) as exc:
            entry["error"] = str(exc)
        report["belief_checks"].append(entry)
    if all(entry["passed"] for entry in report["action_checks"] + report["belief_checks"]):
        report["status"] = "passed"
    return finish()

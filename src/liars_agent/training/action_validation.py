"""Unweighted, state-level validation against acceptable complete action answers.

The checkpoint metric is -log(sum P(answer)) over the teacher's acceptable
action set. Each P(answer) uses the sum of its answer-token log probabilities,
including the assistant ending supplied by the training chat template. The
softmax over legal candidates is reported separately as a diagnostic.
"""
from __future__ import annotations

import json
import math
from typing import Any

GROUPS = ("clear", "tied", "small_gap", "unscored")
METRICS = ("action_set_nll", "candidate_normalized_set_probability",
           "action_set_accuracy", "estimated_regret", "canonical_mean_token_nll")


def _list(value: Any) -> list:
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _logsumexp(values: list[float]) -> float:
    largest = max(values)
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def candidate_completion(row: dict[str, Any], action_id: str) -> str:
    """Preserve answer formatting so canonical token NLL needs no extra forward."""
    canonical = row["completion"]
    start = canonical.index(":") + 1
    while canonical[start].isspace():
        start += 1
    current_id, length = json.JSONDecoder().raw_decode(canonical[start:])
    if current_id == action_id:
        return canonical
    return canonical[:start] + json.dumps(action_id, ensure_ascii=False) + canonical[start + length:]


def _summarize_state(scores: dict[str, tuple[float, int]], descriptor: dict,
                     canonical_id: str) -> dict[str, Any]:
    target_ids = descriptor["target_action_ids"]
    if not target_ids or len(set(target_ids)) != len(target_ids) or not set(target_ids) <= scores.keys():
        raise ValueError("Acceptable action IDs must be a nonempty unique subset of legal candidates.")
    if canonical_id not in scores:
        raise ValueError("The canonical action is not a legal candidate.")
    legal_logmass = _logsumexp([score[0] for score in scores.values()])
    if legal_logmass > 1e-8:
        raise RuntimeError("Legal complete-answer probabilities sum to more than one.")
    set_logmass = _logsumexp([scores[action_id][0] for action_id in target_ids])
    # A tolerance is needed only for floating-point summation at probability 1.
    set_nll = -min(0.0, set_logmass)
    normalized = math.exp(min(0.0, set_logmass - legal_logmass))
    chosen = max(scores, key=lambda action_id: scores[action_id][0])
    values = descriptor.get("candidate_values")
    regret = None
    if values is not None:
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
            raise ValueError("Stored candidate values must be finite numbers.")
        if chosen in values:
            regret = max(values.values()) - values[chosen]
    canonical_logp, canonical_tokens = scores[canonical_id]
    return {"group": descriptor["group"], "num_candidate_sequences": len(scores),
            "action_set_nll": set_nll, "candidate_normalized_set_probability": normalized,
            "action_set_accuracy": float(chosen in target_ids), "estimated_regret": regret,
            "canonical_mean_token_nll": -canonical_logp / canonical_tokens}


def _aggregate(states: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"num_samples": len(states),
               "num_candidate_sequences": sum(state["num_candidate_sequences"] for state in states),
               "estimated_regret_num_samples": sum(state["estimated_regret"] is not None for state in states)}
    summary["mean_candidate_count"] = summary["num_candidate_sequences"] / len(states) if states else None
    for key in METRICS:
        available = [state[key] for state in states if state[key] is not None]
        summary[key] = sum(available) / len(available) if available else None
    return summary


def evaluate_action_validation(
    rows: list[dict[str, Any]], backend: Any, client: Any, batch_size: int,
    max_seq_length: int, policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score every legal candidate once, batching forwards without optimizer steps.

    Every state contributes equally, regardless of ambiguity loss_weight or its
    candidate count. Unscored legacy rows have a singleton target and undefined
    regret. Regret is also undefined when the selected candidate has no stored
    value. Canonical mean-token NLL remains a separate legacy diagnostic.
    """
    from liars_agent.training.action_supervision import normalize_action_policy, resolve_action_supervision
    from liars_agent.training.tinker_train import build_sft_datum, validate_training_sample

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Action validation batch_size must be a positive integer.")
    normalized_policy = normalize_action_policy(policy)
    descriptors, canonical_ids, observations = [], [], []
    # Check schemas before any forward; do not convert a stale row into a valid
    # one by overriding its completion with a newly constructed candidate.
    for row in rows:
        if row.get("task") != "action":
            raise ValueError("Action validation accepts action rows only.")
        obs = validate_training_sample(row, action_supervision_policy=normalized_policy)
        descriptor = resolve_action_supervision(row, normalized_policy)
        if descriptor["group"] not in GROUPS:
            raise ValueError("Unknown action supervision validation group.")
        descriptors.append(descriptor)
        canonical_ids.append(json.loads(row["completion"])["chosen_action_id"])
        observations.append(obs)

    scores: list[dict[str, tuple[float, int]]] = [{} for _ in rows]
    pending = []

    def forward() -> None:
        if not pending:
            return
        data = [item[2] for item in pending]
        result = backend.run_training_batch(client, data, learning_rate=None)
        outputs = getattr(result, "loss_fn_outputs", None)
        if outputs is None or len(outputs) != len(data):
            raise RuntimeError("Action validation logprob output count differs from candidate batch size.")
        for (row_index, action_id, datum), output in zip(pending, outputs):
            if "logprobs" not in output:
                raise RuntimeError("Action validation requires per-token logprobs.")
            logprobs = _list(output["logprobs"])
            weights = _list(datum.loss_fn_inputs["weights"])
            if len(logprobs) != len(weights):
                raise RuntimeError("Action validation logprob lengths do not match answer masks.")
            answer = []
            for raw_logprob, raw_weight in zip(logprobs, weights):
                if type(raw_logprob) not in (int, float) or not math.isfinite(raw_logprob) or raw_logprob > 0:
                    raise RuntimeError("Action validation received an invalid token logprob (expected finite <= 0).")
                weight = float(raw_weight)
                if not math.isfinite(weight) or weight < 0:
                    raise RuntimeError("Action validation received an invalid answer mask.")
                if weight > 0:
                    answer.append(float(raw_logprob))
            if not answer:
                raise RuntimeError("Action validation candidate has no answer tokens.")
            sequence_logp = sum(answer)
            if not math.isfinite(sequence_logp):
                raise RuntimeError("Action validation complete-answer logprob is not finite.")
            scores[row_index][action_id] = sequence_logp, len(answer)
        pending.clear()

    for row_index, (row, obs) in enumerate(zip(rows, observations)):
        for action in obs.legal_actions:
            completion = candidate_completion(row, action.action_id)
            datum = build_sft_datum(row, backend, max_seq_length,
                completion_override=completion, sample_weight=1.0,
                action_supervision_policy=normalized_policy)
            pending.append((row_index, action.action_id, datum))
            if len(pending) == batch_size:
                forward()
    forward()
    states = [_summarize_state(state_scores, descriptor, canonical_id)
              for state_scores, descriptor, canonical_id in zip(scores, descriptors, canonical_ids)]
    return {**_aggregate(states), "policy": normalized_policy,
            "weighting": "equal_per_state_without_ambiguity_weights",
            "checkpoint_metric": "action_set_nll",
            "sequence_probability": "exp(sum of complete answer token logprobs, including chat-template assistant ending)",
            "candidate_probability": "renormalized over the enumerated legal JSON answers only",
            "action_set_accuracy_prediction": "legal_candidate_argmax; not unconstrained generated actions",
            "regret_definition": "max stored teacher candidate value minus chosen candidate value; undefined if unavailable",
            "argmax_tie_break": "first candidate in the legal-action order",
            "subgroups": {group: _aggregate([state for state in states if state["group"] == group])
                          for group in GROUPS}}

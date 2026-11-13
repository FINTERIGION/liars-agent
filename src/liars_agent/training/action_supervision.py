"""Empirical tied-action targets and explicit, configurable sample weights."""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

from liars_agent.agents.prompt_builder import validate_action_decision
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.state import _derive_seed

ACTION_SUPERVISION_VERSION = "empirical_ties_weighted_actions_v1"
DEFAULT_ACTION_SUPERVISION_POLICY = {
    "version": ACTION_SUPERVISION_VERSION,
    "positive_gap_weight": .5,
    "tie_tolerance": 1e-12,
    "gap_threshold": .05,
    "target_selection": "seeded_cycle_per_observation_epoch",
    "action_weight_normalization": "train_mean",
    "checkpoint_selection": "val_action_set_nll",
}


def normalize_action_policy(policy: dict | None = None) -> dict:
    if policy is None:
        return dict(DEFAULT_ACTION_SUPERVISION_POLICY)
    if not isinstance(policy, dict) or set(policy) - DEFAULT_ACTION_SUPERVISION_POLICY.keys():
        raise ValueError("Unknown action supervision policy fields.")
    result = {**DEFAULT_ACTION_SUPERVISION_POLICY, **policy}
    for key, value in DEFAULT_ACTION_SUPERVISION_POLICY.items():
        if key != "positive_gap_weight" and result[key] != value:
            raise ValueError(f"Incompatible action supervision {key}; use a new supported method version.")
    weight = result["positive_gap_weight"]
    if type(weight) not in (float, int) or not math.isfinite(weight) or not 0 < weight <= 1:
        raise ValueError("positive_gap_weight must be in (0, 1].")
    result["positive_gap_weight"] = float(weight)
    return result


def make_action_supervision(sample: dict, candidate_values: dict[str, float] | None,
                            policy: dict | None = None, *, source="recorded_teacher_scores") -> dict:
    """Describe empirical maxima; a small positive gap never adds extra targets."""
    policy = normalize_action_policy(policy)
    if sample.get("task") != "action":
        raise ValueError("Action supervision cannot be attached to a belief sample.")
    obs = PlayerObservation.from_dict(sample["observation"])
    canonical = validate_action_decision(obs, sample["completion"])["chosen_action_id"]
    legal_ids = {action.action_id for action in obs.legal_actions}
    gap = None
    if candidate_values is None:
        targets, group = [canonical], "unscored"
        values = None
    else:
        if (not isinstance(candidate_values, dict) or not candidate_values
                or not set(candidate_values) <= legal_ids or canonical not in candidate_values):
            raise ValueError("Teacher scores must be a nonempty legal candidate subset containing the recorded target.")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 + 1e-12
               for v in candidate_values.values()):
            raise ValueError("Teacher candidate values must be finite mixed rewards in [0, 1].")
        values = {key: float(value) for key, value in sorted(candidate_values.items())}
        best = max(values.values())
        targets = sorted(key for key, value in values.items() if abs(value - best) <= policy["tie_tolerance"])
        if canonical not in targets:
            raise ValueError("Recorded action is outside the empirical best-action set.")
        ordered = sorted(values.values(), reverse=True)
        if len(ordered) > 1:
            gap = ordered[0] - ordered[1]
        group = "tied" if len(targets) > 1 else ("small_gap" if gap is not None
            and gap <= policy["gap_threshold"] + policy["tie_tolerance"] else "clear")
    return {"version": ACTION_SUPERVISION_VERSION, "policy": policy,
        "group": group, "target_action_ids": targets, "candidate_values": values,
        "loss_weight": policy["positive_gap_weight"] if group == "small_gap" else 1.0,
        "best_vs_runnerup_gap": gap, "source": source}


def resolve_action_supervision(sample: dict, policy: dict | None = None) -> dict:
    policy = normalize_action_policy(policy)
    stored = sample.get("action_supervision")
    if stored is not None:
        if not isinstance(stored, dict) or stored.get("version") != ACTION_SUPERVISION_VERSION:
            raise ValueError("Missing or stale action supervision version.")
        expected = make_action_supervision(sample, stored.get("candidate_values"), policy,
                                          source=stored.get("source"))
        if stored != expected:
            raise ValueError("Action targets/weights do not match the recorded scores and policy.")
        return expected
    if sample.get("action_refinement"):
        raise ValueError("Refined samples need explicit action_supervision; prepare an uncertainty-aware run first.")
    values = sample.get("rollout_candidate_values")
    return make_action_supervision(sample, values, policy,
        source="recorded_teacher_scores" if values is not None else "single_recorded_target_without_scores")


def select_action_completion(sample: dict, seed: int, epoch: int, policy: dict | None = None) -> str:
    if type(seed) is not int or type(epoch) is not int or epoch < 0:
        raise ValueError("Target selection requires an integer seed and nonnegative epoch.")
    targets = list(resolve_action_supervision(sample, policy)["target_action_ids"])
    key = hashlib.sha256((sample["system"] + "\n" + sample["prompt"]).encode()).hexdigest()
    random.Random(_derive_seed(seed, "action_target_cycle", key)).shuffle(targets)
    return json.dumps({"chosen_action_id": targets[epoch % len(targets)]}, ensure_ascii=False)


def mean_action_weight(rows: list[dict], policy: dict | None = None) -> float:
    if not rows:
        raise ValueError("Cannot normalize an empty action training set.")
    return sum(resolve_action_supervision(row, policy)["loss_weight"] for row in rows) / len(rows)


def require_training_objective(metadata: dict, policy: dict | None = None) -> None:
    """Reject old checkpoint selection/weighting semantics before a paid call."""
    expected = normalize_action_policy(policy)
    if metadata.get("action_supervision_policy") != expected:
        raise ValueError("Checkpoint uses a missing or incompatible action supervision objective.")

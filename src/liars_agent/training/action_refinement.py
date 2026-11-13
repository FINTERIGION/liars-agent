"""Local paired-rollout refinement of observation-only teacher preferences.

The fixed gap rule controls simulation effort; it is not a significance test.
All candidate rewards are retained so an independent reader can reproduce the
means, deterministic label and refinement decisions without another rollout.
"""

from __future__ import annotations

import math
from typing import Any

from liars_agent.engine.observation import PlayerObservation
from liars_agent.training.data_generator import evaluate_observation_action_rewards

ACTION_REFINEMENT_VERSION = "paired_adaptive_8_32_64_v2"
GAP_THRESHOLD = 0.05
ADVANTAGE_MARGIN = 0.05
COMPARISON_TOLERANCE = 1e-12
ROLLOUT_STAGES = (8, 32, 64)


def _summarize(obs: PlayerObservation, rewards: dict[str, list[float]], count: int) -> dict[str, Any]:
    # Preserve the original teacher's floating-point accumulation and tie break.
    means = {action.action_id: sum(value / count for value in rewards[action.action_id][:count])
             for action in obs.legal_actions}
    chosen = max(obs.legal_actions, key=lambda action: (means[action.action_id], action.action_type, action.cards))
    best = means[chosen.action_id]
    ordered_means = sorted(means.values(), reverse=True)
    gap = best - ordered_means[1] if len(ordered_means) > 1 else None
    ties = sorted(action_id for action_id, mean in means.items()
                  if abs(best - mean) <= COMPARISON_TOLERANCE)
    advantage = best - sum(means.values()) / len(means)
    return {
        "num_rollouts": count,
        "candidate_means": means,
        "chosen_action_id": chosen.action_id,
        "best_vs_runnerup_gap": gap,
        "tied_best_action_ids": ties,
        "best_minus_mean_advantage": advantage,
        "meets_advantage_margin": advantage >= ADVANTAGE_MARGIN - COMPARISON_TOLERANCE,
        "ambiguous": len(ties) > 1 or (gap is not None and gap <= GAP_THRESHOLD + COMPARISON_TOLERANCE),
    }


def _upgrade_reasons(summary: dict[str, Any], prefix: str) -> list[str]:
    reasons = []
    if len(summary["tied_best_action_ids"]) > 1:
        reasons.append(f"{prefix}_top_tie")
    gap = summary["best_vs_runnerup_gap"]
    if gap is not None and gap <= GAP_THRESHOLD + COMPARISON_TOLERANCE:
        reasons.append(f"{prefix}_gap_at_or_below_threshold")
    return reasons


def _assemble_report(obs: PlayerObservation, rewards: dict[str, list[float]], seed: int,
                     force_refine: bool, reference_action_id: str | None) -> dict[str, Any]:
    """Reconstruct the only valid stopping path from the stored reward prefixes."""
    count = len(next(iter(rewards.values())))
    screen = _summarize(obs, rewards, 8)
    label_changed = reference_action_id is not None and reference_action_id not in screen["tied_best_action_ids"]
    stages = [screen]
    reasons = _upgrade_reasons(screen, "screen")
    if force_refine:
        reasons.append("forced_refinement")
    if label_changed:
        reasons.append("screen_label_changed")
    expected_count = 8
    if screen["ambiguous"] or force_refine or label_changed:
        if count < 32:
            raise ValueError("Refinement stopped before the required 32 trajectories.")
        stage32 = _summarize(obs, rewards, 32)
        stages.append(stage32)
        expected_count = 32
        if stage32["ambiguous"]:
            reasons.extend(_upgrade_reasons(stage32, "stage32"))
            if count < 64:
                raise ValueError("Refinement stopped before the required 64 trajectories.")
            stages.append(_summarize(obs, rewards, 64))
            expected_count = 64
    if count != expected_count:
        raise ValueError("Stored rollout count does not match the adaptive stopping rule.")
    final = stages[-1]
    if final["ambiguous"]:
        reason = "maximum_rollouts_ambiguous"
    elif count == 8:
        reason = "screen_gap_above_threshold" if len(rewards) > 1 else "single_legal_action"
    else:
        reason = "refined_gap_above_threshold" if len(rewards) > 1 else "single_legal_action"
    return {
        "version": ACTION_REFINEMENT_VERSION,
        "seed": seed,
        "force_refine": force_refine,
        "reference_action_id": reference_action_id,
        "screen_label_changed": label_changed,
        "gap_threshold": GAP_THRESHOLD,
        "advantage_margin": ADVANTAGE_MARGIN,
        "comparison_tolerance": COMPARISON_TOLERANCE,
        "candidate_count": len(rewards),
        "screen": {**screen, "candidate_rewards": {key: list(values[:8]) for key, values in rewards.items()}},
        "final": {**final, "candidate_rewards": {key: list(values) for key, values in rewards.items()}},
        "stages": stages,
        "chosen_action_id": final["chosen_action_id"],
        "rollouts_per_candidate": count,
        "total_rollouts": count * len(rewards),
        "refinement_reason": reasons,
        "reason": reason,
        "ambiguous": final["ambiguous"],
        "meets_advantage_margin": final["meets_advantage_margin"],
    }


def refine_action_observation(obs: PlayerObservation, teacher_config: dict | None, seed: int,
                              force_refine: bool = False,
                              reference_action_id: str | None = None) -> dict[str, Any]:
    """Evaluate 8 paired trajectories, then append up to 32 and 64 when needed.

    ``force_refine`` requests at least 32 trajectories. A reference action outside
    the screen's tied-best set also requests at least 32; it does not affect any
    reward or the final choice. Further expansion occurs
    only when the current top is tied (within tolerance) or its runner-up gap is
    at most .05. ``total_rollouts`` counts candidate continuations, whereas
    ``rollouts_per_candidate`` counts shared trajectory groups. A low final
    advantage is reported explicitly; this function does not accept/reject data.
    """
    if type(seed) is not int or type(force_refine) is not bool:
        raise ValueError("Refinement requires an integer seed and boolean force_refine.")
    if reference_action_id is not None and (not isinstance(reference_action_id, str)
            or reference_action_id not in {a.action_id for a in obs.legal_actions}):
        raise ValueError("reference_action_id must be a legal candidate ID or None.")
    rewards = evaluate_observation_action_rewards(obs, teacher_config, 8, seed)
    current = _summarize(obs, rewards, 8)
    label_changed = reference_action_id is not None and reference_action_id not in current["tied_best_action_ids"]
    for target in ROLLOUT_STAGES[1:]:
        if not current["ambiguous"] and not (target == 32 and (force_refine or label_changed)):
            break
        previous = current["num_rollouts"]
        additional = evaluate_observation_action_rewards(obs, teacher_config, target - previous, seed,
                                                         start_index=previous)
        for action_id in rewards:
            rewards[action_id].extend(additional[action_id])
        current = _summarize(obs, rewards, target)
    return _assemble_report(obs, rewards, seed, force_refine, reference_action_id)


def validate_action_refinement_report(obs: PlayerObservation, report: dict[str, Any]) -> None:
    """Check saved arithmetic and protocol without resampling hidden worlds.

    This verifies all summaries, the screen prefix, deterministic selection and
    stopping rule. It does not prove that the saved rewards came from a rollout;
    seed-based regeneration is a separate reproducibility check.
    """
    if obs.ablation_mode != "full" or not obs.legal_actions:
        raise ValueError("Refinement verification requires a full legal observation.")
    if not isinstance(report, dict) or report.get("version") != ACTION_REFINEMENT_VERSION:
        raise ValueError("Missing or incompatible action refinement version.")
    if type(report.get("seed")) is not int or type(report.get("force_refine")) is not bool:
        raise ValueError("Refinement seed/force_refine metadata is invalid.")
    reference_action_id = report.get("reference_action_id")
    if reference_action_id is not None and (not isinstance(reference_action_id, str)
            or reference_action_id not in {a.action_id for a in obs.legal_actions}):
        raise ValueError("reference_action_id must be a legal candidate ID or None.")
    final = report.get("final")
    rewards = final.get("candidate_rewards") if isinstance(final, dict) else None
    if not isinstance(rewards, dict) or set(rewards) != {a.action_id for a in obs.legal_actions}:
        raise ValueError("Refinement rewards must cover exactly the legal candidate IDs.")
    lengths = set()
    for values in rewards.values():
        if not isinstance(values, list):
            raise ValueError("Candidate rewards must be JSON lists.")
        lengths.add(len(values))
        if any(type(value) not in (int, float) or not math.isfinite(value) or value not in (0.0, .35, .65, 1.0)
               for value in values):
            raise ValueError("Candidate reward is not a possible win/round-safety score.")
    if len(lengths) != 1 or next(iter(lengths)) not in ROLLOUT_STAGES:
        raise ValueError("Candidate trajectories must have a shared count of 8, 32 or 64.")
    expected = _assemble_report(obs, rewards, report["seed"], report["force_refine"], reference_action_id)
    if set(report) != set(expected):
        raise ValueError("Action refinement report fields differ from its versioned schema.")
    for key, value in expected.items():
        if report[key] != value:
            raise ValueError(f"Action refinement report disagrees with saved rewards: {key}.")

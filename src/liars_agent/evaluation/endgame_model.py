"""Model diagnostics on the independent, deliberately selected endgame check set.

Only public PlayerObservation objects reach agents. Hidden labels are joined
after predictions, for belief calibration; there are no optimal-action labels.
An agent's returned action is checked after its own fallback, without stepping
a game or estimating a win rate from these selected observations.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from liars_agent.agents.base import BaseAgent
from liars_agent.agents.prompt_builder import validate_action_decision
from liars_agent.contracts import current_contract, require_contract
from liars_agent.engine.observation import PlayerObservation
from liars_agent.evaluation.metrics import compute_probability_calibration


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize(rows: list[dict]) -> dict[str, Any]:
    forced = [row for row in rows if row["forced_challenge"]]
    claims = [row for row in rows if row["label"] is not None]
    calibration = compute_probability_calibration(
        [row["native_belief_probability"] for row in claims], [row["label"] for row in claims])
    return {
        "num_samples": len(rows), "num_claim_samples": len(claims),
        "num_no_claim_samples": len(rows) - len(claims),
        "native_action_legal_count": sum(row["native_action_legal"] for row in rows),
        "native_action_legal_rate": _ratio(sum(row["native_action_legal"] for row in rows), len(rows)),
        "returned_action_legal_count": sum(row["returned_action_legal"] for row in rows),
        "returned_action_legal_rate": _ratio(sum(row["returned_action_legal"] for row in rows), len(rows)),
        "fallback_count": sum(row["was_fallback"] for row in rows),
        "fallback_rate": _ratio(sum(row["was_fallback"] for row in rows), len(rows)),
        "returned_action_types": dict(Counter(row["returned_action_type"] for row in rows)),
        "forced_challenge": {
            "num_samples": len(forced),
            "native_compliance_count": sum(row["native_action_legal"] for row in forced),
            "native_compliance_rate": _ratio(sum(row["native_action_legal"] for row in forced), len(forced)),
            "returned_compliance_count": sum(row["returned_action_legal"] for row in forced),
            "returned_compliance_rate": _ratio(sum(row["returned_action_legal"] for row in forced), len(forced)),
        },
        "native_belief": {
            "sources": dict(Counter(row["belief_source"] for row in claims)),
            "num_claim_samples": len(claims), "num_valid_predictions": calibration["num_samples"],
            "calibration": calibration,
        },
    }


def _model_metadata(key: str, agent: BaseAgent) -> dict[str, Any]:
    parameter_names = ("seed", "temperature", "top_p", "max_new_tokens", "bluff_weights",
        "challenge_prob_cutoff", "risk_aversion", "posterior_challenge_cutoff",
        "bluff_prob_when_forced", "challenge_threshold", "bluff_rate", "multi_card_bias",
        "multi_claim_challenge_prob", "single_claim_challenge_prob", "high_risk_shots_threshold",
        "low_risk_bluff_rate", "high_risk_bluff_rate")
    return {"model_key": key, "agent_name": agent.name,
            "agent_class": f"{type(agent).__module__}.{type(agent).__qualname__}",
            "checkpoint": getattr(agent, "checkpoint", None),
            "base_model": getattr(getattr(agent, "backend", None), "base_model", None),
            "parameters": {name: getattr(agent, name) for name in parameter_names if hasattr(agent, name)}}


def evaluate_endgame_models(
    coverage_dir: str | Path, agents: Mapping[str, BaseAgent], *,
    cached_results: Mapping[str, dict] | None = None,
) -> dict[str, Any]:
    """Evaluate each real agent interface on exactly the same frozen public rows.

    Callers construct agents (including their checkpoints), validate source-game
    independence with validate_endgame_coverage, and persist this returned report.
    Agent/service exceptions propagate instead of becoming successful predictions.
    Native belief uses the outer decision record only. A model's independent
    belief remains native even when its action falls back; a fallback bot's score
    is excluded. select_action handles belief only when a pending claim exists.
    Cache values are previous models[key] entries; both data and agent metadata
    must match before they can bypass model calls.
    """
    directory = Path(coverage_dir)
    coverage = json.loads((directory / "coverage_report.json").read_text(encoding="utf-8"))
    require_contract(coverage, "endgame coverage report")
    filenames = ("observations.jsonl", "no_pending_claim.jsonl")
    rows = [row for name in filenames for row in _read_rows(directory / name)]
    labels = _read_rows(directory / "labels.jsonl")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids) or len({row["sample_id"] for row in labels}) != len(labels):
        raise ValueError("Endgame observations and labels require unique sample IDs.")
    label_by_id = {row["sample_id"]: row["label"] for row in labels}
    if set(ids) != label_by_id.keys():
        raise ValueError("Endgame observations and labels differ by sample ID.")
    for row in rows:
        has_claim = row["observation"]["last_claim"] is not None
        label = label_by_id[row["sample_id"]]
        if (has_claim and (type(label) is not int or label not in (0, 1))) or (not has_claim and label is not None):
            raise ValueError("Endgame truth labels must match pending-claim applicability.")
    digest = hashlib.sha256()
    for name in (*filenames, "labels.jsonl"):
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    dataset = {"num_samples": len(rows), "num_claim": sum(label is not None for label in label_by_id.values()),
               "num_no_claim": sum(label is None for label in label_by_id.values()),
               "input_sha256": digest.hexdigest()}

    results = {}
    scenarios = sorted({tag for row in rows for tag in row["scenario_tags"]})
    for key, agent in agents.items():
        metadata = _model_metadata(key, agent)
        if cached_results is not None and key in cached_results:
            cached = cached_results[key]
            require_contract(cached, "cached endgame model diagnostics")
            if cached.get("dataset") != dataset or cached.get("model_metadata") != metadata:
                raise ValueError(f"Cached endgame data/model metadata differ for {key}.")
            results[key] = cached
            continue
        predictions = []
        for row in rows:
            # No labels, source game IDs, seeds, or private replay data are sent.
            obs = PlayerObservation.from_dict(row["observation"])
            action, record = agent.select_action(obs)
            legal = {candidate.action_id: candidate for candidate in obs.legal_actions}
            returned_legal = legal.get(action.action_id) == action
            native_id = None
            native_error = record.fallback_reason if record.was_fallback else ""
            if not record.was_fallback:
                if record.raw_output or hasattr(agent, "checkpoint"):
                    try:
                        native_id = validate_action_decision(obs, record.raw_output)["chosen_action_id"]
                    except ValueError as exc:
                        native_error = str(exc)
                elif returned_legal:
                    # Rule/classifier agents return a structured candidate directly.
                    native_id = action.action_id
            native_legal = native_id in legal
            probability = None
            if obs.last_claim is not None and record.belief_source in {"model", "classifier", "heuristic"}:
                if not record.was_fallback or record.belief_source == "model":
                    probability = record.belief_bluff_prob
            forced = len(obs.legal_actions) == 1 and obs.legal_actions[0].action_type == "CHALLENGE"
            predictions.append({
                "sample_id": row["sample_id"], "scenario_tags": row["scenario_tags"],
                "alive_players": row["alive_players"], "forced_challenge": forced,
                "native_action_id": native_id, "native_action_legal": native_legal,
                "native_action_error": native_error, "raw_output": record.raw_output,
                "returned_action_id": action.action_id, "returned_action_type": action.action_type,
                "returned_action_legal": returned_legal, "was_fallback": record.was_fallback,
                "fallback_reason": record.fallback_reason,
                "belief_source": record.belief_source, "native_belief_probability": probability,
                "label": label_by_id[row["sample_id"]],
            })
        tracker = getattr(agent, "tracker", None)
        results[key] = {
            "contract": current_contract(), "dataset": dataset,
            "summary": _summarize(predictions),
            "by_scenario": {tag: _summarize([row for row in predictions if tag in row["scenario_tags"]])
                            for tag in scenarios},
            "by_alive_players": {str(alive): _summarize([row for row in predictions if row["alive_players"] == alive])
                                 for alive in (2, 3, 4)},
            "by_scenario_alive_players": {
                tag: {str(alive): _summarize([row for row in predictions
                                             if tag in row["scenario_tags"] and row["alive_players"] == alive])
                      for alive in (2, 3, 4)} for tag in scenarios},
            "predictions": predictions,
            "model_metadata": metadata,
            "usage": tracker.to_dict() if tracker is not None else {"api_calls": 0},
        }
    return {
        "contract": current_contract(), "purpose": "independent_endgame_model_diagnostics",
        "natural_distribution": False,
        "dataset": dataset,
        "interpretation": "Truth-balanced selected strata; no optimal free-action labels or aggregate win rate. "
                          "Returned-action legality is checked after the agent's fallback; no game is stepped. "
                          "Scenario strata can overlap; their counts must not be added.",
        "models": results,
    }

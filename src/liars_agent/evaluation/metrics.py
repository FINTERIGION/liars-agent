"""评测统计指标：配对 Bootstrap 置信区间、行为画像、风险分层诊断与信念概率校准."""

from __future__ import annotations

import math
import random
from typing import Any

from liars_agent.engine.types import are_cards_truthful, is_card_truthful


def _interval(values: list[float]) -> tuple[float, float]:
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_block_bootstrap_ci(
    block_winrates: list[float], num_resamples: int = 10000, seed: int = 42,
) -> dict[str, Any]:
    """Resample complete four-seat blocks, never the dependent games separately."""
    if not block_winrates or num_resamples < 1:
        raise ValueError("Bootstrap requires non-empty blocks and positive resamples.")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in block_winrates):
        raise ValueError("Block win rates must be finite probabilities.")
    n = len(block_winrates)
    rng = random.Random(seed)
    low, high = _interval([
        sum(block_winrates[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(num_resamples)
    ])
    return {"num_blocks": n, "num_games": n * 4, "mean_winrate": sum(block_winrates) / n,
            "ci_95_low": low, "ci_95_high": high, "num_resamples": num_resamples, "bootstrap_seed": seed,
            "bootstrap": {"unit": "block", "num_resamples": num_resamples, "seed": seed}}


def _indexed_records(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get(key), str):
            raise ValueError(f"Every paired record requires a string {key}.")
        if record[key] in indexed:
            raise ValueError(f"Duplicate paired record: {record[key]}")
        indexed[record[key]] = record
    return indexed


def compute_paired_bootstrap_ci(
    block_records_a: list[dict[str, Any]], block_records_b: list[dict[str, Any]],
    num_resamples: int = 10000, seed: int = 42,
) -> dict[str, Any]:
    """Align blocks by identity and reject different seeds, pools or permutations."""
    a, b = _indexed_records(block_records_a, "block_id"), _indexed_records(block_records_b, "block_id")
    if not a or a.keys() != b.keys() or num_resamples < 1:
        raise ValueError("Paired results require the same non-empty block IDs and positive resamples.")
    ordered = sorted(a)
    for key in ordered:
        for field in ("block_seed", "opponent_specs", "opponent_order", "rules_version"):
            if field not in a[key] or field not in b[key] or a[key][field] != b[key][field]:
                raise ValueError(f"Block {key} differs or lacks pairing field {field}.")
        if sorted(a[key]["opponent_order"]) != [0, 1, 2] or len(a[key]["opponent_specs"]) != 3:
            raise ValueError(f"Block {key} does not describe three opponents.")
    wa, wb = [a[k]["win_rate"] for k in ordered], [b[k]["win_rate"] for k in ordered]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in wa + wb):
        raise ValueError("Block win rates must be finite probabilities.")
    differences = [x - y for x, y in zip(wa, wb)]
    n, rng = len(ordered), random.Random(seed)
    low, high = _interval([sum(differences[rng.randrange(n)] for _ in range(n)) / n
                           for _ in range(num_resamples)])
    return {"num_blocks": n, "num_games_per_model": n * 4,
            "mean_winrate_a": sum(wa) / n, "mean_winrate_b": sum(wb) / n,
            "mean_diff": sum(differences) / n, "ci_95_low": low, "ci_95_high": high,
            "num_resamples": num_resamples, "bootstrap_seed": seed,
            "bootstrap": {"unit": "block", "num_resamples": num_resamples, "seed": seed}}


def compute_behavioral_profile(
    replay_dicts: list[dict[str, Any]],
    target_agent_name: str,
) -> dict[str, Any]:
    """统计目标智能体的四维核心行为画像：诈唬率、被抓率、质疑率、质疑命中率."""
    total_plays = 0
    bluff_plays = 0
    bluff_caught = 0
    challenge_opportunities = 0
    voluntary_challenges = 0
    challenge_hits = 0

    for bundle in replay_dicts:
        seat_names = bundle.get("seat_agent_names", [])
        target_seats = {
            idx for idx, name in enumerate(seat_names) if name == target_agent_name
        }
        if not target_seats:
            continue

        steps = bundle.get("steps", [])
        for i, step in enumerate(steps):
            seat = step["actor_seat"]
            if seat not in target_seats:
                continue

            obs = step["observation"]
            action = step["chosen_action"]
            target_rank = step["target_rank"]
            legal_actions = obs.get("legal_actions", [])
            is_forced = len(legal_actions) == 1 and legal_actions[0]["action_type"] == "CHALLENGE"

            # 统计面临上家声称时的自愿质疑率与命中率
            if obs.get("last_claim") is not None and not is_forced:
                challenge_opportunities += 1
                if action["action_type"] == "CHALLENGE":
                    voluntary_challenges += 1
                    res = step.get("round_resolution")
                    if res and res.get("challenge_succeeded"):
                        challenge_hits += 1

            # 统计出牌时的诈唬率与被下家抓现行率
            if action["action_type"] == "PLAY":
                total_plays += 1
                cards = tuple(action.get("cards", ()))
                is_bluff = not are_cards_truthful(cards, target_rank)
                if is_bluff:
                    bluff_plays += 1
                    # 检查紧接着的下一步是否被下家 CHALLENGE
                    if i + 1 < len(steps):
                        next_step = steps[i + 1]
                        if (
                            next_step["round_id"] == step["round_id"]
                            and next_step["chosen_action"]["action_type"] == "CHALLENGE"
                        ):
                            bluff_caught += 1


    return {
        "total_plays": total_plays,
        "bluff_plays": bluff_plays,
        "bluff_rate": _ratio(bluff_plays, total_plays),
        "caught_when_bluffing_rate": _ratio(bluff_caught, bluff_plays),
        "challenge_opportunities": challenge_opportunities,
        "voluntary_challenges": voluntary_challenges,
        "challenge_rate": _ratio(voluntary_challenges, challenge_opportunities),
        "challenge_hit_rate": _ratio(challenge_hits, voluntary_challenges),
        "bluff_caught": bluff_caught, "challenge_hits": challenge_hits,
        "denominators": {"bluff_rate": total_plays, "caught_when_bluffing_rate": bluff_plays,
                         "challenge_rate": challenge_opportunities, "challenge_hit_rate": voluntary_challenges},
    }


def compute_risk_and_hand_stratified_diagnosis(
    replay_dicts: list[dict[str, Any]],
    target_agent_name: str,
) -> dict[str, Any]:
    """按淘汰风险（已开枪次数 0, 1, 2+）与手牌真牌数量（0, 1~2, 3+）分层分析质疑与诈唬行为."""
    risk_bins = ["0枪(下枪致死16.7%)", "1枪(下枪致死20.0%)", "2+枪(下枪致死≥25.0%)"]
    hand_bins = ["0张真牌(无真牌)", "1~2张真牌(常规)", "3+张真牌(富余)"]

    grid_stats: dict[str, dict[str, dict[str, int]]] = {
        r: {
            h: {
                "opp_count": 0,
                "challenges": 0,
                "challenge_hits": 0,
                "true_opp_bluffs": 0,
                "plays": 0,
                "bluffs": 0,
            }
            for h in hand_bins
        }
        for r in risk_bins
    }

    for bundle in replay_dicts:
        seat_names = bundle.get("seat_agent_names", [])
        target_seats = {
            idx for idx, name in enumerate(seat_names) if name == target_agent_name
        }
        for step in bundle.get("steps", []):
            seat = step["actor_seat"]
            if seat not in target_seats:
                continue
            obs = step["observation"]
            target_rank = step["target_rank"]
            own_hand = obs.get("own_hand", [])
            own_truth = sum(1 for c in own_hand if is_card_truthful(c, target_rank))

            shots = 0
            for p in obs.get("players_public_status", []):
                if p["seat"] == seat:
                    shots = p["shots_taken"]
                    break

            r_key = (
                risk_bins[0]
                if shots == 0
                else (risk_bins[1] if shots == 1 else risk_bins[2])
            )
            h_key = (
                hand_bins[0]
                if own_truth == 0
                else (hand_bins[1] if own_truth <= 2 else hand_bins[2])
            )
            cell = grid_stats[r_key][h_key]

            legal = obs.get("legal_actions", [])
            is_forced = len(legal) == 1 and legal[0]["action_type"] == "CHALLENGE"
            action = step["chosen_action"]

            if obs.get("last_claim") is not None and not is_forced:
                cell["opp_count"] += 1
                if step.get("pending_claim_was_bluff"):
                    cell["true_opp_bluffs"] += 1
                if action["action_type"] == "CHALLENGE":
                    cell["challenges"] += 1
                    if step.get("pending_claim_was_bluff"):
                        cell["challenge_hits"] += 1

            if action["action_type"] == "PLAY":
                cell["plays"] += 1
                cards = tuple(action.get("cards", ()))
                if not are_cards_truthful(cards, target_rank):
                    cell["bluffs"] += 1

    summary_table: dict[str, dict[str, dict[str, Any]]] = {}
    for r_key in risk_bins:
        summary_table[r_key] = {}
        for h_key in hand_bins:
            c = grid_stats[r_key][h_key]
            summary_table[r_key][h_key] = {
                "opportunities": c["opp_count"],
                "challenge_rate": _ratio(c["challenges"], c["opp_count"]),
                "challenge_precision": _ratio(c["challenge_hits"], c["challenges"]),
                "ground_truth_bluff_rate": _ratio(c["true_opp_bluffs"], c["opp_count"]),
                "play_bluff_rate": _ratio(c["bluffs"], c["plays"]),
                "challenges": c["challenges"], "plays": c["plays"],
                "challenge_hits": c["challenge_hits"], "bluffs": c["bluffs"],
                "denominators": {"challenge_rate": c["opp_count"], "challenge_precision": c["challenges"],
                                 "ground_truth_bluff_rate": c["opp_count"], "play_bluff_rate": c["plays"]},
            }

    return {"risk_bins": risk_bins, "hand_bins": hand_bins, "grid": summary_table}


def compute_probability_calibration(
    probabilities: list[float | None], labels: list[bool | int], num_bins: int = 5,
) -> dict[str, Any]:
    """Score native predictions; None denotes an unavailable probability, not zero.

    Log loss bounds each logarithm's argument below by 1e-15.
    All metrics weight decisions equally; coverage reports missing predictions.
    """
    if len(probabilities) != len(labels) or num_bins < 1:
        raise ValueError("Probabilities/labels must align and num_bins must be positive.")
    pairs = []
    for probability, label in zip(probabilities, labels):
        if label not in (0, 1) or not isinstance(label, (int, bool)):
            raise ValueError("Calibration labels must be binary observed outcomes.")
        if probability is None:
            continue
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Predictions must be finite probabilities or None.")
        pairs.append((float(probability), int(label)))
    n = len(pairs)
    bins = []
    ece = 0.0
    for index in range(num_bins):
        selected = [(p, y) for p, y in pairs if min(int(p * num_bins), num_bins - 1) == index]
        count = len(selected)
        mean_p = sum(p for p, _ in selected) / count if count else None
        mean_y = sum(y for _, y in selected) / count if count else None
        if count:
            ece += count / n * abs(mean_p - mean_y)
        bins.append({"bin_low": index / num_bins, "bin_high": (index + 1) / num_bins,
                     "count": count, "mean_predicted_prob": mean_p, "empirical_bluff_freq": mean_y})
    clip = 1e-15
    log_loss = sum(-(y * math.log(max(clip, p)) + (1 - y) * math.log(max(clip, 1 - p))) for p, y in pairs) / n if n else None
    return {"num_candidates": len(labels), "num_samples": n, "coverage": _ratio(n, len(labels)),
            "ece": ece if n else None,
            "brier_score": sum((p - y) ** 2 for p, y in pairs) / n if n else None,
            "log_loss": log_loss, "log_loss_clip": clip, "bins": bins,
            "weighting": "decision"}


def compute_belief_calibration(
    replay_dicts: list[dict[str, Any]], target_agent_name: str, num_bins: int = 5,
    *, belief_source: str = "model",
) -> dict[str, Any]:
    """Score the named source's native probability, never fallback_record beliefs.

    An action fallback may still have a valid independent native model belief.
    Such a prediction remains eligible when its source is explicitly 'model'.
    """
    probabilities, labels = [], []
    source_counts: dict[str, int] = {}
    action_fallbacks = 0
    for bundle in replay_dicts:
        seats = {i for i, name in enumerate(bundle.get("seat_agent_names", [])) if name == target_agent_name}
        for step in bundle.get("steps", []):
            if step["actor_seat"] not in seats or step.get("pending_claim_was_bluff") is None:
                continue
            record = step.get("decision_record", {})
            source = record.get("belief_source", "unavailable")
            source_counts[source] = source_counts.get(source, 0) + 1
            action_fallbacks += bool(record.get("was_fallback"))
            probabilities.append(record.get("belief_bluff_prob") if source == belief_source else None)
            labels.append(bool(step["pending_claim_was_bluff"]))
    result = compute_probability_calibration(probabilities, labels, num_bins)
    result.update({"belief_source": belief_source, "source_counts": source_counts,
                   "action_fallbacks": action_fallbacks, "population": "on_policy_decisions"})
    return result


def compute_paired_probability_metrics(
    records_a: list[dict[str, Any]], records_b: list[dict[str, Any]],
    num_resamples: int = 10000, seed: int = 42, num_bins: int = 5,
) -> dict[str, Any]:
    """Compare fixed-bank probabilities with a paired whole-game bootstrap.

    Each record contains sample_id, game_id, label and probability (or None).
    Paired differences use the intersection of available native predictions;
    separate model summaries retain all eligible records and coverage.
    """
    a, b = _indexed_records(records_a, "sample_id"), _indexed_records(records_b, "sample_id")
    if a.keys() != b.keys() or num_resamples < 1:
        raise ValueError("Probability comparisons require matching sample IDs and positive resamples.")
    ids = sorted(a)
    for key in ids:
        if (not isinstance(a[key].get("game_id"), str) or a[key].get("game_id") != b[key].get("game_id")
                or a[key].get("label") != b[key].get("label")):
            raise ValueError(f"Probability sample {key} has a different game or label.")
    labels = [a[k]["label"] for k in ids]
    pa, pb = [a[k].get("probability") for k in ids], [b[k].get("probability") for k in ids]
    summary_a = compute_probability_calibration(pa, labels, num_bins)
    summary_b = compute_probability_calibration(pb, labels, num_bins)
    paired_ids = [k for k in ids if a[k].get("probability") is not None and b[k].get("probability") is not None]
    groups: dict[str, list[str]] = {}
    for key in paired_ids:
        groups.setdefault(a[key]["game_id"], []).append(key)
    names = sorted(groups)
    keys = ("brier_score", "log_loss", "ece")
    result = {"num_candidates": len(ids), "num_samples": len(paired_ids), "num_games": len(names),
              "paired_coverage": _ratio(len(paired_ids), len(ids)), "model_a": summary_a, "model_b": summary_b,
              "difference": "a_minus_b; lower_is_better",
              "bootstrap": {"unit": "game", "num_resamples": num_resamples, "seed": seed},
              "paired_metrics": {key: {"mean_a": None, "mean_b": None, "mean_diff": None,
                                       "ci_95_low": None, "ci_95_high": None} for key in keys}}
    if not names:
        return result
    # Aggregate each game's sufficient statistics; resampling does not split games.
    game_stats = []
    for name in names:
        rows = groups[name]
        stats = [float(len(rows)), 0.0, 0.0, 0.0, 0.0] + [0.0] * (num_bins * 6)
        for key in rows:
            y = a[key]["label"]
            for model, record in enumerate((a[key], b[key])):
                p = record["probability"]
                stats[1 + model] += (p - y) ** 2
                stats[3 + model] += -(y * math.log(max(1e-15, p)) + (1-y) * math.log(max(1e-15, 1-p)))
                offset = 5 + model * num_bins * 3 + min(int(p * num_bins), num_bins - 1) * 3
                stats[offset] += 1
                stats[offset + 1] += p
                stats[offset + 2] += y
        game_stats.append(stats)
    def scores(stats):
        total = stats[0]
        eces = [sum(abs(stats[5 + model * num_bins * 3 + index * 3 + 1] - stats[5 + model * num_bins * 3 + index * 3 + 2])
                    for index in range(num_bins)) / total for model in range(2)]
        return [(stats[1] / total, stats[2] / total), (stats[3] / total, stats[4] / total), tuple(eces)]
    totals = [sum(row[j] for row in game_stats) for j in range(len(game_stats[0]))]
    means = scores(totals)
    rng, differences = random.Random(seed), [[] for _ in keys]
    for _ in range(num_resamples):
        drawn = [game_stats[rng.randrange(len(names))] for _ in names]
        sums = [sum(row[j] for row in drawn) for j in range(len(totals))]
        for index, (x, y) in enumerate(scores(sums)):
            differences[index].append(x - y)
    for index, key in enumerate(keys):
        low, high = _interval(differences[index])
        x, y = means[index]
        result["paired_metrics"][key] = {"mean_a": x, "mean_b": y, "mean_diff": x-y,
                                          "ci_95_low": low, "ci_95_high": high}
    return result

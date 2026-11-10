"""评测统计指标：配对 Bootstrap 置信区间、行为画像、风险分层诊断与信念概率校准."""

from __future__ import annotations

import math
import random
from typing import Any

from liars_agent.engine.types import are_cards_truthful, is_card_truthful


def compute_paired_bootstrap_ci(
    block_winrates_a: list[float],
    block_winrates_b: list[float],
    num_resamples: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """以四局座位轮换对局块（Block）为单位，计算模型 A 相对模型 B 的配对胜率差及 95% Bootstrap 置信区间."""
    if len(block_winrates_a) != len(block_winrates_b) or not block_winrates_a:
        raise ValueError("Paired block lists must be non-empty and of equal length.")

    n = len(block_winrates_a)
    diffs = [a - b for a, b in zip(block_winrates_a, block_winrates_b)]
    mean_a = sum(block_winrates_a) / n
    mean_b = sum(block_winrates_b) / n
    mean_diff = sum(diffs) / n

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(num_resamples):
        sample_sum = sum(diffs[rng.randrange(n)] for _ in range(n))
        boot_means.append(sample_sum / n)

    boot_means.sort()
    low_idx = max(0, int(0.025 * num_resamples))
    high_idx = min(num_resamples - 1, int(0.975 * num_resamples))
    ci_low = boot_means[low_idx]
    ci_high = boot_means[high_idx]

    # 计算单侧经验 p-value (P(diff <= 0))
    non_positive = sum(1 for m in boot_means if m <= 0.0) / num_resamples

    return {
        "num_blocks": n,
        "num_games_per_model": n * 4,
        "mean_winrate_a": round(mean_a, 4),
        "mean_winrate_b": round(mean_b, 4),
        "mean_diff": round(mean_diff, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "p_value_one_sided": round(non_positive, 4),
    }


def compute_behavioral_profile(
    replay_dicts: list[dict[str, Any]],
    target_agent_name: str,
) -> dict[str, float]:
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

    bluff_rate = bluff_plays / max(1, total_plays)
    caught_rate = bluff_caught / max(1, bluff_plays)
    challenge_rate = voluntary_challenges / max(1, challenge_opportunities)
    challenge_hit_rate = challenge_hits / max(1, voluntary_challenges)

    return {
        "total_plays": total_plays,
        "bluff_plays": bluff_plays,
        "bluff_rate": round(bluff_rate, 4),
        "caught_when_bluffing_rate": round(caught_rate, 4),
        "challenge_opportunities": challenge_opportunities,
        "voluntary_challenges": voluntary_challenges,
        "challenge_rate": round(challenge_rate, 4),
        "challenge_hit_rate": round(challenge_hit_rate, 4),
    }


def compute_risk_and_hand_stratified_diagnosis(
    replay_dicts: list[dict[str, Any]],
    target_agent_name: str,
) -> dict[str, Any]:
    """按淘汰风险（已开枪次数 0, 1, 2+）与手牌真牌数量（0, 1~2, 3+）分层分析质疑与诈唬行为."""
    risk_bins = ["0枪(低风险16.7%)", "1枪(中风险20.0%)", "2+枪(高危>=25.0%)"]
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

    summary_table: dict[str, dict[str, dict[str, float]]] = {}
    for r_key in risk_bins:
        summary_table[r_key] = {}
        for h_key in hand_bins:
            c = grid_stats[r_key][h_key]
            opp_n = max(1, c["opp_count"])
            ch_n = max(1, c["challenges"])
            pl_n = max(1, c["plays"])
            summary_table[r_key][h_key] = {
                "opportunities": c["opp_count"],
                "challenge_rate": round(c["challenges"] / opp_n, 4),
                "challenge_precision": round(c["challenge_hits"] / ch_n, 4),
                "ground_truth_bluff_rate": round(c["true_opp_bluffs"] / opp_n, 4),
                "play_bluff_rate": round(c["bluffs"] / pl_n, 4),
            }

    return {"risk_bins": risk_bins, "hand_bins": hand_bins, "grid": summary_table}


def compute_belief_calibration(
    replay_dicts: list[dict[str, Any]],
    target_agent_name: str,
    num_bins: int = 5,
) -> dict[str, Any]:
    """评估智能体输出的信念估计 `belief_bluff_prob` 与事后真值 `pending_claim_was_bluff` 的校准度 (ECE & Brier Score)."""
    pairs: list[tuple[float, float]] = []
    for bundle in replay_dicts:
        seat_names = bundle.get("seat_agent_names", [])
        target_seats = {
            idx for idx, name in enumerate(seat_names) if name == target_agent_name
        }
        for step in bundle.get("steps", []):
            if step["actor_seat"] not in target_seats:
                continue
            was_bluff = step.get("pending_claim_was_bluff")
            pred_prob = step.get("decision_record", {}).get("belief_bluff_prob")
            if was_bluff is not None and isinstance(pred_prob, (int, float)):
                pairs.append((float(pred_prob), 1.0 if was_bluff else 0.0))

    if not pairs:
        return {"num_samples": 0, "ece": 0.0, "brier_score": 0.0, "bins": []}

    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    bins_data: list[dict[str, float]] = []
    ece = 0.0

    for b_idx in range(num_bins):
        low = b_idx / num_bins
        high = (b_idx + 1) / num_bins
        in_bin = [
            (p, y)
            for p, y in pairs
            if (low <= p < high) or (b_idx == num_bins - 1 and p == high)
        ]
        if in_bin:
            avg_conf = sum(p for p, _ in in_bin) / len(in_bin)
            emp_freq = sum(y for _, y in in_bin) / len(in_bin)
            weight = len(in_bin) / len(pairs)
            ece += weight * abs(avg_conf - emp_freq)
        else:
            avg_conf = (low + high) / 2.0
            emp_freq = avg_conf
        bins_data.append(
            {
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "count": len(in_bin),
                "mean_predicted_prob": round(avg_conf, 4),
                "empirical_bluff_freq": round(emp_freq, 4),
            }
        )

    return {
        "num_samples": len(pairs),
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "bins": bins_data,
    }

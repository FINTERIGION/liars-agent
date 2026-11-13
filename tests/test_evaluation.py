import random
import math
from copy import deepcopy
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from liars_agent.agents.base import BaseAgent
from liars_agent.engine.replay import verify_replay_determinism
from liars_agent.engine.types import AgentDecisionRecord
from liars_agent.evaluation.duplicate_runner import run_duplicate_evaluation
from liars_agent.evaluation.metrics import (
    compute_block_bootstrap_ci, compute_paired_bootstrap_ci, compute_probability_calibration,
    compute_paired_probability_metrics, compute_belief_calibration,
    compute_behavioral_profile, compute_risk_and_hand_stratified_diagnosis,
)
from liars_agent.evaluation.plots import (
    generate_fig1_winrate_svg, generate_fig2_history_ablation_svg,
    generate_fig3_risk_stratified_svg, generate_fig4_calibration_and_profile_svg,
)


class HistoryBlindAgent(BaseAgent):
    """A stochastic policy independent of the history under test."""
    def __init__(self, seed, name="history_blind"):
        super().__init__(name)
        self.seed = seed

    def select_action(self, obs):
        rng = random.Random(f"{self.seed}:{obs.round_id}:{obs.step_id}:{obs.observer_seat}")
        return rng.choice(obs.legal_actions), AgentDecisionRecord(
            completion_tokens=3, belief_completion_tokens_upper_bound=7,
        )


def blocks(rates):
    return [{"block_id": f"block_{index}", "block_seed": index, "win_rate": value,
             "opponent_specs": [{"id": str(i)} for i in range(3)], "opponent_order": [0, 1, 2],
             "rules_version": "test_rules"} for index, value in enumerate(rates)]


class EvaluationTests(unittest.TestCase):
    def test_bootstrap_constant_and_paired_results(self):
        ci = compute_block_bootstrap_ci([0.75] * 4, num_resamples=100)
        self.assertEqual(ci["mean_winrate"], 0.75)
        self.assertEqual(ci["ci_95_low"], 0.75)
        self.assertEqual(ci["ci_95_high"], 0.75)
        paired = compute_paired_bootstrap_ci(blocks([0.75] * 4), list(reversed(blocks([0.25] * 4))), num_resamples=100)
        self.assertEqual(paired["mean_diff"], 0.5)
        self.assertEqual(paired["ci_95_low"], 0.5)
        self.assertEqual(paired["ci_95_high"], 0.5)
        self.assertNotIn("p_value_one_sided", paired)
        with self.assertRaises(ValueError):
            compute_block_bootstrap_ci([])
        with self.assertRaises(ValueError):
            compute_paired_bootstrap_ci(blocks([1.0]), blocks([0.0, 1.0]))

    def test_ablation_preserves_pairing_and_logs_actual_observation(self):
        kwargs = dict(candidate_factory=HistoryBlindAgent, opponent_factories=[lambda seed, i=i: HistoryBlindAgent(seed, f"opponent_{i}") for i in range(3)],
                      num_blocks=4, base_seed=51, model_metadata={"backend": "test", "checkpoint": None},
                      num_resamples=67, bootstrap_seed=9)
        full = run_duplicate_evaluation(**kwargs, pool_name="main", ablation_mode="full")
        reduced = run_duplicate_evaluation(**kwargs, pool_name="ablation", ablation_mode="last_action_only")
        self.assertEqual(full.block_records, reduced.block_records)
        self.assertEqual(full.block_win_rates, reduced.block_win_rates)
        for a, b in zip(full.replays, reduced.replays):
            self.assertEqual([step["chosen_action"] for step in a.steps], [step["chosen_action"] for step in b.steps])
            self.assertEqual([step["round_resolution"] for step in a.steps], [step["round_resolution"] for step in b.steps])
            self.assertTrue(verify_replay_determinism(b))
        for block in reduced.block_records:
            self.assertEqual([game["candidate_seat"] for game in block["games"]], [0, 1, 2, 3])
        for i, replay in enumerate(reduced.replays):
            for step in replay.steps:
                if step["actor_seat"] == i % 4:
                    obs = step["observation"]
                    self.assertEqual(obs["ablation_mode"], "last_action_only")
                    self.assertFalse(obs["past_rounds_history"])
                    self.assertLessEqual(len(obs["current_round_history"]), 1)
        self.assertGreater(len({tuple(block["opponent_order"]) for block in full.block_records}), 1)
        for index, replay in enumerate(full.replays):
            candidate_seat = index % 4
            order = full.block_records[index // 4]["opponent_order"]
            for offset, opponent in enumerate(order, 1):
                self.assertEqual(replay.seat_agent_names[(candidate_seat + offset) % 4], f"opponent_{opponent}")
        with patch("liars_agent.evaluation.duplicate_runner.compute_block_bootstrap_ci", side_effect=AssertionError("CI recomputed")):
            exported = reduced.to_dict()
        self.assertEqual(exported["win_rate_ci"]["bootstrap"], {"unit": "block", "num_resamples": 67, "seed": 9})
        self.assertEqual(exported["model_metadata"]["backend"], "test")
        self.assertEqual(exported["total_games"], 16)
        self.assertEqual(exported["win_rate_ci"]["mean_winrate"], reduced.overall_win_rate)
        self.assertNotIn("replays", exported)
        self.assertGreater(exported["decision_metrics"]["total_decisions"], 0)
        decisions = exported["decision_metrics"]["total_decisions"]
        self.assertEqual(exported["decision_metrics"]["completion_tokens"], decisions * 3)
        self.assertEqual(exported["decision_metrics"]["belief_completion_tokens_upper_bound"], decisions * 7)

    def test_pairing_rejects_different_experiment_conditions(self):
        original = blocks([0.5, 0.25])
        for field, value in [("block_id", "other"), ("block_seed", 99),
                             ("opponent_specs", [{"id": "changed"}] * 3),
                             ("opponent_order", [1, 0, 2]), ("rules_version", "other_rules")]:
            changed = deepcopy(original)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compute_paired_bootstrap_ci(original, changed, num_resamples=50)
        with self.assertRaises(ValueError):
            compute_paired_bootstrap_ci(original, [original[0], original[0]])

    def test_probability_scores_and_empty_populations(self):
        scores = compute_probability_calibration([0.0, 1.0, 0.25, None], [0, 1, 1, 0])
        self.assertEqual(scores["num_samples"], 3)
        self.assertEqual(scores["coverage"], 0.75)
        self.assertAlmostEqual(scores["brier_score"], 0.1875)
        self.assertAlmostEqual(scores["log_loss"], -math.log(0.25) / 3)
        self.assertAlmostEqual(scores["ece"], 0.25)
        for entry in scores["bins"]:
            if entry["count"] == 0:
                self.assertIsNone(entry["mean_predicted_prob"])
                self.assertIsNone(entry["empirical_bluff_freq"])
        empty = compute_probability_calibration([], [])
        for key in ("ece", "brier_score", "log_loss", "coverage"):
            self.assertIsNone(empty[key])
        missing = compute_probability_calibration([None], [True])
        self.assertEqual(missing["coverage"], 0.0)
        self.assertIsNone(missing["brier_score"])
        self.assertTrue(math.isfinite(compute_probability_calibration([0.0, 1.0], [1, 0])["log_loss"]))
        for invalid in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                compute_probability_calibration([invalid], [1])

    def test_calibration_uses_native_source_even_if_action_falls_back(self):
        replay = {"seat_agent_names": ["model"], "steps": [
            {"actor_seat": 0, "pending_claim_was_bluff": True, "decision_record": {
                "belief_source": "heuristic", "belief_bluff_prob": 0.9, "was_fallback": True,
                "fallback_record": {"belief_bluff_prob": 0.99}}},
            {"actor_seat": 0, "pending_claim_was_bluff": True, "decision_record": {
                "belief_source": "model", "belief_bluff_prob": 0.2, "was_fallback": True,
                "fallback_record": {"belief_bluff_prob": 0.99}}},
        ]}
        scores = compute_belief_calibration([replay], "model")
        self.assertEqual(scores["num_candidates"], 2)
        self.assertEqual(scores["num_samples"], 1)
        self.assertEqual(scores["coverage"], 0.5)
        self.assertAlmostEqual(scores["brier_score"], 0.64)
        self.assertEqual(scores["source_counts"], {"heuristic": 1, "model": 1})
        self.assertEqual(scores["action_fallbacks"], 2)

    def test_probability_comparison_bootstraps_games_and_reports_coverage(self):
        a = [{"sample_id": f"s{i}", "game_id": "many" if i < 10 else "single",
              "label": True, "probability": 1.0 if i < 10 else 0.0} for i in range(11)]
        b = [{**row, "probability": 0.5} for row in a]
        result = compute_paired_probability_metrics(a, list(reversed(b)), num_resamples=500)
        self.assertEqual(result["bootstrap"]["unit"], "game")
        self.assertEqual(result["num_games"], 2)
        brier = result["paired_metrics"]["brier_score"]
        self.assertAlmostEqual(brier["mean_a"], 1 / 11)
        self.assertEqual((brier["ci_95_low"], brier["ci_95_high"]), (-0.25, 0.75))
        b[-1]["probability"] = None
        incomplete = compute_paired_probability_metrics(a, b, num_resamples=20)
        self.assertEqual(incomplete["num_samples"], 10)
        self.assertEqual(incomplete["model_b"]["coverage"], 10 / 11)
        self.assertEqual(incomplete["paired_metrics"]["brier_score"]["mean_a"], 0.0)
        b[0]["game_id"] = "changed"
        with self.assertRaises(ValueError):
            compute_paired_probability_metrics(a, b)

    def test_empty_metrics_and_plots_do_not_invent_zero_rates(self):
        profile = compute_behavioral_profile([], "missing")
        stratified = compute_risk_and_hand_stratified_diagnosis([], "missing")
        calibration = compute_probability_calibration([], [])
        for key in profile["denominators"]:
            self.assertIsNone(profile[key])
            self.assertEqual(profile["denominators"][key], 0)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [
                generate_fig2_history_ablation_svg([{"label": "empty", "win_rate": 0.25,
                    "challenge_hit_rate": None, "denominators": {"win_rate": 4, "challenge_hit_rate": 0}}], Path(tmp) / "f2.svg"),
                generate_fig3_risk_stratified_svg(stratified, Path(tmp) / "f3.svg"),
                generate_fig4_calibration_and_profile_svg(calibration, calibration,
                    {"zero_shot": profile}, Path(tmp) / "f4.svg"),
            ]
            for path in paths:
                svg = path.read_text()
                ElementTree.fromstring(svg)
                self.assertIn("N/A", svg)
                self.assertIn("n=0", svg)
                self.assertNotIn("nan", svg.lower())

    def test_plot_keeps_rates_above_fifty_percent(self):
        seen = [{"key": "a", "label": "Model & baseline", "win_rate": 0.8, "ci_95_low": 0.7, "ci_95_high": 0.95}]
        unseen = [{"key": "a", "label": "Model & baseline", "win_rate": 0.9, "ci_95_low": 0.8, "ci_95_high": 1.0}]
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_fig1_winrate_svg(seen, unseen, Path(tmp) / "fig.svg")
            svg = path.read_text()
            self.assertIn("80.0%", svg)
            self.assertIn("90.0%", svg)
            self.assertNotIn("随机均线", svg)
            ElementTree.fromstring(svg)

    def test_ablation_plot_draws_supplied_winrate_ci_only(self):
        record = {"label": "full", "win_rate": 0.8, "challenge_hit_rate": 0.9,
                  "denominators": {"win_rate": 40, "challenge_hit_rate": 20},
                  "win_rate_ci": {"ci_95_low": 0.65, "ci_95_high": 0.95}}
        with tempfile.TemporaryDirectory() as tmp:
            svg = generate_fig2_history_ablation_svg([record], Path(tmp) / "ci.svg").read_text()
            root = ElementTree.fromstring(svg)
            error_lines = [node for node in root if node.tag.endswith("line") and node.get("stroke-width") == "1.8"]
            self.assertEqual(len(error_lines), 3)
            vertical = [node for node in error_lines if node.get("x1") == node.get("x2")]
            self.assertEqual(len(vertical), 1)
            # The 0–100% plot runs from y=360 to y=75, so both supplied bounds
            # must appear above the 50% level (y=217.5), without clipping.
            self.assertEqual((vertical[0].get("y1"), vertical[0].get("y2")), ("90", "175"))
            self.assertIn("80.0%", svg)
            self.assertIn("90.0%", svg)
            no_ci = generate_fig2_history_ablation_svg(
                [{key: value for key, value in record.items() if key != "win_rate_ci"}],
                Path(tmp) / "no-ci.svg",
            ).read_text()
            self.assertNotIn('stroke-width="1.8"', no_ci)


if __name__ == "__main__":
    unittest.main()

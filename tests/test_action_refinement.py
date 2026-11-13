"""Paired adaptive teacher effort, reproducibility and persisted-score checks."""

from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from liars_agent.engine.observation import apply_history_ablation
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.types import CandidateAction
from liars_agent.training import action_refinement as refinement
from liars_agent.training.data_generator import (
    DEFAULT_TEACHER_CONFIG,
    evaluate_observation_action_rewards,
    evaluate_observation_action_values,
)


class TestAdaptiveActionRefinement(unittest.TestCase):
    def setUp(self):
        engine = LiarsBarEngine(game_seed=72)
        engine.step(engine.get_legal_actions()[0])
        self.full_obs = engine.get_observation()
        self.obs = replace(self.full_obs, legal_actions=self.full_obs.legal_actions[:2])
        self.a, self.b = (a.action_id for a in self.obs.legal_actions)

    def fake_rewards(self, rewards, calls):
        def score(obs, config, num_rollouts, rollout_seed, candidate_actions=None, *, start_index=0):
            calls.append((start_index, num_rollouts, rollout_seed))
            self.assertEqual(obs, self.obs)
            return {a.action_id: rewards[a.action_id][start_index:start_index + num_rollouts]
                    for a in obs.legal_actions}
        return score

    def run_scripted(self, a, b, force_refine=False, reference_action_id=None):
        calls = []
        with patch.object(refinement, "evaluate_observation_action_rewards",
                          side_effect=self.fake_rewards({self.a: a, self.b: b}, calls)):
            report = refinement.refine_action_observation(self.obs, DEFAULT_TEACHER_CONFIG, 317, force_refine,
                                                          reference_action_id)
        refinement.validate_action_refinement_report(self.obs, json.loads(json.dumps(report, allow_nan=False)))
        return report, calls

    def test_clear_screen_stops_after_eight(self):
        report, calls = self.run_scripted([1.0] * 64, [.65] * 64)
        self.assertEqual(calls, [(0, 8, 317)])
        self.assertEqual(report["rollouts_per_candidate"], 8)
        self.assertEqual(report["total_rollouts"], 16)
        self.assertEqual(report["chosen_action_id"], self.a)
        self.assertFalse(report["ambiguous"])
        self.assertTrue(report["meets_advantage_margin"])

    def test_force_refine_appends_only_twenty_four_then_stops(self):
        report, calls = self.run_scripted([1.0] * 64, [.65] * 64, force_refine=True)
        self.assertEqual(calls, [(0, 8, 317), (8, 24, 317)])
        self.assertEqual(report["rollouts_per_candidate"], 32)
        self.assertIn("forced_refinement", report["refinement_reason"])
        self.assertFalse(report["ambiguous"])

    def test_changed_reference_forces_thirty_two_despite_clear_screen_gap(self):
        report, calls = self.run_scripted([1.0] * 64, [.65] * 64, reference_action_id=self.b)
        self.assertEqual(calls, [(0, 8, 317), (8, 24, 317)])
        self.assertGreater(report["screen"]["best_vs_runnerup_gap"], .05)
        self.assertEqual(report["reference_action_id"], self.b)
        self.assertTrue(report["screen_label_changed"])
        self.assertIn("screen_label_changed", report["refinement_reason"])
        self.assertEqual(report["chosen_action_id"], self.a)
        self.assertEqual(report["rollouts_per_candidate"], 32)
        changed = copy.deepcopy(report)
        changed["screen_label_changed"] = False
        with self.assertRaisesRegex(ValueError, "screen_label_changed"):
            refinement.validate_action_refinement_report(self.obs, changed)

    def test_reference_in_screen_ties_is_not_reported_as_label_change(self):
        report, _ = self.run_scripted([.65] * 64, [.65] * 64, reference_action_id=self.a)
        self.assertIn(self.a, report["screen"]["tied_best_action_ids"])
        self.assertFalse(report["screen_label_changed"])
        self.assertNotIn("screen_label_changed", report["refinement_reason"])

    def test_illegal_reference_is_rejected_before_scoring(self):
        with patch.object(refinement, "evaluate_observation_action_rewards") as score:
            with self.assertRaisesRegex(ValueError, "reference_action_id"):
                refinement.refine_action_observation(self.obs, DEFAULT_TEACHER_CONFIG, 317,
                                                     reference_action_id="NOT_LEGAL")
            score.assert_not_called()
        report, _ = self.run_scripted([1.0] * 64, [.65] * 64)
        report["reference_action_id"] = "NOT_LEGAL"
        with self.assertRaisesRegex(ValueError, "reference_action_id"):
            refinement.validate_action_refinement_report(self.obs, report)

    def test_screen_tie_can_resolve_at_thirty_two_and_preserves_prefix(self):
        report, calls = self.run_scripted([1.0] * 64, [1.0] * 8 + [0.0] * 56)
        self.assertEqual(calls, [(0, 8, 317), (8, 24, 317)])
        self.assertEqual(report["screen"]["tied_best_action_ids"], sorted([self.a, self.b]))
        self.assertEqual(report["screen"]["candidate_rewards"][self.b], [1.0] * 8)
        self.assertEqual(report["final"]["candidate_rewards"][self.b], [1.0] * 8 + [0.0] * 24)
        self.assertEqual(report["final"]["best_vs_runnerup_gap"], .75)
        self.assertEqual([s["num_rollouts"] for s in report["stages"]], [8, 32])

    def test_persistent_tie_reaches_sixty_four_without_repeating_trajectories(self):
        report, calls = self.run_scripted([.65] * 64, [.65] * 64)
        self.assertEqual(calls, [(0, 8, 317), (8, 24, 317), (32, 32, 317)])
        indices = [i for start, count, _ in calls for i in range(start, start + count)]
        self.assertEqual(indices, list(range(64)))
        self.assertEqual(report["total_rollouts"], 128)
        self.assertTrue(report["ambiguous"])
        self.assertFalse(report["meets_advantage_margin"])
        self.assertEqual(report["reason"], "maximum_rollouts_ambiguous")
        expected = max(self.obs.legal_actions, key=lambda a: (a.action_type, a.cards))
        self.assertEqual(report["chosen_action_id"], expected.action_id)

    def test_small_gap_at_thirty_two_can_resolve_at_sixty_four(self):
        report, calls = self.run_scripted([1.0] * 64, [1.0] * 31 + [0.0] * 33)
        self.assertEqual(calls[-1], (32, 32, 317))
        self.assertEqual(report["stages"][1]["best_vs_runnerup_gap"], 1 / 32)
        self.assertFalse(report["final"]["ambiguous"])
        self.assertEqual(report["final"]["best_vs_runnerup_gap"], 33 / 64)

    def test_threshold_equality_expands_with_float_tolerance(self):
        # Mean difference is (.35 * 3 - .65) / 8 = .05, with float rounding.
        report, calls = self.run_scripted(([.35] * 3 + [0.0] * 5) * 8,
                                         ([.65] + [0.0] * 7) * 8)
        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(report["final"]["best_vs_runnerup_gap"], .05)
        self.assertTrue(report["ambiguous"])

    def test_clear_runnerup_gap_does_not_mask_low_mean_advantage(self):
        report, _ = self.run_scripted([.65] + [0.0] * 63, [0.0] * 64)
        self.assertGreater(report["final"]["best_vs_runnerup_gap"], .05)
        self.assertLess(report["final"]["best_minus_mean_advantage"], .05)
        self.assertFalse(report["ambiguous"])
        self.assertFalse(report["meets_advantage_margin"])

    def test_saved_summary_corruption_is_rejected(self):
        report, _ = self.run_scripted([.65] * 64, [.65] * 64)
        mutations = (
            lambda r: r["final"]["candidate_means"].__setitem__(self.a, .99),
            lambda r: r["screen"]["candidate_rewards"][self.a].__setitem__(0, 0.0),
            lambda r: r.__setitem__("chosen_action_id", "NOT_LEGAL"),
            lambda r: r.__setitem__("ambiguous", False),
            lambda r: r.__setitem__("total_rollouts", 999),
            lambda r: r.__setitem__("gap_threshold", .1),
            lambda r: r["stages"][1].__setitem__("num_rollouts", 16),
            lambda r: r.__setitem__("version", "old_version"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = copy.deepcopy(report)
                mutate(changed)
                with self.assertRaises(ValueError):
                    refinement.validate_action_refinement_report(self.obs, changed)

    def test_verifier_rejects_stopping_before_required_expansion(self):
        report, _ = self.run_scripted([.65] * 64, [.65] * 64)
        for action_id in report["final"]["candidate_rewards"]:
            report["final"]["candidate_rewards"][action_id] = [.65] * 32
        with self.assertRaisesRegex(ValueError, "required 64"):
            refinement.validate_action_refinement_report(self.obs, report)

    def test_segmented_real_rollouts_match_one_call_and_candidate_order(self):
        whole = evaluate_observation_action_rewards(self.obs, num_rollouts=8, rollout_seed=317)
        first = evaluate_observation_action_rewards(self.obs, num_rollouts=3, rollout_seed=317)
        rest = evaluate_observation_action_rewards(self.obs, num_rollouts=5, rollout_seed=317,
                                                    candidate_actions=self.obs.legal_actions[::-1], start_index=3)
        self.assertEqual(whole, {key: first[key] + rest[key] for key in first})
        values = evaluate_observation_action_values(self.obs, num_rollouts=8, rollout_seed=317)
        self.assertEqual(values, {key: sum(x / 8 for x in rewards) for key, rewards in whole.items()})

    def test_same_real_observation_and_seed_produce_identical_reports(self):
        first = refinement.refine_action_observation(self.obs, DEFAULT_TEACHER_CONFIG, 317)
        second = refinement.refine_action_observation(self.obs, DEFAULT_TEACHER_CONFIG, 317)
        self.assertEqual(first, second)
        refinement.validate_action_refinement_report(self.obs, first)

    def test_reward_api_rejects_unknown_actions_and_nonfull_observations(self):
        with self.assertRaisesRegex(ValueError, "distinct legal actions"):
            evaluate_observation_action_rewards(self.obs, candidate_actions=[CandidateAction("invalid", "PLAY", ("Q",))])
        with self.assertRaisesRegex(ValueError, "full observations"):
            refinement.refine_action_observation(apply_history_ablation(self.obs, "last_action_only"),
                                                  DEFAULT_TEACHER_CONFIG, 317)
        with self.assertRaisesRegex(ValueError, "start_index"):
            evaluate_observation_action_rewards(self.obs, start_index=-1)


if __name__ == "__main__":
    unittest.main()

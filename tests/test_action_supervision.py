import copy
import json
import unittest

from liars_agent.engine.state import LiarsBarEngine
from liars_agent.training.action_supervision import (
    make_action_supervision, resolve_action_supervision, select_action_completion,
    normalize_action_policy, mean_action_weight,
)
from liars_agent.training.data_generator import make_action_sample


class TestActionSupervision(unittest.TestCase):
    def setUp(self):
        obs = LiarsBarEngine(game_seed=23).get_observation()
        self.row = make_action_sample(obs, obs.legal_actions[0], "source")
        self.ids = [a.action_id for a in obs.legal_actions]

    def attach(self, scores):
        row = copy.deepcopy(self.row)
        row["action_supervision"] = make_action_supervision(row, scores)
        return row

    def test_ties_cycle_without_increasing_total_observation_weight(self):
        row = self.attach({self.ids[0]: .7, self.ids[1]: .7, self.ids[2]: .1})
        first = select_action_completion(row, 13, 0)
        second = select_action_completion(row, 13, 1)
        self.assertNotEqual(first, second)
        self.assertEqual(first, select_action_completion(row, 13, 2))
        self.assertEqual(first, select_action_completion(dict(row, sample_id="different_game_id"), 13, 0))
        self.assertEqual(resolve_action_supervision(row)["loss_weight"], 1)
        self.assertEqual({json.loads(v)["chosen_action_id"] for v in (first, second)}, set(self.ids[:2]))

    def test_small_positive_gap_downweights_but_never_creates_extra_targets(self):
        row = self.attach({self.ids[0]: .70, self.ids[1]: .68, self.ids[2]: .1})
        descriptor = resolve_action_supervision(row)
        self.assertEqual(descriptor["group"], "small_gap")
        self.assertEqual(descriptor["target_action_ids"], [self.ids[0]])
        self.assertEqual(descriptor["loss_weight"], .5)
        clear = self.attach({self.ids[0]: .9, self.ids[1]: .3})
        mean = mean_action_weight([row, clear])
        self.assertEqual(mean, .75)
        self.assertAlmostEqual(sum(resolve_action_supervision(r)["loss_weight"] / mean for r in (row, clear)), 2)

    def test_stale_or_malformed_targets_are_rejected(self):
        row = self.attach({self.ids[0]: .7, self.ids[1]: .7})
        for field, value in (("target_action_ids", ["A999"]), ("loss_weight", .1),
                             ("version", "old"), ("group", "clear")):
            changed = copy.deepcopy(row); changed["action_supervision"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_action_supervision(changed)
        with self.assertRaises(ValueError):
            resolve_action_supervision(dict(self.row, action_refinement={"version":"v2"}))
        with self.assertRaises(ValueError):
            normalize_action_policy({"positive_gap_weight": float("nan")})
        with self.assertRaises(ValueError):
            make_action_supervision(self.row, {self.ids[0]:.2,self.ids[1]:.8})

    def test_legacy_single_recorded_answer_is_explicitly_unscored(self):
        descriptor = resolve_action_supervision(self.row)
        self.assertEqual(descriptor["group"], "unscored")
        self.assertIsNone(descriptor["candidate_values"])
        self.assertEqual(descriptor["target_action_ids"], [self.ids[0]])

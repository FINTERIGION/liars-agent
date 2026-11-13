"""Teacher information boundaries and honest action/belief supervision contracts."""

from __future__ import annotations

import copy
import json
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from liars_agent.agents.base import describe_action, public_opponent_profile
from liars_agent.agents.classifier_bot import SmallClassifierBot
from liars_agent.agents.prompt_builder import validate_action_decision
from liars_agent.agents.rule_bots import (
    AggressiveBlufferBot,
    CardCounterBot,
    ChameleonAdaptiveBot,
    HonestConservativeBot,
    ParanoidHunterBot,
)
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.types import CandidateAction, PublicEvent, RoundResolution
from liars_agent.training import data_generator as data


RULE_CLASSES = (HonestConservativeBot, AggressiveBlufferBot, CardCounterBot,
                ParanoidHunterBot, ChameleonAdaptiveBot)


def opening_observation():
    engine = LiarsBarEngine(game_id="PRIVATE_SEED_998877", game_seed=72)
    return engine, engine.get_observation()


def pending_observation():
    engine, obs = opening_observation()
    engine.step(obs.legal_actions[0])
    return engine, engine.get_observation()


class TestTeacherBoundaries(unittest.TestCase):
    def test_paired_values_are_invariant_to_candidate_order_and_subset(self):
        _, obs = pending_observation()
        actions = obs.legal_actions[:3]
        values = data.evaluate_observation_action_values(
            obs, num_rollouts=2, rollout_seed=91, candidate_actions=actions)
        reversed_values = data.evaluate_observation_action_values(
            obs, num_rollouts=2, rollout_seed=91, candidate_actions=actions[::-1])
        singleton = data.evaluate_observation_action_values(
            obs, num_rollouts=2, rollout_seed=91, candidate_actions=actions[1:2])
        self.assertEqual(values, reversed_values)
        self.assertEqual(singleton, {actions[1].action_id: values[actions[1].action_id]})

    def test_actual_game_agents_cannot_be_supplied_to_teacher(self):
        _, obs = pending_observation()
        agents = [CardCounterBot(seed=seat) for seat in range(4)]
        with patch.object(CardCounterBot, "select_action", side_effect=AssertionError("actual policy called")):
            with self.assertRaisesRegex(TypeError, "never actual-game agent"):
                data.evaluate_observation_action_values(obs, agents, num_rollouts=1)
        invalid = copy.deepcopy(data.DEFAULT_TEACHER_CONFIG)
        invalid["actor"]["class"] = "AggressiveBlufferBot"
        with self.assertRaisesRegex(ValueError, "continuation actor must be CardCounterBot"):
            data.build_rollout_agents(invalid, obs.observer_seat, 3)

    def test_candidates_share_fresh_policy_seeds_within_each_trajectory(self):
        _, obs = pending_observation()
        runs = []
        state_seeds = []
        original_sample = data.sample_rollout_state

        def sample(observation, seed):
            state_seeds.append(seed)
            return original_sample(observation, seed)

        def score(simulation, actor_seat, action, agents):
            runs.append((tuple(type(a) for a in agents), tuple(a.seed for a in agents)))
            self.assertIsInstance(agents[actor_seat], CardCounterBot)
            self.assertEqual({type(a) for seat, a in enumerate(agents) if seat != actor_seat},
                             {HonestConservativeBot, AggressiveBlufferBot, CardCounterBot})
            return 0.5

        with patch.object(data, "sample_rollout_state", side_effect=sample), \
                patch.object(data, "_score_continuation", side_effect=score):
            data.evaluate_observation_action_values(obs, num_rollouts=3, rollout_seed=9,
                                                    candidate_actions=obs.legal_actions[:2])
        self.assertEqual(len(runs), 6)
        for index in range(0, 6, 2):
            self.assertEqual(runs[index], runs[index + 1])
            self.assertEqual(len(set(runs[index][1])), 4)
        self.assertEqual(len({runs[index][1] for index in (0, 2, 4)}), 3)
        self.assertTrue(set(state_seeds).isdisjoint(seed for _, seeds in runs for seed in seeds))

    def test_hidden_world_and_private_identifier_do_not_change_teacher(self):
        engine, obs = pending_observation()
        candidates = obs.legal_actions[:2]
        before = data.evaluate_observation_action_values(
            obs, num_rollouts=1, rollout_seed=19, candidate_actions=candidates)
        others = [seat for seat in range(4) if seat != obs.observer_seat]
        left, right = others[:2]
        # Preserve public hand counts while changing private assignments.
        engine.hands[left][0], engine.hands[right][0] = engine.hands[right][0], engine.hands[left][0]
        engine.fatal_chamber_slot = [6, 5, 4, 3]
        engine.game_id, engine.game_seed = "CHANGED_PRIVATE_IDENTIFIER", 123456789
        after_obs = engine.get_observation()
        self.assertEqual(obs, after_obs)
        self.assertEqual(before, data.evaluate_observation_action_values(
            after_obs, num_rollouts=1, rollout_seed=19, candidate_actions=candidates))


class TestSupervisionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replays = [r.to_dict() for r in data.generate_rule_replays(2, base_seed=173,
            opponent_factories=[lambda seed: AggressiveBlufferBot(seed=seed)])]

    def test_every_pending_claim_can_supply_truth_independently_of_challenges(self):
        rows = data.extract_belief_samples(self.replays, 100000, seed=7)
        points = {(r["game_id"], s["observation"]["round_id"], s["observation"]["step_id"]): s
                  for r in self.replays for s in r["steps"]
                  if s["pending_claim_was_bluff"] is not None}
        self.assertEqual(len(rows), len(points))
        self.assertTrue(any(s["chosen_action"]["action_type"] == "PLAY" for s in points.values()))
        self.assertEqual({row["completion"] for row in rows}, {"0", "1"})
        for row in rows:
            truth = points[row["game_id"], row["round_id"], row["step_id"]]["pending_claim_was_bluff"]
            self.assertEqual(row["task"], "belief")
            self.assertEqual(row["completion"], str(int(truth)))
            self.assertEqual(row["label"], int(truth))
            self.assertEqual(row["source"], "engine_pending_claim_truth")

    def test_heuristic_probability_style_and_rationale_never_become_belief_targets(self):
        changed = copy.deepcopy(self.replays)
        for bundle in changed:
            for step in bundle["steps"]:
                step["decision_record"] = {
                    "belief_bluff_prob": 1.0 if not step["pending_claim_was_bluff"] else 0.0,
                    "estimated_opponent_style": "FABRICATED_PRIVATE_IDENTITY",
                    "rationale": "FABRICATED_TEACHER_REASON",
                }
        self.assertEqual(data.extract_belief_samples(self.replays, 20, seed=14),
                         data.extract_belief_samples(changed, 20, seed=14))

    def test_action_and_belief_texts_exclude_private_ids_and_auxiliary_labels(self):
        engine, obs = pending_observation()
        action_row = data.make_action_sample(obs, obs.legal_actions[0], engine.game_id)
        self.assertEqual(json.loads(action_row["completion"]),
                         {"chosen_action_id": obs.legal_actions[0].action_id})
        rows = [action_row, *data.extract_belief_samples(self.replays, 4)]
        for row in rows:
            visible = json.dumps({k: row[k] for k in ("system", "prompt", "completion", "observation")})
            self.assertNotIn(row["game_id"], visible)
            self.assertNotIn("game_seed", visible)
            self.assertNotIn("game_id", visible)
            self.assertNotIn("belief_bluff_prob", row["completion"])
            self.assertNotIn("estimated_opponent_style", row["completion"])
            self.assertNotIn("rationale", row["completion"])
        with self.assertRaises(ValueError):
            validate_action_decision(obs, json.dumps({"chosen_action_id": obs.legal_actions[0].action_id,
                                                      "belief_bluff_prob": .9}))

    def test_nonboolean_truth_is_rejected_and_classifier_requires_binary_task(self):
        changed = copy.deepcopy(self.replays)
        for bundle in changed:
            for step in bundle["steps"]:
                if step["pending_claim_was_bluff"] is not None:
                    step["pending_claim_was_bluff"] = 0.4
        with self.assertRaisesRegex(ValueError, "actual boolean engine truth"):
            data.extract_belief_samples(changed, 1)
        _, obs = pending_observation()
        action_row = data.make_action_sample(obs, obs.legal_actions[0], "private")
        with self.assertRaisesRegex(ValueError, "true binary belief samples"):
            SmallClassifierBot().fit_from_belief_samples([action_row], epochs=1)


class TestBeliefAndExplanationSemantics(unittest.TestCase):
    def test_no_pending_claim_is_null_for_every_baseline(self):
        _, obs = opening_observation()
        # Ensure the classifier enters its normal truthful PLAY branch.
        self.assertTrue(any(all(c in (obs.target_rank, "JOKER") for c in a.cards)
                            for a in obs.legal_actions))
        for bot in [*(cls() for cls in RULE_CLASSES), SmallClassifierBot()]:
            with self.subTest(bot=type(bot).__name__):
                action, record = bot.select_action(obs)
                self.assertIn(action, obs.legal_actions)
                self.assertIsNone(record.belief_bluff_prob)
                self.assertEqual(record.belief_source, "none")
                self.assertEqual(record.estimated_opponent_style, "unknown")

    def test_pending_scores_are_finite_and_explicitly_sourced(self):
        _, obs = pending_observation()
        for bot in [*(cls() for cls in RULE_CLASSES), SmallClassifierBot(),
                    SmallClassifierBot(bluff_weights=[1e6] * 12),
                    SmallClassifierBot(bluff_weights=[-1e6] * 12)]:
            with self.subTest(bot=type(bot).__name__):
                _, record = bot.select_action(obs)
                self.assertTrue(math.isfinite(record.belief_bluff_prob))
                self.assertGreaterEqual(record.belief_bluff_prob, 0)
                self.assertLessEqual(record.belief_bluff_prob, 1)
                self.assertEqual(record.belief_source,
                                 "classifier" if isinstance(bot, SmallClassifierBot) else "heuristic")
                self.assertEqual(record.estimated_opponent_style, "unknown")

    def test_forced_challenge_reason_does_not_claim_threshold_evidence(self):
        engine, _ = opening_observation()
        engine.alive = [True, True, False, False]
        engine.hands = [["Q"], ["K", "A"], [], []]
        engine.current_turn_seat = 0
        engine.target_rank = "Q"
        engine.step(CandidateAction("custom", "PLAY", ("Q",)))
        obs = engine.get_observation()
        self.assertEqual(len(obs.legal_actions), 1)
        for bot in [*(cls() for cls in RULE_CLASSES), SmallClassifierBot(bluff_weights=[-100] * 12)]:
            with self.subTest(bot=type(bot).__name__):
                action, record = bot.select_action(obs)
                self.assertEqual(action.action_type, "CHALLENGE")
                self.assertEqual(record.rationale, describe_action(obs, action))
                self.assertIn("必须质疑", record.rationale)
                self.assertNotIn("阈值", record.rationale)

    def test_public_profile_reports_selected_reveals_and_voluntary_denominators(self):
        _, obs = opening_observation()
        events = (
            PublicEvent(1, 1, 1, "PLAY", 1, "Q", 4),
            PublicEvent(1, 5, 1, "PLAY", 3, "Q", 1),
            PublicEvent(2, 2, 1, "CHALLENGE"),
            PublicEvent(3, 2, 1, "CHALLENGE", forced_challenge=True),
        )
        resolutions = tuple(RoundResolution(
            round_id=index + 1, target_rank="Q", claimer_seat=1, challenger_seat=2,
            claimed_count=1, revealed_cards=("K" if bluff else "Q",), was_bluff=bluff,
            challenge_succeeded=bluff, penalized_seat=1 if bluff else 2,
            shots_taken_before=0, shots_taken_after=1, lethal_shot=False,
        ) for index, bluff in enumerate((True, False)))
        obs = replace(obs, past_public_actions=events, past_rounds_history=resolutions)
        profile = public_opponent_profile(obs)
        seat = profile["1"]
        self.assertEqual((seat["plays"], seat["cards_played"], seat["mean_play_count"]), (2, 4, 2))
        self.assertEqual((seat["voluntary_challenges"], seat["challenge_opportunities"]), (1, 2))
        self.assertEqual(seat["voluntary_challenge_rate"], .5)
        self.assertEqual((seat["reveals"], seat["revealed_bluffs"], seat["revealed_bluff_rate"]), (2, 1, .5))
        self.assertIsNone(profile["0"]["revealed_bluff_rate"])
        self.assertIsNone(profile["0"]["voluntary_challenge_rate"])


if __name__ == "__main__":
    unittest.main()

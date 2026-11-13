"""Information-set teacher, game partitions and learned baseline contracts."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from liars_agent.agents.classifier_bot import SmallClassifierBot, extract_observation_features
from liars_agent.agents.rule_bots import CardCounterBot
from liars_agent.engine.observation import PlayerObservation, apply_history_ablation
from liars_agent.engine.replay import verify_replay_determinism
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.types import CandidateAction, STANDARD_DECK_COUNTS
from liars_agent.training.data_generator import (
    _score_continuation, evaluate_observation_action_values, DEFAULT_TEACHER_CONFIG,
    generate_mixed_opponent_sft_dataset, sample_rollout_state,
)
from liars_agent.training.dataset_splitter import split_dataset_by_game_id, split_game_ids


class TestObservationRollouts(unittest.TestCase):
    def setUp(self):
        self.engine = LiarsBarEngine(game_seed=72)
        self.engine.step(self.engine.get_legal_actions()[0])
        self.obs = self.engine.get_observation()
        self.teacher = DEFAULT_TEACHER_CONFIG

    def test_observation_roundtrip_and_sampled_state_preserve_visible_facts(self):
        self.assertEqual(self.obs, PlayerObservation.from_dict(self.obs.to_dict()))
        for mode in ("full", "current_round_only", "last_action_only"):
            obs = apply_history_ablation(self.obs, mode)
            sim = sample_rollout_state(obs, 10)
            self.assertEqual(apply_history_ablation(sim.get_observation(), mode), obs)
            cards = [c for hand in sim.hands + sim.played_cards_this_round for c in hand]
            self.assertEqual(Counter(cards), Counter(STANDARD_DECK_COUNTS))
            for seat, status in enumerate(obs.players_public_status):
                self.assertGreater(sim.fatal_chamber_slot[seat], status.shots_taken)
            self.assertFalse(sim.record_truth)

    def test_hidden_engine_truth_cannot_change_teacher_values(self):
        action = self.obs.legal_actions[0]
        seat = self.obs.observer_seat
        before = evaluate_observation_action_values(self.engine.get_observation(seat), self.teacher, 2, 99, [action])[action.action_id]
        unknown_seats = [s for s in range(4) if s != seat and self.engine.hands[s]]
        left, right = unknown_seats[:2]
        self.engine.hands[left][0], self.engine.hands[right][0] = self.engine.hands[right][0], self.engine.hands[left][0]
        self.engine.fatal_chamber_slot = [6] * 4
        self.engine.pending_play.actual_cards = tuple("JOKER" for _ in self.engine.pending_play.actual_cards)
        self.assertEqual(self.obs, self.engine.get_observation())
        self.assertEqual(before, evaluate_observation_action_values(self.engine.get_observation(seat), self.teacher, 2, 99, [action])[action.action_id])

    def test_candidate_order_does_not_change_paired_values(self):
        actions = self.obs.legal_actions[:3]
        first = evaluate_observation_action_values(self.obs, self.teacher, 2, 99, actions)
        reverse = evaluate_observation_action_values(self.obs, self.teacher, 2, 99, list(reversed(actions)))
        self.assertEqual(first, reverse)

    def test_survived_chambers_and_eliminated_seats(self):
        self.engine.alive[3] = False
        self.engine.hands[3] = []
        self.engine.shots_taken = [3, 1, 2, 4]
        self.engine.current_turn_seat = 0
        self.engine.pending_play = None
        # Start a proper new round after the elimination.
        self.engine._start_new_round(0)
        obs = self.engine.get_observation()
        for seed in range(8):
            sim = sample_rollout_state(obs, seed)
            self.assertEqual(sim.get_observation(), obs)
            self.assertEqual(sim.fatal_chamber_slot[3], 4)
            self.assertTrue(all(sim.fatal_chamber_slot[s] > sim.shots_taken[s] for s in range(3)))

    def test_reward_stops_penalty_measurement_at_first_round_but_finishes_game(self):
        class ScriptedSimulation:
            current_turn_seat = 1
            winner_seat = 0
            calls = 0
            def step(self, action, record=None):
                self.calls += 1
                # Actor 0 is penalized later, but survives the current round.
                return SimpleNamespace(penalized_seat=1 if self.calls == 2 else 0) if self.calls >= 2 else None
            def is_game_over(self):
                return self.calls >= 125
            def get_observation(self, seat):
                return None
        agent = SimpleNamespace(select_action=lambda obs: (None, None))
        sim = ScriptedSimulation()
        self.assertEqual(_score_continuation(sim, 0, None, [agent] * 4, "full"), 1.0)
        self.assertEqual(sim.calls, 125)

    def test_generation_covers_complete_games_and_keeps_actual_samples(self):
        samples, replays = generate_mixed_opponent_sft_dataset(
            num_games=3, target_samples=6, num_rollouts_per_action=1,
            min_advantage_margin=0, base_seed=19,
            opponent_factories=[lambda seed: CardCounterBot(seed=seed)],
        )
        self.assertEqual(len(samples), 6)
        self.assertEqual(sorted(Counter(row["game_id"] for row in samples).values()), [2, 2, 2])
        self.assertTrue(all(verify_replay_determinism(replay) for replay in replays))
        self.assertTrue(all("observation" in row and "train_19_" in row["game_id"] for row in samples))


class TestTrainingPartitionsAndClassifier(unittest.TestCase):
    def test_shared_partition_and_zero_validation(self):
        ids = [f"game_{index}" for index in range(10)]
        partition = split_game_ids(ids, val_ratio=0, seed=9)
        samples = [{"game_id": gid} for gid in ids for _ in range(2)]
        splits = split_dataset_by_game_id(samples, game_splits=partition)
        self.assertEqual(splits["val"], [])
        for name in partition:
            self.assertEqual({row["game_id"] for row in splits[name]}, set(partition[name]))
        self.assertEqual(sum(map(len, splits.values())), 20)
        with self.assertRaises(ValueError):
            split_dataset_by_game_id(samples, game_splits={"train": ids, "val": [ids[0]], "test": []})

    def test_classifier_uses_live_features_and_weight_roundtrip(self):
        engine = LiarsBarEngine(game_seed=2)
        engine.step(engine.get_legal_actions()[0])
        engine.shots_taken[engine.current_turn_seat] = 5
        obs = engine.get_observation()
        self.assertEqual(extract_observation_features(obs)[3], 1.0)
        bot = SmallClassifierBot()
        with patch("liars_agent.agents.classifier_bot.extract_observation_features", wraps=extract_observation_features) as features:
            stats = bot.fit_from_replays([{"steps": [{"observation": obs.to_dict(), "pending_claim_was_bluff": True}]}], epochs=1)
            self.assertEqual(features.call_args.args[0], obs)
        self.assertEqual(stats["num_samples"], 1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            bot.save_weights(path)
            restored = SmallClassifierBot.load_weights(path)
            self.assertEqual(bot.select_action(obs), restored.select_action(obs))


if __name__ == "__main__":
    unittest.main()

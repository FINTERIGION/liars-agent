"""测试集 3：随机种子确定性、JSONL 回放一致性与按对局划分防泄漏测试."""

from __future__ import annotations

import tempfile
import copy
import unittest
from pathlib import Path

from liars_agent.agents.rule_bots import (
    AggressiveBlufferBot,
    CardCounterBot,
    HonestConservativeBot,
)
from liars_agent.engine.replay import (
    GameReplayBundle,
    ReplayVerificationError,
    load_replays_jsonl,
    save_replays_jsonl,
    verify_replay_determinism,
)
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.observation import apply_history_ablation
from liars_agent.training.dataset_splitter import split_dataset_by_game_id


class TestReplayAndDeterminism(unittest.TestCase):
    """验证相同种子与智能体配置必产生 100% 相同的轨迹指纹，且回放校验器全部通过."""

    def test_historical_samples_are_rejected_before_training(self) -> None:
        historical = {"game_id": "old_game", "historical_example": True}
        with self.assertRaisesRegex(ValueError, "Historical examples"):
            split_dataset_by_game_id([historical])

    def _run_single_game(self, seed: int) -> GameReplayBundle:
        agents = [
            CardCounterBot(name="rule_counter", seed=seed),
            HonestConservativeBot(seed=seed + 1),
            AggressiveBlufferBot(seed=seed + 2),
            CardCounterBot(seed=seed + 3),
        ]
        engine = LiarsBarEngine(game_id=f"det_game_{seed}", game_seed=seed)
        while not engine.is_game_over():
            turn = engine.current_turn_seat
            obs = engine.get_observation(turn)
            act, rec = agents[turn].select_action(obs)
            engine.step(act, rec)
        return GameReplayBundle.from_engine(engine, [a.name for a in agents])

    def test_identical_seeds_produce_identical_trajectory_hashes(self) -> None:
        b1 = self._run_single_game(seed=8888)
        b2 = self._run_single_game(seed=8888)
        self.assertEqual(b1.compute_trajectory_hash(), b2.compute_trajectory_hash())
        self.assertTrue(verify_replay_determinism(b1))

    def test_jsonl_serialization_and_replay_verification(self) -> None:
        bundles = [self._run_single_game(seed=400 + i) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "replays.jsonl"
            save_replays_jsonl(bundles, jsonl_path)
            loaded = load_replays_jsonl(jsonl_path)
            self.assertEqual(len(loaded), 5)
            for item in loaded:
                self.assertTrue(verify_replay_determinism(item))

    def test_verifier_checks_labels_hands_history_and_resolution(self) -> None:
        original = self._run_single_game(seed=72).to_dict()
        pending_index = next(i for i, step in enumerate(original["steps"])
                             if step["pending_claim_was_bluff"] is not None)
        mutations = {
            "pending_claim_was_bluff": lambda d: d["steps"][pending_index].__setitem__(
                "pending_claim_was_bluff", not d["steps"][pending_index]["pending_claim_was_bluff"]),
            "ground_truth_hands": lambda d: d["steps"][0]["ground_truth_hands"].__setitem__(1, ["BROKEN"]),
            "current_round_history": lambda d: d["steps"][pending_index]["observation"].__setitem__("current_round_history", []),
            "rounds_summary": lambda d: d["rounds_summary"][0].__setitem__("was_bluff", not d["rounds_summary"][0]["was_bluff"]),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                mutate(changed)
                self.assertFalse(verify_replay_determinism(changed))
                with self.assertRaises(ReplayVerificationError) as caught:
                    verify_replay_determinism(changed, raise_on_error=True)
                self.assertIn(field, caught.exception.field)

    def test_verifier_rejects_rehashed_corruption_and_changed_decision_metadata(self) -> None:
        bundle = self._run_single_game(seed=73)
        bundle.steps[0]["ground_truth_fatal_chambers"][0] = 99
        # Recomputing the fingerprint must not disguise a deterministic-state error.
        self.assertFalse(verify_replay_determinism(bundle.to_dict()))
        data = self._run_single_game(seed=74).to_dict()
        data["steps"][0]["decision_record"]["rationale"] = "changed recorded explanation"
        with self.assertRaises(ReplayVerificationError) as caught:
            verify_replay_determinism(data, raise_on_error=True)
        self.assertEqual(caught.exception.field, "trajectory_hash")

    def test_actual_ablated_observations_roundtrip(self) -> None:
        for mode in ("current_round_only", "last_action_only"):
            engine = LiarsBarEngine(game_seed=48)
            while not engine.is_game_over():
                observation = apply_history_ablation(engine.get_observation(), mode)
                engine.step(observation.legal_actions[0])
                engine.step_truth_logs[-1].observation = observation.to_dict()
            replay = GameReplayBundle.from_engine(engine)
            self.assertTrue(verify_replay_determinism(replay, raise_on_error=True))

    def test_missing_or_old_replay_versions_are_rejected(self) -> None:
        data = self._run_single_game(seed=44).to_dict()
        for field in ("rules_version", "observation_version"):
            invalid = {**data, field: "old"}
            with self.assertRaises(ReplayVerificationError) as caught:
                verify_replay_determinism(invalid, raise_on_error=True)
            self.assertEqual(caught.exception.field, field)

    def test_game_level_dataset_split_has_zero_leakage(self) -> None:
        fake_samples = [
            {"sample_id": f"g{g}_s{s}", "game_id": f"game_{g:03d}"}
            for g in range(20)
            for s in range(5)
        ]
        splits = split_dataset_by_game_id(fake_samples, train_ratio=0.8, val_ratio=0.1, seed=42)
        train_games = {x["game_id"] for x in splits["train"]}
        val_games = {x["game_id"] for x in splits["val"]}
        test_games = {x["game_id"] for x in splits["test"]}

        self.assertEqual(len(train_games), 16)
        self.assertEqual(len(val_games), 2)
        self.assertEqual(len(test_games), 2)
        self.assertTrue(train_games.isdisjoint(val_games))
        self.assertTrue(train_games.isdisjoint(test_games))
        self.assertTrue(val_games.isdisjoint(test_games))


if __name__ == "__main__":
    unittest.main()

"""测试集 3：随机种子确定性、JSONL 回放一致性与按对局划分防泄漏测试."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from liars_agent.agents.llm_agent import LLMAgent
from liars_agent.agents.rule_bots import (
    AggressiveBlufferBot,
    CardCounterBot,
    HonestConservativeBot,
)
from liars_agent.engine.replay import (
    GameReplayBundle,
    load_replays_jsonl,
    save_replays_jsonl,
    verify_replay_determinism,
)
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.training.dataset_splitter import split_dataset_by_game_id


class TestReplayAndDeterminism(unittest.TestCase):
    """验证相同种子与智能体配置必产生 100% 相同的轨迹指纹，且回放校验器全部通过."""

    def _run_single_game(self, seed: int) -> GameReplayBundle:
        agents = [
            LLMAgent(name="qwen35_9b_lora_sft", mode="lora_sft", seed=seed),
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

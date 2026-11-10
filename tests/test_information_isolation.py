"""测试集 2：严格观测信息隔离与历史消融检查 (Information Isolation & Ablation)."""

from __future__ import annotations

import json
import unittest

from liars_agent.agents.prompt_builder import build_user_prompt
from liars_agent.engine.observation import apply_history_ablation
from liars_agent.engine.state import LiarsBarEngine


class TestInformationIsolation(unittest.TestCase):
    """验证任何座位的 `PlayerObservation` 与模型 Prompt 绝不泄露其他玩家底牌、未翻开暗牌或实弹位置."""

    def test_observation_never_contains_hidden_truth(self) -> None:
        engine = LiarsBarEngine(game_id="test_isolation", game_seed=2026)
        first_seat = engine.current_turn_seat
        first_legal = engine.get_legal_actions(first_seat)
        play_act = [a for a in first_legal if a.action_type == "PLAY"][0]
        engine.step(play_act)

        second_seat = engine.current_turn_seat
        obs = engine.get_observation(second_seat)
        obs_dict = obs.to_dict()
        serialized = json.dumps(obs_dict, ensure_ascii=False)

        # 1. 验证观测字典中不包含实弹位置字段或全局手牌字典
        self.assertNotIn("fatal_chamber", serialized)
        self.assertNotIn("ground_truth", serialized)
        self.assertNotIn("actual_cards", serialized)

        # 2. 验证观测者仅能看到自己的 5 张手牌，看不到上家暗置打出的具体牌面
        self.assertEqual(len(obs.own_hand), 5)
        self.assertIsNotNone(obs.last_claim)
        assert obs.last_claim is not None
        self.assertEqual(obs.last_claim.claimer_seat, first_seat)
        self.assertEqual(obs.last_claim.claimed_count, len(play_act.cards))
        self.assertNotIn("cards", obs.last_claim.to_dict())

        # 3. 验证构造出的 LLM 用户 Prompt 也不包含任何隐藏真值
        prompt_str = build_user_prompt(obs)
        self.assertIn(f"你的座位: Seat {second_seat}", prompt_str)
        self.assertNotIn("fatal_chamber", prompt_str)

    def test_history_ablation_modes_strip_target_history(self) -> None:
        engine = LiarsBarEngine(game_id="test_ablation", game_seed=31415)
        # 推进到至少产生一次质疑结算进入第 2 小局
        while engine.round_id == 1 and not engine.is_game_over():
            legal = engine.get_legal_actions()
            challenges = [a for a in legal if a.action_type == "CHALLENGE"]
            engine.step(challenges[0] if challenges else legal[0])

        # 再走一步出牌使本小局有动作历史
        if not engine.is_game_over():
            engine.step(engine.get_legal_actions()[0])

        obs_full = engine.get_observation()
        self.assertGreaterEqual(len(obs_full.past_rounds_history), 1)
        self.assertGreaterEqual(len(obs_full.current_round_history), 1)

        obs_round_only = apply_history_ablation(obs_full, "current_round_only")
        self.assertEqual(len(obs_round_only.past_rounds_history), 0)
        self.assertEqual(
            len(obs_round_only.current_round_history),
            len(obs_full.current_round_history),
        )

        obs_last_only = apply_history_ablation(obs_full, "last_action_only")
        self.assertEqual(len(obs_last_only.past_rounds_history), 0)
        self.assertLessEqual(len(obs_last_only.current_round_history), 1)
        self.assertEqual(len(obs_last_only.own_played_cards_this_round), 0)


if __name__ == "__main__":
    unittest.main()

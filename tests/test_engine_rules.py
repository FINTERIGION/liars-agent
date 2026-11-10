"""测试集 1：研究用骗子牌规则与边界用例验证 (Rules & Boundary Cases)."""

from __future__ import annotations

import unittest

from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.types import CandidateAction, are_cards_truthful, is_card_truthful


class TestEngineRulesAndBoundaries(unittest.TestCase):
    """验证 20 张标准牌组、JOKER 万能牌、左轮 1/6 递增致死率与强制质疑边界规则."""

    def test_deck_composition_and_initial_deal(self) -> None:
        engine = LiarsBarEngine(game_id="test_deck", game_seed=12345)
        self.assertEqual(engine.num_players, 4)
        for seat in range(4):
            self.assertEqual(len(engine.hands[seat]), 5)
            self.assertTrue(engine.alive[seat])
            self.assertEqual(engine.shots_taken[seat], 0)
            self.assertGreaterEqual(engine.fatal_chamber_slot[seat], 1)
            self.assertLessEqual(engine.fatal_chamber_slot[seat], 6)

        all_dealt = [c for h in engine.hands for c in h]
        self.assertEqual(len(all_dealt), 20)
        self.assertEqual(all_dealt.count("Q"), 6)
        self.assertEqual(all_dealt.count("K"), 6)
        self.assertEqual(all_dealt.count("A"), 6)
        self.assertEqual(all_dealt.count("JOKER"), 2)

    def test_joker_counts_as_truthful_for_all_targets(self) -> None:
        for target in ("Q", "K", "A"):
            self.assertTrue(is_card_truthful("JOKER", target))
            self.assertTrue(is_card_truthful(target, target))
            self.assertTrue(are_cards_truthful((target, "JOKER"), target))
        self.assertFalse(are_cards_truthful(("Q", "K"), "Q"))

    def test_first_player_cannot_challenge_and_play_size_bounded(self) -> None:
        engine = LiarsBarEngine(game_id="test_first_turn", game_seed=99)
        legal = engine.get_legal_actions()
        self.assertGreater(len(legal), 0)
        for act in legal:
            self.assertEqual(act.action_type, "PLAY")
            self.assertGreaterEqual(len(act.cards), 1)
            self.assertLessEqual(len(act.cards), 3)

    def test_forced_challenge_when_all_other_alive_players_empty(self) -> None:
        engine = LiarsBarEngine(game_id="test_forced_challenge", game_seed=777)
        # 模拟仅剩 Seat 0 与 Seat 1 存活，且 Seat 0 打空最后的手牌
        engine.alive = [True, True, False, False]
        engine.hands[0] = ["Q", "JOKER"]
        engine.hands[1] = ["K", "A"]
        engine.hands[2] = []
        engine.hands[3] = []
        engine.current_turn_seat = 0
        engine.target_rank = "Q"

        # Seat 0 一次性打出全部 2 张手牌，此时 Seat 0 手牌变为 0，场上仅 Seat 1 有牌
        play_all = CandidateAction(action_id="custom", action_type="PLAY", cards=("Q", "JOKER"))
        res = engine.step(play_all)
        self.assertIsNone(res)
        self.assertEqual(engine.current_turn_seat, 1)

        # 此时除 Seat 1 外其余存活玩家（Seat 0）手牌均为 0，Seat 1 必须强制质疑
        seat1_legal = engine.get_legal_actions(1)
        self.assertEqual(len(seat1_legal), 1)
        self.assertEqual(seat1_legal[0].action_type, "CHALLENGE")

        # Seat 1 执行质疑：因 Seat 0 打出的是 ("Q", "JOKER") 全真牌，质疑失败，Seat 1 开枪受罚
        resolution = engine.step(seat1_legal[0])
        self.assertIsNotNone(resolution)
        assert resolution is not None
        self.assertFalse(resolution.was_bluff)
        self.assertFalse(resolution.challenge_succeeded)
        self.assertEqual(resolution.penalized_seat, 1)
        self.assertEqual(resolution.shots_taken_after, 1)
        self.assertTrue(resolution.forced_challenge)

    def test_illegal_action_rejected_before_state_mutation(self) -> None:
        engine = LiarsBarEngine(game_id="test_illegal", game_seed=55)
        with self.assertRaises(ValueError):
            engine.step("A999_NONEXISTENT")


if __name__ == "__main__":
    unittest.main()

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

    def test_nonstandard_rule_parameters_are_rejected(self) -> None:
        for kwargs in ({"num_players": 3}, {"hand_size": 6},
                       {"max_play_cards": 4}, {"max_chambers": 5}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "Fixed research rules"):
                LiarsBarEngine(**kwargs)

    def test_finished_round_archives_all_public_actions(self) -> None:
        engine = LiarsBarEngine(game_seed=10)
        for _ in range(4):
            engine.step(next(a for a in engine.get_legal_actions() if a.action_type == "PLAY"))
        plays = tuple(engine.current_round_history)
        engine.step("CHALLENGE")
        self.assertEqual(engine.round_id, 2)
        self.assertEqual(tuple(engine.past_public_actions[:-1]), plays)
        self.assertEqual(engine.past_public_actions[-1].action_type, "CHALLENGE")
        self.assertEqual(engine.current_round_history, [])
        self.assertTrue(all(not cards for cards in engine.played_cards_this_round))
        self.assertEqual(engine.get_observation().past_public_actions, tuple(engine.past_public_actions))

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

    def test_last_holder_must_challenge_with_three_or_four_players_alive(self) -> None:
        for alive_count in (3, 4):
            for last_card in ("Q", "K"):
                with self.subTest(alive_count=alive_count, last_card=last_card):
                    engine = LiarsBarEngine(game_seed=10)
                    engine.alive = [seat < alive_count for seat in range(4)]
                    engine.hands = [[last_card], ["K", "A"], [], []]
                    engine.current_turn_seat = 0
                    engine.target_rank = "Q"
                    engine.step(CandidateAction("custom", "PLAY", (last_card,)))
                    self.assertEqual(engine.current_turn_seat, 1)
                    self.assertEqual(
                        [action.action_type for action in engine.get_legal_actions()],
                        ["CHALLENGE"],
                    )
                    hands_before = [list(hand) for hand in engine.hands]
                    with self.assertRaises(ValueError):
                        engine.step(CandidateAction("custom", "PLAY", ("K",)))
                    self.assertEqual(engine.hands, hands_before)
                    self.assertEqual(engine.current_turn_seat, 1)
                    resolution = engine.step("CHALLENGE")
                    self.assertTrue(resolution.forced_challenge)
                    self.assertEqual(resolution.penalized_seat, 0 if last_card == "K" else 1)

    def test_play_remains_legal_when_another_alive_player_has_cards(self) -> None:
        engine = LiarsBarEngine(game_seed=10)
        engine.alive = [True, True, True, False]
        engine.hands = [["Q"], ["K", "A"], ["A"], []]
        engine.current_turn_seat = 0
        engine.target_rank = "Q"
        engine.step(CandidateAction("custom", "PLAY", ("Q",)))
        self.assertEqual(engine.current_turn_seat, 1)
        self.assertTrue(any(a.action_type == "PLAY" for a in engine.get_legal_actions()))
        self.assertTrue(any(a.action_type == "CHALLENGE" for a in engine.get_legal_actions()))

    def test_empty_player_has_no_challenge_or_play_actions(self) -> None:
        engine = LiarsBarEngine(game_seed=10)
        engine.hands = [["Q"], ["K", "K"], ["A", "A"], ["Q", "Q"]]
        engine.current_turn_seat = 0
        engine.target_rank = "Q"
        engine.step(CandidateAction("custom", "PLAY", ("Q",)))
        engine.step(CandidateAction("custom", "PLAY", ("K",)))
        self.assertEqual(engine.current_turn_seat, 2)
        self.assertFalse(engine._must_force_challenge(0))
        self.assertEqual(engine.get_legal_actions(0), ())
        # 即使错误地选中了空手座位，也不能为它提供质疑动作。
        engine.current_turn_seat = 0
        self.assertEqual(engine.get_legal_actions(), ())
        step_before = engine.step_id
        shots_before = list(engine.shots_taken)
        with self.assertRaisesRegex(RuntimeError, "no legal actions"):
            engine.step("CHALLENGE")
        self.assertEqual(engine.step_id, step_before)
        self.assertEqual(engine.shots_taken, shots_before)

    def test_empty_player_is_skipped_until_next_round(self) -> None:
        engine = LiarsBarEngine(game_seed=10)
        engine.hands = [["Q"], ["K", "K"], ["A", "A"], ["Q", "Q"]]
        engine.fatal_chamber_slot = [6, 6, 6, 6]
        engine.current_turn_seat = 0
        engine.target_rank = "Q"
        for seat, card in ((0, "Q"), (1, "K"), (2, "A"), (3, "Q"), (1, "K"), (2, "A")):
            self.assertEqual(engine.current_turn_seat, seat)
            engine.step(CandidateAction("custom", "PLAY", (card,)))
            self.assertNotEqual(engine.current_turn_seat, 0)
            self.assertEqual(engine.get_legal_actions(0), ())
        self.assertEqual(engine.current_turn_seat, 3)
        self.assertEqual([a.action_type for a in engine.get_legal_actions()], ["CHALLENGE"])
        resolution = engine.step("CHALLENGE")
        self.assertTrue(resolution.forced_challenge)
        self.assertEqual(resolution.claimer_seat, 2)
        self.assertEqual(resolution.challenger_seat, 3)
        self.assertEqual(resolution.penalized_seat, 2)
        self.assertEqual(engine.shots_taken[0], 0)
        self.assertEqual(engine.round_id, 2)
        self.assertEqual(len(engine.hands[0]), 5)

    def test_disabled_truth_logging_preserves_game_behavior(self) -> None:
        recorded = LiarsBarEngine(game_seed=17)
        simulation = LiarsBarEngine(game_seed=17, record_truth=False)
        while not recorded.is_game_over():
            action = recorded.get_legal_actions()[0]
            self.assertEqual(recorded.step(action), simulation.step(action))
            self.assertEqual(recorded.get_observation(), simulation.get_observation())
        self.assertGreater(len(recorded.step_truth_logs), 0)
        self.assertEqual(simulation.step_truth_logs, [])


if __name__ == "__main__":
    unittest.main()

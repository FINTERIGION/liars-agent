"""测试集 2：严格观测信息隔离与历史消融检查 (Information Isolation & Ablation)."""

from __future__ import annotations

import json
import copy
import unittest

from liars_agent.agents.prompt_builder import build_user_prompt
from liars_agent.engine.observation import PlayerObservation, apply_history_ablation
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.ui.app import InteractiveSessionManager, create_ui


class TestInformationIsolation(unittest.TestCase):
    """验证任何座位的 `PlayerObservation` 与模型 Prompt 绝不泄露其他玩家底牌、未翻开暗牌或实弹位置."""

    def test_environment_ids_seeds_and_hidden_state_are_not_observations(self) -> None:
        first = LiarsBarEngine(game_id="train_905_game_00000", game_seed=19)
        second = copy.deepcopy(first)
        second.game_id = "a-different-log-id"
        second.game_seed = 123456
        actor = first.current_turn_seat
        others = [seat for seat in range(4) if seat != actor]
        second.hands[others[0]], second.hands[others[1]] = second.hands[others[1]], second.hands[others[0]]
        second.fatal_chamber_slot = [7 - slot for slot in first.fatal_chamber_slot]
        self.assertEqual(first.get_observation(), second.get_observation())
        self.assertFalse(hasattr(first.get_observation(), "game_id"))
        serialized = first.get_observation().to_dict()
        for key in ("game_id", "game_seed", "fatal_chamber_slot"):
            self.assertNotIn(key, serialized)
        self.assertEqual(PlayerObservation.from_dict(serialized), first.get_observation())
        self.assertIn("hand_capacity_prior", serialized)
        self.assertNotIn("combinatorial_claim_truth_prior", serialized)
        invalid = {**serialized, "observation_version": "old"}
        with self.assertRaisesRegex(ValueError, "version"):
            PlayerObservation.from_dict(invalid)

    def test_public_history_ablation_never_removes_own_known_cards(self) -> None:
        engine = LiarsBarEngine(game_seed=22)
        engine.step(engine.get_legal_actions()[0])
        engine.step("CHALLENGE")
        for _ in range(5):
            engine.step(next(a for a in engine.get_legal_actions() if a.action_type == "PLAY"))
        observation = engine.get_observation()
        self.assertGreater(len(observation.own_played_cards_this_round), 0)
        self.assertGreater(len(observation.past_public_actions), 0)
        for mode in ("current_round_only", "last_action_only"):
            with self.subTest(mode=mode):
                ablated = apply_history_ablation(observation, mode)
                self.assertEqual(ablated.own_hand, observation.own_hand)
                self.assertEqual(ablated.own_played_cards_this_round, observation.own_played_cards_this_round)
                self.assertEqual(ablated.own_known_cards_this_round, observation.own_known_cards_this_round)
                self.assertEqual(ablated.hand_capacity_prior, observation.hand_capacity_prior)
                self.assertEqual(ablated.legal_actions, observation.legal_actions)
                self.assertEqual(ablated.past_public_actions, ())
                self.assertEqual(ablated.past_rounds_history, ())
        self.assertEqual(len(apply_history_ablation(observation, "last_action_only").current_round_history), 1)

    def test_full_prompt_retains_more_than_six_rounds(self) -> None:
        engine = LiarsBarEngine(game_seed=10)
        engine.fatal_chamber_slot = [6, 6, 6, 6]
        while len(engine.past_rounds_history) < 8:
            self.assertFalse(engine.is_game_over())
            engine.step(engine.get_legal_actions()[0])
        obs = engine.get_observation()
        prompt = build_user_prompt(obs)
        for resolution in obs.past_rounds_history:
            self.assertIn(f"R{resolution.round_id}(目标{resolution.target_rank})", prompt)
        for mode in ("current_round_only", "last_action_only"):
            self.assertNotIn("R1(目标", build_user_prompt(apply_history_ablation(obs, mode)))

    def test_replay_truth_only_after_game_end_and_reset_on_new_game(self) -> None:
        session = InteractiveSessionManager()
        session.start_game(seat3_spec="card_counter", game_seed=2026)
        self.assertFalse(session.engine.is_game_over())
        for index in (0, -1, 10**6):
            result = session.inspect_replay_step(index)
            self.assertNotIn("post_hoc_ground_truth", result)
            self.assertNotIn("decision_time_visible", result)
            self.assertNotIn("fatal_chamber", json.dumps(result))
            self.assertEqual(result["current_view"]["observation"]["observer_seat"], 0)
        for _ in range(1000):
            if session.engine.is_game_over():
                break
            session.submit_human_action(session.engine.get_legal_actions()[0].action_id)
        self.assertTrue(session.engine.is_game_over())
        last = session.inspect_replay_step(10**6)
        self.assertEqual(last["step_index"], len(session.engine.step_truth_logs) - 1)
        self.assertIn("post_hoc_ground_truth", last)
        self.assertTrue(session.get_session_profiles())
        session.start_game(seat3_spec="card_counter", game_seed=2026)
        self.assertIsNone(session.latest_replay)
        self.assertEqual(session.get_session_profiles(), {})
        self.assertNotIn("post_hoc_ground_truth", session.inspect_replay_step(0))

    def test_sessions_keep_separate_games(self) -> None:
        first, second = InteractiveSessionManager(), InteractiveSessionManager()
        for session in (first, second):
            session.start_game(seat3_spec="card_counter", game_seed=2026)
        before = second.get_current_view()
        first.submit_human_action(first.engine.get_legal_actions()[0].action_id)
        self.assertEqual(second.get_current_view(), before)

    def test_ui_uses_browser_session_state(self) -> None:
        try:
            import gradio
        except ImportError:
            self.skipTest("Gradio is optional")
        demo = create_ui()
        states = [item for item in demo.config["components"] if item["type"] == "state"]
        self.assertEqual(len(states), 1)
        self.assertIsNone(states[0]["props"].get("value"))
        events = {item["api_name"]: item for item in demo.config["dependencies"]}
        self.assertIn(states[0]["id"], events["start_game"]["outputs"])
        self.assertIn(states[0]["id"], events["inspect_replay"]["inputs"])

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
        self.assertEqual(obs_last_only.own_played_cards_this_round, obs_full.own_played_cards_this_round)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from dataclasses import asdict
from unittest.mock import patch

from liars_agent.agents.base import BaseAgent
from liars_agent.agents.rule_bots import CardCounterBot
from liars_agent.contracts import current_contract
from liars_agent.ui.app import InteractiveSessionManager, configured_agent_factory, create_ui, build_agent_by_spec


class FailOnceAgent(BaseAgent):
    def __init__(self, seed):
        super().__init__("fail_once")
        self.failed = False
        self.policy = CardCounterBot(seed=seed)

    def select_action(self, obs):
        if not self.failed:
            self.failed = True
            raise ConnectionError("Temporary model service error")
        return self.policy.select_action(obs)


class SessionTests(unittest.TestCase):
    def test_pipeline_sampling_config_maps_to_model_arguments(self):
        config = {"contract": current_contract(), "base_model": "Qwen/Qwen3.5-9B", "budget_path": "/tmp/ui-test-budget.json",
                  "budget_limit_usd": 10, "sampling": {"temperature": 0.1, "top_p": 0.8,
                  "max_new_tokens": 128, "fallback_strategy": "card_counter_safe"}}
        with patch("liars_agent.training.tinker_backend.TinkerBackend"), patch("liars_agent.agents.llm_agent.LLMAgent") as agent:
            configured_agent_factory(config)("qwen35_9b_zero_shot", 42)
        kwargs = agent.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["max_new_tokens"], 128)
        self.assertNotIn("fallback_strategy", kwargs)
        with self.assertRaisesRegex(ValueError, "incompatible method contract"):
            configured_agent_factory({"base_model": "old-model"})

    def finish_human_game(self, session):
        for _ in range(2000):
            if session.engine.is_game_over():
                return
            choices = session.human_action_choices()
            self.assertTrue(choices)
            session.submit_human_action(choices[0][1])
        self.fail("Game did not terminate")

    def test_human_game_hides_truth_until_completion(self):
        session = InteractiveSessionManager()
        session.start_game(game_seed=2026)
        self.assertTrue(session.human_action_choices())
        view = session.inspect_replay_step(0)
        self.assertNotIn("post_hoc_ground_truth", view)
        self.assertNotIn("fatal_chamber", json.dumps(view))
        self.assertEqual(view["current_view"]["observation"]["observer_seat"], 0)
        self.finish_human_game(session)
        last = session.inspect_replay_step(10**6)
        self.assertEqual(last["step_index"], len(session.engine.step_truth_logs) - 1)
        self.assertIn("post_hoc_ground_truth", last)
        self.assertTrue(all(isinstance(seat, str) for seat in last["post_hoc_ground_truth"]["all_players_true_hands"]))
        self.assertTrue(session.get_session_profiles()["replay_deterministic_verified"])
        for step in session.engine.step_truth_logs:
            if step.actor_seat == 0:
                self.assertEqual(step.decision_record["estimated_opponent_style"], "unknown")
                self.assertIsNone(step.decision_record["belief_bluff_prob"])
                self.assertEqual(step.decision_record["rationale_source"], "action_description")
                self.assertTrue(step.decision_record["rationale"])
        self.assertEqual(session.human_action_choices(), [])

    def test_new_game_clears_replay_and_sessions_are_independent(self):
        a, b = InteractiveSessionManager(), InteractiveSessionManager()
        a.start_game(seat0_spec="card_counter", game_seed=20)
        self.assertTrue(a.get_session_profiles())
        a.start_game(game_seed=21)
        self.assertIsNone(a.latest_replay)
        self.assertEqual(a.get_session_profiles(), {})
        b.start_game(game_seed=21)
        before = b.get_current_view()
        a.submit_human_action(a.human_action_choices()[0][1])
        self.assertEqual(b.get_current_view(), before)

    def test_invalid_human_action_does_not_advance_state(self):
        session = InteractiveSessionManager()
        session.start_game(game_seed=5)
        before = session.get_current_view()
        with self.assertRaises(ValueError):
            session.submit_human_action("not_a_legal_action")
        self.assertEqual(session.get_current_view(), before)

    def test_default_seed_is_private_random_and_fixed_seed_is_debug_only(self):
        a, b = InteractiveSessionManager(), InteractiveSessionManager()
        a.start_game()
        b.start_game()
        self.assertNotEqual(a.engine.game_seed, b.engine.game_seed)
        self.assertFalse(a.get_current_view()["debug_seed_enabled"])
        view = json.dumps(a.get_current_view())
        self.assertNotIn(str(a.engine.game_seed), view)
        self.assertNotIn("game_id", a.get_current_view()["observation"])
        a.start_game(game_seed=81)
        self.assertEqual(a.engine.game_seed, 81)
        self.assertTrue(a.get_current_view()["debug_seed_enabled"])

    def test_failed_ai_preserves_human_step_and_can_resume(self):
        def factory(spec, seed):
            return FailOnceAgent(seed) if spec == "flaky" else build_agent_by_spec(spec, seed)
        session = InteractiveSessionManager(factory)
        session.start_game(seat1_spec="flaky", game_seed=2)
        result = session.submit_human_action(session.human_action_choices()[0][1])
        self.assertEqual(result["status"], "paused_ai")
        self.assertIn("ConnectionError", result["ai_error"])
        self.assertEqual(len(session.engine.step_truth_logs), 1)
        preserved_step = asdict(session.engine.step_truth_logs[0])
        self.assertFalse(session.human_action_choices())
        self.assertNotIn("post_hoc_ground_truth", session.inspect_replay_step(0))
        session.resume_ai_turns()
        self.assertIsNone(session.ai_error)
        self.assertEqual(asdict(session.engine.step_truth_logs[0]), preserved_step)
        self.assertGreater(len(session.engine.step_truth_logs), 1)
        self.finish_human_game(session)
        self.assertTrue(session.get_session_profiles()["replay_deterministic_verified"])

    def test_ui_has_human_action_and_replay_events(self):
        try:
            import gradio  # noqa: F401
        except ImportError:
            self.skipTest("Install the UI extra to exercise Gradio construction")
        demo = create_ui()
        endpoints = {dependency["api_name"] for dependency in demo.config["dependencies"]}
        self.assertTrue({"start_game", "submit_action", "inspect_replay", "resume_ai"} <= endpoints)
        states = [component for component in demo.config["components"] if component["type"] == "state"]
        self.assertEqual(len(states), 1)
        self.assertIsNone(states[0]["props"].get("value"))


if __name__ == "__main__":
    unittest.main()

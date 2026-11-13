"""The same strict action grammar and observation context serve train and inference."""
from dataclasses import replace
import json
import unittest

from liars_agent.agents.prompt_builder import (SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT,
    build_user_prompt, build_bluff_user_prompt, render_observation, validate_action_decision)
from liars_agent.contracts import current_contract
from liars_agent.engine.observation import apply_history_ablation
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.types import PublicEvent


class TestDecisionProtocol(unittest.TestCase):
    def setUp(self):
        self.engine = LiarsBarEngine(game_seed=123)
        self.engine.step(self.engine.get_legal_actions()[0])
        self.obs = self.engine.get_observation()

    def test_valid_action_is_one_available_string_id(self):
        for action in self.obs.legal_actions:
            result = validate_action_decision(self.obs, json.dumps({"chosen_action_id": action.action_id}))
            self.assertEqual(result, {"chosen_action_id": action.action_id})

    def test_strict_schema_rejects_extra_fields_wrapping_duplicates_and_wrong_types(self):
        for completion in ('{"chosen_action_id":"A0","rationale":"x"}',
            '{"chosen_action_id":"A0","belief_bluff_prob":0.2}',
            '{"chosen_action_id":"A0","chosen_action_id":"A1"}',
            '```json\n{"chosen_action_id":"A0"}\n```',
            '{"chosen_action_id":"A0"} trailing', '{"chosen_action_id":0}',
            '{"chosen_action_id":true}', '{"chosen_action_id":null}',
            '{"chosen_action_id":"missing"}', '{}', '[]', 'null', None):
            with self.subTest(completion=completion), self.assertRaises(ValueError):
                validate_action_decision(self.obs, completion)

    def test_tasks_use_same_context_and_exclude_run_identifiers(self):
        context = render_observation(self.obs)
        action_prompt = build_user_prompt(self.obs)
        belief_prompt = build_bluff_user_prompt(self.obs)
        self.assertTrue(action_prompt.startswith(context + "\n"))
        self.assertTrue(belief_prompt.startswith(context + "\n"))
        self.assertNotIn(self.engine.game_id, action_prompt)
        self.assertNotIn("GameID", action_prompt)
        self.assertNotIn("game_id", self.obs.to_dict())
        self.assertIn("容量概率", context)
        self.assertIn("这不是该手实际诈唬概率", context)
        self.assertIn("chosen_action_id", SYSTEM_PROMPT)
        self.assertIn("0 表示全部为真牌", BELIEF_SYSTEM_PROMPT)
        self.assertNotEqual(SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT)
        self.assertTrue(current_contract()["prompt_sha256"])

    def test_full_context_keeps_all_past_actions_and_ablations_preserve_private_cards(self):
        history = tuple(PublicEvent(round_id=i, step_id=1, actor_seat=0,
                        action_type="PLAY", claimed_count=1, claimed_rank="Q", actor_remaining_cards=4)
                        for i in range(1, 9))
        obs = replace(self.obs, round_id=10, past_public_actions=history,
                      own_played_cards_this_round=("A",))
        full = render_observation(obs)
        for event in history:
            self.assertIn(f"R{event.round_id}/Step1", full)
        for mode in ("current_round_only", "last_action_only"):
            ablated = apply_history_ablation(obs, mode)
            rendered = render_observation(ablated)
            self.assertIn("你本小局已打出的牌: ['A']", rendered)
            self.assertNotIn("R1/Step1:Seat0", rendered)
            self.assertEqual(ablated.own_played_cards_this_round, ("A",))

    def test_bluff_prompt_requires_pending_claim(self):
        with self.assertRaisesRegex(ValueError, "pending claim"):
            build_bluff_user_prompt(LiarsBarEngine(game_seed=1).get_observation())


if __name__ == "__main__":
    unittest.main()

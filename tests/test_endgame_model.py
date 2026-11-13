import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from liars_agent.agents.base import BaseAgent
from liars_agent.agents.llm_agent import LLMAgent
from liars_agent.agents.rule_bots import CardCounterBot
from liars_agent.contracts import current_contract
from liars_agent.engine.replay import GameReplayBundle, save_replays_jsonl
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.types import AgentDecisionRecord
from liars_agent.evaluation.endgame_coverage import build_endgame_coverage
from liars_agent.evaluation.endgame_model import evaluate_endgame_models


class OfflineModelBackend:
    """Explicit model protocol double; its outputs are not model evidence."""
    base_model = "Qwen/Qwen3.5-9B"
    rates = {"prefill": .3, "sample": .807}

    def __init__(self, malformed=False):
        self.malformed = malformed
        self.actions = []
        self.beliefs = []

    def sample(self, messages, **kwargs):
        self.actions.append(messages)
        # A0 always exists; force a real LLMAgent fallback when requested.
        text = "bad JSON" if self.malformed else '{"chosen_action_id":"A0"}'
        return [SimpleNamespace(text=text, stop_reason="stop", latency_ms=0,
                                prompt_tokens=3, completion_tokens=2)]

    def score_bluff(self, messages, **kwargs):
        self.beliefs.append(messages)
        return SimpleNamespace(probability=.25, prompt_tokens=6, latency_ms=0,
                               belief_completion_tokens_upper_bound=2)


class RuleFallbackOnly(BaseAgent):
    def __init__(self):
        super().__init__("fallback-only")

    def select_action(self, obs):
        return obs.legal_actions[0], AgentDecisionRecord(
            was_fallback=True, belief_bluff_prob=.9, belief_source="heuristic",
            fallback_record={"belief_bluff_prob": .99, "belief_source": "heuristic"})


class EndgameModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.source, cls.output = root / "source", root / "coverage"
        (cls.source / "dataset/batches").mkdir(parents=True)
        (cls.source / "config.json").write_text(json.dumps({"contract": current_contract()}))
        (cls.source / "dataset/game_splits.json").write_text(json.dumps({
            "train": ["main_source_game"], "val": [], "test": []}))
        engine = LiarsBarEngine(game_id="main_source_game", game_seed=17)
        while not engine.is_game_over():
            engine.step(engine.get_legal_actions()[0])
        save_replays_jsonl([GameReplayBundle.from_engine(engine)],
                          cls.source / "dataset/batches/replays_0000.jsonl")
        build_endgame_coverage(cls.source, cls.output, seed=2026111301,
                              samples_per_cell=1, max_games=100, no_claim_per_alive=1)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_agent_interface_distinguishes_native_action_and_fallback_belief(self):
        good_backend, bad_backend = OfflineModelBackend(), OfflineModelBackend(malformed=True)
        agents = {"good": LLMAgent(backend=good_backend), "bad": LLMAgent(backend=bad_backend),
                  "rule": CardCounterBot(), "fallback_only": RuleFallbackOnly()}
        result = evaluate_endgame_models(self.output, agents)
        count, claims = result["dataset"]["num_samples"], result["dataset"]["num_claim"]
        self.assertEqual(result["dataset"]["num_no_claim"], 3)
        self.assertFalse(result["natural_distribution"])
        good, bad = result["models"]["good"], result["models"]["bad"]
        self.assertEqual(good["summary"]["native_action_legal_rate"], 1)
        self.assertEqual(bad["summary"]["native_action_legal_rate"], 0)
        self.assertEqual(bad["summary"]["returned_action_legal_rate"], 1)
        self.assertEqual(bad["summary"]["fallback_rate"], 1)
        self.assertEqual(bad["summary"]["forced_challenge"]["native_compliance_rate"], 0)
        self.assertEqual(bad["summary"]["forced_challenge"]["returned_compliance_rate"], 1)
        self.assertEqual(bad["summary"]["native_belief"]["num_valid_predictions"], claims)
        self.assertEqual([p["sample_id"] for p in good["predictions"]],
                         [p["sample_id"] for p in bad["predictions"]])
        self.assertEqual(good_backend.actions, bad_backend.actions)
        for backend in (good_backend, bad_backend):
            self.assertEqual(len(backend.actions), count)
            self.assertEqual(len(backend.beliefs), claims)
            self.assertTrue(all("仅回答一个数字标签" in call[0]["content"] for call in backend.beliefs))
        for row in bad["predictions"]:
            self.assertEqual(row["native_belief_probability"], None if row["label"] is None else .25)
        fallback = result["models"]["fallback_only"]["summary"]["native_belief"]
        self.assertEqual(fallback["num_valid_predictions"], 0)
        self.assertIsNone(fallback["calibration"]["log_loss"])
        self.assertNotIn("accuracy", json.dumps(result))
        self.assertEqual(sum(s["num_samples"] for s in good["by_alive_players"].values()), count)
        empty = good["by_scenario_alive_players"]["empty_claimer_free"]["2"]
        self.assertEqual(empty["num_samples"], 0)
        self.assertIsNone(empty["native_action_legal_rate"])

    def test_changing_labels_cannot_change_agent_inputs_and_stale_cache_is_rejected(self):
        backend = OfflineModelBackend()
        original = evaluate_endgame_models(self.output, {"model": LLMAgent(backend=backend)})
        path = self.output / "labels.jsonl"
        text = path.read_text()
        changed = [json.loads(line) for line in text.splitlines()]
        for row in changed:
            if row["label"] is not None:
                row["label"] = 1 - row["label"]
        path.write_text("\n".join(json.dumps(row) for row in changed) + "\n")
        try:
            second_backend = OfflineModelBackend()
            second = evaluate_endgame_models(self.output, {"model": LLMAgent(backend=second_backend)})
            self.assertEqual(backend.actions, second_backend.actions)
            self.assertEqual(backend.beliefs, second_backend.beliefs)
            self.assertNotEqual(original["dataset"]["input_sha256"], second["dataset"]["input_sha256"])
            untouched = OfflineModelBackend()
            with self.assertRaisesRegex(ValueError, "metadata differ"):
                evaluate_endgame_models(self.output, {"model": LLMAgent(backend=untouched)},
                                        cached_results=original["models"])
            self.assertEqual(untouched.actions, [])
        finally:
            path.write_text(text)

    def test_cache_reuses_matching_checkpoint_configuration_without_model_calls(self):
        report = evaluate_endgame_models(self.output, {"model": LLMAgent(backend=OfflineModelBackend())})
        backend = OfflineModelBackend()
        cached = evaluate_endgame_models(self.output, {"model": LLMAgent(backend=backend)},
                                         cached_results=report["models"])
        self.assertEqual(cached, report)
        self.assertEqual(backend.actions, [])
        for field, value in (("temperature", .7), ("checkpoint", "tinker://other/sampler_weights/new")):
            agent = LLMAgent(backend=backend)
            setattr(agent, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "metadata differ"):
                evaluate_endgame_models(self.output, {"model": agent}, cached_results=report["models"])
        self.assertEqual(backend.actions, [])

    def test_service_errors_propagate_and_no_success_is_manufactured(self):
        backend = OfflineModelBackend()
        def error(*args, **kwargs):
            raise RuntimeError("offline transport failure")
        backend.score_bluff = error
        with self.assertRaisesRegex(RuntimeError, "offline transport failure"):
            evaluate_endgame_models(self.output, {"model": LLMAgent(backend=backend)})


if __name__ == "__main__":
    unittest.main()

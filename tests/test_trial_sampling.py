"""Small offline doubles test the sampler round trip, never model performance."""
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from liars_agent.agents.prompt_builder import BELIEF_SYSTEM_PROMPT, build_bluff_user_prompt
from liars_agent.contracts import current_contract
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.training.data_generator import make_action_sample
from liars_agent.training.tinker_backend import BudgetExceededError, TinkerBackend
from liars_agent.training.trial_sampling import (
    TrialSamplingServiceError, estimate_saved_sampler_verification, verify_saved_sampler,
)
from test_tinker_backend import SDK, Tokenizer, SamplingClient


class BinaryForward:
    def __init__(self):
        self.logprobs = {"0": -1.4, "1": -.4}
        self.calls = []
        self.invalid = None

    def forward(self, data, loss_fn):
        self.calls.append(data)
        outputs = []
        for datum in data:
            value = self.logprobs[chr(datum.loss_fn_inputs["target_tokens"][-1])]
            values = [-.3] * (len(datum.model_input.to_ints()) - 1) + [value]
            if self.invalid == "nan":
                values[-1] = float("nan")
            if self.invalid == "length":
                values.pop()
            outputs.append({"logprobs": values})
        return SimpleNamespace(loss_fn_outputs=outputs)


class FreshService:
    def __init__(self):
        self.clients = []
        self.options = []
        self.configure = lambda client: None

    def create_sampling_client(self, **kwargs):
        client = SamplingClient()
        client.label_logprobs = {"0": -1.4, "1": -.4}
        self.configure(client)
        self.clients.append(client)
        self.options.append(kwargs)
        return client


class TrialSamplingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = FreshService()
        self.backend = TinkerBackend(sdk=SDK, service_client=self.service,
                                    budget_path=Path(self.directory.name) / "budget.jsonl")
        self.backend._tokenizer = Tokenizer()
        self.client = BinaryForward()
        self.path = "tinker://run/sampler_weights/trial"
        engine = LiarsBarEngine(game_seed=31)
        no_claim = engine.get_observation()
        engine.step(no_claim.legal_actions[0])
        claim = engine.get_observation()
        self.probes = {"action": [make_action_sample(obs, obs.legal_actions[0], f"val-{i}")
                                   for i, obs in enumerate((no_claim, claim))],
                       "belief": [{"task": "belief", "sample_id": f"val-belief-{i}",
                            "game_id": f"val-{i}", "completion": str(i), "label": i,
                            "observation": claim.to_dict(), "system": BELIEF_SYSTEM_PROMPT,
                            "prompt": build_bluff_user_prompt(claim), "contract": current_contract()}
                            for i in (0, 1)]}

    def verify(self, **kwargs):
        return verify_saved_sampler(self.backend, self.client, self.path, self.probes,
                                    max_seq_length=16384, **kwargs)

    def test_fresh_saved_sampler_raw_labels_native_actions_and_budget_accounting(self):
        original = copy.deepcopy(self.probes)
        old = self.backend.sampling_client(self.path)
        old.text = "stale cached client"
        estimate = estimate_saved_sampler_verification(self.backend, self.probes,
                                                       max_seq_length=16384, max_new_tokens=64)
        self.assertEqual(self.backend.ledger.snapshot()["estimated_cost_usd"], 0)
        result = self.verify(max_new_tokens=64)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["fresh_sampler_created"])
        self.assertEqual(len(self.service.clients), 2)
        self.assertEqual(old.calls, [])
        current = self.service.clients[-1]
        self.assertIs(self.backend.sampling_client(self.path), current)
        self.assertEqual(len(current.calls), 2)
        # The no-claim action never triggers a belief score; only the two
        # explicitly selected belief rows produce their two labels each.
        self.assertEqual(len(current.logprob_calls), 4)
        self.assertTrue(all(not row["belief_requested"] for row in result["action_checks"]))
        self.assertTrue(all(row["training_logprobs"] == row["sampler_logprobs"] for row in result["belief_checks"]))
        self.assertEqual(self.probes, original)
        usage = result["usage"]
        self.assertEqual(usage["training_tokens"], estimate["training_forward_tokens"])
        self.assertEqual(usage["prompt_tokens"], estimate["sampling_prefill_tokens"])
        self.assertEqual(usage["completion_tokens_upper_bound"], estimate["belief_generated_tokens_upper_bound"])
        self.assertLessEqual(usage["completion_tokens"], estimate["action_generated_tokens_upper_bound"])
        expected_cost = (usage["training_tokens"] * self.backend.rates["train"]
            + usage["prompt_tokens"] * self.backend.rates["prefill"]
            + (usage["completion_tokens"] + usage["completion_tokens_upper_bound"]) * self.backend.rates["sample"]) / 1_000_000
        self.assertAlmostEqual(usage["estimated_cost_usd"], expected_cost)
        self.assertLessEqual(usage["estimated_cost_usd"], estimate["total_usd_upper_bound"])
        self.assertEqual(usage["pending_requests"], 0)
        self.assertEqual(estimate["training_forward_sequences"], 4)

    def test_same_normalized_probability_with_wrong_raw_logprobs_fails(self):
        self.service.configure = lambda client: setattr(client, "label_logprobs", {"0": -2.4, "1": -1.4})
        result = self.verify()
        self.assertEqual(result["status"], "not_passed")
        self.assertTrue(all(row["passed"] for row in result["action_checks"]))
        self.assertTrue(all(not row["passed"] for row in result["belief_checks"]))
        for row in result["belief_checks"]:
            self.assertAlmostEqual(row["sampler_probability"], 1 / (1 + math.exp(-1)))
            self.assertAlmostEqual(row["absolute_errors"]["0"], 1)
            self.assertAlmostEqual(row["absolute_errors"]["1"], 1)

    def test_tolerance_is_recorded_and_not_automatically_relaxed(self):
        self.service.configure = lambda client: setattr(client, "label_logprobs", {"0": -1.40005, "1": -.40005})
        self.assertEqual(self.verify()["status"], "passed")
        strict = self.verify(logprob_atol=1e-7, logprob_rtol=0)
        self.assertEqual(strict["status"], "not_passed")
        self.assertEqual(strict["tolerances"], {"logprob_atol": 1e-7, "logprob_rtol": 0})

    def test_illegal_json_or_token_limit_cannot_be_hidden_by_fallback(self):
        for text, stop in (("```json\n{}\n```", "stop"), ('{"chosen_action_id":"missing"}', "stop"),
                           ('{"chosen_action_id":"A0"}', "length")):
            with self.subTest(text=text, stop=stop):
                def configure(client):
                    client.text, client.stop_reason = text, stop
                self.service.configure = configure
                result = self.verify()
                self.assertEqual(result["status"], "not_passed")
                self.assertTrue(all(not row["passed"] for row in result["action_checks"]))
                self.assertTrue(all(row["passed"] for row in result["belief_checks"]))
                self.assertNotIn("fallback_record", json.dumps(result))

    def test_invalid_training_or_sampling_logprobs_return_not_passed(self):
        for invalid in ("nan", "length"):
            with self.subTest(training_invalid=invalid):
                self.client.invalid = invalid
                result = self.verify()
                self.assertEqual(result["status"], "not_passed")
                self.assertEqual(result["errors"][0]["kind"], "invalid_response")
                self.assertEqual(self.service.clients[-1].calls, [])
        self.client.invalid = None
        self.service.configure = lambda client: setattr(client, "label_logprobs", {"0": float("nan"), "1": -.4})
        result = self.verify()
        self.assertEqual(result["status"], "not_passed")
        self.assertTrue(all(not row["passed"] for row in result["belief_checks"]))
        json.dumps(result, allow_nan=False)

    def test_impossible_pair_of_raw_label_probabilities_is_not_passed(self):
        self.client.logprobs = {"0": -.1, "1": -.1}
        result = self.verify()
        self.assertEqual(result["status"], "not_passed")
        self.assertIn("sum to more than one", result["errors"][0]["message"])
        self.client.logprobs = {"0": -1.4, "1": -.4}
        self.service.configure = lambda client: setattr(client, "label_logprobs", {"0": -.1, "1": -.1})
        result = self.verify()
        self.assertEqual(result["status"], "not_passed")
        self.assertTrue(all("sum to more than one" in row["error"] for row in result["belief_checks"]))

    def test_remote_failures_raise_with_partial_report_and_keep_uncertain_reservations(self):
        for operation in ("error", "logprob_error"):
            with self.subTest(operation=operation):
                error = RuntimeError("uncertain remote failure")
                self.service.configure = lambda client: setattr(client, operation, error)
                with self.assertRaises(TrialSamplingServiceError) as raised:
                    self.verify()
                report = raised.exception.partial_report
                self.assertIs(raised.exception.__cause__, error)
                self.assertEqual(report["status"], "not_passed")
                self.assertEqual(report["usage"]["pending_requests"], 1)
                self.assertGreater(report["usage"]["reserved_unknown_usd"], 0)
                self.assertEqual(report["errors"][-1]["kind"], "service_or_budget_failure")
                current = self.service.clients[-1]
                self.assertEqual(len(current.calls) if operation == "error" else len(current.logprob_calls), 1)

    def test_token_prefix_mismatch_is_rejected_before_model_calls(self):
        self.backend.encode_messages = lambda messages, generation: [42]
        with self.assertRaisesRegex(ValueError, "prompt tokens differ"):
            self.verify()
        self.assertEqual(self.service.clients, [])
        self.assertEqual(self.client.calls, [])

    def test_budget_rejection_blocks_forward_and_is_not_a_successful_trial(self):
        self.backend.ledger.limit_usd = 1e-12
        with self.assertRaises(TrialSamplingServiceError) as raised:
            self.verify()
        self.assertIsInstance(raised.exception.__cause__, BudgetExceededError)
        self.assertEqual(raised.exception.partial_report["status"], "not_passed")
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.service.clients[-1].calls, [])
        self.assertEqual(raised.exception.partial_report["usage"]["estimated_cost_usd"], 0)

    def test_failed_fresh_load_never_uses_cached_sampler(self):
        cached = self.backend.sampling_client(self.path)
        def fail(client):
            raise RuntimeError("saved sampler cannot load")
        self.service.configure = fail
        with self.assertRaises(TrialSamplingServiceError) as raised:
            self.verify()
        self.assertFalse(raised.exception.partial_report["fresh_sampler_created"])
        self.assertEqual(cached.calls, [])
        self.assertEqual(self.client.calls, [])


if __name__ == "__main__":
    unittest.main()

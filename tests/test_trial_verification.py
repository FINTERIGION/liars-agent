"""Offline trial-closure checks with persistent weights and optimizer state."""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from liars_agent.agents.prompt_builder import (SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT,
    build_user_prompt, build_bluff_user_prompt)
from liars_agent.contracts import current_contract
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.training.trial_verification import (TRIAL_VERIFICATION_VERSION,
    TrialVerificationError, normalize_trial_verification_config, select_trial_probe_rows,
    build_trial_probe_data, score_trial_probe, trial_verification_plan,
    verify_trial_reload_and_optimizer)
from trial_fakes import make_trial_backend


class TestTrialVerification(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.backend, self.service = make_trial_backend(self.path / "budget.jsonl")
        engine = LiarsBarEngine(game_seed=29)
        engine.step(engine.get_legal_actions()[0])
        self.obs = engine.get_observation()
        self.splits = {split: [self.row(split, task, index) for task in ("action", "belief")
                              for index in range(3)] for split in ("train", "val", "test")}
        self.probes = select_trial_probe_rows(self.splits, 42)
        self.data = build_trial_probe_data(self.probes, self.backend, 16384)

    def row(self, split, task, index):
        action = task == "action"
        return {"game_id": split, "sample_id": f"{split}-{task}-{index}", "task": task,
            "system": SYSTEM_PROMPT if action else BELIEF_SYSTEM_PROMPT,
            "prompt": build_user_prompt(self.obs) if action else build_bluff_user_prompt(self.obs),
            "completion": '{"chosen_action_id":"A0"}' if action else "1",
            "label": None if action else 1, "observation": self.obs.to_dict(), "contract": current_contract()}

    def train_and_save(self):
        client = self.service.training
        before = score_trial_probe(client, self.backend, self.data)
        control = score_trial_probe(client, self.backend, self.data)
        for rate in (.1, .15, .2):
            self.backend.run_training_batch(client, self.data, learning_rate=rate)
        saved = score_trial_probe(client, self.backend, self.data)
        state_path = client.save_state("trial").path
        return {"backend": self.backend, "original_client": client, "state_path": state_path,
            "probe_data": self.data, "before_update_snapshot": before,
            "baseline_control_snapshot": control, "saved_snapshot": saved,
            "next_batch": self.data, "next_learning_rate": .25}

    def test_fixed_validation_only_probes_and_real_sdk_masks(self):
        reversed_splits = {name: list(reversed(rows)) for name, rows in self.splits.items()}
        self.assertEqual(select_trial_probe_rows(reversed_splits, 42), self.probes)
        self.assertEqual({task: len(rows) for task, rows in self.probes.items()}, {"action": 2, "belief": 2})
        self.assertTrue(all(row["game_id"] == "val" for rows in self.probes.values() for row in rows))
        self.assertEqual(self.data.tasks, ["action", "action", "belief", "belief"])
        for index, datum in enumerate(self.data):
            self.assertEqual(set(datum.loss_fn_inputs), {"target_tokens", "weights"})
            weights = datum.loss_fn_inputs["weights"].tolist()
            targets = datum.loss_fn_inputs["target_tokens"].tolist()
            self.assertEqual(len(weights), len(datum.model_input.to_ints()))
            self.assertAlmostEqual(sum(weights), 1, places=6)
            self.assertEqual(targets[-1], self.backend.tokenizer.eos_token_id if index < 2 else ord("1"))
            self.assertEqual(sum(w > 0 for w in weights) == 1, index >= 2)
        with self.assertRaisesRegex(ValueError, "both tasks"):
            select_trial_probe_rows({"val": self.probes["action"]}, 42)

    def test_config_and_exact_call_token_plan(self):
        config = normalize_trial_verification_config()
        self.assertEqual(config["version"], TRIAL_VERIFICATION_VERSION)
        self.assertEqual(config["probe_per_task"], 2)
        for change in ({"version": "old"}, {"unknown": 1}, {"probe_per_task": True},
                       {"min_update_delta": 0}, {"logprob_atol": math.nan}, {"jitter_multiplier": .5}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_trial_verification_config(change)
        plan = trial_verification_plan(self.data, self.data[:2])
        probe_tokens = sum(len(d.model_input.to_ints()) for d in self.data)
        next_tokens = sum(len(d.model_input.to_ints()) for d in self.data[:2])
        self.assertEqual(plan["total_processed_tokens"], 7 * probe_tokens + 3 * next_tokens)
        self.assertEqual(plan["diagnostic_optimizer_steps"], 3)
        self.assertEqual(plan["sdk_calls"], {"forward": 7, "forward_backward": 3, "optim_step": 3,
                                          "create_training_client_from_state_with_optimizer": 1,
                                          "create_training_client_from_state": 1})

    def test_live_and_restored_updates_match_without_overwriting_saved_state(self):
        args = self.train_and_save()
        saved_state = copy.deepcopy(self.service.saved_states[args["state_path"]])
        usage_before = self.backend.ledger.snapshot()["training_tokens"]
        report = verify_trial_reload_and_optimizer(**args)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(all(check["passed"] for check in report["checks"].values()))
        self.assertEqual(report["diagnostic_optimizer_steps"], {"original": 1, "restored": 1, "weight_only": 1})
        self.assertEqual(len(self.service.training_clients), 3)
        self.assertEqual(self.service.resumed, [(args["state_path"], True), (args["state_path"], False)])
        self.assertIs(args["original_client"], self.service.training)
        original, restored, weight_only = self.service.training_clients
        self.assertEqual(original.state, restored.state)
        self.assertEqual((original.optim_calls, restored.optim_calls), (4, 1))
        self.assertIs(original.batches[-1], restored.batches[-1])
        self.assertIs(original.batches[-1], weight_only.batches[-1])
        self.assertEqual(original.learning_rates[-1], restored.learning_rates[-1])
        self.assertEqual(original.learning_rates[-1], weight_only.learning_rates[-1])
        self.assertEqual(restored.initial_state["weight"], weight_only.initial_state["weight"])
        self.assertEqual(restored.initial_state, saved_state)
        self.assertEqual(weight_only.initial_state["momentum"], 0)
        self.assertNotEqual(original.state["weight"], weight_only.state["weight"])
        self.assertEqual(self.service.saved_states, {args["state_path"]: saved_state})
        self.assertEqual(original.saved_states, ["trial"])
        self.assertFalse(restored.saved_states)
        self.assertFalse(weight_only.saved_states)
        # This helper does four forwards + three updates. The plan's other three
        # forwards are the caller's before/control/saved snapshots.
        tokens = sum(len(d.model_input.to_ints()) for d in self.data)
        self.assertEqual(self.backend.ledger.snapshot()["training_tokens"] - usage_before, 7 * tokens)
        json.dumps(report, allow_nan=False)
        # True-label 1 becomes less likely here. Improvement is not required.
        self.assertLess(report["snapshots"]["saved"]["sequences"][-1]["logprobs"][0],
                        report["snapshots"]["before"]["sequences"][-1]["logprobs"][0])

    def test_weight_only_reload_passes_static_check_but_fails_optimizer_check(self):
        self.service.restore_optimizer = False
        args = self.train_and_save()
        with self.assertRaisesRegex(TrialVerificationError, "Optimizer-restored") as raised:
            verify_trial_reload_and_optimizer(**args)
        report = raised.exception.partial_report
        self.assertTrue(report["checks"]["state_reload"]["passed"])
        self.assertFalse(report["checks"]["optimizer_restore"]["passed"])
        self.assertEqual(report["status"], "not_passed")
        self.assertEqual(report["diagnostic_optimizer_steps"], {"original": 1, "restored": 1, "weight_only": 0})

    def test_corrupt_reload_stops_before_diagnostic_updates(self):
        self.service.corrupt_reload = True
        args = self.train_and_save()
        with self.assertRaisesRegex(TrialVerificationError, "Reloaded trial weights") as raised:
            verify_trial_reload_and_optimizer(**args)
        self.assertFalse(raised.exception.partial_report["checks"]["state_reload"]["passed"])
        self.assertEqual(raised.exception.partial_report["diagnostic_optimizer_steps"], {"original": 0, "restored": 0, "weight_only": 0})
        self.assertEqual(self.service.training.optim_calls, 3)

    def test_no_change_and_jitter_sized_change_cannot_pass_or_add_steps(self):
        self.service.static_updates = True
        args = self.train_and_save()
        with self.assertRaisesRegex(TrialVerificationError, "observable update") as raised:
            verify_trial_reload_and_optimizer(**args)
        check = raised.exception.partial_report["checks"]["observable_update"]
        self.assertEqual(check["update_delta"]["max_abs_delta"], 0)
        # A change larger than 1e-6 but below 3x the control noise is still unverified.
        args["baseline_control_snapshot"] = copy.deepcopy(args["before_update_snapshot"])
        args["saved_snapshot"] = copy.deepcopy(args["before_update_snapshot"])
        args["baseline_control_snapshot"]["sequences"][0]["logprobs"][0] += 1e-5
        args["saved_snapshot"]["sequences"][0]["logprobs"][0] += 2e-5
        with self.assertRaises(TrialVerificationError) as raised:
            verify_trial_reload_and_optimizer(**args)
        check = raised.exception.partial_report["checks"]["observable_update"]
        self.assertGreater(check["update_delta"]["max_abs_delta"], 1e-6)
        self.assertLess(check["update_delta"]["max_abs_delta"], check["required_max_abs_delta_exclusive"])
        self.assertEqual(self.service.training.optim_calls, 3)
        self.assertFalse(self.service.resumed)

    def test_actions_moving_without_belief_is_not_distinguishable_for_sampler_check(self):
        def unchanged_belief(result):
            for output in result.loss_fn_outputs[-2:]:
                scores = output["logprobs"].tolist()
                scores[-1] = -1.0
                output["logprobs"] = scores
            return result

        self.service.training.output_transform = unchanged_belief
        args = self.train_and_save()
        with self.assertRaisesRegex(TrialVerificationError, "all/belief probes") as raised:
            verify_trial_reload_and_optimizer(**args)
        check = raised.exception.partial_report["checks"]["observable_update"]
        self.assertTrue(check["exceeds_reload_tolerance"])
        self.assertFalse(check["belief"]["passed"])
        self.assertFalse(self.service.resumed)

    def test_stable_gradient_cannot_certify_optimizer_restoration(self):
        self.service.constant_gradient = True
        args = self.train_and_save()
        with self.assertRaisesRegex(TrialVerificationError, "weight-only negative control") as raised:
            verify_trial_reload_and_optimizer(**args)
        checks = raised.exception.partial_report["checks"]
        self.assertTrue(checks["observable_update"]["passed"])
        self.assertTrue(checks["state_reload"]["passed"])
        self.assertTrue(checks["optimizer_restore"]["passed"])
        self.assertFalse(checks["optimizer_negative_control"]["passed"])
        self.assertEqual(self.service.training.optim_calls, 4)

    def test_negative_control_must_exceed_measured_baseline_noise(self):
        reference = verify_trial_reload_and_optimizer(**self.train_and_save())
        delta = reference["checks"]["optimizer_negative_control"]["max_abs_delta"]
        self.backend, self.service = make_trial_backend(self.path / "noise_budget.jsonl")
        args = self.train_and_save()
        args["baseline_control_snapshot"]["sequences"][0]["logprobs"][0] += delta / 2
        with self.assertRaisesRegex(TrialVerificationError, "weight-only negative control") as raised:
            verify_trial_reload_and_optimizer(**args)
        check = raised.exception.partial_report["checks"]["optimizer_negative_control"]
        self.assertFalse(check["matched"])  # A tolerance-only check would pass it.
        self.assertFalse(check["passed"])
        self.assertLess(check["max_abs_delta"], check["required_max_abs_delta_exclusive"])

    def test_missing_or_nonfinite_snapshot_is_persistable_failure_before_load(self):
        args = self.train_and_save()
        for value in (None, math.nan, math.inf):
            with self.subTest(value=value):
                bad = copy.deepcopy(args["saved_snapshot"])
                bad["sequences"][0]["logprobs"][0] = value
                with self.assertRaises(TrialVerificationError) as raised:
                    verify_trial_reload_and_optimizer(**{**args, "saved_snapshot": bad})
                json.dumps(raised.exception.partial_report, allow_nan=False)
        self.assertFalse(self.service.resumed)

    def test_service_error_retains_completed_checks_without_retry(self):
        args = self.train_and_save()
        self.service.load_error = RuntimeError("offline injected load failure")
        with self.assertRaisesRegex(TrialVerificationError, "injected load failure") as raised:
            verify_trial_reload_and_optimizer(**args)
        report = raised.exception.partial_report
        self.assertTrue(report["checks"]["observable_update"]["passed"])
        self.assertEqual(report["operations"], [{"name": "load_state_with_optimizer", "status": "failed"}])
        self.assertEqual(len(self.service.resumed), 1)
        self.assertEqual(set(report["snapshots"]), {"before", "baseline_control", "saved"})
        json.dumps(report, allow_nan=False)

    def test_forward_failure_keeps_budget_reservation_and_is_not_retried(self):
        self.service.training.forward_error = RuntimeError("offline injected forward failure")
        with self.assertRaises(TrialVerificationError) as raised:
            score_trial_probe(self.service.training, self.backend, self.data)
        self.assertEqual(self.service.training.val_calls, 1)
        self.assertEqual(self.backend.ledger.snapshot()["pending_requests"], 1)
        self.assertEqual(raised.exception.partial_report["status"], "not_passed")
        self.assertIn("injected forward failure", raised.exception.partial_report["error"]["message"])

    def test_invalid_forward_scores_fail_once_with_persistable_evidence(self):
        def corrupt(result, mode):
            if mode == "missing":
                del result.loss_fn_outputs[0]["logprobs"]
            else:
                count = len(self.data[0].model_input.to_ints())
                result.loss_fn_outputs[0]["logprobs"] = [math.nan] * count
            return result

        for index, mode in enumerate(("missing", "nonfinite"), start=1):
            with self.subTest(mode=mode):
                self.service.training.output_transform = lambda result: corrupt(result, mode)
                with self.assertRaises(TrialVerificationError) as raised:
                    score_trial_probe(self.service.training, self.backend, self.data)
                self.assertEqual(self.service.training.val_calls, index)
                json.dumps(raised.exception.partial_report, allow_nan=False)
                self.assertEqual(raised.exception.partial_report["status"], "not_passed")
        # Responses returned, so their consumed forward tokens remain accounted.
        ledger = self.backend.ledger.snapshot()
        self.assertEqual(ledger["pending_requests"], 0)
        self.assertEqual(ledger["training_tokens"], 2 * sum(len(d.model_input.to_ints()) for d in self.data))


if __name__ == "__main__":
    unittest.main()

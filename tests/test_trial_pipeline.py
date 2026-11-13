"""Small stateful transport tests, never real training or model experiments."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from liars_agent import experiment
from liars_agent.agents.prompt_builder import BELIEF_SYSTEM_PROMPT, build_bluff_user_prompt
from liars_agent.contracts import current_contract
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.training.data_generator import make_action_sample
from liars_agent.training.tinker_train import run_tinker_lora_sft
from trial_fakes import make_trial_backend


def tiny_run(path):
    config = experiment.load_config(experiment.DEFAULT_CONFIG)
    config["training"].update(batch_size=2, num_epochs=3, learning_rate=.1,
                               diagnostic_steps=[], diagnostic_samples_per_task=1)
    engine = LiarsBarEngine(game_seed=29)
    engine.step(engine.get_legal_actions()[0])
    obs = engine.get_observation()
    splits = {}
    for split in ("train", "val", "test"):
        action = make_action_sample(obs, obs.legal_actions[0], split)
        belief = {"task": "belief", "game_id": split, "sample_id": split + "_belief",
            "system": BELIEF_SYSTEM_PROMPT, "prompt": build_bluff_user_prompt(obs), "completion": "1",
            "label": 1, "observation": obs.to_dict(), "contract": current_contract()}
        splits[split] = [action, belief]
        experiment.write_jsonl(path / f"dataset/{split}.jsonl", splits[split])
    experiment.write_json(path / "dataset/data_summary.json", {"contract": current_contract(), "shortfall": 0})
    return config, splits


class TestTrialPipeline(unittest.TestCase):
    def test_complete_closure_uses_saved_weights_and_preserves_trial_checkpoint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            config, splits = tiny_run(path)
            backend, service = make_trial_backend(path / "mock_budget.jsonl")
            with patch.object(experiment, "make_backend", return_value=backend):
                status = experiment.run_pipeline(config, path, "sft_trial", max_train_steps=2)
            self.assertEqual(status["stages"]["sft_trial"], "trial_complete")
            self.assertEqual(status["real_experiments"], "pending")
            metadata = experiment.read_json(path / "models/sft_trial/trial_metadata.json")
            evidence = metadata["trial_verification"]
            self.assertEqual(evidence["status"], "passed")
            self.assertTrue(all(check["passed"] for check in evidence["training"]["checks"].values()))
            self.assertEqual(service.resumed, [(metadata["state_path"], True), (metadata["state_path"], False)])
            self.assertEqual(service.sampling_init[0]["model_path"], metadata["sampler_path"])
            self.assertEqual(service.saved_states[metadata["state_path"]]["step"], 2)
            self.assertEqual(service.saved_samplers[metadata["sampler_path"]]["step"], 2)
            self.assertEqual([client.state["step"] for client in service.training_clients], [3, 3, 1])
            self.assertEqual(metadata["training_metrics"]["training_optimizer_steps"], 2)
            self.assertEqual(metadata["trial_verification_optimizer_steps"], 3)
            experiment.require_verified_trial(config, metadata, splits)
            with patch.object(backend, "preflight", side_effect=AssertionError("Cached trial must not call service")):
                cached = experiment.train_model(config, path, backend, "sft_trial", max_train_steps=2)
            self.assertEqual(cached, metadata)
            # The formal stage can proceed only with matching successful evidence.
            with patch("liars_agent.training.tinker_train.run_tinker_lora_sft", return_value={}) as train:
                experiment.train_model(config, path, backend, "sft")
            self.assertIsNone(train.call_args.kwargs["max_train_steps"])
            for mutate in ("status", "dataset", "lora", "config", "sampler"):
                bad = copy.deepcopy(metadata)
                if mutate == "status": bad["trial_verification"]["status"] = "failed"
                elif mutate == "dataset": bad["dataset_sha256"] = "changed"
                elif mutate == "lora": bad["lora_config"]["r"] = 32
                elif mutate == "config": bad["training_metrics"]["learning_rate"] = .2
                else: bad["sampler_path"] += "wrong"
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    experiment.require_verified_trial(config, bad, splits)

    def test_no_update_or_bad_restore_never_creates_success_metadata(self):
        for options in ({"static_updates": True}, {"corrupt_reload": True}, {"restore_optimizer": False}):
            with self.subTest(options=options), TemporaryDirectory() as directory:
                path = Path(directory)
                config, _ = tiny_run(path)
                backend, _ = make_trial_backend(path / "budget.jsonl", **options)
                with patch.object(experiment, "make_backend", return_value=backend), self.assertRaises(RuntimeError):
                    experiment.run_pipeline(config, path, "sft_trial", max_train_steps=2)
                self.assertFalse((path / "models/sft_trial/trial_metadata.json").exists())
                self.assertFalse((path / "models/sft/adapter_metadata.json").exists())
                report = experiment.read_json(path / "models/sft_trial/trial_verification.json")
                self.assertEqual(report["status"], "failed")
                self.assertIn("checkpoints", report)
                self.assertEqual(experiment.read_json(path / "run_status.json")["stages"]["sft_trial"], "failed")

    def test_formal_stage_rejects_missing_or_old_trial_before_preflight(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            config, _ = tiny_run(path)
            backend, _ = make_trial_backend(path / "budget.jsonl")
            with patch.object(backend, "preflight", side_effect=AssertionError("No paid call")):
                with self.assertRaisesRegex(ValueError, "passed sft_trial"):
                    experiment.train_model(config, path, backend, "sft")
                old = {"contract": current_contract(), "action_supervision_policy": config["training"]["action_supervision"],
                    "max_train_steps": 5, "training_status": "trial_complete"}
                experiment.write_json(path / "models/sft_trial/trial_metadata.json", old)
                with self.assertRaisesRegex(ValueError, "verification is required"):
                    experiment.train_model(config, path, backend, "sft_trial", max_train_steps=5)

    def test_save_or_training_failure_preserves_partial_evidence_and_budget(self):
        for method in ("forward_backward", "save_state", "save_weights_for_sampler"):
            with self.subTest(method=method), TemporaryDirectory() as directory:
                path = Path(directory)
                config, _ = tiny_run(path)
                backend, service = make_trial_backend(path / "budget.jsonl")
                with patch.object(service.training, method, side_effect=RuntimeError("injected service failure")), \
                     self.assertRaisesRegex(RuntimeError, "injected service failure"):
                    experiment.train_model(config, path, backend, "sft_trial", max_train_steps=1)
                report = experiment.read_json(path / "models/sft_trial/trial_verification.json")
                self.assertEqual(report["status"], "failed")
                self.assertFalse((path / "models/sft_trial/trial_metadata.json").exists())
                if method == "save_weights_for_sampler":
                    self.assertIn(report["epoch_checkpoints"][-1]["state_path"], service.saved_states)
                    self.assertEqual(report["phase"], "save_epoch_sampler")
                if method == "forward_backward":
                    self.assertEqual(backend.ledger.snapshot()["pending_requests"], 1)


if __name__ == "__main__":
    unittest.main()

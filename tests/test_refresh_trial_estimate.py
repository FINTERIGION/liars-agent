"""Tiny verified-fixture reuse tests; no real service calls or dataset generation."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from liars_agent import experiment
from liars_agent.agents.prompt_builder import BELIEF_SYSTEM_PROMPT, build_bluff_user_prompt
from liars_agent.contracts import current_contract
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.training.action_supervision import make_action_supervision
from liars_agent.training.data_generator import make_action_sample
from liars_agent.training.tinker_backend import DEFAULT_RATES, PRICING_DATE, PRICING_SOURCE, TinkerBackend
from liars_agent.training.tinker_train import next_logical_batch_indices, training_diagnostic_plan
from scripts.refresh_trial_estimate import VERIFIED_FILES, _sha256, refresh_trial_estimate
from test_tinker_backend import SDK, Tokenizer


class TestRefreshTrialEstimate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        self.report_path = self.root / "verified.json"
        self.config = experiment.load_config(experiment.DEFAULT_CONFIG)
        self.config["training"].update(diagnostic_steps=[50, 100], diagnostic_samples_per_task=2)
        engine = LiarsBarEngine(game_seed=19)
        engine.step(engine.get_legal_actions()[0])
        obs = engine.get_observation()

        def row(split, task, index):
            gid = f"fixture_{split}_{index}"
            if task == "action":
                result = make_action_sample(obs, obs.legal_actions[0], gid)
                values = {a.action_id: .8 if i < 2 else .1 for i, a in enumerate(obs.legal_actions)}
                result["action_supervision"] = make_action_supervision(result, values)
                return result
            return {"task": "belief", "sample_id": gid + "_belief", "game_id": gid,
                "system": BELIEF_SYSTEM_PROMPT, "prompt": build_bluff_user_prompt(obs),
                "observation": obs.to_dict(), "completion": "1", "label": 1, "contract": current_contract()}

        sizes = {"train": {"action": 48, "belief": 45},
                 "val": {"action": 4, "belief": 4}, "test": {"action": 1, "belief": 1}}
        self.splits = {split: [row(split, task, i) for task, n in tasks.items() for i in range(n)]
                       for split, tasks in sizes.items()}
        self.config["dataset"].update(total_target_samples=53, belief_target_samples=50)
        experiment.write_json(self.run / "config.json", self.config)
        for split, rows in self.splits.items():
            experiment.write_jsonl(self.run / f"dataset/{split}.jsonl", rows)
        experiment.write_json(self.run / "dataset/game_splits.json",
                              {s: sorted({r["game_id"] for r in rows}) for s, rows in self.splits.items()})
        for relative in ("augmentation/replacements.jsonl", "augmentation/added_actions.jsonl"):
            experiment.write_jsonl(self.run / relative, [])
        plan = training_diagnostic_plan(sizes["train"], 3, 32, [50, 100], max_train_steps=5)
        actions = [r for r in self.splits["train"] if r["task"] == "action"]
        expanded = sum(len(actions[i]["action_supervision"]["target_action_ids"])
                       for step in range(5) for i in next_logical_batch_indices(sizes["train"], 32, 42, step)["action"])
        self.baseline = {"balanced_training_tokens": 500, "full_validation_tokens": 100,
            "diagnostic_tokens": 150, "validation_tokens": 250, "training_optimizer_steps": 5,
            "training_logical_state_visits": {"action": 80, "belief": 80},
            "training_expanded_sequences": {"action": expanded, "belief": 80},
            "training_plan": plan, "sft_train_and_validation_usd": 750 * DEFAULT_RATES["train"] / 1e6,
            "rates_per_million_tokens": dict(DEFAULT_RATES), "pricing_date": PRICING_DATE,
            "pricing_source": PRICING_SOURCE, "endgame_evaluation": {"status": "not_requested"}}
        self.previous = {"contract": current_contract(), "model": self.config["base_model"],
            "action_supervision_policy": self.config["training"]["action_supervision"],
            **self.baseline}
        experiment.write_json(self.run / "trial_cost_estimate.json", self.previous)
        self.verified = {"dataset_files_sha256": {name: _sha256(self.run / name) for name in VERIFIED_FILES},
            "five_step_trial_estimate": copy.deepcopy(self.baseline),
            "augmentation_verification": {"contract": current_contract(), "status": "passed", "task_samples_per_split": sizes},
            "supervision": {"contract": current_contract(), "policy": self.config["training"]["action_supervision"]},
            "tokenizer_check": {"status": "passed", "policy": self.config["training"]["action_supervision"],
                "all_action_targets_stop_at_first_assistant_eos": True,
                "belief_unit_weight_verified_for_all_rows": True, "max_seq_length": self.config["training"]["max_seq_length"]}}
        experiment.write_json(self.report_path, self.verified)
        self.backend = TinkerBackend(sdk=SDK, budget_path=self.root / "unused_budget.jsonl")
        self.backend._tokenizer = Tokenizer()

    def test_reuses_verified_body_and_only_estimates_small_closure(self):
        with patch.object(experiment, "make_backend", return_value=self.backend) as constructor, \
                patch.object(experiment, "estimate_cost", side_effect=AssertionError("Full tokenization")), \
                patch.object(self.backend, "preflight", side_effect=AssertionError("Paid API")), \
                patch.object(self.backend, "run_training_batch", side_effect=AssertionError("Model forward")), \
                patch.object(experiment, "estimate_trial_closure", wraps=experiment.estimate_trial_closure) as closure:
            result = refresh_trial_estimate(self.run, self.report_path)
            repeated = refresh_trial_estimate(self.run, self.report_path)
        self.assertEqual(constructor.call_count, 2)
        self.assertEqual(closure.call_count, 2)
        self.assertEqual(result, repeated)
        for key, value in self.baseline.items():
            self.assertEqual(result[key], value)
        self.assertGreater(result["trial_verification"]["total_usd_upper_bound"], 0)
        self.assertAlmostEqual(result["trial_total_usd_upper_bound"],
            self.baseline["sft_train_and_validation_usd"] + result["trial_verification"]["total_usd_upper_bound"])
        self.assertEqual(result["trial_verification"]["state_and_optimizer"]["diagnostic_training_batches"], 3)
        self.assertEqual(result["trial_verification"]["state_and_optimizer"]["probe_forward_calls"], 7)
        self.assertFalse(result["verified_baseline_reuse"]["full_dataset_retokenized"])
        self.assertEqual(result["verified_baseline_reuse"]["model_api_calls"], 0)
        self.assertFalse((self.root / "unused_budget.jsonl").exists())

    def test_hash_or_unverified_tokens_are_rejected_before_backend(self):
        for case in ("config_hash", "data_hash", "token_total", "missing_hash", "policy", "not_certified"):
            with self.subTest(case=case):
                config_bytes = (self.run / "config.json").read_bytes()
                train_bytes = (self.run / "dataset/train.jsonl").read_bytes()
                estimate = copy.deepcopy(self.previous); report = copy.deepcopy(self.verified)
                if case == "config_hash":
                    (self.run / "config.json").write_bytes(config_bytes + b"\n")
                elif case == "data_hash":
                    (self.run / "dataset/train.jsonl").write_bytes(train_bytes + b"\n")
                elif case == "token_total":
                    estimate["balanced_training_tokens"] += 1
                elif case == "missing_hash":
                    report["dataset_files_sha256"].pop("dataset/train.jsonl")
                elif case == "policy":
                    estimate["action_supervision_policy"] = {"version": "old"}
                else:
                    report["tokenizer_check"]["status"] = "not_passed"
                experiment.write_json(self.run / "trial_cost_estimate.json", estimate)
                experiment.write_json(self.report_path, report)
                with patch.object(experiment, "make_backend") as constructor:
                    with self.assertRaises(ValueError):
                        refresh_trial_estimate(self.run, self.report_path)
                    constructor.assert_not_called()
                (self.run / "config.json").write_bytes(config_bytes)
                (self.run / "dataset/train.jsonl").write_bytes(train_bytes)

    def test_wrong_task_counts_plan_expansion_or_rates_are_rejected_before_backend(self):
        for case in ("task_counts", "plan", "expanded", "rates"):
            with self.subTest(case=case):
                report = copy.deepcopy(self.verified)
                estimate = copy.deepcopy(self.previous)
                if case == "task_counts":
                    report["augmentation_verification"]["task_samples_per_split"]["train"]["belief"] += 1
                elif case == "plan":
                    report["five_step_trial_estimate"]["training_plan"]["planned_steps"] = 6
                    estimate["training_plan"]["planned_steps"] = 6
                elif case == "expanded":
                    report["five_step_trial_estimate"]["training_expanded_sequences"]["action"] += 1
                    estimate["training_expanded_sequences"]["action"] += 1
                else:
                    report["five_step_trial_estimate"]["rates_per_million_tokens"]["train"] = 99.
                    estimate["rates_per_million_tokens"]["train"] = 99.
                experiment.write_json(self.report_path, report)
                experiment.write_json(self.run / "trial_cost_estimate.json", estimate)
                with patch.object(experiment, "make_backend") as constructor:
                    with self.assertRaises(ValueError):
                        refresh_trial_estimate(self.run, self.report_path)
                    constructor.assert_not_called()

    def test_failed_small_estimate_does_not_replace_previous_file(self):
        before = (self.run / "trial_cost_estimate.json").read_bytes()
        with patch.object(experiment, "make_backend", return_value=self.backend), \
                patch.object(experiment, "estimate_trial_closure", side_effect=ValueError("Uncached tokenizer")):
            with self.assertRaisesRegex(ValueError, "Uncached tokenizer"):
                refresh_trial_estimate(self.run, self.report_path)
        self.assertEqual((self.run / "trial_cost_estimate.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

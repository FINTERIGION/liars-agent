"""Offline integration for empirical action sets, weighting and epoch resume."""

from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from liars_agent.agents.prompt_builder import BELIEF_SYSTEM_PROMPT, build_bluff_user_prompt
from liars_agent.contracts import current_contract
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.training.action_supervision import (
    action_target_completions, make_action_supervision, normalize_action_policy,
)
from liars_agent.training.data_generator import make_action_sample
from liars_agent.training.tinker_backend import TinkerBackend
from liars_agent.training.tinker_train import (build_sft_datum, run_tinker_lora_sft,
    select_diagnostic_rows, training_diagnostic_plan, validate_training_sample)
from test_tinker_backend import SDK, Service, Tokenizer, TrainingClient


class SetPreferenceClient(TrainingClient):
    """Epoch 1 has better set mass; epoch 2 has better canonical token NLL."""
    def __init__(self):
        super().__init__()
        self.validation_epoch = 0
        self.fail_after_train_calls = None
        self.learning_rates = []

    def forward_backward(self, data, loss_fn):
        if self.fail_after_train_calls is not None and self.train_calls >= self.fail_after_train_calls:
            raise RuntimeError("interrupted between completed epochs")
        super().forward_backward(data, loss_fn)
        # Distinct task losses expose slicing by the old, unexpanded half-size.
        from types import SimpleNamespace
        return SimpleNamespace(loss_fn_outputs=[{"logprobs": [-.7 if
            sum(w > 0 for w in datum.loss_fn_inputs["weights"]) == 1 else -.4]
            * len(datum.model_input.tokens)} for datum in data])

    def optim_step(self, params):
        self.learning_rates.append(params.learning_rate)
        return super().optim_step(params)

    def forward(self, data, loss_fn):
        self.val_calls += 1
        self.validation_epoch = sum("-epoch-" in name for name in self.saved_states)
        belief = sum(w > 0 for w in data[0].loss_fn_inputs["weights"]) == 1
        if belief:
            return self.output(data, .2 / (self.validation_epoch + 1))
        outputs = []
        probabilities = ({"A0": .1, "A1": .6}, {"A0": .3, "A1": .05}, {"A0": .2, "A1": .2})
        for datum in data:
            text = "".join(chr(token) for token in datum.loss_fn_inputs["target_tokens"])
            action_id = re.findall(r'"chosen_action_id"\s*:\s*"([^"]+)"', text)[-1]
            count = sum(w > 0 for w in datum.loss_fn_inputs["weights"])
            logprob = math.log(probabilities[self.validation_epoch % 3].get(action_id, .001)) / count
            outputs.append({"logprobs": [logprob if w > 0 else 0.0
                                          for w in datum.loss_fn_inputs["weights"]]})
        from types import SimpleNamespace
        return SimpleNamespace(loss_fn_outputs=outputs)


class TestUncertainTraining(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        engine = LiarsBarEngine(game_seed=29)
        engine.step(engine.get_legal_actions()[0])
        self.obs = engine.get_observation()
        self.policy = normalize_action_policy()
        self.backend, self.service = self.new_backend("main")

    def new_backend(self, key):
        service = Service()
        service.training = SetPreferenceClient()
        backend = TinkerBackend(budget_path=self.path / f"{key}.jsonl", sdk=SDK, service_client=service)
        backend._tokenizer = Tokenizer()
        return backend, service

    def action(self, game, group="tied", suffix="", ties=2):
        row = make_action_sample(self.obs, self.obs.legal_actions[0], game)
        row["sample_id"] += suffix
        values = {a.action_id: .2 for a in self.obs.legal_actions}
        values["A0"] = .8
        values["A1"] = .8 if group == "tied" else (.78 if group == "small_gap" else .2)
        if group == "tied":
            for action in self.obs.legal_actions[:ties]:
                values[action.action_id] = .8
        row["action_supervision"] = make_action_supervision(row, values, self.policy)
        return row

    def belief(self, game, suffix=""):
        return {"task": "belief", "game_id": game, "sample_id": game + "_belief" + suffix,
                "system": BELIEF_SYSTEM_PROMPT, "prompt": build_bluff_user_prompt(self.obs),
                "completion": "1", "label": 1, "observation": self.obs.to_dict(),
                "contract": current_contract()}

    def splits(self):
        return {key: [self.action(key), self.belief(key)] for key in ("train", "val", "test")}

    @staticmethod
    def action_answer(datum):
        text = "".join(chr(token) for token in datum.loss_fn_inputs["target_tokens"])
        return re.findall(r'"chosen_action_id"\s*:\s*"([^"]+)"', text)[-1]

    def test_override_and_sample_weight_preserve_prompt_mask_and_source(self):
        row = self.action("train")
        original = row["completion"]
        override = json.dumps({"chosen_action_id": "A1"})
        datum = build_sft_datum(row, self.backend, 16384, completion_override=override,
                                sample_weight=.6, action_supervision_policy=self.policy)
        self.assertEqual(row["completion"], original)
        self.assertEqual(self.action_answer(datum), "A1")
        weights = datum.loss_fn_inputs["weights"]
        self.assertAlmostEqual(sum(weights), .6)
        prefix = self.backend.tokenizer.apply_chat_template(
            [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["prompt"]}],
            tokenize=True, add_generation_prompt=True)
        self.assertTrue(all(w == 0 for w in weights[:len(prefix) - 1]))
        self.assertTrue(all(w > 0 for w in weights[len(prefix) - 1:]))
        with self.assertRaisesRegex(ValueError, "unweighted"):
            build_sft_datum(self.belief("train"), self.backend, 16384, sample_weight=.5)

    def test_action_masks_stop_at_first_eos_and_missing_eos_fails(self):
        row = self.action("train")
        original = build_sft_datum(row, self.backend, 16384)
        class TrailingTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                return result + "\nTEMPLATE_SEPARATOR<end>" if messages[-1]["role"] == "assistant" else result
        self.backend._tokenizer = TrailingTokenizer()
        actual = build_sft_datum(row, self.backend, 16384)
        self.assertEqual(actual.loss_fn_inputs, original.loss_fn_inputs)
        self.assertEqual(actual.model_input.tokens, original.model_input.tokens)
        self.assertEqual(actual.loss_fn_inputs["target_tokens"][-1], self.backend.tokenizer.eos_token_id)
        self.backend.tokenizer.eos_token_id = 999999
        with self.assertRaisesRegex(ValueError, "assistant EOS"):
            build_sft_datum(row, self.backend, 16384)

    def test_diagnostic_plan_and_fixed_validation_subset(self):
        plan = training_diagnostic_plan({"action": 17, "belief": 31}, 3, 32, [1, 5], 3)
        self.assertEqual(plan, {"total_steps": 6, "steps_per_epoch": 2, "planned_steps": 3,
            "full_validation_epochs": 1, "diagnostic_steps": [0, 1, 3], "is_trial": True})
        full = training_diagnostic_plan({"action": 8000, "belief": 7995}, 3, 32)
        self.assertEqual(full["diagnostic_steps"], [0, 50, 100])
        self.assertEqual(full["total_steps"], 1500)
        splits = self.splits()
        splits["val"] = [row for index in range(8) for row in
            (self.action("val", suffix=str(index)), self.belief("val", str(index)))]
        selected = select_diagnostic_rows(splits, 19, 3)
        reordered = {**splits, "val": list(reversed(splits["val"]))}
        self.assertEqual(selected, select_diagnostic_rows(reordered, 19, 3))
        self.assertEqual({task: len(rows) for task, rows in selected.items()}, {"action": 3, "belief": 3})
        self.assertTrue(all(row["game_id"] == "val" for rows in selected.values() for row in rows))

    def test_diagnostics_never_select_checkpoint_and_use_one_fixed_subset(self):
        splits = self.splits()
        splits["val"].append(self.action("val", suffix="extra"))
        diagnostics_seen = []
        def evaluate(rows, backend, client, *args):
            if len(rows) == 1:
                diagnostics_seen.append([row["sample_id"] for row in rows])
                nll = .001  # Diagnostic would incorrectly win if treated as full validation.
            else:
                nll = 4 - client.train_calls
            return {"action_set_nll": nll, "canonical_mean_token_nll": nll / 4,
                    "num_samples": len(rows), "subgroups": {}}
        with patch("liars_agent.training.action_validation.evaluate_action_validation", side_effect=evaluate):
            result = run_tinker_lora_sft([], self.path / "diagnostics", backend=self.backend,
                splits=splits, num_epochs=2, batch_size=2, diagnostic_steps=[1], diagnostic_samples_per_task=1)
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual(result["best_val_loss"], 2)
        self.assertEqual(diagnostics_seen[0], diagnostics_seen[1])
        records = result["early_diagnostics"]["records"]
        self.assertEqual([row["step"] for row in records], [0, 1])
        self.assertTrue(all(not row["used_for_checkpoint_selection"] for row in records))
        self.assertEqual(self.service.training.saved_samplers, ["sft-best-epoch-1", "sft-best-epoch-2"])

    def test_step_limited_trial_preserves_schedule_and_never_writes_formal_adapter(self):
        splits = self.splits()
        splits["train"] += [self.belief("train", str(i)) for i in range(2)]
        out = self.path / "trial"
        result = run_tinker_lora_sft([], out, backend=self.backend, splits=splits,
            num_epochs=3, batch_size=2, learning_rate=2e-4, warmup_ratio=.5,
            diagnostic_steps=[1, 50], max_train_steps=2, stage="sft_trial")
        self.assertEqual(self.service.training.optim_calls, 2)
        self.assertEqual(self.service.training.learning_rates, [2e-4 / 5, 2e-4 * 2 / 5])
        self.assertEqual([row["step"] for row in result["early_diagnostics"]["records"]], [0, 1, 2])
        self.assertEqual(result["training_metrics"]["training_expanded_sequences"], {"action": 4, "belief": 2})
        self.assertEqual(result["training_metrics"]["training_logical_state_visits"], {"action": 2, "belief": 2})
        self.assertEqual(result["training_metrics"]["full_validation_epochs"], 0)
        self.assertEqual(result["max_train_steps"], 2)
        self.assertFalse(result["formal_training_complete"])
        self.assertIsNone(result["best_epoch"])
        self.assertIsNone(result["training_metrics"]["final_val_loss"])
        self.assertTrue(result["sampler_path"].endswith("sft_trial-trial-step-2"))
        self.assertTrue((out / "trial_metadata.json").exists())
        self.assertFalse((out / "adapter_metadata.json").exists())
        self.assertFalse((out / "training_progress.json").exists())
        with patch.object(self.backend, "preflight", side_effect=AssertionError("API")):
            with self.assertRaisesRegex(ValueError, "separate"):
                run_tinker_lora_sft([], out, backend=self.backend, splits=splits)

    def test_source_lora_mismatch_is_rejected_before_backend_or_preflight(self):
        source = {"contract": current_contract(), "base_model": self.backend.base_model,
            "state_path": "tinker://source/weights/epoch", "action_supervision_policy": self.policy,
            "lora_config": {"r": 16, "train_attn": True, "train_mlp": True, "train_unembed": False}}
        for override in ({"lora_rank": 32}, {"train_attn": False}, {"train_mlp": False}, {"train_unembed": True}):
            with self.subTest(override=override), patch("liars_agent.training.tinker_train.TinkerBackend") as constructor:
                with self.assertRaisesRegex(ValueError, "LoRA configuration"):
                    run_tinker_lora_sft([], self.path / "wrong_lora", splits=self.splits(),
                        resume_from=source["state_path"], resume_metadata=source, **override)
                constructor.assert_not_called()

    def test_trial_reaching_full_epoch_retains_validation_but_stays_a_trial(self):
        result = run_tinker_lora_sft([], self.path / "whole_epoch_trial", backend=self.backend,
            splits=self.splits(), num_epochs=1, batch_size=2, max_train_steps=5, diagnostic_steps=[])
        self.assertEqual(result["training_metrics"]["completed_steps"], 1)
        self.assertEqual(result["training_metrics"]["full_validation_epochs"], 1)
        self.assertEqual([row["step"] for row in result["early_diagnostics"]["records"]], [0, 1])
        self.assertFalse(result["formal_training_complete"])
        self.assertEqual(result["checkpoint_role"], "trial_final_step")
        self.assertIsNone(result["best_epoch"])
        self.assertEqual(result["full_validation_best"]["epoch"], 1)
        self.assertTrue(result["sampler_path"].endswith("trial-step-1"))
        self.assertTrue(result["full_validation_best"]["sampler_path"].endswith("best-epoch-1"))

    def test_small_gap_is_downweighted_then_action_mean_normalized(self):
        splits = self.splits()
        splits["train"] = [self.action("train", group, group) for group in ("tied", "clear", "small_gap")]
        splits["train"].append(self.belief("train"))
        result = run_tinker_lora_sft([], self.path / "weighted", backend=self.backend,
                                    splits=splits, num_epochs=1, batch_size=2)
        weights = [sum(sum(datum.loss_fn_inputs["weights"]) for datum in batch[:-1])
                   for batch in self.service.training.batches]
        for actual, expected in zip(sorted(weights), [.6, 1.2, 1.2]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(weights), 3.0)
        self.assertTrue(all(sum(batch[-1].loss_fn_inputs["weights"]) == 1.0
                            for batch in self.service.training.batches))
        tied_batch = next(batch for batch in self.service.training.batches if len(batch) == 3)
        self.assertEqual([self.action_answer(datum) for datum in tied_batch[:-1]], ["A0", "A1"])
        for datum in tied_batch[:-1]:
            self.assertAlmostEqual(sum(datum.loss_fn_inputs["weights"]), .6)
        self.assertAlmostEqual(result["training_metrics"]["train_action_weight_mean"], 5 / 6)
        curve = result["training_metrics"]["loss_curve"][0]
        self.assertAlmostEqual(curve["train_action_loss"], .4)
        self.assertAlmostEqual(curve["train_belief_loss"], .7)
        self.assertEqual(curve["training_logical_state_visits"], {"action": 3, "belief": 3})
        self.assertEqual(curve["training_expanded_sequences"], {"action": 4, "belief": 3})

    def test_four_and_five_answers_are_all_supervised_at_every_state_visit(self):
        for ties in (4, 5):
            with self.subTest(ties=ties):
                backend, service = self.new_backend(f"coverage_{ties}")
                splits = self.splits()
                splits["train"][0] = self.action("train", ties=ties)
                splits["train"] += [self.belief("train", str(i)) for i in range(3)]
                result = run_tinker_lora_sft([], self.path / f"all_{ties}", backend=backend,
                    splits=splits, num_epochs=3, batch_size=2, seed=91)
                self.assertEqual(len(service.training.batches), 12)
                for batch in service.training.batches:
                    self.assertEqual([self.action_answer(datum) for datum in batch[:-1]],
                                     [f"A{i}" for i in range(ties)])
                    for datum in batch[:-1]:
                        self.assertAlmostEqual(sum(datum.loss_fn_inputs["weights"]), 1 / ties)
                    self.assertAlmostEqual(sum(sum(d.loss_fn_inputs["weights"]) for d in batch[:-1]), 1)
                    self.assertEqual(sum(batch[-1].loss_fn_inputs["weights"]), 1)
                metrics = result["training_metrics"]
                self.assertEqual(metrics["training_logical_state_visits"], {"action": 12, "belief": 12})
                self.assertEqual(metrics["training_expanded_sequences"], {"action": 12 * ties, "belief": 12})
                self.assertEqual(metrics["training_optimizer_steps"], 12)

    def test_default_logical_batch_and_optimizer_schedule_do_not_expand(self):
        splits = self.splits()
        splits["train"][0] = self.action("train", ties=5)
        splits["train"] += [self.belief("train", str(i)) for i in range(16)]
        rate = 2e-4
        result = run_tinker_lora_sft([], self.path / "logical_batch", backend=self.backend,
            splits=splits, num_epochs=3, learning_rate=rate, warmup_ratio=.5)
        self.assertEqual(self.service.training.train_calls, 6)
        self.assertEqual(self.service.training.optim_calls, 6)
        for batch in self.service.training.batches:
            self.assertEqual(len(batch), 16 * 5 + 16)
            self.assertEqual([self.action_answer(d) for d in batch[:80]], [f"A{i}" for i in range(5)] * 16)
            self.assertAlmostEqual(sum(sum(d.loss_fn_inputs["weights"]) for d in batch[:80]), 16)
            self.assertEqual(sum(sum(d.loss_fn_inputs["weights"]) for d in batch[80:]), 16)
        for actual, fraction in zip(self.service.training.learning_rates, (1/3, 2/3, 1, 1, 1, 1)):
            self.assertAlmostEqual(actual, rate * fraction)
        metrics = result["training_metrics"]
        self.assertEqual(metrics["logical_batch_task_counts"], {"action": 16, "belief": 16})
        self.assertEqual(metrics["training_optimizer_steps_per_epoch"], 2)
        self.assertEqual(metrics["training_optimizer_steps"], 6)
        self.assertEqual(metrics["training_logical_state_visits"], {"action": 96, "belief": 96})
        self.assertEqual(metrics["training_expanded_sequences"], {"action": 480, "belief": 96})

    def test_set_nll_selects_checkpoint_when_canonical_nll_prefers_other_epoch(self):
        result = run_tinker_lora_sft([], self.path / "selected", backend=self.backend,
                                    splits=self.splits(), num_epochs=2, batch_size=2)
        curve = result["training_metrics"]["loss_curve"]
        self.assertGreater(curve[0]["val_action_canonical_token_nll"], curve[1]["val_action_canonical_token_nll"])
        self.assertAlmostEqual(curve[0]["val_action_set_nll"], -math.log(.7))
        self.assertAlmostEqual(curve[1]["val_action_set_nll"], -math.log(.35))
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["action_supervision_policy"], self.policy)
        self.assertEqual(result["training_metrics"]["checkpoint_selection"], "val_action_set_nll")
        for epoch in curve:
            self.assertEqual(epoch["val_action_loss"], epoch["val_action_set_nll"])
            self.assertEqual(epoch["val_loss"], epoch["val_action_set_nll"])
            self.assertIn("subgroups", epoch["action_validation"])

    def test_epoch_resume_reproduces_state_order_all_answers_weights_and_lr(self):
        splits = self.splits()
        splits["train"] += [self.action("train", "clear", "clear"),
                            self.action("train", "small_gap", "small"),
                            self.belief("train", "extra")]
        uninterrupted, reference_service = self.new_backend("reference")
        reference = run_tinker_lora_sft([], self.path / "reference_model", backend=uninterrupted,
                            splits=splits, num_epochs=3, batch_size=2, seed=23, warmup_ratio=.5)
        self.service.training.fail_after_train_calls = 3
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            run_tinker_lora_sft([], self.path / "resumed", backend=self.backend,
                                splits=splits, num_epochs=3, batch_size=2, seed=23, warmup_ratio=.5)
        self.service.training.fail_after_train_calls = None
        result = run_tinker_lora_sft([], self.path / "resumed", backend=self.backend,
                                    splits=splits, num_epochs=3, batch_size=2, seed=23, resume=True, warmup_ratio=.5)
        self.assertEqual(self.service.resumed, [("tinker://run/weights/sft-epoch-1", True)])
        expected = [[(d.model_input.tokens, d.loss_fn_inputs) for d in batch]
                    for batch in reference_service.training.batches]
        actual = [[(d.model_input.tokens, d.loss_fn_inputs) for d in batch]
                  for batch in self.service.training.batches]
        self.assertEqual(actual, expected)
        self.assertEqual(self.service.training.learning_rates, reference_service.training.learning_rates)
        self.assertEqual(result["training_metrics"], reference["training_metrics"])
        self.assertEqual(result["best_epoch"], 1)

    def test_missing_old_objective_is_rejected_before_preflight(self):
        metadata = {"contract": current_contract(), "base_model": self.backend.base_model,
                    "state_path": "tinker://old/weights/sft"}
        with patch.object(self.backend, "preflight", side_effect=AssertionError("API")):
            with self.assertRaisesRegex(ValueError, "supervision objective"):
                run_tinker_lora_sft([], self.path / "bad_source", backend=self.backend,
                    splits=self.splits(), resume_from=metadata["state_path"], resume_metadata=metadata)
            out = self.path / "old_progress"
            out.mkdir()
            (out / "training_progress.json").write_text(json.dumps({"contract": current_contract()}))
            with self.assertRaisesRegex(ValueError, "supervision objective"):
                run_tinker_lora_sft([], out, backend=self.backend, splits=self.splits(), resume=True)

    def test_previous_epoch_rotation_objective_is_rejected_before_preflight(self):
        old_policy = {**self.policy, "version": "empirical_ties_weighted_actions_v1",
                      "target_selection": "seeded_cycle_per_observation_epoch"}
        old_policy.pop("answer_weighting")
        metadata = {"contract": current_contract(), "base_model": self.backend.base_model,
            "state_path": "tinker://old/weights/sft", "action_supervision_policy": old_policy}
        with patch.object(self.backend, "preflight", side_effect=AssertionError("API")):
            with self.assertRaisesRegex(ValueError, "supervision objective"):
                run_tinker_lora_sft([], self.path / "old_rotation", backend=self.backend,
                    splits=self.splits(), resume_from=metadata["state_path"], resume_metadata=metadata)
            out = self.path / "old_rotation_progress"
            out.mkdir()
            (out / "training_progress.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "supervision objective"):
                run_tinker_lora_sft([], out, backend=self.backend, splits=self.splits(), resume=True)

    def test_legacy_refined_rows_remain_readable_but_need_explicit_training_supervision(self):
        row = self.action("train")
        row.pop("action_supervision")
        row["action_refinement"] = {"origin": "original", "ambiguous": True}
        self.assertEqual(validate_training_sample(row), self.obs)
        build_sft_datum(row, self.backend, 16384)
        splits = self.splits()
        splits["train"][0] = row
        with patch("liars_agent.training.tinker_train.TinkerBackend") as constructor:
            with self.assertRaisesRegex(ValueError, "explicit action_supervision"):
                run_tinker_lora_sft([], self.path / "missing_descriptor", splits=splits)
            constructor.assert_not_called()

    def test_all_validation_candidates_are_tokenized_before_preflight(self):
        class LongAlternateTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                text = super().apply_chat_template(messages, **kwargs)
                if messages[-1]["role"] == "assistant" and '"A1"' in messages[-1]["content"]:
                    text = text.removesuffix("<end>") + "X" * 20000 + "<end>"
                return text
        self.backend._tokenizer = LongAlternateTokenizer()
        splits = self.splits()
        splits["train"][0] = self.action("train", "clear")
        with patch.object(self.backend, "preflight") as preflight:
            with self.assertRaisesRegex(ValueError, "no JSON was truncated"):
                run_tinker_lora_sft([], self.path / "long_val", backend=self.backend,
                                    splits=splits, num_epochs=1, batch_size=2)
            preflight.assert_not_called()
        self.assertEqual(self.service.training.train_calls, 0)

    def test_every_tied_answer_is_tokenized_before_preflight_even_for_one_epoch(self):
        class LongAlternateTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                text = super().apply_chat_template(messages, **kwargs)
                if messages[-1]["role"] == "assistant" and '"A1"' in messages[-1]["content"]:
                    text = text.removesuffix("<end>") + "X" * 20000 + "<end>"
                return text
        self.backend._tokenizer = LongAlternateTokenizer()
        splits = self.splits()
        self.assertEqual(len(action_target_completions(splits["train"][0], self.policy)), 2)
        # Validation has only A0; the oversized A1 belongs to training only.
        singleton_obs = replace(self.obs, legal_actions=self.obs.legal_actions[:1])
        val = make_action_sample(singleton_obs, singleton_obs.legal_actions[0], "val")
        val["action_supervision"] = make_action_supervision(val, {"A0": .8}, self.policy)
        splits["val"][0] = val
        with patch.object(self.backend, "preflight") as preflight:
            with self.assertRaisesRegex(ValueError, "no JSON was truncated"):
                run_tinker_lora_sft([], self.path / "long_next_epoch", backend=self.backend,
                                    splits=splits, num_epochs=1, batch_size=2)
            preflight.assert_not_called()
        self.assertEqual(self.service.training.train_calls, 0)


if __name__ == "__main__":
    unittest.main()

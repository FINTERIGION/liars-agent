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
    make_action_supervision, normalize_action_policy, select_action_completion,
)
from liars_agent.training.data_generator import make_action_sample
from liars_agent.training.tinker_backend import TinkerBackend
from liars_agent.training.tinker_train import build_sft_datum, run_tinker_lora_sft, validate_training_sample
from test_tinker_backend import SDK, Service, Tokenizer, TrainingClient


class SetPreferenceClient(TrainingClient):
    """Epoch 1 has better set mass; epoch 2 has better canonical token NLL."""
    def __init__(self):
        super().__init__()
        self.validation_epoch = 0
        self.fail_after_train_calls = None

    def forward_backward(self, data, loss_fn):
        if self.fail_after_train_calls is not None and self.train_calls >= self.fail_after_train_calls:
            raise RuntimeError("interrupted between completed epochs")
        return super().forward_backward(data, loss_fn)

    def forward(self, data, loss_fn):
        self.val_calls += 1
        belief = sum(w > 0 for w in data[0].loss_fn_inputs["weights"]) == 1
        if belief:
            self.validation_epoch += 1
            return self.output(data, .2 / self.validation_epoch)
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

    def action(self, game, group="tied", suffix=""):
        row = make_action_sample(self.obs, self.obs.legal_actions[0], game)
        row["sample_id"] += suffix
        values = {a.action_id: .2 for a in self.obs.legal_actions}
        values["A0"] = .8
        values["A1"] = .8 if group == "tied" else (.78 if group == "small_gap" else .2)
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

    def test_small_gap_is_downweighted_then_action_mean_normalized(self):
        splits = self.splits()
        splits["train"] = [self.action("train", group, group) for group in ("tied", "clear", "small_gap")]
        splits["train"].append(self.belief("train"))
        result = run_tinker_lora_sft([], self.path / "weighted", backend=self.backend,
                                    splits=splits, num_epochs=1, batch_size=2)
        weights = [sum(batch[0].loss_fn_inputs["weights"]) for batch in self.service.training.batches]
        for actual, expected in zip(sorted(weights), [.6, 1.2, 1.2]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(weights), 3.0)
        self.assertTrue(all(sum(batch[1].loss_fn_inputs["weights"]) == 1.0
                            for batch in self.service.training.batches))
        self.assertAlmostEqual(result["training_metrics"]["train_action_weight_mean"], 5 / 6)

    def test_tied_target_rotates_by_epoch_but_short_pool_reuses_same_target(self):
        splits = self.splits()
        splits["train"] += [self.belief("train", str(i)) for i in range(3)]
        run_tinker_lora_sft([], self.path / "rotated", backend=self.backend,
                            splits=splits, num_epochs=2, batch_size=2, seed=91)
        actual = [self.action_answer(batch[0]) for batch in self.service.training.batches]
        expected = [json.loads(select_action_completion(splits["train"][0], 91, epoch, self.policy))["chosen_action_id"]
                    for epoch in range(2)]
        self.assertNotEqual(expected[0], expected[1])
        self.assertEqual(actual, [expected[0]] * 4 + [expected[1]] * 4)

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

    def test_epoch_resume_reproduces_target_rotation_and_weights(self):
        splits = self.splits()
        splits["train"].append(self.belief("train", "extra"))
        uninterrupted, reference_service = self.new_backend("reference")
        run_tinker_lora_sft([], self.path / "reference_model", backend=uninterrupted,
                            splits=splits, num_epochs=3, batch_size=2, seed=23)
        self.service.training.fail_after_train_calls = 2
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            run_tinker_lora_sft([], self.path / "resumed", backend=self.backend,
                                splits=splits, num_epochs=3, batch_size=2, seed=23)
        self.service.training.fail_after_train_calls = None
        result = run_tinker_lora_sft([], self.path / "resumed", backend=self.backend,
                                    splits=splits, num_epochs=3, batch_size=2, seed=23, resume=True)
        self.assertEqual(self.service.resumed, [("tinker://run/weights/sft-epoch-1", True)])
        expected = [(self.action_answer(batch[0]), batch[0].loss_fn_inputs["weights"])
                    for batch in reference_service.training.batches]
        actual = [(self.action_answer(batch[0]), batch[0].loss_fn_inputs["weights"])
                  for batch in self.service.training.batches]
        self.assertEqual(actual, expected)
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
                    text += "X" * 20000
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

    def test_later_epoch_tied_targets_are_tokenized_before_preflight(self):
        class LongAlternateTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                text = super().apply_chat_template(messages, **kwargs)
                if messages[-1]["role"] == "assistant" and '"A1"' in messages[-1]["content"]:
                    text += "X" * 20000
                return text
        self.backend._tokenizer = LongAlternateTokenizer()
        splits = self.splits()
        seed = next(seed for seed in range(20)
                    if json.loads(select_action_completion(splits["train"][0], seed, 0, self.policy))["chosen_action_id"] == "A0")
        # Validation has only A0, so the oversized A1 occurs only next epoch.
        singleton_obs = replace(self.obs, legal_actions=self.obs.legal_actions[:1])
        val = make_action_sample(singleton_obs, singleton_obs.legal_actions[0], "val")
        val["action_supervision"] = make_action_supervision(val, {"A0": .8}, self.policy)
        splits["val"][0] = val
        with patch.object(self.backend, "preflight") as preflight:
            with self.assertRaisesRegex(ValueError, "no JSON was truncated"):
                run_tinker_lora_sft([], self.path / "long_next_epoch", backend=self.backend,
                                    splits=splits, num_epochs=2, batch_size=2, seed=seed)
            preflight.assert_not_called()
        self.assertEqual(self.service.training.train_calls, 0)


if __name__ == "__main__":
    unittest.main()

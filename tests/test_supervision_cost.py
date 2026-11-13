"""Cost estimates must match actual all-answer expanded training forwards.

All service calls use the explicit local transport from the backend tests. The
temporary ledger records mock tokens, never real model usage or research data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from liars_agent import experiment
from liars_agent.agents.prompt_builder import BELIEF_SYSTEM_PROMPT, build_bluff_user_prompt
from liars_agent.contracts import current_contract
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.training.action_supervision import make_action_supervision
from liars_agent.training.data_generator import make_action_sample
from liars_agent.training.tinker_backend import TinkerBackend
from liars_agent.training.tinker_train import run_tinker_lora_sft
from test_tinker_backend import SDK, Service, Tokenizer


class TestSupervisionCost(unittest.TestCase):
    def test_estimate_matches_expanded_transport_and_unchanged_logical_steps(self):
        engine = LiarsBarEngine(game_seed=29)
        engine.step(engine.get_legal_actions()[0])
        observation = engine.get_observation()
        legal_ids = [action.action_id for action in observation.legal_actions]
        # Different answer lengths and a four-target tie expose any remaining
        # single-target-per-epoch accounting: all four must appear in one batch.
        self.assertGreater(len(legal_ids[-1]), len(legal_ids[0]))
        tied_ids = {legal_ids[0], *legal_ids[-3:]}
        self.assertEqual(len(tied_ids), 4)

        def sample(split, index, task, group=None):
            game_id = f"{split}-{index}"
            if task == "belief":
                return {"game_id": game_id, "sample_id": game_id + "-belief", "task": task,
                    "system": BELIEF_SYSTEM_PROMPT, "prompt": build_bluff_user_prompt(observation),
                    "completion": "1", "label": 1, "observation": observation.to_dict(),
                    "contract": current_contract()}
            row = make_action_sample(observation, observation.legal_actions[0], game_id)
            values = {action_id: .1 for action_id in legal_ids}
            values[legal_ids[0]] = .8
            if group == "tied":
                for action_id in tied_ids:
                    values[action_id] = .8
            elif group == "small_gap":
                values[legal_ids[1]] = .78
            row["action_supervision"] = make_action_supervision(row, values)
            return row

        splits = {
            "train": [sample("train", 0, "action", "tied"),
                      sample("train", 1, "action", "small_gap"),
                      sample("train", 2, "action", "clear"),
                      sample("train", 0, "belief"), sample("train", 1, "belief")],
            "val": [sample("val", 0, "action", "tied"),
                    sample("val", 1, "action", "small_gap"), sample("val", 0, "belief")],
            "test": [sample("test", 0, "action", "clear"), sample("test", 0, "belief")],
        }
        config = experiment.load_config(experiment.DEFAULT_CONFIG)
        config["training"].update(batch_size=4, num_epochs=3, seed=23)
        with TemporaryDirectory() as temporary:
            path = Path(temporary)
            service = Service()
            backend = TinkerBackend(budget_path=path / "mock_budget.jsonl", sdk=SDK,
                                    service_client=service)
            backend._tokenizer = Tokenizer()
            for split, rows in splits.items():
                experiment.write_jsonl(path / f"dataset/{split}.jsonl", rows)
            with patch.object(experiment, "make_backend", return_value=backend):
                estimate = experiment.estimate_cost(config, path)
            self.assertEqual(service.training.train_calls, 0)
            self.assertEqual(service.training.val_calls, 0)
            self.assertEqual(backend.ledger.snapshot()["estimated_cost_usd"], 0)

            metadata = run_tinker_lora_sft([], path / "model", splits=splits, backend=backend,
                                           batch_size=4, num_epochs=3, seed=23)
            reservations, tokens = {}, {"train": 0, "validation": 0}
            for event in experiment.read_jsonl(path / "mock_budget.jsonl"):
                if event["event"] == "reserve":
                    reservations[event["request_id"]] = event["kind"]
                elif event["event"] == "complete":
                    tokens[reservations[event["request_id"]]] += event["usage"]["training_tokens"]
            self.assertEqual(tokens["train"], estimate["balanced_training_tokens"])
            self.assertEqual(tokens["validation"], estimate["validation_tokens"])
            self.assertEqual(estimate["validation_action_candidate_sequences_per_epoch"],
                             len(legal_ids) * 2)
            self.assertEqual(service.training.train_calls, 6)  # ceil(3/2) steps x 3 epochs.
            self.assertEqual(service.training.optim_calls, 6)
            self.assertEqual(estimate["training_optimizer_steps_per_epoch"], 2)
            self.assertEqual(estimate["training_optimizer_steps"], 6)
            self.assertEqual(estimate["training_logical_state_visits"], {"action": 12, "belief": 12})
            actual_expanded = {"action": 0, "belief": 0}
            transport_tokens = 0
            tied_state_visits = 0
            for batch in service.training.batches:
                tied_answers, singleton_actions, beliefs = [], 0, 0
                for datum in batch:
                    transport_tokens += len(datum.model_input.to_ints())
                    weights = datum.loss_fn_inputs["weights"]
                    targets = datum.loss_fn_inputs["target_tokens"]
                    answer = backend.tokenizer.decode([token for token, weight in zip(targets, weights) if weight > 0])
                    total_weight = sum(weights)
                    if answer == "1":
                        self.assertAlmostEqual(total_weight, 1.0)
                        beliefs += 1
                        actual_expanded["belief"] += 1
                    else:
                        action_id = json.loads(answer.removesuffix("<end>"))["chosen_action_id"]
                        actual_expanded["action"] += 1
                        if math.isclose(total_weight, (1 / (5 / 6)) / 4):
                            tied_answers.append(action_id)
                        else:
                            self.assertTrue(math.isclose(total_weight, 1 / (5 / 6))
                                            or math.isclose(total_weight, .5 / (5 / 6)))
                            singleton_actions += 1
                self.assertEqual(beliefs, 2)
                self.assertIn(len(tied_answers), (0, 4))
                if tied_answers:
                    self.assertEqual(set(tied_answers), tied_ids)
                    tied_state_visits += 1
                self.assertEqual(singleton_actions + len(tied_answers) // 4, 2)
            self.assertGreaterEqual(tied_state_visits, 3)  # At least once in each logical epoch.
            self.assertEqual(actual_expanded, estimate["training_expanded_sequences"])
            self.assertEqual(actual_expanded["action"], 12 + 3 * tied_state_visits)
            self.assertEqual(transport_tokens, estimate["balanced_training_tokens"])
            self.assertAlmostEqual(metadata["training_metrics"]["train_action_weight_mean"], 5 / 6)
            self.assertAlmostEqual(backend.ledger.snapshot()["estimated_cost_usd"],
                                   estimate["sft_train_and_validation_usd"])


if __name__ == "__main__":
    unittest.main()

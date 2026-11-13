"""Offline probability-arithmetic tests; fake scores are not model results."""
import copy
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.training.action_supervision import make_action_supervision
from liars_agent.training.action_validation import candidate_completion, evaluate_action_validation
from liars_agent.training.data_generator import make_action_sample


class FakeForwardBackend:
    def __init__(self, probabilities, token_counts=None, invalid=None):
        self.probabilities = probabilities
        self.token_counts = token_counts or {}
        self.invalid = invalid
        self.calls = []
        self.built = []

    def build(self, row, backend, max_seq_length, *, completion_override,
              sample_weight, action_supervision_policy):
        assert backend is self
        action_id = json.loads(completion_override)["chosen_action_id"]
        key = (row["sample_id"], action_id)
        count = self.token_counts.get(key, 4)
        self.built.append((key, completion_override, sample_weight))
        return SimpleNamespace(key=key, answer_tokens=count,
            loss_fn_inputs={"weights": [0.0, 0.0] + [sample_weight / count] * count})

    def run_training_batch(self, client, data, *, learning_rate):
        assert learning_rate is None
        self.calls.append([datum.key for datum in data])
        outputs = []
        for datum in data:
            probability = self.probabilities[datum.key]
            logs = [-100.0, -200.0] + [math.log(probability) / datum.answer_tokens] * datum.answer_tokens
            if self.invalid == "nan":
                logs[-1] = float("nan")
            elif self.invalid == "positive":
                logs[-1] = 0.01
            elif self.invalid == "infinite":
                logs[-1] = float("-inf")
            elif self.invalid == "length":
                logs.pop()
            outputs.append({"logprobs": logs})
        if self.invalid == "count":
            outputs.pop()
        return SimpleNamespace(loss_fn_outputs=outputs)


class ActionValidationTests(unittest.TestCase):
    def sample(self, name, seed, values=None, policy=None):
        observation = LiarsBarEngine(game_seed=seed).get_observation()
        row = make_action_sample(observation, observation.legal_actions[0], name)
        row["sample_id"] = name
        # Preserve legacy compact formatting, including trailing whitespace.
        row["completion"] = '{"chosen_action_id":"A0"} \n'
        row["action_supervision"] = make_action_supervision(row, values, policy)
        return row

    def probabilities(self, row, leading):
        actions = row["observation"]["legal_actions"]
        remaining = [action["action_id"] for action in actions if action["action_id"] not in leading]
        probabilities = {**{action_id: .05 / len(remaining) for action_id in remaining}, **leading}
        return {(row["sample_id"], action_id): probability for action_id, probability in probabilities.items()}

    def evaluate(self, rows, backend, policy=None, batch_size=3):
        # These tests isolate scoring arithmetic; shared schema validation still
        # runs through resolve_action_supervision on versioned real observations.
        with patch("liars_agent.training.tinker_train.build_sft_datum", side_effect=backend.build), patch(
                "liars_agent.training.tinker_train.validate_training_sample",
                side_effect=lambda row, **kwargs: PlayerObservation.from_dict(row["observation"])):
            return evaluate_action_validation(rows, backend, object(), batch_size, 16384, policy)

    def test_complete_sequence_set_probability_not_mean_token_softmax(self):
        row = self.sample("tied", 2, {"A0": .8, "A1": .7, "A2": .8})
        original = copy.deepcopy(row)
        backend = FakeForwardBackend(self.probabilities(row, {"A0": .2, "A1": .1, "A2": .3}),
            {("tied", "A0"): 2, ("tied", "A1"): 8, ("tied", "A2"): 4})
        result = self.evaluate([row], backend)
        self.assertAlmostEqual(result["action_set_nll"], -math.log(.5))
        self.assertAlmostEqual(result["candidate_normalized_set_probability"], .5 / .65)
        self.assertEqual(result["action_set_accuracy"], 1.0)
        self.assertAlmostEqual(result["estimated_regret"], 0.0)
        self.assertAlmostEqual(result["canonical_mean_token_nll"], -math.log(.2) / 2)
        self.assertEqual(result["subgroups"]["tied"]["num_samples"], 1)
        self.assertEqual(result["subgroups"]["clear"]["num_samples"], 0)
        self.assertIsNone(result["subgroups"]["clear"]["action_set_nll"])
        count = len(row["observation"]["legal_actions"])
        self.assertEqual(result["num_candidate_sequences"], count)
        self.assertEqual(sum(map(len, backend.calls)), count)
        self.assertTrue(all(len(batch) <= 3 for batch in backend.calls))
        canonical = next(completion for key, completion, _ in backend.built if key[1] == "A0")
        self.assertEqual(canonical, row["completion"])
        self.assertEqual(candidate_completion(row, "A0"), row["completion"])
        self.assertEqual(candidate_completion(row, "A1"), '{"chosen_action_id":"A1"} \n')
        self.assertEqual(row, original)
        self.assertIn("legal_candidate_argmax", result["action_set_accuracy_prediction"])

    def test_state_mean_ignores_ambiguity_weights_and_candidate_count(self):
        policy = {"positive_gap_weight": .1}
        clear = self.sample("clear", 2, {"A0": 1.0, "A1": .2}, policy)
        small = self.sample("small", 0, {"A0": .8, "A1": .78}, policy)
        self.assertNotEqual(len(clear["observation"]["legal_actions"]), len(small["observation"]["legal_actions"]))
        probabilities = {**self.probabilities(clear, {"A0": .1, "A1": .2}),
                         **self.probabilities(small, {"A0": .4, "A1": .1})}
        backend = FakeForwardBackend(probabilities)
        result = self.evaluate([clear, small], backend, policy)
        self.assertAlmostEqual(result["action_set_nll"], (-math.log(.1) - math.log(.4)) / 2)
        self.assertAlmostEqual(result["action_set_accuracy"], .5)
        self.assertAlmostEqual(result["estimated_regret"], .4)
        self.assertEqual(result["estimated_regret_num_samples"], 2)
        self.assertEqual(result["subgroups"]["small_gap"]["num_samples"], 1)
        self.assertTrue(all(weight == 1.0 for _, _, weight in backend.built))

    def test_unscored_and_missing_chosen_candidate_value_have_null_regret(self):
        unscored = self.sample("legacy", 2)
        partial = self.sample("partial", 2, {"A0": 1.0})
        probabilities = {**self.probabilities(unscored, {"A0": .1, "A1": .4}),
                         **self.probabilities(partial, {"A0": .1, "A1": .4})}
        result = self.evaluate([unscored, partial], FakeForwardBackend(probabilities))
        self.assertIsNone(result["estimated_regret"])
        self.assertEqual(result["estimated_regret_num_samples"], 0)
        self.assertEqual(result["subgroups"]["unscored"]["num_samples"], 1)
        self.assertIsNone(result["subgroups"]["unscored"]["estimated_regret"])

    def test_invalid_logprobs_and_output_lengths_are_errors(self):
        row = self.sample("invalid", 2)
        probabilities = self.probabilities(row, {"A0": .1})
        for invalid in ("nan", "positive", "infinite", "length", "count"):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                self.evaluate([row], FakeForwardBackend(probabilities, invalid=invalid))
        impossible = {key: 1.0 for key in probabilities}
        with self.assertRaisesRegex(RuntimeError, "sum to more than one"):
            self.evaluate([row], FakeForwardBackend(impossible))

    def test_extreme_logprobs_use_stable_logsumexp(self):
        row = self.sample("small_probabilities", 2, {"A0": 1.0, "A1": 1.0})
        probabilities = self.probabilities(row, {"A0": 1e-300, "A1": 1e-300})
        result = self.evaluate([row], FakeForwardBackend(probabilities))
        self.assertAlmostEqual(result["action_set_nll"], -math.log(1e-300) - math.log(2))
        self.assertTrue(math.isfinite(result["action_set_nll"]))

    def test_empty_validation_returns_null_metrics_without_forward(self):
        backend = FakeForwardBackend({})
        result = self.evaluate([], backend)
        self.assertEqual(result["num_samples"], 0)
        self.assertIsNone(result["action_set_nll"])
        self.assertIsNone(result["canonical_mean_token_nll"])
        self.assertFalse(backend.calls)

    def test_real_datum_masks_include_complete_answer_and_preserve_canonical(self):
        class Tokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
                result = "".join(f"<{message['role']}>{message['content']}" for message in messages)
                return result + ("<assistant>" if add_generation_prompt else "<end>")

            def encode(self, text, **kwargs):
                return [ord(character) for character in text]

        class Backend:
            tokenizer = Tokenizer()
            sdk = SimpleNamespace(
                ModelInput=SimpleNamespace(from_ints=lambda tokens: SimpleNamespace(tokens=tokens)),
                Datum=lambda **kwargs: SimpleNamespace(**kwargs),
            )

            def __init__(self):
                self.answers = []

            def run_training_batch(self, client, data, *, learning_rate):
                assert learning_rate is None
                outputs = []
                for datum in data:
                    targets = datum.loss_fn_inputs["target_tokens"]
                    weights = datum.loss_fn_inputs["weights"]
                    answer = "".join(chr(token) for token, weight in zip(targets, weights) if weight > 0)
                    self.answers.append(answer)
                    action_id = json.loads(answer.removesuffix("<end>"))["chosen_action_id"]
                    probability = .2 if action_id == "A0" else .01
                    count = sum(weight > 0 for weight in weights)
                    outputs.append({"logprobs": [math.log(probability) / count if weight > 0 else -50.0 for weight in weights]})
                return SimpleNamespace(loss_fn_outputs=outputs)

        row = self.sample("actual-datum", 2)
        backend = Backend()
        result = evaluate_action_validation([row], backend, object(), 5, 16384, None)
        self.assertIn(row["completion"] + "<end>", backend.answers)
        self.assertAlmostEqual(result["action_set_nll"], -math.log(.2))
        self.assertAlmostEqual(result["canonical_mean_token_nll"], -math.log(.2) / len(row["completion"] + "<end>"))
        self.assertEqual(len(backend.answers), len(row["observation"]["legal_actions"]))


if __name__ == "__main__":
    unittest.main()

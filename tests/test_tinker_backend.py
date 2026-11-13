"""Offline protocol tests. Fake transport results are never experimental evidence."""
from __future__ import annotations

import json
import math
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from liars_agent.agents.llm_agent import LLMAgent
from liars_agent.agents.prompt_builder import (SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT,
    build_user_prompt, build_bluff_user_prompt)
from liars_agent.agents.rule_bots import CardCounterBot
from liars_agent.contracts import current_contract
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.training.tinker_backend import BudgetExceededError, BudgetLedger, TinkerBackend
from liars_agent.training.tinker_train import build_sft_datum, run_tinker_lora_sft
from liars_agent.training.action_supervision import normalize_action_policy


class ModelInput:
    def __init__(self, tokens):
        self.tokens = list(tokens)

    @classmethod
    def from_ints(cls, tokens):
        return cls(tokens)

    def to_ints(self):
        return self.tokens


class Tokenizer:
    eos_token_id = ord(">")

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        text += "<assistant>" if add_generation_prompt else "<end>"
        return self.encode(text) if tokenize else text

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class SamplingClient:
    def __init__(self):
        self.calls = []
        self.logprob_calls = []
        self.text = '{"chosen_action_id":"A0"}'
        self.stop_reason = "stop"
        self.error = None
        self.logprob_error = None
        self.label_logprobs = {"0": -1.2, "1": -.2}

    def sample(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[ord(c) for c in self.text],
            stop_reason=self.stop_reason) for _ in range(kwargs["num_samples"])])

    def compute_logprobs(self, prompt, *, prompt_logprobs_last_n):
        self.logprob_calls.append((prompt.tokens, prompt_logprobs_last_n))
        if self.logprob_error:
            raise self.logprob_error
        return [None] * (len(prompt.tokens) - 1) + [self.label_logprobs[chr(prompt.tokens[-1])]]


class TrainingClient:
    def __init__(self):
        self.train_calls = 0
        self.batches = []
        self.optim_calls = 0
        self.val_calls = 0
        self.validation_epoch = 0
        self.saved_samplers = []
        self.saved_states = []

    def output(self, data, loss):
        return SimpleNamespace(loss_fn_outputs=[{"logprobs": [-loss] * len(d.model_input.tokens)} for d in data])

    def forward_backward(self, data, loss_fn):
        self.train_calls += 1
        self.batches.append(data)
        return self.output(data, .4)

    def optim_step(self, params):
        self.optim_calls += 1
        return SimpleNamespace()

    def forward(self, data, loss_fn):
        # Action optimum is epoch 2, belief optimum is epoch 3.
        losses = ([.8, .3, .5], [.2, .9, .1])
        belief = sum(w > 0 for w in data[0].loss_fn_inputs["weights"]) == 1
        # Diagnostics may run before/between full validation. Only a saved
        # complete-epoch checkpoint advances this fake epoch-specific curve.
        self.validation_epoch = sum("-epoch-" in name for name in self.saved_states)
        loss = losses[int(belief)][self.validation_epoch % 3]
        self.val_calls += 1
        return self.output(data, loss)

    def save_state(self, name, **kwargs):
        self.saved_states.append(name)
        return SimpleNamespace(path=f"tinker://run/weights/{name}")

    def save_weights_for_sampler(self, name, **kwargs):
        self.saved_samplers.append(name)
        return SimpleNamespace(path=f"tinker://run/sampler_weights/{name}")


class Service:
    def __init__(self):
        self.sampling = SamplingClient()
        self.training = TrainingClient()
        self.sampling_init = []
        self.training_init = []
        self.resumed = []

    def create_sampling_client(self, **kwargs):
        self.sampling_init.append(kwargs)
        return self.sampling

    def create_lora_training_client(self, **kwargs):
        self.training_init.append(kwargs)
        return self.training

    def create_training_client_from_state(self, path):
        self.resumed.append((path, False))
        return self.training

    def create_training_client_from_state_with_optimizer(self, path):
        self.resumed.append((path, True))
        return self.training


SDK = SimpleNamespace(ModelInput=ModelInput, Datum=lambda **kwargs: SimpleNamespace(**kwargs),
                      SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
                      AdamParams=lambda **kwargs: SimpleNamespace(**kwargs))


def reserve_in_process(path, ready, start, result):
    ledger = BudgetLedger(path, limit_usd=1.0)
    ready.put(True)
    start.wait(5)
    try:
        ledger.reserve("test", .75)
        result.put("reserved")
    except BudgetExceededError:
        result.put("rejected")


class TestTinkerContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.service = Service()
        self.backend = TinkerBackend(budget_path=self.path / "budget.jsonl", sdk=SDK, service_client=self.service)
        self.backend._tokenizer = Tokenizer()
        engine = LiarsBarEngine(game_seed=29)
        engine.step(engine.get_legal_actions()[0])
        self.obs = engine.get_observation()

    def sample(self, game="train", task="action"):
        action = task == "action"
        return {"game_id": game, "sample_id": game + "-" + task, "task": task,
            "system": SYSTEM_PROMPT if action else BELIEF_SYSTEM_PROMPT,
            "prompt": build_user_prompt(self.obs) if action else build_bluff_user_prompt(self.obs),
            "completion": '{"chosen_action_id":"A0"}' if action else "1",
            "label": None if action else 1, "observation": self.obs.to_dict(), "contract": current_contract()}

    def splits(self):
        return {name: [self.sample(name, task) for task in ("action", "belief")]
                for name in ("train", "val", "test")}

    def test_historical_samples_never_reach_training_or_tokenization(self):
        historical = {**self.sample(), "historical_example": True}
        with self.assertRaisesRegex(ValueError, "Historical examples"):
            build_sft_datum(historical, self.backend, 100)
        with self.assertRaisesRegex(ValueError, "Historical examples"):
            run_tinker_lora_sft([historical], self.path / "rejected")
        with self.assertRaisesRegex(ValueError, "Historical examples"):
            run_tinker_lora_sft([], self.path / "rejected", splits={"train": [historical]})
        self.assertFalse(self.service.training_init)
        self.assertFalse((self.path / "rejected").exists())

    def test_action_mean_token_loss_and_belief_single_label_loss(self):
        for task in ("action", "belief"):
            sample = self.sample(task=task)
            datum = build_sft_datum(sample, self.backend, 16384)
            prefix = f"<system>{sample['system']}<user>{sample['prompt']}<assistant>"
            completion = sample["completion"] + ("<end>" if task == "action" else "")
            full = prefix + completion
            self.assertEqual(datum.model_input.tokens, [ord(c) for c in full[:-1]])
            self.assertEqual(datum.loss_fn_inputs["target_tokens"], [ord(c) for c in full[1:]])
            weights = datum.loss_fn_inputs["weights"]
            self.assertTrue(all(w == 0 for w in weights[:len(prefix) - 1]))
            self.assertTrue(all(w == 1 / len(completion) for w in weights[len(prefix) - 1:]))
            self.assertAlmostEqual(sum(weights), 1)
            if task == "belief":
                self.assertEqual(weights[-1], 1)
                self.assertEqual(datum.loss_fn_inputs["target_tokens"][-1], ord("1"))
        with self.assertRaisesRegex(ValueError, "no JSON was truncated"):
            build_sft_datum(self.sample(), self.backend, 4)

    def test_all_split_contracts_prompts_targets_validated_before_api(self):
        changes = [({"contract": {}}, "contract"), ({"completion": '{"chosen_action_id":"missing"}'}, "unavailable"),
            ({"prompt": "changed"}, "prompt"), ({"task": "unknown"}, "task")]
        for update, message in changes:
            with self.subTest(update=update):
                splits = self.splits()
                splits["test"][0] = {**splits["test"][0], **update}
                with patch("liars_agent.training.tinker_train.TinkerBackend") as constructor:
                    with self.assertRaisesRegex(ValueError, message):
                        run_tinker_lora_sft([], self.path / "invalid", splits=splits)
                    constructor.assert_not_called()
        invalid = {**self.sample(task="belief"), "label": 0}
        with self.assertRaisesRegex(ValueError, "matching"):
            build_sft_datum(invalid, self.backend, 16384)
        self.assertFalse(self.service.training_init)

    def test_training_balances_tasks_and_selects_action_validation_best(self):
        splits = self.splits()
        # Longer action pool forces two updates; the single belief sample cycles.
        splits["train"].append({**self.sample(), "sample_id": "train-action-2"})
        result = run_tinker_lora_sft([], self.path / "model", backend=self.backend,
                                    splits=splits, num_epochs=3, batch_size=2)
        self.assertEqual(self.service.training.train_calls, 6)
        self.assertEqual(self.service.training.optim_calls, 6)
        for batch in self.service.training.batches:
            self.assertEqual(len(batch), 2)
            self.assertGreater(sum(w > 0 for w in batch[0].loss_fn_inputs["weights"]), 1)
            self.assertEqual(sum(w > 0 for w in batch[1].loss_fn_inputs["weights"]), 1)
            for datum in batch:
                self.assertAlmostEqual(sum(datum.loss_fn_inputs["weights"]), 1)
        self.assertEqual(result["best_epoch"], 2)
        val_datum = build_sft_datum(splits["val"][0], self.backend, 16384)
        answer_tokens = sum(w > 0 for w in val_datum.loss_fn_inputs["weights"])
        self.assertAlmostEqual(result["best_val_loss"], .3 * answer_tokens)
        self.assertAlmostEqual(result["training_metrics"]["loss_curve"][1]["val_action_canonical_token_nll"], .3)
        self.assertEqual(result["training_metrics"]["loss_curve"][1]["val_belief_loss"], .9)
        self.assertTrue(result["sampler_path"].endswith("sft-best-epoch-2"))
        self.assertTrue(result["state_path"].endswith("sft-epoch-2"))
        self.assertEqual(result["contract"], current_contract())
        self.assertGreater(result["cost_accounting"]["training_tokens"], 0)
        repeated = run_tinker_lora_sft([], self.path / "model", backend=self.backend,
                    splits=splits, num_epochs=3, batch_size=2, resume=True)
        self.assertEqual(self.service.training.train_calls, 6)
        self.assertEqual(repeated["sampler_path"], result["sampler_path"])
        self.assertEqual(self.service.resumed[-1], ("tinker://run/weights/sft-epoch-3", True))
        altered = {**splits, "train": [*splits["train"], {**self.sample(), "sample_id": "extra"}]}
        with self.assertRaisesRegex(ValueError, "changed dataset"):
            run_tinker_lora_sft([], self.path / "model", backend=self.backend,
                               splits=altered, num_epochs=3, batch_size=2, resume=True)

    def test_resume_source_and_progress_need_matching_contract(self):
        state = "tinker://run/weights/sft"
        with self.assertRaisesRegex(ValueError, "contract"):
            run_tinker_lora_sft([], self.path / "rs", backend=self.backend, splits=self.splits(),
                num_epochs=1, batch_size=2, resume_from=state)
        self.assertFalse(self.service.resumed)
        metadata = {"contract": current_contract(), "base_model": self.backend.base_model, "state_path": state,
                    "action_supervision_policy": normalize_action_policy(),
                    "lora_config": {"r": 16, "train_attn": True, "train_mlp": True, "train_unembed": False}}
        run_tinker_lora_sft([], self.path / "rs", backend=self.backend, splits=self.splits(),
                num_epochs=1, batch_size=2, resume_from=state, resume_metadata=metadata, stage="rs")
        self.assertEqual(self.service.resumed, [(state, False)])
        path = self.path / "rs" / "training_progress.json"
        progress = json.loads(path.read_text())
        progress["contract"] = {}
        path.write_text(json.dumps(progress))
        with self.assertRaisesRegex(ValueError, "contract"):
            run_tinker_lora_sft([], self.path / "rs", backend=self.backend, splits=self.splits(),
                num_epochs=1, batch_size=2, resume=True, stage="rs")
        self.assertEqual(self.service.resumed, [(state, False)])

    def test_layer_switches_reach_sdk_and_saved_metadata(self):
        result = run_tinker_lora_sft([], self.path / "switches", backend=self.backend,
                    splits=self.splits(), num_epochs=1, batch_size=2,
                    train_attn=False, train_mlp=True, train_unembed=True)
        self.assertFalse(self.service.training_init[0]["train_attn"])
        self.assertTrue(self.service.training_init[0]["train_unembed"])
        self.assertFalse(result["lora_config"]["train_attn"])
        self.assertTrue(result["lora_config"]["train_unembed"])
        self.assertIsNone(result["checkpoint_ttl_seconds"])

    def test_sampling_consumes_checkpoint_and_counts_exact_tokens(self):
        path = "tinker://run/sampler_weights/sft"
        results = self.backend.sample([{"role": "user", "content": "P"}], checkpoint=path,
                                      num_samples=2, max_tokens=256)
        self.assertEqual(len(self.service.sampling_init), 1)
        self.assertEqual(self.service.sampling_init[0]["model_path"], path)
        self.assertFalse(self.service.sampling_init[0]["retry_config"].enable_retry_logic)
        self.assertEqual(len(results), 2)
        usage = self.backend.ledger.snapshot()
        self.assertEqual(usage["prompt_tokens"], 2 * len("<user>P<assistant>"))
        self.assertEqual(usage["completion_tokens"], 2 * len(self.service.sampling.text))
        self.assertEqual(usage["pending_requests"], 0)

    def test_bluff_logprobs_share_checkpoint_and_account_full_prefix_twice(self):
        path = "tinker://run/sampler_weights/sft"
        result = self.backend.score_bluff([{"role": "user", "content": "P"}], checkpoint=path)
        self.assertAlmostEqual(result.probability, 1 / (1 + math.exp(-1)))
        self.assertEqual(len(self.service.sampling_init), 1)
        self.assertEqual(self.service.sampling_init[0]["model_path"], path)
        self.assertFalse(self.service.sampling_init[0]["retry_config"].enable_retry_logic)
        self.assertEqual([tokens[-1] for tokens, _ in self.service.sampling.logprob_calls], [ord("0"), ord("1")])
        self.assertTrue(all(n == 1 for _, n in self.service.sampling.logprob_calls))
        usage = self.backend.ledger.snapshot()
        self.assertEqual(usage["prompt_tokens"], 2 * len("<user>P<assistant>0"))
        self.assertEqual(usage["completion_tokens"], 0)
        self.assertEqual(usage["completion_tokens_upper_bound"], 2)
        self.assertEqual(result.prompt_tokens, usage["prompt_tokens"])
        self.assertEqual(result.belief_completion_tokens_upper_bound, 2)
        self.assertAlmostEqual(result.estimated_cost_usd, usage["estimated_cost_usd"])
        completed = [json.loads(line) for line in self.backend.ledger.path.read_text().splitlines()
                     if json.loads(line)["event"] == "complete"]
        self.assertTrue(all(event["usage_basis"] == "upper_bound" for event in completed))

    def test_invalid_binary_tokenizer_or_scores_fail_without_probability(self):
        original = self.backend._tokenizer.encode
        self.backend._tokenizer.encode = lambda text, **kw: [4, 5] if text in ("0", "1") else original(text, **kw)
        with self.assertRaisesRegex(ValueError, "single tokens"):
            self.backend.score_bluff([{"role": "user", "content": "P"}])
        self.assertFalse(self.service.sampling_init)
        self.backend._tokenizer.encode = original
        self.service.sampling.label_logprobs["0"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            self.backend.score_bluff([{"role": "user", "content": "P"}])
        self.assertEqual(self.backend.ledger.snapshot()["pending_requests"], 0)
        self.assertGreater(self.backend.ledger.snapshot()["estimated_cost_usd"], 0)

    def test_budget_is_checked_before_call_and_survives_reload(self):
        self.backend.ledger.limit_usd = .000001
        with self.assertRaises(BudgetExceededError):
            self.backend.sample([{"role": "user", "content": "P"}])
        with self.assertRaises(BudgetExceededError):
            self.backend.score_bluff([{"role": "user", "content": "P"}])
        self.assertEqual(self.service.sampling.calls, [])
        self.assertEqual(self.service.sampling.logprob_calls, [])
        self.backend.ledger.limit_usd = 1
        self.service.sampling.error = RuntimeError("transport outcome unknown")
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            self.backend.sample([{"role": "user", "content": "P"}])
        reloaded = BudgetLedger(self.path / "budget.jsonl", limit_usd=1)
        self.assertEqual(reloaded.snapshot()["pending_requests"], 1)
        self.assertGreater(reloaded.snapshot()["reserved_unknown_usd"], 0)

    def test_invalid_action_retains_independent_model_belief_and_fallback_record(self):
        agent = LLMAgent(backend=self.backend, seed=9)
        self.service.sampling.text = "not json"
        action, record = agent.select_action(self.obs)
        self.assertEqual(action, CardCounterBot(seed=9).select_action(self.obs)[0])
        self.assertTrue(record.was_fallback)
        self.assertEqual(record.raw_output, "not json")
        self.assertEqual(record.belief_source, "model")
        self.assertAlmostEqual(record.belief_bluff_prob, 1 / (1 + math.exp(-1)))
        self.assertIsInstance(record.fallback_record, dict)
        self.service.sampling.error = RuntimeError("authentication failed")
        with self.assertRaisesRegex(RuntimeError, "authentication"):
            agent.select_action(self.obs)
        self.assertEqual(agent.tracker.fallback_count, 1)

    def test_candidates_share_one_belief_and_frozen_prediction_never_generates_actions(self):
        agent = LLMAgent(backend=self.backend)
        candidates = agent.sample_candidates(self.obs, num_samples=3)
        self.assertEqual(len(self.service.sampling.logprob_calls), 2)
        self.assertEqual(len(self.service.sampling.calls), 1)
        self.assertEqual(len({r.belief_bluff_prob for _, r in candidates}), 1)
        self.assertEqual(agent.tracker.api_calls, 3)
        self.assertEqual(agent.tracker.belief_calls, 1)
        self.assertEqual(agent.tracker.belief_generated_tokens_upper_bound, 2)
        observed_completions = 3 * len(self.service.sampling.text)
        self.assertEqual(agent.tracker.total_completion_tokens, observed_completions)
        self.assertEqual(sum(record.completion_tokens for _, record in candidates), observed_completions)
        self.assertEqual(sum(record.belief_completion_tokens_upper_bound for _, record in candidates), 2)
        ledger = self.backend.ledger.snapshot()
        self.assertEqual(agent.tracker.total_prompt_tokens, ledger["prompt_tokens"])
        self.assertEqual(agent.tracker.total_completion_tokens, ledger["completion_tokens"])
        self.assertAlmostEqual(agent.tracker.estimated_cost_usd, ledger["estimated_cost_usd"])
        agent.sample_candidates(self.obs, num_samples=3, include_belief=False)
        self.assertEqual(len(self.service.sampling.logprob_calls), 2)
        probability = agent.predict_bluff_prob(self.obs)
        self.assertEqual(probability, candidates[0][1].belief_bluff_prob)
        self.assertEqual(len(self.service.sampling.calls), 2)
        self.assertEqual(len(self.service.sampling.logprob_calls), 4)
        self.assertIsNone(agent.predict_bluff_prob(LiarsBarEngine(game_seed=1).get_observation()))
        self.assertEqual(len(self.service.sampling.logprob_calls), 4)
        self.service.sampling.logprob_error = RuntimeError("belief transport error")
        with self.assertRaisesRegex(RuntimeError, "belief transport"):
            agent.predict_bluff_prob(self.obs)

    def test_adapter_requires_current_contract_and_real_sampler_path(self):
        with self.assertRaisesRegex(ValueError, "requires a trained"):
            LLMAgent(mode="lora_sft", backend=self.backend)
        with self.assertRaisesRegex(ValueError, "contract-bearing"):
            LLMAgent(mode="lora_sft", adapter_path="tinker://run/sampler_weights/sft", backend=self.backend)
        path = self.path / "adapter_metadata.json"
        path.write_text(json.dumps({"base_model": "Qwen/Qwen3.5-9B", "contract": current_contract()}))
        with self.assertRaisesRegex(ValueError, "real Tinker sampler_path"):
            LLMAgent(mode="lora_sft", adapter_path=path, backend=self.backend)
        path.write_text(json.dumps({"base_model": "Qwen/Qwen3.5-9B", "sampler_path": "tinker://run/sampler_weights/sft"}))
        with self.assertRaisesRegex(ValueError, "contract"):
            LLMAgent(mode="lora_sft", adapter_path=path, backend=self.backend)

    def test_valid_action_uses_public_profile_and_deterministic_summary(self):
        from liars_agent.engine.observation import apply_history_ablation
        obs = LiarsBarEngine(game_seed=23).get_observation()
        agent = LLMAgent(backend=self.backend)
        action, record = agent.select_action(obs)
        self.assertEqual(action.action_id, "A0")
        self.assertFalse(record.was_fallback)
        self.assertEqual(record.belief_source, "not_applicable")
        self.assertEqual(record.rationale_source, "deterministic_action_summary")
        self.assertTrue(record.opponent_profile)
        self.assertEqual(agent._sampling_seed(obs), agent._sampling_seed(apply_history_ablation(obs, "last_action_only")))

    def test_two_processes_cannot_overbook_shared_budget(self):
        context = multiprocessing.get_context("spawn")
        ready, result, start = context.Queue(), context.Queue(), context.Event()
        workers = [context.Process(target=reserve_in_process,
                   args=(self.path / "shared.jsonl", ready, start, result)) for _ in range(2)]
        for worker in workers:
            worker.start()
        self.assertTrue(ready.get(timeout=5))
        self.assertTrue(ready.get(timeout=5))
        start.set()
        statuses = [result.get(timeout=5), result.get(timeout=5)]
        for worker in workers:
            worker.join(5)
            self.assertEqual(worker.exitcode, 0)
        self.assertCountEqual(statuses, ["reserved", "rejected"])
        self.assertEqual(BudgetLedger(self.path / "shared.jsonl", 1).snapshot()["reserved_unknown_usd"], .75)

    def test_actual_qwen_tokenizer_and_sdk_mask_when_locally_available(self):
        try:
            import tinker
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("Tinker/Transformers optional dependencies not installed.")
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", local_files_only=True)
        except OSError:
            self.skipTest("Qwen tokenizer is not cached; test never downloads it.")
        backend = TinkerBackend(budget_path=self.path / "sdk.jsonl", sdk=tinker, service_client=self.service)
        backend._tokenizer = tokenizer
        for task in ("action", "belief"):
            sample = self.sample(task=task)
            datum = build_sft_datum(sample, backend, 16384)
            messages = [{"role": "system", "content": sample["system"]}, {"role": "user", "content": sample["prompt"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=True,
                        add_generation_prompt=True, enable_thinking=False, return_dict=False)
            if task == "action":
                full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": sample["completion"]}],
                            tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False)
                end = full.index(tokenizer.eos_token_id, len(prompt)) + 1
                self.assertGreater(len(full), end)  # Real Qwen appends a separator after EOS.
                full = full[:end]
                self.assertEqual(datum.loss_fn_inputs["target_tokens"].tolist()[-1], tokenizer.eos_token_id)
            else:
                full = prompt + tokenizer.encode(sample["completion"], add_special_tokens=False)
                self.assertEqual(len(full), len(prompt) + 1)
            self.assertEqual(datum.model_input.to_ints(), full[:-1])
            self.assertEqual(datum.loss_fn_inputs["target_tokens"].tolist(), full[1:])
            weights = datum.loss_fn_inputs["weights"].tolist()
            self.assertEqual(weights[:len(prompt) - 1], [0.] * (len(prompt) - 1))
            self.assertAlmostEqual(sum(weights), 1., places=6)
            self.assertEqual(sum(w > 0 for w in weights), len(full) - len(prompt))


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Tinker online RL (Paired GRPO/PPO) trainer, datum alignment, and resume safety."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import threading
import unittest

import tinker

from liars_agent.agents.llm_agent import LLMAgent
from liars_agent.contracts import current_contract
from liars_agent.training.adaptive_opponents import ParameterizedArchetypeBot, V4_TRAIN_ARCHETYPES
from liars_agent.training.adaptive_teacher import extract_v4_belief_samples, generate_adaptive_v4_replays
from liars_agent.training.tinker_backend import TinkerBackend
from liars_agent.training.tinker_rl import (
    RLEpisodeResult,
    RLStepRecord,
    build_rl_datum,
    compute_group_relative_advantages,
    compute_step_returns,
    run_tinker_lora_rl,
)


class _DummyTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False, return_dict=False):
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        text = "\n".join(parts)
        if not tokenize:
            return text
        return self.encode(text, add_special_tokens=False)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "0":
            return [10]
        if text == "1":
            return [11]
        if text.startswith('{"chosen_action_id":'):
            return [20, 21, 22, 2]
        # Deterministic compact byte-bucket encoding
        return [100 + (b % 50) for b in text.encode("utf-8")[:64]] + ([2] if "</assistant>" in text else [])

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        toks = list(tokens)
        if toks == [20, 21, 22, 2]:
            return '{"chosen_action_id":"A0"}'
        if toks == [20, 21, 23, 2]:
            return '{"chosen_action_id":"A1"}'
        return "INVALID_JSON"


@dataclass
class _FakeSeq:
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str = "stop"


@dataclass
class _FakeSampleResponse:
    sequences: list[_FakeSeq]


@dataclass
class _FakePathResp:
    path: str


class _FakeSamplingClient:
    def __init__(self, call_counter: list[int], lock: threading.Lock | None = None) -> None:
        self.call_counter = call_counter
        self._lock = lock or threading.Lock()

    def sample(self, *, prompt, num_samples: int, sampling_params):
        with self._lock:
            idx = self.call_counter[0]
            self.call_counter[0] += 1
        # Alternate between A0 (CHALLENGE) and A1 (PLAY) so paired rollouts exhibit genuine variance
        toks = [20, 21, 22, 2] if (idx % 2 == 0) else [20, 21, 23, 2]
        return _FakeSampleResponse([_FakeSeq(tokens=toks, logprobs=[-0.4, -0.3, -0.2, -0.1]) for _ in range(num_samples)])


class _FakeTrainingClient:
    def __init__(self, call_log: list[tuple[str, str]], sample_counter: list[int]) -> None:
        self.call_log = call_log
        self.sample_counter = sample_counter
        self._lock = threading.Lock()

    def forward_backward(self, data, loss_fn: str, **kwargs):
        self.call_log.append(("forward_backward", loss_fn))
        return {"loss": 0.1, "num_datums": len(data)}

    def optim_step(self, adam_params):
        self.call_log.append(("optim_step", f"lr={adam_params.learning_rate:.2e}"))
        return {"ok": True}

    def save_weights_and_get_sampling_client(self):
        self.call_log.append(("refresh_sampler", "inflight"))
        return _FakeSamplingClient(self.sample_counter, self._lock)

    def save_state(self, name: str):
        self.call_log.append(("save_state", name))
        return _FakePathResp(f"tinker://mock-rl:train:0/weights/{name}")

    def save_weights_for_sampler(self, name: str):
        self.call_log.append(("save_weights_for_sampler", name))
        return _FakePathResp(f"tinker://mock-rl:train:0/sampler_weights/{name}")


class _FakeServiceClient:
    def __init__(self, call_log: list[tuple[str, str]], sample_counter: list[int]) -> None:
        self.call_log = call_log
        self.sample_counter = sample_counter

    def create_training_client_from_state(self, state_path: str):
        self.call_log.append(("from_state", state_path))
        return _FakeTrainingClient(self.call_log, self.sample_counter)

    def create_training_client_from_state_with_optimizer(self, state_path: str):
        self.call_log.append(("from_state_with_optimizer", state_path))
        return _FakeTrainingClient(self.call_log, self.sample_counter)

    def create_sampling_client(self, **kwargs):
        self.call_log.append(("create_sampling_client", str(kwargs.get("model_path"))))
        return _FakeSamplingClient(self.sample_counter)


def _make_mock_backend(budget_path: Path, call_log: list[tuple[str, str]], sample_counter: list[int]) -> TinkerBackend:
    service = _FakeServiceClient(call_log, sample_counter)
    backend = TinkerBackend(
        base_model="Qwen/Qwen3.5-9B",
        budget_path=budget_path,
        budget_limit_usd=5000.0,
        sdk=tinker,
        service_client=service,
    )
    backend._tokenizer = _DummyTokenizer()
    return backend


class TinkerRLTests(unittest.TestCase):
    def test_build_rl_datum_alignment_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _make_mock_backend(Path(tmp) / "budget.jsonl", [], [0])
            datum = build_rl_datum(
                prompt_token_ids=[101, 102, 103],
                completion_token_ids=[20, 21, 2],
                completion_logprobs=[-0.5, -0.25, -0.1],
                advantage=1.25,
                backend=backend,
                max_seq_length=32,
            )
            self.assertEqual(list(datum.model_input.to_ints()), [101, 102, 103, 20, 21])
            self.assertEqual(list(datum.loss_fn_inputs["target_tokens"].data), [102, 103, 20, 21, 2])
            for got, exp in zip(datum.loss_fn_inputs["logprobs"].data, [0.0, 0.0, -0.5, -0.25, -0.1]):
                self.assertAlmostEqual(got, exp, places=5)
            for got, exp in zip(datum.loss_fn_inputs["advantages"].data, [0.0, 0.0, 1.25, 1.25, 1.25]):
                self.assertAlmostEqual(got, exp, places=5)

            with self.assertRaises(ValueError):
                build_rl_datum([101], [20, 21], [-0.5], 1.0, backend, 32)
            with self.assertRaises(ValueError):
                build_rl_datum([101], [20], [0.5], 1.0, backend, 32)  # positive logprob
            with self.assertRaises(ValueError):
                build_rl_datum([101, 102], [20, 21], [-0.2, -0.3], 1.0, backend, 3)  # exceeds max_seq_length

    def test_compute_group_relative_advantages_and_zero_variance_skip(self):
        step_ok = RLStepRecord(1, 2, (1, 2), (20, 2), (-0.4, -0.1), "PLAY", False)
        step_fb = RLStepRecord(1, 2, (1, 2), (99, 2), (-0.8, -0.2), "PLAY", True)
        ep_win = RLEpisodeResult("g0", 0, True, {1: 1}, (step_ok,))
        ep_lose = RLEpisodeResult("g1", 0, False, {1: 0}, (step_fb,))

        pairs, stats = compute_group_relative_advantages([ep_win, ep_lose])
        self.assertEqual(len(pairs), 2)
        self.assertGreater(pairs[0][1], 0.0)
        self.assertLess(pairs[1][1], 0.0)
        self.assertGreater(stats["std_return"], 0.0)

        # Identical episodes -> zero variance -> skipped cleanly
        pairs_zero, stats_zero = compute_group_relative_advantages([ep_win, ep_win])
        self.assertEqual(pairs_zero, [])
        self.assertEqual(stats_zero["kept_steps"], 0)

    def test_run_tinker_lora_rl_end_to_end_belief_anchor_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replays = generate_adaptive_v4_replays(4, base_seed=20261124)
            belief_rows = extract_v4_belief_samples(replays, target_samples=12, seed=42)
            belief_path = root / "belief_train.jsonl"
            belief_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in belief_rows) + "\n",
                encoding="utf-8",
            )

            source_meta = {
                "contract": current_contract("adaptive_v4"),
                "base_model": "Qwen/Qwen3.5-9B",
                "state_path": "tinker://mock-sft:train:0/weights/sft-epoch-3",
                "sampler_path": "tinker://mock-sft:train:0/sampler_weights/sft-best-epoch-3",
                "lora_config": {"r": 16, "train_attn": True, "train_mlp": True, "train_unembed": False},
                "formal_training_complete": True,
            }

            # Contract & LoRA mismatch must fail BEFORE any API call
            bad_log: list[tuple[str, str]] = []
            bad_backend = _make_mock_backend(root / "bad_budget.jsonl", bad_log, [0])
            with self.assertRaises(ValueError):
                run_tinker_lora_rl(
                    output_dir=root / "bad_rl",
                    source_metadata={**source_meta, "lora_config": {"r": 8, "train_attn": True, "train_mlp": True, "train_unembed": False}},
                    opponent_pool_sampler=lambda s: [lambda sd, sp=V4_TRAIN_ARCHETYPES[i]: ParameterizedArchetypeBot(sp, seed=sd) for i in range(3)],
                    backend=bad_backend,
                )
            self.assertEqual(bad_log, [])

            # Run 2 steps with checkpoint_every_steps=1 + belief aux anchor
            call_log: list[tuple[str, str]] = []
            sample_counter = [0]
            backend = _make_mock_backend(root / "budget.jsonl", call_log, sample_counter)
            opp_sampler = lambda s: [
                lambda sd, sp=V4_TRAIN_ARCHETYPES[i]: ParameterizedArchetypeBot(sp, seed=sd)
                for i in range(3)
            ]
            rl_dir = root / "models/rl"
            meta = run_tinker_lora_rl(
                output_dir=rl_dir,
                source_metadata=source_meta,
                opponent_pool_sampler=opp_sampler,
                belief_train_path=belief_path,
                num_steps=2,
                groups_per_step=1,
                group_size=2,
                belief_aux_batch_size=2,
                belief_aux_weight=0.25,
                checkpoint_every_steps=1,
                seed=77,
                method_version="adaptive_v4",
                backend=backend,
            )
            self.assertEqual(meta["stage"], "rl")
            self.assertTrue(meta["formal_training_complete"])
            self.assertTrue(meta["sampler_path"].startswith("tinker://"))
            self.assertTrue((rl_dir / "adapter_metadata.json").exists())
            self.assertTrue((rl_dir / "rl_metrics.jsonl").exists())
            self.assertTrue((rl_dir / "rl_progress.json").exists())
            self.assertIn(("from_state", source_meta["state_path"]), call_log)
            self.assertIn(("forward_backward", "ppo"), call_log)
            self.assertIn(("forward_backward", "cross_entropy"), call_log)
            self.assertGreater(backend.ledger.snapshot()["estimated_cost_usd"], 0.0)

            # Verify LLMAgent loads the resulting lora_rl adapter metadata cleanly
            rl_agent = LLMAgent(
                name="lora_rl",
                mode="lora_rl",
                model_version="Qwen/Qwen3.5-9B",
                adapter_path=rl_dir / "adapter_metadata.json",
                backend=backend,
                method_version="adaptive_v4",
            )
            self.assertEqual(rl_agent.checkpoint, meta["sampler_path"])

            # Resume from step 2 -> step 3 must use create_training_client_from_state_with_optimizer
            resume_log: list[tuple[str, str]] = []
            resume_backend = _make_mock_backend(root / "budget.jsonl", resume_log, [0])
            meta_resumed = run_tinker_lora_rl(
                output_dir=rl_dir,
                source_metadata=source_meta,
                opponent_pool_sampler=opp_sampler,
                belief_train_path=belief_path,
                num_steps=3,
                groups_per_step=1,
                group_size=2,
                belief_aux_batch_size=2,
                checkpoint_every_steps=1,
                seed=77,
                method_version="adaptive_v4",
                backend=resume_backend,
            )
            self.assertEqual(meta_resumed["summary"]["total_steps"], 3)
            self.assertTrue(any(op == "from_state_with_optimizer" for op, _ in resume_log))

    def test_compute_step_returns_discounting_and_shaping(self):
        # Round 1 voluntary caught bluff (had_truth_card=True) vs Round 3 successful challenge
        st_early_bluff = RLStepRecord(
            1, 1, (1, 2), (20, 2), (-0.4, -0.1), "PLAY", False,
            was_bluff_play=True, had_truth_card=True, caught_bluff_this_round=True,
        )
        st_forced_bluff = RLStepRecord(
            2, 1, (1, 2), (20, 2), (-0.4, -0.1), "PLAY", False,
            was_bluff_play=True, had_truth_card=False, caught_bluff_this_round=True,
        )
        st_late_hit = RLStepRecord(
            3, 2, (1, 2), (20, 2), (-0.4, -0.1), "CHALLENGE", False,
            challenge_succeeded=True, had_truth_card=True,
        )
        ep = RLEpisodeResult(
            game_id="g_shape",
            seat=0,
            won=True,
            round_penalized_seat={1: 0, 2: 0, 3: 1},
            steps=(st_early_bluff, st_forced_bluff, st_late_hit),
        )
        returns = compute_step_returns(
            ep,
            win_weight=0.70,
            round_survival_weight=0.30,
            gamma=0.90,
            challenge_hit_bonus=0.15,
            caught_bluff_penalty=-0.20,
        )
        # Round 1: 0.70 * 0.90^2 + 0.0 - 0.20 = 0.367 (penalized + caught voluntary bluff)
        # Round 2: 0.70 * 0.90^1 + 0.0 (penalized, no caught_bluff_penalty because had_truth_card=False) = 0.63
        # Round 3: 0.70 * 0.90^0 + 0.30 + 0.15 = 1.15 (survived + challenge hit bonus)
        self.assertAlmostEqual(returns[0], 0.70 * 0.81 - 0.20, places=5)
        self.assertAlmostEqual(returns[1], 0.70 * 0.90, places=5)
        self.assertAlmostEqual(returns[2], 0.70 + 0.30 + 0.15, places=5)
        self.assertGreater(returns[2], returns[1])
        self.assertGreater(returns[1], returns[0])

    def test_run_tinker_lora_rl_v2_on_policy_belief_concurrent_and_val_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replays = generate_adaptive_v4_replays(4, base_seed=20261124)
            belief_rows = extract_v4_belief_samples(replays, target_samples=12, seed=42)
            belief_path = root / "belief_train.jsonl"
            belief_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in belief_rows) + "\n",
                encoding="utf-8",
            )
            source_meta = {
                "contract": current_contract("adaptive_v4"),
                "base_model": "Qwen/Qwen3.5-9B",
                "state_path": "tinker://mock-sft:train:0/weights/sft-epoch-3",
                "sampler_path": "tinker://mock-sft:train:0/sampler_weights/sft-best-epoch-3",
                "lora_config": {"r": 16, "train_attn": True, "train_mlp": True, "train_unembed": False},
                "formal_training_complete": True,
            }
            call_log: list[tuple[str, str]] = []
            backend = _make_mock_backend(root / "budget.jsonl", call_log, [0])
            opp_sampler = lambda s: [
                lambda sd, sp=V4_TRAIN_ARCHETYPES[i]: ParameterizedArchetypeBot(sp, seed=sd)
                for i in range(3)
            ]
            rl_v2_dir = root / "models/rl_v2"
            meta = run_tinker_lora_rl(
                output_dir=rl_v2_dir,
                source_metadata=source_meta,
                opponent_pool_sampler=opp_sampler,
                eval_opponent_sampler=opp_sampler,
                belief_train_path=belief_path,
                num_steps=2,
                groups_per_step=1,
                group_size=2,
                gamma=0.90,
                challenge_hit_bonus=0.15,
                caught_bluff_penalty=-0.20,
                belief_aux_batch_size=4,
                belief_aux_weight=0.25,
                on_policy_belief_ratio=0.50,
                rollout_workers=2,
                eval_every_steps=1,
                eval_blocks=1,
                eval_temperature=0.2,
                checkpoint_every_steps=1,
                seed=99,
                method_version="adaptive_v4",
                stage_name="rl_v2",
                backend=backend,
            )
            self.assertEqual(meta["stage"], "rl")
            self.assertEqual(meta["rl_variant"], "rl_v2")
            self.assertIsNotNone(meta["best_checkpoint"])
            self.assertEqual(meta["sampler_path"], meta["best_checkpoint"]["sampler_path"])
            self.assertTrue(meta["final_sampler_path"].startswith("tinker://"))

            metric_rows = [json.loads(line) for line in (rl_v2_dir / "rl_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(metric_rows), 2)
            self.assertTrue(all(r["validation"] is not None for r in metric_rows))
            self.assertTrue(any(r["on_policy_beliefs_used"] > 0 for r in metric_rows))

            v2_agent = LLMAgent(
                name="lora_rl_v2",
                mode="lora_rl_v2",
                model_version="Qwen/Qwen3.5-9B",
                adapter_path=rl_v2_dir / "adapter_metadata.json",
                backend=backend,
                method_version="adaptive_v4",
            )
            self.assertEqual(v2_agent.checkpoint, meta["sampler_path"])


if __name__ == "__main__":
    unittest.main()

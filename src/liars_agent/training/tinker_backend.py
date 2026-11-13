"""Tinker token inference and durable, token-based spending accounting.

Costs here are conservative estimates using published prices, not provider
invoices. Uncertain requests keep their reservation until explicitly reconciled.
"""

from __future__ import annotations

import json
import fcntl
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RATES = {
    "prefill": 0.30,
    "sample": 0.807,
    "train": 1.463,
}
PRICING_SOURCE = "https://tinker-docs.thinkingmachines.ai/tinker/models.json"
PRICING_DATE = "2026-11-14"


class BudgetExceededError(RuntimeError):
    pass


class BudgetLedger:
    """Append-only request reservations shared by one experiment's backend."""

    def __init__(self, path: str | Path, limit_usd: float = 5000.0) -> None:
        if not math.isfinite(limit_usd) or limit_usd <= 0 or limit_usd > 5000:
            raise ValueError("budget_limit_usd must be in (0, 5000].")
        self.path = Path(path)
        self.limit_usd = float(limit_usd)
        self._lock = threading.RLock()
        self._pending: dict[str, float] = {}
        self._spent = 0.0
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0,
                       "completion_tokens_upper_bound": 0, "training_tokens": 0}
        self._offset = 0
        if self.path.exists():
            with self._transaction():
                pass

    def _consume(self, event: dict[str, Any]) -> None:
        if event["event"] == "reserve":
            self._pending[event["request_id"]] = event["reserved_usd"]
        elif event["event"] == "complete":
            self._pending.pop(event["request_id"], None)
            self._spent += event["estimated_cost_usd"]
            for key in self._usage:
                self._usage[key] += int(event.get("usage", {}).get(key, 0))

    @contextmanager
    def _transaction(self):
        # UI and CLI can spend from the same ledger. The read/check/append must
        # be one process-locked operation, not just a lock around the append.
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a+", encoding="utf-8") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    stream.seek(self._offset)
                    for line in stream.readlines():
                        self._consume(json.loads(line))
                    self._offset = stream.tell()
                    yield stream
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _append(self, stream: Any, event: dict[str, Any]) -> None:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        self._offset = stream.tell()
        self._consume(event)

    def reserve(self, kind: str, max_cost_usd: float, **metadata: Any) -> str:
        if not math.isfinite(max_cost_usd) or max_cost_usd < 0:
            raise ValueError("Invalid cost estimate.")
        with self._transaction() as stream:
            if self._spent + sum(self._pending.values()) + max_cost_usd > self.limit_usd:
                raise BudgetExceededError(f"Request would exceed ${self.limit_usd:.2f} experiment budget.")
            request_id = uuid.uuid4().hex
            self._append(stream, {"event": "reserve", "request_id": request_id, "kind": kind,
                          "reserved_usd": max_cost_usd, "time": time.time(), **metadata})
            return request_id

    def complete(self, request_id: str, estimated_cost_usd: float, *,
                 usage_basis: str = "observed_tokens", **usage: int) -> None:
        with self._transaction() as stream:
            reserved = self._pending[request_id]
            if not math.isfinite(estimated_cost_usd) or estimated_cost_usd < 0:
                raise ValueError("Invalid final cost estimate.")
            if estimated_cost_usd > reserved + 1e-10:
                raise RuntimeError("Actual token estimate exceeded reservation; keep reservation for reconciliation.")
            self._append(stream, {"event": "complete", "request_id": request_id,
                          "estimated_cost_usd": estimated_cost_usd, "usage": usage,
                          "usage_basis": usage_basis, "time": time.time()})

    def snapshot(self) -> dict[str, Any]:
        with self._transaction():
            return {"budget_limit_usd": self.limit_usd,
                    "estimated_cost_usd": self._spent,
                    "reserved_unknown_usd": sum(self._pending.values()),
                    "pending_requests": len(self._pending), **self._usage,
                    "accounting": "token-based estimate; generated-token upper bounds are separate from observed completion_tokens; provider invoice not inferred",
                    "ledger_path": str(self.path)}


@dataclass(frozen=True)
class SampleResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    stop_reason: str
    latency_ms: float


@dataclass(frozen=True)
class BluffScoreResult:
    probability: float
    logprob_zero: float
    logprob_one: float
    prompt_tokens: int
    belief_completion_tokens_upper_bound: int
    latency_ms: float
    estimated_cost_usd: float


def resolve_future(value: Any) -> Any:
    return value.result() if hasattr(value, "result") else value


class TinkerBackend:
    """One authenticated session, tokenizer and checkpoint clients per experiment."""

    def __init__(self, base_model: str = "Qwen/Qwen3.5-9B", *,
                 budget_path: str | Path = "outputs/budget_usage.jsonl",
                 budget_limit_usd: float = 5000.0, api_key: str | None = None,
                 rates: dict[str, float] | None = None,
                 sdk: Any = None, service_client: Any = None) -> None:
        if sdk is None:
            try:
                import tinker as sdk
            except ImportError as exc:
                raise RuntimeError("Real model execution requires the tinker optional dependency.") from exc
        self.sdk = sdk
        self.base_model = base_model
        if rates is None and base_model != "Qwen/Qwen3.5-9B":
            raise ValueError("Supply published token rates for a different base model.")
        self.rates = dict(DEFAULT_RATES if rates is None else rates)
        for key in ("prefill", "sample", "train"):
            if key not in self.rates or not math.isfinite(self.rates[key]) or self.rates[key] < 0:
                raise ValueError(f"Invalid rate: {key}")
        self.ledger = BudgetLedger(budget_path, budget_limit_usd)
        self._api_key = api_key
        self._service = service_client
        self._sampling_clients: dict[str, Any] = {}
        self._tokenizer: Any = None
        self._preflight_done = service_client is not None

    def preflight(self) -> dict[str, Any]:
        if not self._preflight_done:
            kwargs = {"api_key": self._api_key} if self._api_key else {}
            rest = self.sdk.RestClient(**kwargs)
            capabilities = resolve_future(rest.get_server_capabilities())
            names = [m.model_name for m in capabilities.supported_models]
            if self.base_model not in names:
                raise ValueError(f"Tinker account does not support {self.base_model}.")
            self._service = self.sdk.ServiceClient(display_name="liars-agent", **kwargs)
            self._preflight_done = True
        return {"backend": "tinker", "base_model": self.base_model,
                "rates_per_million_tokens": self.rates, "pricing_source": PRICING_SOURCE,
                "pricing_date": PRICING_DATE, "budget": self.ledger.snapshot()}

    @property
    def service(self) -> Any:
        self.preflight()
        return self._service

    def sampling_client(self, checkpoint: str | None = None) -> Any:
        key = checkpoint or self.base_model
        if checkpoint is not None and (not checkpoint.startswith("tinker://") or "/sampler_weights/" not in checkpoint):
            raise ValueError("Inference requires a tinker://.../sampler_weights/... checkpoint.")
        if key not in self._sampling_clients:
            kwargs = {"model_path": checkpoint} if checkpoint else {"base_model": self.base_model}
            self._sampling_clients[key] = resolve_future(self.service.create_sampling_client(**kwargs))
        return self._sampling_clients[key]

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            # Tokenization and cost estimation do not need a paid API session.
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return self._tokenizer

    def encode_messages(self, messages: list[dict[str, str]], *, generation: bool) -> list[int]:
        return list(self.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=generation, enable_thinking=False, return_dict=False))

    def estimate_training_cost(self, tokens: int) -> float:
        return tokens * self.rates["train"] / 1_000_000

    def score_bluff(self, messages: list[dict[str, str]], *,
                    checkpoint: str | None = None) -> BluffScoreResult:
        """Normalize P('1'|context) against P('0'|context) on the same adapter.

        Tinker's compute_logprobs internally samples up to one extra token.
        Each call reserves and charges the full uncached prefix plus that token
        as a conservative upper bound; it never assumes free prefix caching.
        """
        prefix = self.encode_messages(messages, generation=True)
        label_tokens = [self.tokenizer.encode(label, add_special_tokens=False) for label in ("0", "1")]
        if any(len(tokens) != 1 for tokens in label_tokens) or label_tokens[0] == label_tokens[1]:
            raise ValueError("Binary belief labels 0 and 1 must be distinct single tokens.")
        client = self.sampling_client(checkpoint)
        logprobs = []
        prompt_tokens = 0
        total_cost = 0.0
        start = time.perf_counter()
        for label, tokens in zip(("0", "1"), label_tokens):
            sequence = prefix + tokens
            cost = (len(sequence) * self.rates["prefill"] + self.rates["sample"]) / 1_000_000
            reservation = self.ledger.reserve("belief_logprob", cost, model=self.base_model,
                checkpoint=checkpoint, label=label, rates=self.rates,
                accounting="full-prefix plus one generated token upper bound",
                pricing_source=PRICING_SOURCE, pricing_date=PRICING_DATE)
            result = resolve_future(client.compute_logprobs(
                self.sdk.ModelInput.from_ints(sequence), prompt_logprobs_last_n=1))
            # A returned response incurred the call even if its scores are unusable.
            self.ledger.complete(reservation, cost, prompt_tokens=len(sequence),
                                 completion_tokens_upper_bound=1, usage_basis="upper_bound")
            if len(result) != len(sequence) or not isinstance(result[-1], (int, float)) or not math.isfinite(result[-1]):
                raise RuntimeError("Tinker did not return a finite log probability for the belief label.")
            logprobs.append(float(result[-1]))
            prompt_tokens += len(sequence)
            total_cost += cost
        difference = logprobs[1] - logprobs[0]
        probability = 1 / (1 + math.exp(-difference)) if difference >= 0 else math.exp(difference) / (1 + math.exp(difference))
        return BluffScoreResult(probability, logprobs[0], logprobs[1], prompt_tokens, 2,
                                (time.perf_counter() - start) * 1000, total_cost)

    def sample(self, messages: list[dict[str, str]], *, checkpoint: str | None = None,
               num_samples: int = 1, temperature: float = 0.2, top_p: float = 0.9,
               max_tokens: int = 256, seed: int | None = None) -> list[SampleResult]:
        if num_samples < 1 or max_tokens < 1:
            raise ValueError("num_samples and max_tokens must be positive.")
        tokens = self.encode_messages(messages, generation=True)
        client = self.sampling_client(checkpoint)
        max_cost = (len(tokens) * num_samples * self.rates["prefill"] +
                    max_tokens * num_samples * self.rates["sample"]) / 1_000_000
        request_id = self.ledger.reserve("sample", max_cost, model=self.base_model,
                                        checkpoint=checkpoint, num_samples=num_samples,
                                        rates=self.rates, pricing_source=PRICING_SOURCE,
                                        pricing_date=PRICING_DATE)
        params = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
        if seed is not None:
            params["seed"] = seed
        start = time.perf_counter()
        result = resolve_future(client.sample(prompt=self.sdk.ModelInput.from_ints(tokens),
                                num_samples=num_samples, sampling_params=self.sdk.SamplingParams(**params)))
        latency = (time.perf_counter() - start) * 1000
        outputs = [SampleResult(self.tokenizer.decode(s.tokens, skip_special_tokens=True),
                               len(tokens), len(s.tokens), s.stop_reason, latency / num_samples)
                   for s in result.sequences]
        total_completion = sum(item.completion_tokens for item in outputs)
        cost = (len(tokens) * num_samples * self.rates["prefill"] +
                total_completion * self.rates["sample"]) / 1_000_000
        self.ledger.complete(request_id, cost, prompt_tokens=len(tokens) * num_samples,
                             completion_tokens=total_completion)
        if len(outputs) != num_samples:
            raise RuntimeError(f"Tinker returned {len(outputs)} samples; expected {num_samples}.")
        return outputs

    def run_training_batch(self, client: Any, data: list[Any], *,
                           learning_rate: float | None) -> Any:
        token_count = sum(len(item.model_input.to_ints()) for item in data)
        cost = self.estimate_training_cost(token_count)
        request_id = self.ledger.reserve("train" if learning_rate is not None else "validation",
                                        cost, model=self.base_model, rates=self.rates,
                                        pricing_source=PRICING_SOURCE, pricing_date=PRICING_DATE)
        if learning_rate is None:
            result = resolve_future(client.forward(data, "cross_entropy"))
        else:
            result = resolve_future(client.forward_backward(data, "cross_entropy"))
            resolve_future(client.optim_step(self.sdk.AdamParams(learning_rate=learning_rate)))
        self.ledger.complete(request_id, cost, training_tokens=token_count)
        return result

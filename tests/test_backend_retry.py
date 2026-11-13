"""Exercise the installed SDK's retry wrapper with in-memory transport only."""
import asyncio
from concurrent.futures import Future
from contextlib import asynccontextmanager
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import tinker
from tinker.lib.public_interfaces.sampling_client import SamplingClient
from tinker.lib.retry_handler import RetryHandler
from tinker.lib.retryable_exception import RetryableException

from liars_agent.training.tinker_backend import TinkerBackend


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [10, 11, 12]

    def encode(self, label, **kwargs):
        return [int(label)]


class MemoryHolder:
    """No HTTP client, service session, credential or tokenizer download exists."""
    _sample_backoff_until = None
    _client_config = SimpleNamespace(sample_no_retries=False, sample_use_retrieve_futures=False)

    def get_client_config(self):
        return self._client_config

    def get_telemetry(self):
        return None

    def estimate_bytes_count_in_model_input(self, prompt):
        return 1

    @asynccontextmanager
    async def sample_dispatch_rate_limit(self, size):
        yield

    async def execute_with_retries(self, function, *args):
        return await function(*args)

    def run_coroutine_threadsafe(self, coroutine):
        future = Future()
        try:
            future.set_result(asyncio.run(coroutine))
        except BaseException as error:
            future.set_exception(error)
        return SimpleNamespace(future=lambda: future)


class MemoryService:
    def __init__(self):
        self.clients = []
        self.requests = []
        self.options = []

    def create_sampling_client(self, **kwargs):
        self.options.append(kwargs)
        client = object.__new__(SamplingClient)
        client.holder = MemoryHolder()
        client._cloned_sampler_id = 0
        client._request_id_counter = 7
        client.retry_handler = RetryHandler(config=kwargs["retry_config"], name="offline-retry-test")

        async def submit(request_id, *args):
            self.requests.append(request_id)
            return SimpleNamespace(sample_sequence_ids=[f"offline-sequence-{request_id}"])

        client._send_asample_request = submit
        self.clients.append(client)
        return client


class FailedResult:
    def __init__(self, *args, **kwargs):
        pass

    async def result_async(self):
        # Submission succeeded but the result is uncertain. The real SDK's
        # enabled outer RetryHandler would call _sample_async_impl again.
        raise RetryableException("uncertain result after submission")


class BackendRetryTests(unittest.TestCase):
    def test_sampling_and_belief_uncertain_results_do_not_submit_another_sequence(self):
        for operation in ("sample", "score_bluff"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                service = MemoryService()
                backend = TinkerBackend(sdk=tinker, service_client=service,
                                        budget_path=Path(directory) / "ledger.jsonl")
                backend._tokenizer = Tokenizer()
                with patch("tinker.lib.public_interfaces.sampling_client._APIFuture", FailedResult):
                    with self.assertRaisesRegex(RetryableException, "uncertain result"):
                        getattr(backend, operation)([{"role": "user", "content": "public input"}])
                self.assertEqual(service.requests, [7])
                self.assertEqual(service.clients[0]._request_id_counter, 8)
                self.assertFalse(service.options[0]["retry_config"].enable_retry_logic)
                usage = backend.ledger.snapshot()
                self.assertEqual(usage["pending_requests"], 1)
                self.assertGreater(usage["reserved_unknown_usd"], 0)
                self.assertEqual(usage["estimated_cost_usd"], 0)
                self.assertEqual(len((Path(directory) / "ledger.jsonl").read_text().splitlines()), 1)

    def test_base_and_adapter_clients_both_disable_outer_retries_and_are_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            service = MemoryService()
            backend = TinkerBackend(sdk=tinker, service_client=service,
                                    budget_path=Path(directory) / "ledger.jsonl")
            for checkpoint in (None, "tinker://run/sampler_weights/checkpoint"):
                self.assertIs(backend.sampling_client(checkpoint), backend.sampling_client(checkpoint))
            self.assertEqual(len(service.clients), 2)
            self.assertTrue(all(not options["retry_config"].enable_retry_logic for options in service.options))
            self.assertEqual(service.requests, [])


if __name__ == "__main__":
    unittest.main()

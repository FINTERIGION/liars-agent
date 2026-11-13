"""Stateful offline transport for trial tests; scores are not model evidence.

Use the installed SDK's real Datum/ModelInput/TensorData/AdamParams classes and
the real budgeted backend. Only transport and tokenization are local substitutes.
"""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import tinker

from liars_agent.training.tinker_backend import TinkerBackend


class TrialTokenizer:
    eos_token_id = ord(">")

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        text += "<assistant>" if add_generation_prompt else "<end>"
        return self.encode(text) if tokenize else text

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


def _values(tensor):
    return tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)


def token_logprob(token, weight):
    # P(0) + P(1) = .8 leaves probability for the other vocabulary tokens.
    if token == ord("0"):
        return math.log(.8) - max(weight, 0) - math.log1p(math.exp(-abs(weight)))
    if token == ord("1"):
        return math.log(.8) - max(-weight, 0) - math.log1p(math.exp(-abs(weight)))
    return -6.0 - .1 * weight


class StatefulTrainingClient:
    def __init__(self, service, index, state=None):
        self.service, self.index = service, index
        self.state = copy.deepcopy(state or {"weight": .2, "momentum": 0.0, "step": 0})
        self.initial_state = copy.deepcopy(self.state)
        self.train_calls = self.optim_calls = self.val_calls = 0
        self.batches, self.learning_rates = [], []
        self.saved_states, self.saved_samplers = [], []
        self.forward_error = None
        self.output_transform = None
        self.pending_gradient = 0.0

    def output(self, data):
        result = SimpleNamespace(loss_fn_outputs=[{"logprobs": tinker.TensorData(
            [token_logprob(token, self.state["weight"]) for token in _values(d.loss_fn_inputs["target_tokens"])],
            dtype="float32")} for d in data])
        return self.output_transform(result) if self.output_transform else result

    def forward(self, data, loss_fn):
        assert loss_fn == "cross_entropy"
        self.val_calls += 1
        if self.forward_error:
            raise self.forward_error
        return self.output(data)

    def forward_backward(self, data, loss_fn):
        assert loss_fn == "cross_entropy"
        self.train_calls += 1
        self.batches.append(data)
        # A deterministic changing gradient makes lost momentum observable.
        self.pending_gradient = sum(sum(_values(d.loss_fn_inputs["weights"])) for d in data)
        if not self.service.constant_gradient:
            self.pending_gradient += .5 * self.state["weight"]
        return self.output(data)

    def optim_step(self, params):
        self.optim_calls += 1
        self.learning_rates.append(params.learning_rate)
        self.state["step"] += 1
        self.state["momentum"] = .9 * self.state["momentum"] + .1 * self.pending_gradient
        if not self.service.static_updates:
            self.state["weight"] -= params.learning_rate * self.state["momentum"] / (1 - .9 ** self.state["step"])
        self.pending_gradient = 0.0
        return SimpleNamespace()

    def save_state(self, name, **kwargs):
        path = f"tinker://trial-fake-{self.index}/weights/{name}"
        self.saved_states.append(name)
        self.service.saved_states[path] = copy.deepcopy(self.state)
        return SimpleNamespace(path=path)

    def save_weights_for_sampler(self, name, **kwargs):
        path = f"tinker://trial-fake-{self.index}/sampler_weights/{name}"
        self.saved_samplers.append(name)
        self.service.saved_samplers[path] = copy.deepcopy(self.state)
        return SimpleNamespace(path=path)


class StatefulSamplingClient:
    def __init__(self, state):
        self.state = copy.deepcopy(state)
        self.calls, self.logprob_calls = [], []
        self.text, self.stop_reason = '{"chosen_action_id":"A0"}', "stop"
        self.error = self.logprob_error = None

    def sample(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[ord(c) for c in self.text],
            stop_reason=self.stop_reason) for _ in range(kwargs["num_samples"])])

    def compute_logprobs(self, prompt, *, prompt_logprobs_last_n):
        tokens = prompt.to_ints()
        self.logprob_calls.append((tokens, prompt_logprobs_last_n))
        if self.logprob_error:
            raise self.logprob_error
        assert prompt_logprobs_last_n == 1
        return [None] * (len(tokens) - 1) + [token_logprob(tokens[-1], self.state["weight"])]


class StatefulService:
    def __init__(self, *, restore_optimizer=True, static_updates=False, corrupt_reload=False):
        self.restore_optimizer, self.static_updates = restore_optimizer, static_updates
        self.corrupt_reload = corrupt_reload
        self.constant_gradient = False
        self.training_init, self.sampling_init, self.resumed = [], [], []
        self.saved_states, self.saved_samplers = {}, {}
        self.training = StatefulTrainingClient(self, 0)
        self.training_clients = [self.training]
        self.sampling_clients = []
        self.load_error = None

    def create_lora_training_client(self, **kwargs):
        self.training_init.append(kwargs)
        return self.training

    def _load(self, path, with_optimizer):
        self.resumed.append((path, with_optimizer))
        if self.load_error:
            raise self.load_error
        state = copy.deepcopy(self.saved_states[path])
        if not with_optimizer or not self.restore_optimizer:
            state.update(momentum=0.0, step=0)
        if self.corrupt_reload:
            state["weight"] += .1
        client = StatefulTrainingClient(self, len(self.training_clients), state)
        self.training_clients.append(client)
        return client

    def create_training_client_from_state(self, path):
        return self._load(path, False)

    def create_training_client_from_state_with_optimizer(self, path):
        return self._load(path, True)

    def create_sampling_client(self, **kwargs):
        self.sampling_init.append(kwargs)
        state = self.saved_samplers[kwargs["model_path"]] if "model_path" in kwargs else self.training.state
        self.sampling = StatefulSamplingClient(state)
        self.sampling_clients.append(self.sampling)
        return self.sampling


def make_trial_backend(path, *, restore_optimizer=True, static_updates=False, corrupt_reload=False):
    service = StatefulService(restore_optimizer=restore_optimizer,
                             static_updates=static_updates, corrupt_reload=corrupt_reload)
    backend = TinkerBackend(budget_path=path, sdk=tinker, service_client=service)
    backend._tokenizer = TrialTokenizer()
    return backend, service

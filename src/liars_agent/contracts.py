"""Small, explicit method contract shared by persisted experiment artifacts."""
from __future__ import annotations

import hashlib
import json
from typing import Any

RULES_VERSION = "liars_bar_research_v2"
OBSERVATION_VERSION = "public_history_v2"
TEACHER_VERSION = "pool_prior_counter_continuation_v2"
DECISION_SCHEMA_VERSION = "action_and_binary_belief_v2"

V4_OBSERVATION_VERSION = "public_history_count_profile_v4"
V4_TEACHER_VERSION = "calibrated_posterior_continuation_v4"
SUPPORTED_METHOD_VERSIONS = ("v2", "adaptive_v4")


def current_contract(method_version: str = "v2") -> dict[str, str]:
    # Lazy import keeps engine constants independent of agent initialization.
    from liars_agent.agents.prompt_builder import (
        BELIEF_SYSTEM_PROMPT,
        BELIEF_SYSTEM_PROMPT_V4,
        SYSTEM_PROMPT,
        SYSTEM_PROMPT_V4,
    )
    if method_version == "v2":
        prompts = json.dumps([SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT], ensure_ascii=False)
        return {
            "rules_version": RULES_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "teacher_version": TEACHER_VERSION,
            "decision_schema_version": DECISION_SCHEMA_VERSION,
            "prompt_sha256": hashlib.sha256(prompts.encode()).hexdigest(),
        }
    if method_version == "adaptive_v4":
        prompts = json.dumps([SYSTEM_PROMPT_V4, BELIEF_SYSTEM_PROMPT_V4], ensure_ascii=False)
        return {
            "rules_version": RULES_VERSION,
            "observation_version": V4_OBSERVATION_VERSION,
            "teacher_version": V4_TEACHER_VERSION,
            "decision_schema_version": DECISION_SCHEMA_VERSION,
            "prompt_sha256": hashlib.sha256(prompts.encode()).hexdigest(),
        }
    raise ValueError(f"Unsupported method_version: {method_version}")


def detect_method_version(contract: dict[str, Any] | None) -> str:
    if not isinstance(contract, dict):
        raise ValueError("Missing method contract.")
    for version in SUPPORTED_METHOD_VERSIONS:
        if contract == current_contract(version):
            return version
    raise ValueError("Incompatible method contract; regenerate in a new run directory.")


def require_contract(
    data: dict[str, Any],
    context: str = "artifact",
    *,
    method_version: str = "v2",
) -> None:
    if data.get("contract") != current_contract(method_version):
        raise ValueError(f"{context}: missing or incompatible method contract; regenerate in a new run directory.")


def require_supported_contract(data: dict[str, Any], context: str = "artifact") -> str:
    try:
        return detect_method_version(data.get("contract"))
    except ValueError as exc:
        raise ValueError(f"{context}: missing or incompatible method contract; regenerate in a new run directory.") from exc

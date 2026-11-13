"""Small, explicit method contract shared by persisted experiment artifacts."""
from __future__ import annotations

import hashlib
import json
from typing import Any

RULES_VERSION = "liars_bar_research_v2"
OBSERVATION_VERSION = "public_history_v2"
TEACHER_VERSION = "pool_prior_counter_continuation_v2"
DECISION_SCHEMA_VERSION = "action_and_binary_belief_v2"


def current_contract() -> dict[str, str]:
    # Lazy import keeps engine constants independent of agent initialization.
    from liars_agent.agents.prompt_builder import SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT
    prompts = json.dumps([SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT], ensure_ascii=False)
    return {
        "rules_version": RULES_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "teacher_version": TEACHER_VERSION,
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "prompt_sha256": hashlib.sha256(prompts.encode()).hexdigest(),
    }


def require_contract(data: dict[str, Any], context: str = "artifact") -> None:
    if data.get("contract") != current_contract():
        raise ValueError(f"{context}: missing or incompatible method contract; regenerate in a new run directory.")

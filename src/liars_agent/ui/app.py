"""Gradio human games and post-game decision replay workbench."""

from __future__ import annotations

import argparse
from pathlib import Path
import secrets
from typing import Any, Callable
from uuid import uuid4

from liars_agent.agents.base import BaseAgent, describe_action, public_opponent_profile
from liars_agent.agents.classifier_bot import SmallClassifierBot
from liars_agent.agents.rule_bots import create_rule_agent
from liars_agent.contracts import require_contract
from liars_agent.engine.replay import GameReplayBundle, verify_replay_determinism
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.types import AgentDecisionRecord
from liars_agent.evaluation.metrics import compute_behavioral_profile

AgentFactory = Callable[[str, int], BaseAgent]
RULE_CHOICES = [
    "honest_conservative", "aggressive_bluffer", "card_counter",
    "paranoid_hunter", "chameleon_adaptive",
]
MODEL_MODES = {
    "qwen35_9b_zero_shot": "zero_shot",
    "qwen35_9b_lora_sft": "lora_sft",
    "qwen35_9b_lora_rs": "lora_rs_improved",
}


class _HumanPlayer(BaseAgent):
    def __init__(self) -> None:
        super().__init__("human_player", "human")

    def select_action(self, obs):
        raise RuntimeError("The human player must choose an action in the UI.")


def build_agent_by_spec(spec: str, seed: int = 42) -> BaseAgent:
    """Build agents that need no model credentials or training checkpoint."""
    if spec == "human":
        return _HumanPlayer()
    if spec in MODEL_MODES:
        raise ValueError("A model configuration is required for a language-model player.")
    if spec == "small_classifier":
        raise ValueError("Trained classifier weights are required.")
    return create_rule_agent(spec, name=spec, seed=seed)


def configured_agent_factory(config: dict[str, Any]) -> AgentFactory:
    """Share a model service while keeping each player's game state separate."""
    require_contract(config, "UI model config")
    backend = None
    sampling = config.get("sampling", {})
    if sampling.get("fallback_strategy", "card_counter_safe") != "card_counter_safe":
        raise ValueError("The supported model fallback is card_counter_safe.")

    def factory(spec: str, seed: int) -> BaseAgent:
        nonlocal backend
        if spec == "small_classifier":
            return SmallClassifierBot.load_weights(config["classifier_weights"])
        if spec not in MODEL_MODES:
            return build_agent_by_spec(spec, seed)
        from liars_agent.agents.llm_agent import LLMAgent
        from liars_agent.training.tinker_backend import TinkerBackend

        if backend is None:
            backend = TinkerBackend(
                base_model=config["base_model"],
                budget_path=config["budget_path"],
                budget_limit_usd=config["budget_limit_usd"],
            )
        adapter = None
        if spec.endswith("lora_sft"):
            adapter = config["sft_adapter"]
        elif spec.endswith("lora_rs"):
            adapter = config["rs_adapter"]
        return LLMAgent(
            name=spec, mode=MODEL_MODES[spec], model_version=config["base_model"],
            backend=backend, adapter_path=adapter, seed=seed,
            temperature=sampling.get("temperature", 0.2),
            top_p=sampling.get("top_p", 0.9),
            max_new_tokens=sampling.get("max_new_tokens", 256),
        )

    return factory


class InteractiveSessionManager:
    """One browser session's game, human seat and completed replay."""

    def __init__(self, agent_factory: AgentFactory = build_agent_by_spec) -> None:
        self.agent_factory = agent_factory
        self.engine: LiarsBarEngine | None = None
        self.agents: list[BaseAgent] = []
        self.human_seat: int | None = 0
        self.latest_replay: GameReplayBundle | None = None
        self.ai_error: str | None = None
        self.debug_seed_enabled = False

    def start_game(
        self,
        seat0_spec: str = "human",
        seat1_spec: str = "honest_conservative",
        seat2_spec: str = "aggressive_bluffer",
        seat3_spec: str = "card_counter",
        game_seed: int | None = None,
    ) -> dict[str, Any]:
        """Create an unpredictable game by default; an explicit seed is debugging only."""
        specs = [seat0_spec, seat1_spec, seat2_spec, seat3_spec]
        if "human" in specs[1:]:
            raise ValueError("Only Seat 0 can be controlled by a human.")
        self.debug_seed_enabled = game_seed is not None
        environment_seed = secrets.randbits(128) if game_seed is None else game_seed
        agents = [self.agent_factory(spec, secrets.randbits(64) if game_seed is None else game_seed + idx)
                  for idx, spec in enumerate(specs)]
        self.engine = LiarsBarEngine(game_id=f"ui_{uuid4().hex}", game_seed=environment_seed)
        self.human_seat = 0 if seat0_spec == "human" else None
        self.agents = agents
        self.latest_replay = None
        self.ai_error = None
        self._advance_ai_turns_until_human_or_end()
        return self.get_current_view()

    def _advance_ai_turns_until_human_or_end(self) -> None:
        assert self.engine is not None
        self.ai_error = None
        while not self.engine.is_game_over():
            turn = self.engine.current_turn_seat
            if self.human_seat is not None and turn == self.human_seat:
                break
            obs = self.engine.get_observation(turn)
            try:
                act, rec = self.agents[turn].select_action(obs)
            except Exception as exc:
                # Retain the exact pending turn and every completed step for retry.
                self.ai_error = f"Seat {turn}: {type(exc).__name__}: {exc}"
                return
            self.engine.step(act, rec)
        if self.engine.is_game_over():
            self.latest_replay = GameReplayBundle.from_engine(
                self.engine, seat_agent_names=[a.name for a in self.agents]
            )

    def resume_ai_turns(self) -> dict[str, Any]:
        if self.engine is None or self.engine.is_game_over():
            raise ValueError("Game is not active.")
        if self.human_seat is not None and self.engine.current_turn_seat == self.human_seat:
            raise ValueError("It is the human player's turn.")
        self._advance_ai_turns_until_human_or_end()
        return self.get_current_view()

    def submit_human_action(self, action_id: str) -> dict[str, Any]:
        if self.engine is None or self.engine.is_game_over():
            raise ValueError("Game is not active.")
        if self.human_seat is None or self.engine.current_turn_seat != self.human_seat:
            raise ValueError("It is not the human player's turn.")
        obs = self.engine.get_observation(self.human_seat)
        action = next((action for action in obs.legal_actions if action.action_id == action_id), None)
        if action is None:
            raise ValueError("Action is not legal in the current state.")
        self.engine.step(action, decision_record=AgentDecisionRecord(
            rationale=describe_action(obs, action), rationale_source="action_description",
            belief_source="not_requested", opponent_profile=public_opponent_profile(obs),
        ))
        self._advance_ai_turns_until_human_or_end()
        return self.get_current_view()

    def get_current_view(self) -> dict[str, Any]:
        if self.engine is None:
            return {"status": "idle"}
        # The human always sees their own seat, including after elimination.
        obs_seat = self.human_seat if self.human_seat is not None else 0
        obs = self.engine.get_observation(obs_seat)
        return {
            "status": "paused_ai" if self.ai_error else "finished" if self.engine.is_game_over() else "active",
            "game_id": self.engine.game_id,
            "debug_seed_enabled": self.debug_seed_enabled,
            "ai_error": self.ai_error,
            "is_game_over": self.engine.is_game_over(),
            "winner_seat": self.engine.winner_seat,
            "current_turn_seat": self.engine.current_turn_seat,
            "observation": obs.to_dict(),
            "total_logged_steps": len(self.engine.step_truth_logs),
        }

    def human_action_choices(self) -> list[tuple[str, str]]:
        if (self.engine is None or self.engine.is_game_over() or self.human_seat is None
                or self.engine.current_turn_seat != self.human_seat):
            return []
        actions = self.engine.get_observation(self.human_seat).legal_actions
        return [
            ("质疑上家" if action.action_type == "CHALLENGE"
             else f"出 {len(action.cards)} 张：{' / '.join(action.cards)}", action.action_id)
            for action in actions
        ]

    def inspect_replay_step(self, step_index: int) -> dict[str, Any]:
        """Private cards and decision-time observations become visible at game end."""
        if self.engine is None:
            return {"status": "idle"}
        if not self.engine.is_game_over():
            return {
                "status": "对局结束后开放完整决策复盘。",
                "current_view": self.get_current_view(),
            }
        if not self.engine.step_truth_logs:
            return {"error": "No steps recorded yet."}
        idx = max(0, min(len(self.engine.step_truth_logs) - 1, int(step_index)))
        step = self.engine.step_truth_logs[idx]
        return {
            "step_index": idx,
            "total_steps": len(self.engine.step_truth_logs),
            "round_id": step.round_id,
            "step_id": step.step_id,
            "actor_seat": step.actor_seat,
            "actor_agent": self.agents[step.actor_seat].name,
            "action_description": step.decision_record.get("rationale", ""),
            "description_source": step.decision_record.get("rationale_source", "unknown"),
            "decision_time_visible": step.observation,
            "agent_decision_and_belief": {
                "chosen_action": step.chosen_action,
                **step.decision_record,
            },
            "post_hoc_ground_truth": {
                "all_players_true_hands": {str(seat): cards for seat, cards in step.ground_truth_hands.items()},
                "all_players_fatal_chambers": {str(seat): slot for seat, slot in step.ground_truth_fatal_chambers.items()},
                "pending_claim_actual_cards": step.pending_claim_actual_cards,
                "pending_claim_was_bluff": step.pending_claim_was_bluff,
                "round_resolution": step.round_resolution,
            },
        }

    def get_session_profiles(self) -> dict[str, Any]:
        if self.latest_replay is None:
            return {}
        replay = self.latest_replay.to_dict()
        return {
            "replay_deterministic_verified": verify_replay_determinism(self.latest_replay),
            "trajectory_hash": self.latest_replay.compute_trajectory_hash(),
            "profiles": {
                name: compute_behavioral_profile([replay], name)
                for name in sorted(set(self.latest_replay.seat_agent_names))
            },
        }


def create_ui(model_config: dict[str, Any] | None = None):
    """Construct the UI without starting a server (also used by UI tests)."""
    import gradio as gr

    config = dict(model_config or {})
    agent_factory = configured_agent_factory(config) if config else build_agent_by_spec
    ai_choices = list(RULE_CHOICES)
    if config.get("classifier_weights"):
        ai_choices.append("small_classifier")
    if config.get("base_model"):
        ai_choices.append("qwen35_9b_zero_shot")
    if config.get("sft_adapter"):
        ai_choices.append("qwen35_9b_lora_sft")
    if config.get("rs_adapter"):
        ai_choices.append("qwen35_9b_lora_rs")

    with gr.Blocks(title="骗子酒馆 AI 决策复盘工作台") as demo:
        session_state = gr.State(None)
        gr.Markdown("# 骗子酒馆 AI — 对局练习与决策复盘\n选择合法动作出牌；完整对局结束后可查看每一步的决策和底牌。")
        with gr.Row():
            s0 = gr.Dropdown(["human", *ai_choices], value="human", label="Seat 0")
            s1 = gr.Dropdown(ai_choices, value="honest_conservative", label="Seat 1")
            s2 = gr.Dropdown(ai_choices, value="aggressive_bluffer", label="Seat 2")
            s3 = gr.Dropdown(ai_choices, value="card_counter", label="Seat 3")
            start_btn = gr.Button("开始新对局", variant="primary")
        with gr.Row():
            debug_seed = gr.Checkbox(value=False, label="调试：使用固定环境种子")
            seed_in = gr.Number(value=2026, label="仅用于复现调试的种子", precision=0, interactive=False)
        state_out = gr.JSON(label="当前玩家视图 / 结局")
        with gr.Row():
            action_in = gr.Dropdown([], label="选择合法动作", interactive=False)
            action_btn = gr.Button("提交动作", interactive=False)
            resume_btn = gr.Button("继续 AI 回合", interactive=False)
        ai_status = gr.Markdown()
        with gr.Row():
            step_slider = gr.Slider(0, 1, value=0, step=1, interactive=False, label="复盘步骤")
            inspect_btn = gr.Button("查看决策复盘", interactive=False)
        replay_out = gr.JSON(label="终局复盘：动作说明、概率来源与事后真值")
        profile_out = gr.JSON(label="本局行为画像与回放校验")

        def render(session: InteractiveSessionManager):
            choices = session.human_action_choices()
            finished = session.engine is not None and session.engine.is_game_over()
            total_steps = len(session.engine.step_truth_logs) if session.engine else 0
            return (
                session, session.get_current_view(),
                gr.Dropdown(choices=choices, value=choices[0][1] if choices else None, interactive=bool(choices)),
                gr.Button(interactive=bool(choices)),
                gr.Button(interactive=bool(session.ai_error)),
                f"AI 回合暂停：{session.ai_error}。已完成步骤保留，可点击继续。" if session.ai_error else "",
                gr.Slider(minimum=0, maximum=max(1, total_steps - 1), value=0, interactive=finished),
                gr.Button(interactive=finished),
                session.inspect_replay_step(0), session.get_session_profiles(),
            )

        def on_start(a0, a1, a2, a3, use_debug_seed, seed):
            session = InteractiveSessionManager(agent_factory)
            session.start_game(a0, a1, a2, a3, int(seed) if use_debug_seed else None)
            return render(session)

        def on_action(session, action_id):
            if session is None:
                raise gr.Error("请先开始对局。")
            try:
                session.submit_human_action(action_id)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            return render(session)

        def on_resume(session):
            if session is None:
                raise gr.Error("请先开始对局。")
            try:
                session.resume_ai_turns()
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            return render(session)

        outputs = [session_state, state_out, action_in, action_btn, resume_btn, ai_status, step_slider, inspect_btn, replay_out, profile_out]
        start_btn.click(on_start, [s0, s1, s2, s3, debug_seed, seed_in], outputs, api_name="start_game")
        action_btn.click(on_action, [session_state, action_in], outputs, api_name="submit_action")
        resume_btn.click(on_resume, [session_state], outputs, api_name="resume_ai")
        debug_seed.change(lambda enabled: gr.Number(interactive=enabled), debug_seed, seed_in, api_name=False)
        inspect_btn.click(
            lambda session, index: session.inspect_replay_step(int(index)) if session else {"status": "idle"},
            [session_state, step_slider], replay_out, api_name="inspect_replay",
        )
    return demo


def launch_ui(host: str = "127.0.0.1", port: int = 7860, model_config: dict[str, Any] | None = None) -> None:
    create_ui(model_config).launch(server_name=host, server_port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Liar's Bar AI Workbench UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model-config", type=Path, help="Pipeline-generated model/checkpoint YAML")
    args = parser.parse_args()
    config = None
    if args.model_config:
        import yaml
        config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    launch_ui(host=args.host, port=args.port, model_config=config)

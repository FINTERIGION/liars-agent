"""Gradio 对局与终局复盘工作台；未安装 Gradio 时提供命令行演示."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from liars_agent.agents.base import BaseAgent
from liars_agent.agents.classifier_bot import SmallClassifierBot
from liars_agent.agents.llm_agent import LLMAgent
from liars_agent.agents.rule_bots import create_rule_agent
from liars_agent.engine.replay import GameReplayBundle, verify_replay_determinism
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.evaluation.metrics import compute_behavioral_profile


def build_agent_by_spec(spec: str, seed: int = 42) -> BaseAgent:
    """根据界面下拉菜单选择的名称构造智能体."""
    if spec == "qwen35_9b_lora_sft":
        return LLMAgent(name=spec, mode="lora_sft", seed=seed)
    if spec == "qwen35_9b_lora_rs":
        return LLMAgent(name=spec, mode="lora_rs_improved", seed=seed)
    if spec == "qwen35_9b_zero_shot":
        return LLMAgent(name=spec, mode="zero_shot", seed=seed)
    if spec == "small_classifier":
        return SmallClassifierBot(name=spec)
    return create_rule_agent(spec, name=spec, seed=seed)


class InteractiveSessionManager:
    """管理单人练习（Seat 0 为人类或指定 AI，Seat 1..3 为可配置 AI 对手）与决策复盘."""

    def __init__(self) -> None:
        self.engine: Optional[LiarsBarEngine] = None
        self.agents: list[BaseAgent] = []
        self.human_seat: Optional[int] = 0
        self.latest_replay: Optional[GameReplayBundle] = None

    def start_game(
        self,
        seat0_spec: str = "human",
        seat1_spec: str = "honest_conservative",
        seat2_spec: str = "aggressive_bluffer",
        seat3_spec: str = "qwen35_9b_lora_sft",
        game_seed: int = 2026,
    ) -> dict[str, Any]:
        self.engine = LiarsBarEngine(game_id=f"ui_game_{game_seed}", game_seed=game_seed)
        self.latest_replay = None
        self.human_seat = 0 if seat0_spec == "human" else None
        specs = [seat0_spec, seat1_spec, seat2_spec, seat3_spec]
        self.agents = [
            build_agent_by_spec(
                "qwen35_9b_lora_sft" if s == "human" else s, seed=game_seed + idx
            )
            for idx, s in enumerate(specs)
        ]
        if self.human_seat is not None:
            self.agents[0].name = "human_player"

        self._advance_ai_turns_until_human_or_end()
        return self.get_current_view()

    def _advance_ai_turns_until_human_or_end(self) -> None:
        assert self.engine is not None
        while not self.engine.is_game_over():
            turn = self.engine.current_turn_seat
            if self.human_seat is not None and turn == self.human_seat:
                break
            obs = self.engine.get_observation(turn)
            act, rec = self.agents[turn].select_action(obs)
            self.engine.step(act, rec)

        if self.engine.is_game_over():
            self.latest_replay = GameReplayBundle.from_engine(
                self.engine, seat_agent_names=[a.name for a in self.agents]
            )

    def submit_human_action(self, action_id: str) -> dict[str, Any]:
        if self.engine is None or self.engine.is_game_over():
            return {"error": "Game is not active."}
        obs = self.engine.get_observation(self.engine.current_turn_seat)
        _, ai_rec = self.agents[0].select_action(obs)
        self.engine.step(action_id, decision_record=ai_rec)
        self._advance_ai_turns_until_human_or_end()
        return self.get_current_view()

    def get_current_view(self) -> dict[str, Any]:
        if self.engine is None:
            return {"status": "idle"}
        obs_seat = self.human_seat if self.human_seat is not None else self.engine.current_turn_seat
        obs = self.engine.get_observation(obs_seat)
        return {
            "game_id": self.engine.game_id,
            "is_game_over": self.engine.is_game_over(),
            "winner_seat": self.engine.winner_seat,
            "current_turn_seat": self.engine.current_turn_seat,
            "observation": obs.to_dict(),
            "total_logged_steps": len(self.engine.step_truth_logs),
        }

    def inspect_replay_step(self, step_index: int) -> dict[str, Any]:
        """仅整场结束后开放所有玩家的决策时观测和事后真值."""
        if self.engine is None:
            return {"status": "idle"}
        if not self.engine.is_game_over():
            return {"status": "整场对局结束后开放完整复盘。", "current_view": self.get_current_view()}
        if not self.engine.step_truth_logs:
            return {"error": "No steps recorded yet."}
        idx = max(0, min(len(self.engine.step_truth_logs) - 1, step_index))
        step_log = self.engine.step_truth_logs[idx]
        obs = step_log.observation
        actor = step_log.actor_seat
        actor_shots = 0
        actor_lethal_prob = 1.0 / 6.0
        for p in obs["players_public_status"]:
            if p["seat"] == actor:
                actor_shots = p["shots_taken"]
                actor_lethal_prob = p["next_shot_lethal_prob"]
                break

        return {
            "step_index": idx,
            "total_steps": len(self.engine.step_truth_logs),
            "round_id": step_log.round_id,
            "step_id": step_log.step_id,
            "actor_seat": actor,
            "actor_agent": self.agents[actor].name if actor < len(self.agents) else f"Seat_{actor}",
            "decision_time_visible": {
                "target_rank": step_log.target_rank,
                "own_hand": obs["own_hand"],
                "own_shots_taken": actor_shots,
                "own_next_shot_lethal_prob": actor_lethal_prob,
                "last_claim": obs["last_claim"],
                "combinatorial_claim_truth_prior": obs["combinatorial_claim_truth_prior"],
                "legal_actions": obs["legal_actions"],
            },
            "agent_decision_and_belief": {
                "chosen_action": step_log.chosen_action,
                "belief_bluff_prob": step_log.decision_record.get("belief_bluff_prob"),
                "estimated_opponent_style": step_log.decision_record.get("estimated_opponent_style"),
                "rationale": step_log.decision_record.get("rationale"),
            },
            "post_hoc_ground_truth": {
                "all_players_true_hands": {str(s): cards for s, cards in step_log.ground_truth_hands.items()},
                "all_players_fatal_chambers": {str(s): slot for s, slot in step_log.ground_truth_fatal_chambers.items()},
                "pending_claim_actual_cards": step_log.pending_claim_actual_cards,
                "pending_claim_was_bluff": step_log.pending_claim_was_bluff,
                "round_resolution": step_log.round_resolution,
            },
        }

    def get_session_profiles(self) -> dict[str, Any]:
        if self.engine is None or not self.engine.is_game_over() or self.latest_replay is None:
            return {}
        rep_dict = self.latest_replay.to_dict()
        profiles = {}
        for name in sorted(set(self.latest_replay.seat_agent_names)):
            profiles[name] = compute_behavioral_profile([rep_dict], name)
        return {
            "replay_deterministic_verified": verify_replay_determinism(self.latest_replay),
            "trajectory_hash": self.latest_replay.compute_trajectory_hash(),
            "profiles": profiles,
        }


def create_ui():
    """构建界面；每个浏览器会话独立保存对局，终局前禁止查看真值。"""
    import gradio as gr

    agent_choices = [
        "human", "qwen35_9b_lora_sft", "qwen35_9b_lora_rs", "qwen35_9b_zero_shot",
        "small_classifier", "honest_conservative", "aggressive_bluffer",
        "card_counter", "paranoid_hunter", "chameleon_adaptive",
    ]
    with gr.Blocks(title="骗子酒馆 AI 决策复盘工作台") as demo:
        session_state = gr.State(None)
        gr.Markdown("# 骗子酒馆 AI — 对局与终局复盘\n对局中仅显示玩家可见信息；整场结束后开放所有步骤的真值复盘。")
        with gr.Row():
            s0 = gr.Dropdown(agent_choices, value="qwen35_9b_lora_sft", label="Seat 0")
            s1 = gr.Dropdown(agent_choices[1:], value="honest_conservative", label="Seat 1")
            s2 = gr.Dropdown(agent_choices[1:], value="aggressive_bluffer", label="Seat 2")
            s3 = gr.Dropdown(agent_choices[1:], value="card_counter", label="Seat 3")
            seed_in = gr.Number(value=2026, label="对局随机种子", precision=0)
            start_btn = gr.Button("开始/模拟对局", variant="primary")
        state_out = gr.JSON(label="当前玩家视图 / 结局")
        with gr.Row():
            step_slider = gr.Slider(0, 1, value=0, step=1, interactive=False, label="终局复盘步骤")
            inspect_btn = gr.Button("查看终局复盘", interactive=False)
        replay_out = gr.JSON(label="单步决策复盘")
        profile_out = gr.JSON(label="本局行为画像与回放校验")

        def on_start(a0, a1, a2, a3, seed):
            session = InteractiveSessionManager()
            view = session.start_game(a0, a1, a2, a3, int(seed))
            finished = session.engine.is_game_over()
            total_steps = len(session.engine.step_truth_logs)
            return (
                session, view, session.inspect_replay_step(0), session.get_session_profiles(),
                gr.Slider(minimum=0, maximum=max(1, total_steps - 1), value=0, interactive=finished),
                gr.Button(interactive=finished),
            )

        start_btn.click(
            on_start, [s0, s1, s2, s3, seed_in],
            [session_state, state_out, replay_out, profile_out, step_slider, inspect_btn],
            api_name="start_game",
        )
        inspect_btn.click(
            lambda session, index: session.inspect_replay_step(int(index)) if session else {"status": "idle"},
            [session_state, step_slider], replay_out, api_name="inspect_replay",
        )
    return demo


def launch_ui(host: str = "127.0.0.1", port: int = 7860) -> None:
    """启动 Gradio；若未安装，则运行完整 AI 对局的命令行演示。"""
    try:
        demo = create_ui()
    except ImportError:
        print("[Info] Gradio not installed; running CLI demo instead.")
        session = InteractiveSessionManager()
        view = session.start_game(
            seat0_spec="qwen35_9b_lora_sft", seat1_spec="honest_conservative",
            seat2_spec="aggressive_bluffer", seat3_spec="card_counter", game_seed=2026,
        )
        print(json.dumps(view, indent=2, ensure_ascii=False))
        return
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Liar's Bar AI Workbench UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch_ui(host=args.host, port=args.port)

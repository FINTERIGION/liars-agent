"""Gradio 交互式对局、AI 补位、决策复盘与行为画像工作台（内置零依赖 Web 服务回退支持）."""

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
        """决策复盘器：对比第 `step_index` 步决策时的可见信息、模型信念/风险评估与事后底牌真值."""
        if self.engine is None or not self.engine.step_truth_logs:
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
                "all_players_true_hands": step_log.ground_truth_hands,
                "all_players_fatal_chambers": step_log.ground_truth_fatal_chambers,
                "pending_claim_actual_cards": step_log.pending_claim_actual_cards,
                "pending_claim_was_bluff": step_log.pending_claim_was_bluff,
                "round_resolution": step_log.round_resolution,
            },
        }

    def get_session_profiles(self) -> dict[str, Any]:
        if self.latest_replay is None:
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


def launch_ui(host: str = "127.0.0.1", port: int = 7860) -> None:
    """启动 Gradio 界面；若当前环境未安装 Gradio，则启动内置轻量 Web 复盘工作台."""
    try:
        import gradio as gr  # type: ignore[import-not-found]

        session = InteractiveSessionManager()
        agent_choices = [
            "human",
            "qwen35_9b_lora_sft",
            "qwen35_9b_lora_rs",
            "qwen35_9b_zero_shot",
            "small_classifier",
            "honest_conservative",
            "aggressive_bluffer",
            "card_counter",
            "paranoid_hunter",
            "chameleon_adaptive",
        ]

        with gr.Blocks(title="骗子酒馆 AI 决策智能体训练与评测工作台") as demo:
            gr.Markdown("# 骗子酒馆 AI 决策智能体 — 对局练习、决策复盘与行为画像工作台")
            with gr.Row():
                s0 = gr.Dropdown(agent_choices, value="qwen35_9b_lora_sft", label="Seat 0")
                s1 = gr.Dropdown(agent_choices[1:], value="honest_conservative", label="Seat 1")
                s2 = gr.Dropdown(agent_choices[1:], value="aggressive_bluffer", label="Seat 2")
                s3 = gr.Dropdown(agent_choices[1:], value="card_counter", label="Seat 3")
                seed_in = gr.Number(value=2026, label="对局随机种子 (Seed)", precision=0)
                start_btn = gr.Button("开始/模拟对局", variant="primary")

            state_out = gr.JSON(label="当前对局视图 / 结局摘要")
            with gr.Row():
                step_slider = gr.Slider(0, 40, value=0, step=1, label="决策复盘步进索引 (Step Index)")
                inspect_btn = gr.Button("查看该步决策复盘（对比可见信息 vs 事后真值）")
            replay_out = gr.JSON(label="单步决策复盘详情")
            profile_out = gr.JSON(label="本局各智能体四维行为画像与回放哈希校验")

            def _on_start(a0, a1, a2, a3, sd):
                view = session.start_game(a0, a1, a2, a3, int(sd))
                rep = session.inspect_replay_step(0)
                prof = session.get_session_profiles()
                return view, rep, prof

            start_btn.click(_on_start, [s0, s1, s2, s3, seed_in], [state_out, replay_out, profile_out])
            inspect_btn.click(lambda idx: session.inspect_replay_step(int(idx)), [step_slider], [replay_out])

        demo.launch(server_name=host, server_port=port)
    except ImportError:
        print(f"[Info] Gradio not installed; running CLI demo & verification session instead.")
        session = InteractiveSessionManager()
        view = session.start_game(
            seat0_spec="qwen35_9b_lora_sft",
            seat1_spec="honest_conservative",
            seat2_spec="aggressive_bluffer",
            seat3_spec="card_counter",
            game_seed=2026,
        )
        print(json.dumps(view, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Liar's Bar AI Workbench UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch_ui(host=args.host, port=args.port)

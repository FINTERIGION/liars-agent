"""Gradio human games and post-game decision replay workbench."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import html
import json
from pathlib import Path
import random
import secrets
from typing import Any, Callable
from uuid import uuid4

from liars_agent.agents.base import BaseAgent, describe_action, public_opponent_profile
from liars_agent.agents.classifier_bot import SmallClassifierBot
from liars_agent.agents.rule_bots import create_rule_agent
from liars_agent.contracts import require_supported_contract
from liars_agent.engine.replay import GameReplayBundle, verify_replay_determinism
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.types import AgentDecisionRecord, are_cards_truthful, is_card_truthful
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
AGENT_LABELS = {
    "human": "真人玩家",
    "human_player": "真人玩家",
    "honest_conservative": "稳健诚实 (rule)",
    "aggressive_bluffer": "激进诈唬 (rule)",
    "card_counter": "记牌专家 (rule)",
    "paranoid_hunter": "多疑猎手 (rule)",
    "chameleon_adaptive": "变色龙 (rule)",
    "small_classifier": "小分类器 (ML)",
    "qwen35_9b_zero_shot": "Qwen3.5-9B Zero-Shot",
    "qwen35_9b_lora_sft": "Qwen3.5-9B LoRA-SFT",
    "qwen35_9b_lora_rs": "Qwen3.5-9B LoRA-RS",
}
DEFAULT_HUMAN_LOG_PATH = Path("outputs/human_games/human_games.jsonl")


def _agent_label(name: str) -> str:
    return AGENT_LABELS.get(name, name)


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
    method_version = require_supported_contract(config, "UI model config")
    backend = None
    sampling = config.get("sampling", {})
    if sampling.get("fallback_strategy", "card_counter_safe") != "card_counter_safe":
        raise ValueError("The supported model fallback is card_counter_safe.")

    def factory(spec: str, seed: int) -> BaseAgent:
        nonlocal backend
        if spec == "small_classifier":
            return SmallClassifierBot.load_weights(config["classifier_weights"], method_version=method_version)
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
            method_version=method_version,
        )

    return factory


def append_game_log(log_path: str | Path, record: dict[str, Any]) -> Path:
    """Append one completed UI game record to a JSONL file with an exclusive file lock."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(line)
            stream.flush()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return path


def load_human_game_logs(log_path: str | Path | None) -> list[dict[str, Any]]:
    if not log_path:
        return []
    path = Path(log_path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return records


def summarize_human_game_logs(log_path: str | Path | None) -> str:
    """Render a compact Markdown summary of saved human games."""
    if not log_path:
        return "未启用本地对局日志保存。"
    path = Path(log_path)
    records = load_human_game_logs(path)
    human_records = [r for r in records if r.get("human_seat") is not None]
    if not human_records:
        return (
            f"📁 日志路径：`{path}`\n\n"
            "尚无已完成的真人对局记录。每局结束后会自动保存完整回放与战绩。"
        )
    total = len(human_records)
    wins = sum(1 for r in human_records if r.get("human_won"))
    win_rate = wins / total * 100.0
    seat_stats: dict[int, list[int]] = {}
    for rec in human_records:
        seat = int(rec["human_seat"])
        seat_stats.setdefault(seat, [0, 0])
        seat_stats[seat][1] += 1
        if rec.get("human_won"):
            seat_stats[seat][0] += 1
    seat_summary = " / ".join(
        f"Seat {s}: {w}/{n}" for s, (w, n) in sorted(seat_stats.items())
    )
    recent_lines = []
    for rec in human_records[-5:][::-1]:
        seat = rec.get("human_seat")
        won = "🏆 胜" if rec.get("human_won") else "💀 负"
        names = rec.get("seat_agent_names", [])
        opponents = ", ".join(
            f"S{i}:{_agent_label(n)}" for i, n in enumerate(names) if i != seat
        )
        ts = str(rec.get("saved_at", ""))[:19].replace("T", " ")
        recent_lines.append(
            f"- `{ts}` | **{won}** (你=Seat {seat}, {rec.get('total_rounds', '?')} 小局) | 对手：{opponents}"
        )
    return (
        f"📁 日志路径：`{path}`（已保存 **{len(records)}** 局，其中真人局 **{total}** 局）\n\n"
        f"- **真人胜率**：**{wins} / {total} ({win_rate:.1f}%)**（四人随机基准 25.0%）\n"
        f"- **各座位战绩**：{seat_summary}\n\n"
        "**最近 5 局真人记录：**\n" + "\n".join(recent_lines)
    )


class InteractiveSessionManager:
    """One browser session's game, human seat and completed replay."""

    def __init__(
        self,
        agent_factory: AgentFactory = build_agent_by_spec,
        log_path: str | Path | None = None,
    ) -> None:
        self.agent_factory = agent_factory
        self.log_path = Path(log_path) if log_path is not None else None
        self.engine: LiarsBarEngine | None = None
        self.agents: list[BaseAgent] = []
        self.human_seat: int | None = 0
        self.latest_replay: GameReplayBundle | None = None
        self.ai_error: str | None = None
        self.debug_seed_enabled = False
        self.last_saved_game_id: str | None = None
        self.last_log_error: str | None = None

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
        human_seats = [idx for idx, spec in enumerate(specs) if spec == "human"]
        if len(human_seats) > 1:
            raise ValueError("At most one seat can be controlled by a human.")
        self.debug_seed_enabled = game_seed is not None
        environment_seed = secrets.randbits(128) if game_seed is None else game_seed
        agents = [self.agent_factory(spec, secrets.randbits(64) if game_seed is None else game_seed + idx)
                  for idx, spec in enumerate(specs)]
        self.engine = LiarsBarEngine(game_id=f"ui_{uuid4().hex}", game_seed=environment_seed)
        self.human_seat = human_seats[0] if human_seats else None
        self.agents = agents
        self.latest_replay = None
        self.ai_error = None
        self.last_log_error = None
        self._advance_ai_turns_until_human_or_end()
        return self.get_current_view()

    def _persist_completed_game_if_needed(self) -> None:
        if self.log_path is None or self.engine is None or self.latest_replay is None:
            return
        if self.last_saved_game_id == self.engine.game_id:
            return
        try:
            profiles_info = self.get_session_profiles()
            record = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "game_id": self.engine.game_id,
                "human_seat": self.human_seat,
                "seat_agent_names": list(self.latest_replay.seat_agent_names),
                "winner_seat": self.engine.winner_seat,
                "human_won": (
                    self.engine.winner_seat == self.human_seat
                    if self.human_seat is not None
                    else None
                ),
                "total_rounds": self.latest_replay.total_rounds,
                "total_steps": self.latest_replay.total_steps,
                "debug_seed_enabled": self.debug_seed_enabled,
                "trajectory_hash": profiles_info.get("trajectory_hash"),
                "replay_deterministic_verified": profiles_info.get("replay_deterministic_verified"),
                "profiles": profiles_info.get("profiles", {}),
                "replay": self.latest_replay.to_dict(),
            }
            append_game_log(self.log_path, record)
            self.last_saved_game_id = self.engine.game_id
            self.last_log_error = None
        except Exception as exc:
            self.last_log_error = f"{type(exc).__name__}: {exc}"

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
            self._persist_completed_game_if_needed()

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
            "human_seat": self.human_seat,
            "seat_agents": [a.name for a in self.agents],
            "current_turn_seat": self.engine.current_turn_seat,
            "observation": obs.to_dict(),
            "total_logged_steps": len(self.engine.step_truth_logs),
        }

    def human_action_choices(self) -> list[tuple[str, str]]:
        if (self.engine is None or self.engine.is_game_over() or self.human_seat is None
                or self.engine.current_turn_seat != self.human_seat):
            return []
        obs = self.engine.get_observation(self.human_seat)
        actions = obs.legal_actions
        only_challenge = len(actions) == 1 and actions[0].action_type == "CHALLENGE"
        choices: list[tuple[str, str]] = []
        for action in actions:
            if action.action_type == "CHALLENGE":
                if obs.last_claim is not None:
                    prefix = "⚠️ 强制质疑" if only_challenge else "🎯 质疑上家"
                    label = (
                        f"{prefix}（Seat {obs.last_claim.claimer_seat} 声称 "
                        f"{obs.last_claim.claimed_count} 张 {obs.last_claim.claimed_rank}）"
                    )
                else:
                    label = "质疑上家"
            else:
                truthful = are_cards_truthful(action.cards, obs.target_rank)
                tag = "全真" if truthful else "诈唬"
                label = f"出 {len(action.cards)} 张：{' / '.join(action.cards)}（{tag}）"
            choices.append((label, action.action_id))
        return choices

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


def _render_card_chip(card: str, target_rank: str, muted: bool = False) -> str:
    truthful = is_card_truthful(card, target_rank)
    if muted:
        bg, border, fg = "#1e293b", "#475569", "#cbd5e1"
    elif card == "JOKER":
        bg, border, fg = "#3b0764", "#c084fc", "#f3e8ff"
    elif truthful:
        bg, border, fg = "#064e3b", "#34d399", "#ecfdf5"
    else:
        bg, border, fg = "#27272a", "#71717a", "#e4e4e7"
    badge = "★万能" if card == "JOKER" else ("✓真牌" if truthful else "假牌")
    return (
        f'<div style="display:inline-flex;flex-direction:column;align-items:center;'
        f'min-width:60px;padding:8px 10px;border-radius:10px;border:1.5px solid {border};'
        f'background:{bg};color:{fg};font-family:ui-monospace,SFMono-Regular,monospace;">'
        f'<span style="font-size:18px;font-weight:800;line-height:1.1;">{html.escape(card)}</span>'
        f'<span style="font-size:11px;opacity:0.9;margin-top:3px;">{badge}</span></div>'
    )


def render_table_html(view: dict[str, Any]) -> str:
    """Render a strictly information-isolated Chinese tavern board from get_current_view()."""
    if not view or view.get("status") == "idle":
        return (
            '<div style="padding:20px;border-radius:12px;background:#0f172a;color:#e2e8f0;'
            'border:1px solid #334155;text-align:center;">'
            '<div style="font-size:18px;font-weight:700;margin-bottom:6px;">🃏 准备开局</div>'
            '<div style="font-size:14px;color:#94a3b8;">在上方选择座位阵容或点击快速预设，然后点击「开始新对局」。</div>'
            '</div>'
        )
    obs = view["observation"]
    target_rank = obs["target_rank"]
    human_seat = view.get("human_seat")
    winner_seat = view.get("winner_seat")
    turn_seat = view.get("current_turn_seat")
    seat_agents = view.get("seat_agents", [f"Seat {i}" for i in range(4)])
    is_over = bool(view.get("is_game_over"))
    status = view.get("status")

    if is_over:
        winner_name = _agent_label(seat_agents[winner_seat]) if winner_seat is not None else "未知"
        if human_seat is not None and winner_seat == human_seat:
            banner_bg, banner_text = "#065f46", f"🏆 恭喜获胜！你（Seat {human_seat}）赢下了整场对局！"
        elif human_seat is not None:
            banner_bg, banner_text = "#7f1d1d", f"💀 对局结束：Seat {winner_seat}（{html.escape(winner_name)}）获胜。下方已解锁全量底牌复盘！"
        else:
            banner_bg, banner_text = "#1e3a8a", f"🏁 AI 观战结束：Seat {winner_seat}（{html.escape(winner_name)}）获胜。"
    elif status == "paused_ai":
        banner_bg, banner_text = "#78350f", "⚠️ AI 回合因服务异常暂停，已完成动作已保留，请点击「继续 AI 回合」。"
    elif human_seat is not None and turn_seat == human_seat:
        banner_bg, banner_text = "#1d4ed8", f"👉 轮到你行动（Seat {human_seat}）！请在下方选择出牌或质疑。"
    else:
        banner_bg, banner_text = "#334155", f"⏳ 等待 Seat {turn_seat} 行动…"

    # Seat cards
    seat_cards_html = []
    for p in obs["players_public_status"]:
        seat = int(p["seat"])
        alive = bool(p["alive"])
        shots = int(p["shots_taken"])
        max_c = int(p.get("max_chambers", 6))
        hand_cnt = int(p["hand_count"])
        lethal_pct = p.get("next_shot_lethal_prob", 0.0) * 100.0
        agent_name = _agent_label(seat_agents[seat] if seat < len(seat_agents) else f"Seat {seat}")
        is_human = human_seat is not None and seat == human_seat
        is_turn = not is_over and seat == turn_seat
        is_winner = is_over and seat == winner_seat

        border = "#38bdf8" if is_turn else ("#34d399" if is_winner else ("#475569" if alive else "#7f1d1d"))
        bg = "#1e293b" if alive else "#1c1917"
        opacity = "1.0" if alive else "0.62"
        tags = []
        if is_human:
            tags.append('<span style="background:#2563eb;color:#fff;padding:1px 6px;border-radius:999px;font-size:11px;">你</span>')
        if is_turn:
            tags.append('<span style="background:#0284c7;color:#fff;padding:1px 6px;border-radius:999px;font-size:11px;">当前回合</span>')
        if is_winner:
            tags.append('<span style="background:#059669;color:#fff;padding:1px 6px;border-radius:999px;font-size:11px;">👑 胜者</span>')
        if not alive:
            tags.append('<span style="background:#991b1b;color:#fecaca;padding:1px 6px;border-radius:999px;font-size:11px;">☠ 淘汰</span>')
        elif hand_cnt == 0:
            tags.append('<span style="background:#475569;color:#e2e8f0;padding:1px 6px;border-radius:999px;font-size:11px;">已空手</span>')

        chambers = "".join(
            f'<span style="color:{"#ef4444" if i < shots else "#64748b"};">{"●" if i < shots else "○"}</span>'
            for i in range(max_c)
        )
        cards_vis = "🂠 " * hand_cnt if hand_cnt > 0 else "（0 张）"
        seat_cards_html.append(
            f'<div style="background:{bg};border:2px solid {border};border-radius:12px;padding:12px;opacity:{opacity};">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;">'
            f'<strong style="font-size:15px;color:#f8fafc;">Seat {seat}</strong>'
            f'<div style="display:flex;gap:4px;flex-wrap:wrap;">{"".join(tags)}</div></div>'
            f'<div style="font-size:12px;color:#94a3b8;margin:4px 0 8px;">{html.escape(agent_name)}</div>'
            f'<div style="font-size:13px;color:#e2e8f0;margin-bottom:4px;">手牌：<b>{hand_cnt}</b> 张 <span style="letter-spacing:1px;">{cards_vis}</span></div>'
            f'<div style="font-size:13px;color:#cbd5e1;">左轮：{chambers} <b>{shots}/{max_c}</b>'
            + (f' <span style="color:#fca5a5;font-size:12px;">(下枪致死 {lethal_pct:.1f}%)</span>' if alive else '')
            + '</div></div>'
        )

    # Pending claim box
    last_claim = obs.get("last_claim")
    legal_actions = obs.get("legal_actions", [])
    forced = len(legal_actions) == 1 and legal_actions[0].get("action_type") == "CHALLENGE"
    if last_claim:
        claimer = last_claim["claimer_seat"]
        claimer_label = _agent_label(seat_agents[claimer] if claimer < len(seat_agents) else f"Seat {claimer}")
        claim_html = (
            f'<div style="background:#172554;border:1.5px solid #3b82f6;border-radius:10px;padding:12px 14px;margin-top:12px;">'
            f'<div style="font-size:13px;color:#93c5fd;">待回应声称（最近一手出牌）</div>'
            f'<div style="font-size:16px;font-weight:700;color:#fff;margin-top:4px;">'
            f'Seat {claimer}（{html.escape(claimer_label)}）暗置打出 <span style="color:#fde047;">{last_claim["claimed_count"]} 张 {html.escape(last_claim["claimed_rank"])}</span>'
            f' <span style="font-size:13px;font-weight:400;color:#cbd5e1;">（出后剩余 {last_claim["claimer_remaining_cards"]} 张）</span></div>'
            + (
                '<div style="margin-top:6px;color:#fecaca;font-size:13px;font-weight:700;">'
                '⚠️ 其余玩家均已空手或淘汰，按规则你必须质疑这一手！</div>'
                if forced and not is_over and human_seat == turn_seat else ""
            )
            + '</div>'
        )
    else:
        claim_html = (
            '<div style="background:#1e293b;border:1px dashed #475569;border-radius:10px;padding:10px 14px;margin-top:12px;'
            'font-size:13px;color:#94a3b8;">本小局开局先手：尚无待回应声称，只能出牌（1~3 张）。</div>'
        )

    # Own hand & played cards
    own_hand = obs.get("own_hand", [])
    own_played = obs.get("own_played_cards_this_round", [])
    hand_chips = "".join(_render_card_chip(c, target_rank) for c in own_hand) or '<span style="color:#94a3b8;">（已打空手牌，等待本小局结算）</span>'
    played_chips = "".join(_render_card_chip(c, target_rank, muted=True) for c in own_played) or '<span style="color:#64748b;font-size:13px;">本小局尚未出牌</span>'
    obs_seat = obs["observer_seat"]
    hand_title = f"你的手牌（Seat {obs_seat}）" if human_seat is not None else f"Seat {obs_seat} 手牌"

    # Current round public history
    cur_hist = obs.get("current_round_history", [])
    if cur_hist:
        hist_items = []
        for ev in cur_hist:
            aseat = ev["actor_seat"]
            if ev["action_type"] == "PLAY":
                txt = f"<b>Seat {aseat}</b> 声称出 <b>{ev['claimed_count']} 张 {html.escape(ev['claimed_rank'])}</b>（余 {ev['actor_remaining_cards']} 张）"
            else:
                tag = "强制质疑" if ev.get("forced_challenge") else "主动质疑"
                txt = f"<b>Seat {aseat}</b> 发起 <span style='color:#fca5a5;'>{tag}</span>！"
            hist_items.append(f"<li style='margin:2px 0;'>{txt}</li>")
        cur_hist_html = "<ol style='margin:6px 0 0 18px;padding:0;font-size:13px;color:#e2e8f0;'>" + "".join(hist_items) + "</ol>"
    else:
        cur_hist_html = "<div style='font-size:13px;color:#94a3b8;margin-top:4px;'>本小局尚无公开行动。</div>"

    # Past rounds resolutions (latest 4)
    past_res = obs.get("past_rounds_history", [])
    if past_res:
        res_items = []
        for r in past_res[-4:][::-1]:
            cards_str = " / ".join(r["revealed_cards"])
            verdict = "<span style='color:#f87171;font-weight:700;'>诈唬成立</span>" if r["was_bluff"] else "<span style='color:#34d399;font-weight:700;'>全为真牌</span>"
            shot_res = "<span style='color:#ef4444;font-weight:700;'>💥 击中实弹淘汰</span>" if r["lethal_shot"] else "幸存"
            res_items.append(
                f"<li style='margin:3px 0;'><b>第 {r['round_id']} 小局</b>（目标 {html.escape(r['target_rank'])}）："
                f"Seat {r['challenger_seat']} 质疑 Seat {r['claimer_seat']} → 亮牌 <code>[{html.escape(cards_str)}]</code>（{verdict}）"
                f"→ <b>Seat {r['penalized_seat']}</b> 开枪 ({r['shots_taken_after']}/6，{shot_res})</li>"
            )
        past_res_html = "<ul style='margin:6px 0 0 18px;padding:0;font-size:13px;color:#cbd5e1;'>" + "".join(res_items) + "</ul>"
    else:
        past_res_html = "<div style='font-size:13px;color:#94a3b8;margin-top:4px;'>暂无历史小局结算。</div>"

    return (
        '<div style="background:#0f172a;color:#f8fafc;border:1px solid #334155;border-radius:14px;padding:16px;font-family:system-ui,-apple-system,sans-serif;">'
        f'<div style="background:{banner_bg};color:#fff;padding:10px 14px;border-radius:10px;font-weight:700;font-size:15px;'
        f'display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
        f'<span>{banner_text}</span>'
        f'<span style="font-size:13px;background:rgba(0,0,0,0.28);padding:4px 10px;border-radius:999px;">'
        f'第 {obs["round_id"]} 小局 · 步 {obs["step_id"]} · 目标牌：<b style="color:#fde047;font-size:15px;">{html.escape(target_rank)}</b>（JOKER=真牌）</span>'
        '</div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-top:12px;">'
        f'{"".join(seat_cards_html)}</div>'
        f'{claim_html}'
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-top:12px;">'
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;">'
        f'<div style="font-size:13px;color:#94a3b8;margin-bottom:8px;">{hand_title}（共 {len(own_hand)} 张）</div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{hand_chips}</div></div>'
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;">'
        f'<div style="font-size:13px;color:#94a3b8;margin-bottom:8px;">本小局自己已打出（记牌参考）</div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{played_chips}</div></div>'
        '</div>'
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-top:12px;">'
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;">'
        f'<div style="font-size:13px;color:#94a3b8;">本小局公开出牌流</div>{cur_hist_html}</div>'
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px;">'
        f'<div style="font-size:13px;color:#94a3b8;">近期小局亮牌结算（最近 4 局）</div>{past_res_html}</div>'
        '</div></div>'
    )


def render_replay_markdown(step_data: dict[str, Any]) -> str:
    """Render a readable Chinese summary for a single post-game replay step."""
    if not step_data or "post_hoc_ground_truth" not in step_data:
        status = step_data.get("status", "对局结束后开放完整决策复盘。") if isinstance(step_data, dict) else "未开局"
        return f"*{status}*"
    dec = step_data["agent_decision_and_belief"]
    truth = step_data["post_hoc_ground_truth"]
    obs = step_data["decision_time_visible"]
    actor = step_data["actor_seat"]
    agent_name = _agent_label(step_data["actor_agent"])
    chosen = dec.get("chosen_action", {})
    if chosen.get("action_type") == "CHALLENGE":
        act_str = "质疑上家"
    else:
        cards = chosen.get("cards", [])
        act_str = f"打出 {len(cards)} 张：`{' / '.join(cards)}`"
    belief_prob = dec.get("belief_bluff_prob")
    belief_str = f"**{belief_prob * 100:.1f}%**（来源：`{dec.get('belief_source')}`）" if belief_prob is not None else "无（首手或人类回合）"
    latency = dec.get("latency_ms", 0.0)
    fallback_str = "⚠️ 发生回退" if dec.get("was_fallback") else "正常输出"
    hands_lines = []
    fatal = truth.get("all_players_fatal_chambers", {})
    for s in sorted(truth["all_players_true_hands"].keys(), key=int):
        cards = truth["all_players_true_hands"][s]
        c_str = " / ".join(cards) if cards else "（空手）"
        hands_lines.append(f"- **Seat {s}**：`{c_str}` ｜ 实弹弹膛：`第 {int(fatal.get(s, 0)) + 1} 枪`")
    pending_actual = truth.get("pending_claim_actual_cards")
    pending_line = ""
    if pending_actual is not None:
        bluff_tag = "诈唬（含假牌）" if truth.get("pending_claim_was_bluff") else "全为真牌"
        pending_line = f"\n- **上家暗置实际牌面**：`{' / '.join(pending_actual)}`（**{bluff_tag}**）"
    res = truth.get("round_resolution")
    res_line = ""
    if res:
        shot_tag = "💥 击中实弹淘汰" if res["lethal_shot"] else "幸存"
        res_line = (
            f"\n- **本步触发小局结算**：Seat {res['penalized_seat']} 开枪 "
            f"({res['shots_taken_after']}/6，**{shot_tag}**)"
        )
    return (
        f"### 第 {step_data['step_index'] + 1} / {step_data['total_steps']} 步（第 {step_data['round_id']} 小局 · 目标牌 `{obs['target_rank']}`）\n\n"
        f"- **行动方**：Seat {actor}（{agent_name}）\n"
        f"- **实际动作**：{act_str}（{fallback_str}，耗时 {latency:.0f} ms）\n"
        f"- **AI 独立诈唬概率估计**：{belief_str}"
        f"{pending_line}{res_line}\n\n"
        "**此时全场真实底牌与左轮实弹位置（仅终局可见）：**\n" + "\n".join(hands_lines)
    )


def create_ui(model_config: dict[str, Any] | None = None, log_path: str | Path | None = None):
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

    all_seat_choices = [("真人玩家 (human)", "human")] + [(_agent_label(c), c) for c in ai_choices]
    default_s1 = "qwen35_9b_lora_sft" if "qwen35_9b_lora_sft" in ai_choices else "honest_conservative"
    default_s2 = "qwen35_9b_lora_rs" if "qwen35_9b_lora_rs" in ai_choices else "aggressive_bluffer"
    default_s3 = "card_counter"

    with gr.Blocks(title="骗子酒馆 AI 决策复盘工作台") as demo:
        session_state = gr.State(None)
        gr.Markdown(
            "# 🃏 骗子酒馆 AI — 真人对局测试与决策复盘工作台\n"
            "支持 **任意座位（或随机座位）** 参与对局、中文可视化牌桌、终局全量底牌/信念复盘，以及自动保存真人战绩日志。"
        )
        with gr.Row():
            s0 = gr.Dropdown(all_seat_choices, value="human", label="Seat 0")
            s1 = gr.Dropdown(all_seat_choices, value=default_s1, label="Seat 1")
            s2 = gr.Dropdown(all_seat_choices, value=default_s2, label="Seat 2")
            s3 = gr.Dropdown(all_seat_choices, value=default_s3, label="Seat 3")
            start_btn = gr.Button("🎮 开始新对局", variant="primary")
        with gr.Row():
            preset_mixed_btn = gr.Button("预设：综合体验桌 (SFT+RS+记牌)", size="sm")
            preset_hard_btn = gr.Button("预设：纯强模型高压桌 (2×SFT+RS)", size="sm")
            preset_rule_btn = gr.Button("预设：规则基线对照桌", size="sm")
            random_seat_cb = gr.Checkbox(value=False, label="开局随机分配 4 个座位（消除 Seat 0 先手偏差）")
            debug_seed = gr.Checkbox(value=False, label="调试：固定环境种子")
            seed_in = gr.Number(value=2026, label="调试种子", precision=0, interactive=False)

        table_html = gr.HTML(value=render_table_html({"status": "idle"}))

        with gr.Row():
            action_in = gr.Radio([], label="你的合法动作（单选后点击提交）", interactive=False)
        with gr.Row():
            action_btn = gr.Button("✅ 提交动作", variant="primary", interactive=False)
            resume_btn = gr.Button("🔄 继续 AI 回合", interactive=False)
        ai_status = gr.Markdown()

        with gr.Accordion("📜 终局全量复盘（对局结束后解锁：底牌、实弹位置、AI 信念概率）", open=True):
            with gr.Row():
                step_slider = gr.Slider(0, 1, value=0, step=1, interactive=False, label="复盘步骤（拖动或点击查看）")
                inspect_btn = gr.Button("🔍 查看该步复盘", interactive=False)
            replay_md = gr.Markdown("*对局结束后开放完整决策复盘。*")
            with gr.Accordion("展开该步原始复盘 JSON 与本局行为画像", open=False):
                replay_out = gr.JSON(label="终局复盘：动作说明、概率来源与事后真值")
                profile_out = gr.JSON(label="本局行为画像与回放校验")

        with gr.Accordion("📊 本地真人战绩统计（自动落盘）", open=True):
            log_summary_md = gr.Markdown(value=summarize_human_game_logs(log_path))

        with gr.Accordion("🔧 原始玩家观测 JSON（调试视图）", open=False):
            state_out = gr.JSON(label="当前玩家视图 / 结局")

        def render(session: InteractiveSessionManager):
            view = session.get_current_view()
            choices = session.human_action_choices()
            finished = session.engine is not None and session.engine.is_game_over()
            total_steps = len(session.engine.step_truth_logs) if session.engine else 0
            first_replay = session.inspect_replay_step(0)
            status_parts = []
            if session.ai_error:
                status_parts.append(f"⚠️ AI 回合暂停：{session.ai_error}。已完成步骤保留，可点击「继续 AI 回合」。")
            if session.last_log_error:
                status_parts.append(f"⚠️ 日志写入失败：{session.last_log_error}")
            elif finished and session.log_path is not None:
                status_parts.append(f"💾 本局回放已自动保存至 `{session.log_path}`。")
            return (
                session,
                render_table_html(view),
                view,
                gr.Radio(choices=choices, value=choices[0][1] if choices else None, interactive=bool(choices)),
                gr.Button(interactive=bool(choices)),
                gr.Button(interactive=bool(session.ai_error)),
                "\n\n".join(status_parts),
                gr.Slider(minimum=0, maximum=max(1, total_steps - 1), value=0, interactive=finished),
                gr.Button(interactive=finished),
                render_replay_markdown(first_replay),
                first_replay,
                session.get_session_profiles(),
                summarize_human_game_logs(log_path),
            )

        def on_start(a0, a1, a2, a3, randomize_seats, use_debug_seed, seed):
            specs = [a0, a1, a2, a3]
            if randomize_seats:
                rng = random.Random(int(seed)) if use_debug_seed else random.SystemRandom()
                rng.shuffle(specs)
            session = InteractiveSessionManager(agent_factory, log_path=log_path)
            try:
                session.start_game(
                    specs[0], specs[1], specs[2], specs[3],
                    int(seed) if use_debug_seed else None,
                )
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
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

        def on_inspect(session, index):
            data = session.inspect_replay_step(int(index)) if session else {"status": "idle"}
            return render_replay_markdown(data), data

        def preset_mixed():
            return (
                "human",
                "qwen35_9b_lora_sft" if "qwen35_9b_lora_sft" in ai_choices else "card_counter",
                "qwen35_9b_lora_rs" if "qwen35_9b_lora_rs" in ai_choices else "chameleon_adaptive",
                "card_counter",
            )

        def preset_hard():
            if "qwen35_9b_lora_sft" in ai_choices and "qwen35_9b_lora_rs" in ai_choices:
                return ("human", "qwen35_9b_lora_sft", "qwen35_9b_lora_rs", "qwen35_9b_lora_sft")
            return ("human", "card_counter", "chameleon_adaptive", "paranoid_hunter")

        def preset_rule():
            return ("human", "card_counter", "aggressive_bluffer", "chameleon_adaptive")

        outputs = [
            session_state, table_html, state_out, action_in, action_btn, resume_btn,
            ai_status, step_slider, inspect_btn, replay_md, replay_out, profile_out, log_summary_md,
        ]
        start_btn.click(
            on_start, [s0, s1, s2, s3, random_seat_cb, debug_seed, seed_in], outputs, api_name="start_game"
        )
        action_btn.click(on_action, [session_state, action_in], outputs, api_name="submit_action")
        resume_btn.click(on_resume, [session_state], outputs, api_name="resume_ai")
        debug_seed.change(lambda enabled: gr.Number(interactive=enabled), debug_seed, seed_in, api_name=False)
        inspect_btn.click(on_inspect, [session_state, step_slider], [replay_md, replay_out], api_name="inspect_replay")
        step_slider.release(on_inspect, [session_state, step_slider], [replay_md, replay_out], api_name=False)
        preset_mixed_btn.click(preset_mixed, [], [s0, s1, s2, s3], api_name=False)
        preset_hard_btn.click(preset_hard, [], [s0, s1, s2, s3], api_name=False)
        preset_rule_btn.click(preset_rule, [], [s0, s1, s2, s3], api_name=False)
    return demo


def launch_ui(
    host: str = "127.0.0.1",
    port: int = 7860,
    model_config: dict[str, Any] | None = None,
    log_path: str | Path | None = DEFAULT_HUMAN_LOG_PATH,
) -> None:
    create_ui(model_config, log_path=log_path).launch(server_name=host, server_port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Liar's Bar AI Workbench UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model-config", type=Path, help="Pipeline-generated model/checkpoint YAML")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_HUMAN_LOG_PATH,
        help="JSONL file to auto-save completed human/UI games",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable automatic game log saving")
    args = parser.parse_args()
    config = None
    if args.model_config:
        import yaml
        config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    launch_ui(
        host=args.host,
        port=args.port,
        model_config=config,
        log_path=None if args.no_log else args.log_path,
    )

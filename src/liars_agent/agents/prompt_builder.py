"""面向 Qwen3.5-9B 的结构化提示词模板与 JSON 输出解析器."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from liars_agent.engine.observation import PlayerObservation

SYSTEM_PROMPT = """你是一名精通《骗子酒馆（Liar's Bar）》博弈的 AI 决策智能体。
游戏规则摘要：
1. 牌组共 20 张：Q×6、K×6、A×6、JOKER（恶魔/万能牌，视为任意目标牌真牌）×2。
2. 每小局指定目标牌 T ∈ {Q, K, A}，存活玩家各发 5 张牌。
3. 轮到你行动时，你只能从给定的合法候选动作列表 (A0, A1, ...) 中选择一个动作：
   - CHALLENGE：质疑上家刚才打出的牌。若上家打出的牌中存在非 T 且非 JOKER 的假牌，则上家开一枪；若全为真牌，则你开一枪。
   - PLAY(cards)：从手牌中暗置打出 1~3 张牌，声称全部为目标牌 T。
4. 每名玩家左轮手枪共 6 个枪膛（其中 1 发实弹），已开枪 k 次的玩家下一枪致死概率为 1/(6-k)。

请严格输出一行 JSON 对象（不要输出额外 Markdown 解释），格式如下：
{"belief_bluff_prob": 0.0到1.0之间的浮点数, "estimated_opponent_style": "honest_conservative|aggressive_bluffer|card_counter|paranoid_hunter|chameleon_adaptive|unknown", "chosen_action_id": "合法动作ID如A0", "rationale": "简短决策依据"}"""


def build_user_prompt(obs: PlayerObservation) -> str:
    """将 `PlayerObservation` 格式化为紧凑、无信息泄漏的模型输入提示词."""
    lines: list[str] = [
        f"=== 对局状态 [Game={obs.game_id} | Round={obs.round_id} | Step={obs.step_id}] ===",
        f"你的座位: Seat {obs.observer_seat} | 本局目标牌: {obs.target_rank} (JOKER视为真牌)",
        f"你的当前手牌 ({len(obs.own_hand)}张): {list(obs.own_hand)}",
    ]
    if obs.own_played_cards_this_round:
        lines.append(f"你本小局已打出的牌: {list(obs.own_played_cards_this_round)}")

    status_strs = [
        f"Seat {p.seat}: {'存活' if p.alive else '淘汰'}(手牌={p.hand_count}, 已开枪={p.shots_taken}/6, 下枪致死率={p.next_shot_lethal_prob:.1%})"
        for p in obs.players_public_status
    ]
    lines.append("牌桌公开状态: " + " | ".join(status_strs))

    if obs.last_claim is not None:
        c = obs.last_claim
        comb_prior = obs.combinatorial_claim_truth_prior
        prior_str = f"{1.0 - comb_prior:.2%}" if comb_prior is not None else "N/A"
        lines.append(
            f"当前待应对声称: 上家 Seat {c.claimer_seat} 声称打出 {c.claimed_count} 张 {c.claimed_rank} "
            f"(其剩余手牌 {c.claimer_remaining_cards} 张 | 仅基于你已知牌的组合诈唬先验={prior_str})"
        )
    else:
        lines.append("当前待应对声称: 无（你是本小局首位出牌玩家）")

    if obs.ablation_mode != "last_action_only":
        cur_hist = [
            f"Step{e.step_id}:Seat{e.actor_seat}->{e.action_type}"
            + (f"({e.claimed_count}张{e.claimed_rank},余{e.actor_remaining_cards})" if e.action_type == "PLAY" else "")
            for e in obs.current_round_history
        ]
        lines.append("本小局公开动作流: " + (" -> ".join(cur_hist) if cur_hist else "无"))

    if obs.ablation_mode == "full" and obs.past_rounds_history:
        past_lines = [
            f"R{r.round_id}(目标{r.target_rank}): Seat{r.challenger_seat}质疑Seat{r.claimer_seat}声称{r.claimed_count}张, "
            f"亮牌={list(r.revealed_cards)}({'诈唬' if r.was_bluff else '真话'}), Seat{r.penalized_seat}开枪({r.shots_taken_after}/6)"
            for r in obs.past_rounds_history[-6:]
        ]
        lines.append("过往小局亮牌历史: " + " ; ".join(past_lines))
    else:
        lines.append(f"过往小局亮牌历史: (消融模式={obs.ablation_mode}，未提供或尚无历史)")

    lines.append("可选合法候选动作:")
    for act in obs.legal_actions:
        if act.action_type == "CHALLENGE":
            lines.append(f"  - {act.action_id}: CHALLENGE (质疑上家出牌)")
        else:
            lines.append(
                f"  - {act.action_id}: PLAY {list(act.cards)} (声称打出 {len(act.cards)} 张 {obs.target_rank})"
            )

    lines.append("请输出 JSON 决策:")
    return "\n".join(lines)


def parse_llm_decision_json(raw_text: str) -> Optional[dict[str, Any]]:
    """解析模型输出的 JSON 字符串（支持从代码块或杂讯文本中提取第一个 JSON 对象）."""
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "chosen_action_id" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "chosen_action_id" in parsed:
                return parsed
        except json.JSONDecodeError:
            return None
    return None

"""Shared public observation rendering and strict action/binary-belief protocols."""
from __future__ import annotations

import json
from typing import Any

from liars_agent.engine.observation import PlayerObservation

RULES_PROMPT = "你是一名精通《骗子酒馆（Liar's Bar）》博弈的 AI 决策智能体。\n游戏规则（liars_bar_research_v2）：\n1. 四人开始，死亡者淘汰，最后一名存活玩家赢得整场游戏。打光手牌只退出当前小局行动，不代表整场获胜。\n2. 牌组共 20 张：Q×6、K×6、A×6、JOKER×2。JOKER 是万能真牌，没有额外惩罚效果。每小局重新洗牌，每名存活玩家各发 5 张，多余牌不使用；随机选择目标牌 T ∈ {Q, K, A}，可与上一小局相同，本小局内不变。\n3. 第一小局随机先手，按座位顺序循环，跳过死亡和空手玩家。轮到你时，只能从当前合法候选动作列表 (A0, A1, ...) 中选择一个动作；没有摸牌或主动跳过。\n   - PLAY(cards)：从手牌中暗置打出 1~3 张牌，声称全部为 T；张数如实公开，可以全真、全假或真假混出，无需与上一手张数相同或递增。\n   - CHALLENGE：质疑最近一次出牌，只翻开该手牌。若其中存在非 T 且非 JOKER 的假牌，则出牌者开一枪；若全部为真牌，则质疑者开一枪。小局第一手只能出牌；继续出牌后，不能再回头质疑更早的一手。\n4. 空手玩家不再出牌或质疑，但刚打出的最后一手仍可被下一名行动玩家质疑；若是假牌且被质疑，空手出牌者仍受罚。下家继续出牌后，该空手玩家一直被跳过，直到下一小局重新发牌。\n5. 存在上一手出牌且仅当前行动玩家还有牌时，当前玩家必须质疑上一手，不能继续出牌。此规则适用于两人、三人、四人仍存活的情况；例如 A 刚打空、其余人已空手、仅 B 有牌且轮到 B，则 B 必须质疑 A。\n6. 每人左轮有 6 个弹膛、1 发实弹，开局随机确定实弹位置，整场不重置。受罚开一枪不一定死亡；仍存活且已开枪 k 次者下一枪致死概率为 1/(6-k)。实弹位置不向任何玩家公开，包括枪的持有者。\n7. 每次质疑产生一次开枪惩罚并结束当前小局，无论是否有人死亡。若剩余多名玩家，重新发牌，由受罚者（若存活）或其下一名存活玩家先手；已开枪次数跨小局累计。只剩一人时整场结束。\n8. 你只能使用自己的手牌和本小局出过的牌、目标牌、公开行动、各人的剩余牌数/存活状态/已开枪次数，以及过往质疑的公开亮牌与结算。其他人的底牌、未翻开的出牌和所有实弹位置均不可见；完整真值复盘仅在整场结束后开放。"

SYSTEM_PROMPT = RULES_PROMPT + '\n\n仅输出一行 JSON，且只能包含 chosen_action_id 字段，例如 {"chosen_action_id":"A0"}。动作必须来自当前合法候选列表；不要输出概率、画像或解释。'
BELIEF_SYSTEM_PROMPT = RULES_PROMPT + '\n\n任务：判断最近一手牌是否含有假牌。仅回答一个数字标签：0 表示全部为真牌，1 表示至少一张假牌。不要输出动作、JSON、概率或解释。'


def _event_text(event: Any) -> str:
    detail = (f"({event.claimed_count}张{event.claimed_rank},余{event.actor_remaining_cards})"
              if event.action_type == "PLAY" else f"({'强制' if event.forced_challenge else '自主'}质疑)")
    return f"R{event.round_id}/Step{event.step_id}:Seat{event.actor_seat}->{event.action_type}{detail}"


def render_observation(obs: PlayerObservation) -> str:
    """Same factual context for both tasks; no private run identifier or seed."""
    lines = [f"=== 对局状态 [Round={obs.round_id} | Step={obs.step_id}] ===",
             f"你的座位: Seat {obs.observer_seat} | 本局目标牌: {obs.target_rank} (JOKER视为真牌)",
             f"你的当前手牌 ({len(obs.own_hand)}张): {list(obs.own_hand)}"]
    if obs.own_played_cards_this_round:
        lines.append(f"你本小局已打出的牌: {list(obs.own_played_cards_this_round)}")
    lines.append("牌桌公开状态: " + " | ".join(
        f"Seat {p.seat}: {'存活' if p.alive else '淘汰'}(手牌={p.hand_count}, 已开枪={p.shots_taken}/{p.max_chambers}, 下枪致死率={p.next_shot_lethal_prob:.1%})"
        for p in obs.players_public_status))
    if obs.last_claim is not None:
        c = obs.last_claim
        lines.append(f"当前待应对声称: 上家 Seat {c.claimer_seat} 声称打出 {c.claimed_count} 张 {c.claimed_rank} (其剩余手牌 {c.claimer_remaining_cards} 张)")
        capacity = obs.hand_capacity_prior
        if capacity is not None:
            lines.append(f"均匀分配参考模型下，上家出牌前手牌至少含{c.claimed_count}张真牌的容量概率={capacity:.2%}；这不是该手实际诈唬概率。")
    else:
        lines.append("当前待应对声称: 无（你是本小局首位出牌玩家）")
    if obs.ablation_mode != "last_action_only":
        lines.append("本小局公开动作流: " + (" -> ".join(_event_text(e) for e in obs.current_round_history) or "无"))
    if obs.ablation_mode == "full":
        lines.append("过往小局完整公开动作: " + (" ; ".join(_event_text(e) for e in obs.past_public_actions) or "无"))
        lines.append("过往小局亮牌历史: " + (" ; ".join(
            f"R{r.round_id}(目标{r.target_rank}): Seat{r.challenger_seat}质疑Seat{r.claimer_seat}声称{r.claimed_count}张, "
            f"亮牌={list(r.revealed_cards)}({'诈唬' if r.was_bluff else '真话'}), Seat{r.penalized_seat}开枪({r.shots_taken_after}/6, {'死亡' if r.lethal_shot else '存活'})"
            for r in obs.past_rounds_history) or "无"))
    else:
        lines.append(f"过往小局历史: (消融模式={obs.ablation_mode}，未提供)")
    return "\n".join(lines)


def build_user_prompt(obs: PlayerObservation) -> str:
    lines = [render_observation(obs), "可选合法候选动作:"]
    for action in obs.legal_actions:
        detail = "CHALLENGE (质疑上家出牌)" if action.action_type == "CHALLENGE" else f"PLAY {list(action.cards)} (声称打出 {len(action.cards)} 张 {obs.target_rank})"
        lines.append(f"  - {action.action_id}: {detail}")
    lines.append('仅输出 {"chosen_action_id":"合法动作ID"}:')
    return "\n".join(lines)


def build_bluff_user_prompt(obs: PlayerObservation) -> str:
    if obs.last_claim is None:
        raise ValueError("Binary bluff scoring requires a pending claim.")
    return render_observation(obs) + "\n最近一手是否含假牌？仅回答 0（全真）或 1（诈唬）："


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def validate_action_decision(obs: PlayerObservation, completion: str) -> dict[str, str]:
    """Strict shared validator for supervised targets and model inference."""
    if not isinstance(completion, str):
        raise ValueError("Action completion must be a JSON string.")
    try:
        parsed = json.loads(completion, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Action output must be one strict JSON object.") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"chosen_action_id"}:
        raise ValueError("Action output must contain only chosen_action_id.")
    action_id = parsed["chosen_action_id"]
    if not isinstance(action_id, str) or action_id not in {a.action_id for a in obs.legal_actions}:
        raise ValueError("Model chose an unavailable action ID.")
    return parsed


def parse_llm_decision_json(raw_text: str) -> dict[str, Any] | None:
    """Legacy syntax-only parser; validation requires validate_action_decision."""
    try:
        parsed = json.loads(raw_text, object_pairs_hook=_unique_object)
        return parsed if isinstance(parsed, dict) and set(parsed) == {"chosen_action_id"} else None
    except (ValueError, TypeError):
        return None

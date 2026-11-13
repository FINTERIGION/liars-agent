"""四人骗子酒馆确定性状态机与规则引擎."""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.types import (
    STANDARD_DECK_COUNTS,
    TARGET_RANKS,
    AgentDecisionRecord,
    CandidateAction,
    ClaimEvent,
    PlayerPublicInfo,
    PublicEvent,
    RoundResolution,
    are_cards_truthful,
    sort_cards,
)


def _derive_seed(*parts: int | str) -> int:
    """通过 SHA256 将任意元组派生为确定性的 64-bit 整数随机种子."""
    payload = ":".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class PendingPlayTruth:
    """内部真值：记录上家实际打出的牌面（仅在质疑翻牌结算或事后复盘中读取）."""

    round_id: int
    step_id: int
    claimer_seat: int
    claimed_rank: str
    actual_cards: tuple[str, ...]

    @property
    def claimed_count(self) -> int:
        return len(self.actual_cards)

    @property
    def is_bluff(self) -> bool:
        return not are_cards_truthful(self.actual_cards, self.claimed_rank)


@dataclass
class StepTruthRecord:
    """单步全量真值快照（用于评测复盘、风险分层诊断与回放校验）."""

    game_id: str
    round_id: int
    step_id: int
    actor_seat: int
    target_rank: str
    observation: dict[str, Any]
    chosen_action: dict[str, Any]
    decision_record: dict[str, Any]
    ground_truth_hands: dict[int, list[str]]
    ground_truth_fatal_chambers: dict[int, int]
    pending_claim_actual_cards: Optional[list[str]]
    pending_claim_was_bluff: Optional[bool]
    round_resolution: Optional[dict[str, Any]] = None


class LiarsBarEngine:
    """研究用四人骗子酒馆确定性规则引擎.

    核心特性：
    1. 确定性可复现：给定 `game_seed`，每小局的洗牌、目标牌以及各座位的左轮实弹位置完全确定；
    2. 座位槽位绑定：支持 `seat_offset` 用于配对评测轮换，确保轮换座位时发牌槽位与实弹槽位严格对齐；
    3. 强制质疑：存在上一手出牌且除当前行动玩家外其余存活玩家均已空手时，当前玩家必须质疑上一手；
    4. 严格观测隔离：`get_observation(seat)` 仅暴露玩家自身手牌与公开历史。
    """

    def __init__(
        self,
        game_id: str = "game_0001",
        game_seed: int = 42,
        num_players: int = 4,
        hand_size: int = 5,
        max_play_cards: int = 3,
        max_chambers: int = 6,
    ) -> None:
        if num_players != 4:
            raise ValueError("Research rules specify a 4-player game (num_players=4).")
        self.game_id = game_id
        self.game_seed = int(game_seed)
        self.num_players = num_players
        self.hand_size = hand_size
        self.max_play_cards = max_play_cards
        self.max_chambers = max_chambers

        # 玩家存活与左轮状态
        self.alive: list[bool] = [True] * num_players
        self.shots_taken: list[int] = [0] * num_players
        # 每个座位的实弹槽位 (1..6)，由 (game_seed, "chamber", seat) 独立派生
        self.fatal_chamber_slot: list[int] = [
            random.Random(_derive_seed(self.game_seed, "chamber", seat)).randint(1, self.max_chambers)
            for seat in range(num_players)
        ]

        # 小局与回合状态
        self.round_id: int = 0
        self.step_id: int = 0
        self.target_rank: str = "Q"
        self.hands: list[list[str]] = [[] for _ in range(num_players)]
        self.played_cards_this_round: list[list[str]] = [[] for _ in range(num_players)]
        self.current_turn_seat: int = 0
        self.pending_play: Optional[PendingPlayTruth] = None
        self.current_round_history: list[PublicEvent] = []
        self.past_rounds_history: list[RoundResolution] = []
        self.step_truth_logs: list[StepTruthRecord] = []
        self.winner_seat: Optional[int] = None

        # 初始化第一小局
        self._start_new_round(starter_seat=self._initial_starter_seat())

    def _initial_starter_seat(self) -> int:
        rng = random.Random(_derive_seed(self.game_seed, "initial_starter"))
        return rng.randint(0, self.num_players - 1)

    def _build_full_deck(self) -> list[str]:
        deck: list[str] = []
        for rank, count in STANDARD_DECK_COUNTS.items():
            deck.extend([rank] * count)
        return deck

    def _next_alive_seat(self, from_seat: int) -> int:
        for offset in range(1, self.num_players + 1):
            candidate = (from_seat + offset) % self.num_players
            if self.alive[candidate]:
                return candidate
        raise RuntimeError("No alive players remaining.")

    def _next_alive_seat_with_cards(self, from_seat: int) -> Optional[int]:
        for offset in range(1, self.num_players):
            candidate = (from_seat + offset) % self.num_players
            if self.alive[candidate] and len(self.hands[candidate]) > 0:
                return candidate
        return None

    def _start_new_round(self, starter_seat: int) -> None:
        self.round_id += 1
        self.step_id = 0
        self.pending_play = None
        self.current_round_history = []
        self.played_cards_this_round = [[] for _ in range(self.num_players)]

        round_rng = random.Random(_derive_seed(self.game_seed, "round", self.round_id))
        self.target_rank = round_rng.choice(TARGET_RANKS)

        deck = self._build_full_deck()
        round_rng.shuffle(deck)

        # 按固定座位槽位 0..3 发牌（保证同种子下每个座位在各小局拿到的牌完全确定）
        self.hands = [[] for _ in range(self.num_players)]
        for seat in range(self.num_players):
            slot_cards = deck[seat * self.hand_size : (seat + 1) * self.hand_size]
            if self.alive[seat]:
                self.hands[seat] = list(sort_cards(slot_cards))

        if not self.alive[starter_seat]:
            starter_seat = self._next_alive_seat(starter_seat)
        self.current_turn_seat = starter_seat

    def is_game_over(self) -> bool:
        return self.winner_seat is not None

    def alive_seats(self) -> list[int]:
        return [s for s in range(self.num_players) if self.alive[s]]

    def _must_force_challenge(self, seat: int) -> bool:
        """检查当前行动玩家是否因边界条件必须执行 CHALLENGE."""
        if self.pending_play is None or not self.alive[seat] or not self.hands[seat]:
            return False
        # 仅当前行动玩家还有牌时，必须立即质疑，不能再出牌。
        other_alive_with_cards = [
            s
            for s in range(self.num_players)
            if self.alive[s] and s != seat and len(self.hands[s]) > 0
        ]
        return len(other_alive_with_cards) == 0

    def get_legal_actions(self, seat: Optional[int] = None) -> tuple[CandidateAction, ...]:
        """枚举当前行动玩家的所有合法候选动作（按牌面多重集去重并固定编号 A0, A1, ...）."""
        if self.is_game_over():
            return ()
        target_seat = self.current_turn_seat if seat is None else seat
        if (
            target_seat != self.current_turn_seat
            or not self.alive[target_seat]
            or not self.hands[target_seat]
        ):
            return ()

        actions: list[CandidateAction] = []
        idx = 0

        # 1. 有手牌的当前玩家可以质疑上一手；空手玩家不再参与本小局行动。
        if self.pending_play is not None:
            actions.append(
                CandidateAction(action_id=f"A{idx}", action_type="CHALLENGE", cards=())
            )
            idx += 1

        # 若触发边界强制质疑规则，则只允许 CHALLENGE
        if self._must_force_challenge(target_seat):
            return tuple(actions)

        # 2. 枚举打出 1..max_play_cards 张手牌的所有唯一牌面组合
        hand = list(sort_cards(self.hands[target_seat]))
        max_k = min(self.max_play_cards, len(hand))
        seen_multisets: set[tuple[str, ...]] = set()

        for k in range(1, max_k + 1):
            for combo in itertools.combinations(hand, k):
                sorted_combo = sort_cards(combo)
                if sorted_combo not in seen_multisets:
                    seen_multisets.add(sorted_combo)
                    actions.append(
                        CandidateAction(
                            action_id=f"A{idx}",
                            action_type="PLAY",
                            cards=sorted_combo,
                        )
                    )
                    idx += 1

        return tuple(actions)

    def get_observation(self, seat: Optional[int] = None) -> PlayerObservation:
        """构造指定座位的局部可见状态（严格隔离其他玩家手牌、上家暗置牌面与实弹槽位）."""
        obs_seat = self.current_turn_seat if seat is None else seat
        public_status = tuple(
            PlayerPublicInfo(
                seat=s,
                alive=self.alive[s],
                hand_count=len(self.hands[s]),
                shots_taken=self.shots_taken[s],
                max_chambers=self.max_chambers,
            )
            for s in range(self.num_players)
        )
        last_claim: Optional[ClaimEvent] = None
        if self.pending_play is not None:
            last_claim = ClaimEvent(
                round_id=self.pending_play.round_id,
                step_id=self.pending_play.step_id,
                claimer_seat=self.pending_play.claimer_seat,
                claimed_rank=self.pending_play.claimed_rank,
                claimed_count=self.pending_play.claimed_count,
                claimer_remaining_cards=len(self.hands[self.pending_play.claimer_seat]),
            )

        legal = self.get_legal_actions(obs_seat) if obs_seat == self.current_turn_seat else ()

        return PlayerObservation(
            game_id=self.game_id,
            round_id=self.round_id,
            step_id=self.step_id,
            observer_seat=obs_seat,
            target_rank=self.target_rank,
            own_hand=sort_cards(self.hands[obs_seat]),
            own_played_cards_this_round=sort_cards(self.played_cards_this_round[obs_seat]),
            players_public_status=public_status,
            last_claim=last_claim,
            current_round_history=tuple(self.current_round_history),
            past_rounds_history=tuple(self.past_rounds_history),
            legal_actions=legal,
            ablation_mode="full",
        )

    def step(
        self,
        action: CandidateAction | str,
        decision_record: Optional[AgentDecisionRecord] = None,
    ) -> Optional[RoundResolution]:
        """执行一步合法动作并推进状态机.

        支持传入 `CandidateAction` 对象或 `action_id` 字符串（如 `"A0"`）。
        返回：若本步触发了质疑结算，返回 `RoundResolution`；若为常规出牌，返回 `None`。
        """
        if self.is_game_over():
            raise RuntimeError("Cannot call step() on a finished game.")

        actor_seat = self.current_turn_seat
        obs_before = self.get_observation(actor_seat)
        legal_actions = obs_before.legal_actions
        if not legal_actions:
            raise RuntimeError(f"Seat {actor_seat} has no legal actions.")

        resolved_action: Optional[CandidateAction] = None
        if isinstance(action, str):
            for cand in legal_actions:
                if cand.action_id == action or (
                    action == "CHALLENGE" and cand.action_type == "CHALLENGE"
                ):
                    resolved_action = cand
                    break
        elif isinstance(action, CandidateAction):
            for cand in legal_actions:
                if (
                    cand.action_type == action.action_type
                    and sort_cards(cand.cards) == sort_cards(action.cards)
                ):
                    resolved_action = cand
                    break

        if resolved_action is None:
            raise ValueError(
                f"Illegal action {action!r} for seat {actor_seat}. "
                f"Legal actions: {[a.to_dict() for a in legal_actions]}"
            )

        # 出牌只能交给仍有手牌的存活下家，不能让空手玩家重新参与质疑。
        next_seat = None
        if resolved_action.action_type == "PLAY":
            next_seat = self._next_alive_seat_with_cards(actor_seat)
            if next_seat is None:
                raise RuntimeError("Cannot play without another alive player holding cards.")

        rec = decision_record or AgentDecisionRecord()
        hands_snapshot = {s: list(self.hands[s]) for s in range(self.num_players)}
        chambers_snapshot = {s: self.fatal_chamber_slot[s] for s in range(self.num_players)}
        pending_cards_snapshot = (
            list(self.pending_play.actual_cards) if self.pending_play is not None else None
        )
        pending_bluff_snapshot = (
            self.pending_play.is_bluff if self.pending_play is not None else None
        )

        self.step_id += 1
        forced_flag = len(legal_actions) == 1 and legal_actions[0].action_type == "CHALLENGE"

        if resolved_action.action_type == "PLAY":
            # 从当前玩家手牌移除打出的牌
            for card in resolved_action.cards:
                self.hands[actor_seat].remove(card)
            self.played_cards_this_round[actor_seat].extend(resolved_action.cards)

            self.pending_play = PendingPlayTruth(
                round_id=self.round_id,
                step_id=self.step_id,
                claimer_seat=actor_seat,
                claimed_rank=self.target_rank,
                actual_cards=sort_cards(resolved_action.cards),
            )
            pub_event = PublicEvent(
                round_id=self.round_id,
                step_id=self.step_id,
                actor_seat=actor_seat,
                action_type="PLAY",
                claimed_count=len(resolved_action.cards),
                claimed_rank=self.target_rank,
                actor_remaining_cards=len(self.hands[actor_seat]),
                forced_challenge=False,
            )
            self.current_round_history.append(pub_event)

            self.step_truth_logs.append(
                StepTruthRecord(
                    game_id=self.game_id,
                    round_id=self.round_id,
                    step_id=self.step_id,
                    actor_seat=actor_seat,
                    target_rank=self.target_rank,
                    observation=obs_before.to_dict(),
                    chosen_action=resolved_action.to_dict(),
                    decision_record=rec.to_dict(),
                    ground_truth_hands=hands_snapshot,
                    ground_truth_fatal_chambers=chambers_snapshot,
                    pending_claim_actual_cards=pending_cards_snapshot,
                    pending_claim_was_bluff=pending_bluff_snapshot,
                    round_resolution=None,
                )
            )

            # 跳过已空手玩家；若下家是最后一名有牌者，则其必须质疑本次出牌。
            self.current_turn_seat = next_seat
            return None

        # CHALLENGE 分支
        assert self.pending_play is not None
        pub_event = PublicEvent(
            round_id=self.round_id,
            step_id=self.step_id,
            actor_seat=actor_seat,
            action_type="CHALLENGE",
            claimed_count=0,
            claimed_rank=self.target_rank,
            actor_remaining_cards=len(self.hands[actor_seat]),
            forced_challenge=forced_flag,
        )
        self.current_round_history.append(pub_event)

        was_bluff = self.pending_play.is_bluff
        challenge_succeeded = was_bluff
        penalized_seat = self.pending_play.claimer_seat if challenge_succeeded else actor_seat

        shots_before = self.shots_taken[penalized_seat]
        shots_after = shots_before + 1
        self.shots_taken[penalized_seat] = shots_after
        lethal = shots_after >= self.fatal_chamber_slot[penalized_seat]
        if lethal:
            self.alive[penalized_seat] = False
            self.hands[penalized_seat] = []

        resolution = RoundResolution(
            round_id=self.round_id,
            target_rank=self.target_rank,
            claimer_seat=self.pending_play.claimer_seat,
            challenger_seat=actor_seat,
            claimed_count=self.pending_play.claimed_count,
            revealed_cards=self.pending_play.actual_cards,
            was_bluff=was_bluff,
            challenge_succeeded=challenge_succeeded,
            penalized_seat=penalized_seat,
            shots_taken_before=shots_before,
            shots_taken_after=shots_after,
            lethal_shot=lethal,
            forced_challenge=forced_flag,
        )
        self.past_rounds_history.append(resolution)

        self.step_truth_logs.append(
            StepTruthRecord(
                game_id=self.game_id,
                round_id=self.round_id,
                step_id=self.step_id,
                actor_seat=actor_seat,
                target_rank=self.target_rank,
                observation=obs_before.to_dict(),
                chosen_action=resolved_action.to_dict(),
                decision_record=rec.to_dict(),
                ground_truth_hands=hands_snapshot,
                ground_truth_fatal_chambers=chambers_snapshot,
                pending_claim_actual_cards=pending_cards_snapshot,
                pending_claim_was_bluff=pending_bluff_snapshot,
                round_resolution=resolution.to_dict(),
            )
        )

        alive_remaining = self.alive_seats()
        if len(alive_remaining) == 1:
            self.winner_seat = alive_remaining[0]
        else:
            # 下一小局由受罚者（若存活）或其紧邻存活下家先手
            next_starter = (
                penalized_seat
                if self.alive[penalized_seat]
                else self._next_alive_seat(penalized_seat)
            )
            self._start_new_round(starter_seat=next_starter)

        return resolution

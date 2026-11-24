"""History-posterior Bayesian opponent inference, calibrated v4 posterior teacher, and count-history classifier."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Sequence, Tuple

from liars_agent.agents.base import (
    BaseAgent,
    describe_action,
    public_opponent_profile,
    split_play_actions,
)
from liars_agent.agents.prompt_builder import (
    BELIEF_SYSTEM_PROMPT_V4,
    SYSTEM_PROMPT_V4,
    build_bluff_user_prompt,
    build_user_prompt,
    validate_action_decision,
)
from liars_agent.contracts import current_contract, require_contract
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.replay import GameReplayBundle, verify_replay_determinism
from liars_agent.engine.state import LiarsBarEngine, PendingPlayTruth, _derive_seed
from liars_agent.engine.types import (
    AgentDecisionRecord,
    CandidateAction,
    ClaimEvent,
    PlayerPublicInfo,
    PublicEvent,
    RoundResolution,
    is_card_truthful,
    sort_cards,
)
from liars_agent.training.adaptive_opponents import (
    ArchetypeSpec,
    ParameterizedArchetypeBot,
    V4_TRAIN_ARCHETYPES,
    extract_count_conditioned_profile,
)


def _clamp(p: float, eps: float = 0.02) -> float:
    return min(1.0 - eps, max(eps, p))


def _logit(p: float) -> float:
    pc = _clamp(p, eps=1e-4)
    return math.log(pc / (1.0 - pc))


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def infer_opponent_posteriors(
    obs: PlayerObservation,
    archetypes: Sequence[ArchetypeSpec] = V4_TRAIN_ARCHETYPES,
    recency_decay: float = 0.90,
) -> dict[int, dict[str, float]]:
    """Pure-observation Bayesian posterior P(archetype | public_history) for each seat.

    Uses ONLY public events and public round resolutions in `obs`. Never reads hidden
    identities, true hands, or seeds. Recency decay (<1.0) enables adapting to mid-game
    strategy switches while retaining earlier evidence.
    """
    if not archetypes:
        raise ValueError("archetypes must be non-empty.")
    events = (*obs.past_public_actions, *obs.current_round_history)
    # Pair each event with the immediately preceding PLAY's claimed_count within the same round
    annotated_events: list[tuple[PublicEvent, int | None]] = []
    prev_round: int | None = None
    prev_claimed_count: int | None = None
    for ev in events:
        if ev.round_id != prev_round:
            prev_round = ev.round_id
            prev_claimed_count = None
        annotated_events.append((ev, prev_claimed_count))
        prev_claimed_count = ev.claimed_count if ev.action_type == "PLAY" else None

    posteriors: dict[int, dict[str, float]] = {}

    for player in obs.players_public_status:
        seat = player.seat
        seat_events = [(ev, pcc) for ev, pcc in annotated_events if ev.actor_seat == seat]
        seat_reveals = [r for r in obs.past_rounds_history if r.claimer_seat == seat]

        log_weights: list[float] = []
        for spec in archetypes:
            lw = 0.0
            # Likelihood from play-count distribution & count-conditioned voluntary challenge behavior
            n_ev = len(seat_events)
            for idx, (ev, prev_cc) in enumerate(seat_events):
                w_rec = recency_decay ** (n_ev - 1 - idx)
                if ev.action_type == "PLAY":
                    p_multi = _clamp(spec.obs_p_multi_play)
                    lw += 0.25 * w_rec * math.log(p_multi if ev.claimed_count >= 2 else (1.0 - p_multi))
                if ev.step_id > 1 and not ev.forced_challenge:
                    if prev_cc == 1:
                        p_ch = _clamp(spec.single_challenge_base)
                    elif prev_cc is not None and prev_cc >= 2:
                        p_ch = _clamp(spec.multi_challenge_base)
                    else:
                        p_ch = _clamp(spec.obs_p_voluntary_challenge)
                    if ev.action_type == "CHALLENGE":
                        lw += 0.65 * w_rec * math.log(p_ch)
                    elif ev.action_type == "PLAY":
                        lw += 0.25 * w_rec * math.log(1.0 - p_ch)

            # Strong likelihood from count-conditioned revealed claims
            n_rev = len(seat_reveals)
            for idx, rev in enumerate(seat_reveals):
                w_rec = recency_decay ** (n_rev - 1 - idx)
                p_bluff = _clamp(
                    spec.obs_p_bluff_given_single if rev.claimed_count == 1 else spec.obs_p_bluff_given_multi
                )
                lw += 1.8 * w_rec * math.log(p_bluff if rev.was_bluff else (1.0 - p_bluff))
            log_weights.append(lw)

        max_lw = max(log_weights)
        exps = [math.exp(lw - max_lw) for lw in log_weights]
        total = sum(exps)
        posteriors[seat] = {
            spec.archetype_id: exps[i] / total for i, spec in enumerate(archetypes)
        }
    return posteriors


def _archetype_bayes_bluff_prob(
    spec: ArchetypeSpec,
    claimed_count: int,
    capacity: float,
    claimer_shots: int,
) -> float:
    """Exact Bayes' rule P(bluff | claimed_count, capacity, archetype) without double-counting (1-capacity).

    Accounts for claim-size selection likelihood: if an archetype rarely bluffs with 2~3 cards
    (q_bluff(multi) << q_truth(multi)), observing a 2~3 card claim is itself strong evidence
    that the claimer held enough truthful cards!
    """
    if capacity <= 0.0:
        return 1.0
    mult = spec.high_risk_bluff_multiplier if claimer_shots >= 2 else 1.0
    if claimed_count == 1:
        p_vol = _clamp(spec.single_bluff_when_has_truth * mult, eps=0.02)
        q_truth = _clamp(1.0 - spec.truth_multi_bias, eps=0.05)
        q_bluff = _clamp(1.0 - spec.bluff_multi_bias, eps=0.05)
    else:
        p_vol = _clamp(spec.multi_bluff_when_has_truth * mult, eps=0.02)
        q_truth = _clamp(spec.truth_multi_bias, eps=0.05)
        q_bluff = _clamp(spec.bluff_multi_bias, eps=0.05)

    # Likelihood of observing this claim size via bluff vs via truthful play
    num = ((1.0 - capacity) + capacity * p_vol) * q_bluff
    den = num + (capacity * (1.0 - p_vol)) * q_truth
    if den <= 0.0:
        return 0.5
    return _clamp(num / den, eps=0.02)


def posterior_bluff_probability(
    obs: PlayerObservation,
    archetypes: Sequence[ArchetypeSpec] = V4_TRAIN_ARCHETYPES,
    recency_decay: float = 0.90,
) -> float | None:
    """Zero-training exact Bayes' rule posterior bluff probability over archetypes + direct reveal shrinkage."""
    if obs.last_claim is None:
        return None
    capacity = obs.hand_capacity_prior
    if capacity is None:
        capacity = 0.5
    if capacity <= 0.0:
        return 1.0

    claim = obs.last_claim
    claimer_shots = 0
    for p in obs.players_public_status:
        if p.seat == claim.claimer_seat:
            claimer_shots = p.shots_taken
            break

    post = infer_opponent_posteriors(obs, archetypes=archetypes, recency_decay=recency_decay)[claim.claimer_seat]
    spec_by_id = {s.archetype_id: s for s in archetypes}

    p_bayes = sum(
        post[aid] * _archetype_bayes_bluff_prob(spec_by_id[aid], claim.claimed_count, capacity, claimer_shots)
        for aid in post
    )

    # Direct Beta-Binomial shrinkage on exact (seat, count_bucket) reveals for unseen extreme tells
    prof = extract_count_conditioned_profile(obs)[str(claim.claimer_seat)]
    if claim.claimed_count == 1 and prof["single_reveals"] > 0:
        n_r = prof["single_reveals"]
        emp = (prof["single_bluffs"] + 2.0 * p_bayes) / (n_r + 2.0)
        w = min(0.65, n_r / (n_r + 2.0))
        p_bayes = (1.0 - w) * p_bayes + w * emp
    elif claim.claimed_count >= 2 and prof["multi_reveals"] > 0:
        n_r = prof["multi_reveals"]
        emp = (prof["multi_bluffs"] + 2.0 * p_bayes) / (n_r + 2.0)
        w = min(0.65, n_r / (n_r + 2.0))
        p_bayes = (1.0 - w) * p_bayes + w * emp

    return min(0.99, max(0.01, p_bayes))


def extract_v4_belief_features(obs: PlayerObservation, include_history: bool = True) -> list[float]:
    """18-dim feature vector adding count-conditioned & recency-window public history stats."""
    hand_len = len(obs.own_hand)
    own_truth = sum(1 for c in obs.own_hand if is_card_truthful(c, obs.target_rank))
    own_truth_ratio = own_truth / max(1, hand_len)
    own_lethal = obs.own_next_shot_lethal_prob

    if obs.last_claim is None:
        return [1.0, hand_len / 5.0, own_truth_ratio, own_lethal] + [0.0] * 14

    claim = obs.last_claim
    cap = obs.hand_capacity_prior if obs.hand_capacity_prior is not None else 0.5
    shortage = 1.0 - cap
    is_single = 1.0 if claim.claimed_count == 1 else 0.0
    is_multi = 1.0 if claim.claimed_count >= 2 else 0.0
    empty_claimer = 1.0 if claim.claimer_remaining_cards == 0 else 0.0

    if not include_history:
        return [
            1.0,
            hand_len / 5.0,
            own_truth_ratio,
            own_lethal,
            claim.claimed_count / 3.0,
            is_single,
            is_multi,
            shortage,
            empty_claimer,
            0.5, 0.0,
            0.5, 0.0,
            0.5, 0.0,
            0.5,
            0.25,
            obs.round_id / 10.0,
        ]

    prof = extract_count_conditioned_profile(obs)[str(claim.claimer_seat)]
    single_rate = (prof["single_bluffs"] + 0.5) / (prof["single_reveals"] + 1.0)
    multi_rate = (prof["multi_bluffs"] + 0.5) / (prof["multi_reveals"] + 1.0)
    matched_rate = single_rate if is_single else multi_rate
    matched_n = prof["single_reveals"] if is_single else prof["multi_reveals"]
    opp_rate = multi_rate if is_single else single_rate
    opp_n = prof["multi_reveals"] if is_single else prof["single_reveals"]

    recent_rate = (prof["recent_bluffs"] + 0.5) / (prof["recent_reveals"] + 1.0)
    recent_strength = min(1.0, prof["recent_reveals"] / 3.0)
    multi_play_rate = prof["multi_play_rate"] if prof["multi_play_rate"] is not None else 0.5
    vol_ch_rate = prof["voluntary_challenge_rate"] if prof["voluntary_challenge_rate"] is not None else 0.25

    return [
        1.0,
        hand_len / 5.0,
        own_truth_ratio,
        own_lethal,
        claim.claimed_count / 3.0,
        is_single,
        is_multi,
        shortage,
        empty_claimer,
        matched_rate,
        min(1.0, matched_n / 3.0),
        opp_rate,
        min(1.0, opp_n / 3.0),
        recent_rate,
        recent_strength,
        multi_play_rate,
        vol_ch_rate,
        obs.round_id / 10.0,
    ]


class CountHistoryLogisticClassifier:
    """Lightweight logistic regression over count-conditioned public history features."""

    def __init__(self, include_history: bool = True) -> None:
        self.include_history = include_history
        self.weights: list[float] = [0.0] * 18

    def predict(self, obs: PlayerObservation) -> float:
        if obs.last_claim is None:
            return 0.0
        if (obs.hand_capacity_prior or 0.0) <= 0.0:
            return 1.0
        x = extract_v4_belief_features(obs, include_history=self.include_history)
        return _sigmoid(sum(w * xi for w, xi in zip(self.weights, x)))

    def fit(
        self,
        samples: Sequence[tuple[PlayerObservation, int]],
        lr: float = 0.08,
        epochs: int = 25,
        l2: float = 1e-3,
    ) -> None:
        feats = [(extract_v4_belief_features(obs, include_history=self.include_history), float(y)) for obs, y in samples]
        w = [0.0] * 18
        for ep in range(epochs):
            step_lr = lr / (1.0 + 0.08 * ep)
            for x, y in feats:
                p = _sigmoid(sum(wi * xi for wi, xi in zip(w, x)))
                err = p - y
                for j in range(len(w)):
                    w[j] -= step_lr * (err * x[j] + l2 * w[j])
        self.weights = w


class CalibratedPosteriorBelief:
    """Calibrates the exact Bayes' rule posterior + count-conditioned history features on training games."""

    def __init__(
        self,
        archetypes: Sequence[ArchetypeSpec] = V4_TRAIN_ARCHETYPES,
        recency_decay: float = 0.90,
    ) -> None:
        self.archetypes = tuple(archetypes)
        self.recency_decay = recency_decay
        self.weights: list[float] = [0.0] * 20

    def _features(self, obs: PlayerObservation) -> list[float]:
        base = extract_v4_belief_features(obs, include_history=True)
        p_bayes = posterior_bluff_probability(obs, archetypes=self.archetypes, recency_decay=self.recency_decay)
        if p_bayes is None:
            return base + [0.0, 0.0]
        # Posterior confidence (1 - normalized entropy) measures how sharp opponent identification is
        assert obs.last_claim is not None
        post = infer_opponent_posteriors(obs, archetypes=self.archetypes, recency_decay=self.recency_decay)[
            obs.last_claim.claimer_seat
        ]
        ent = -sum(v * math.log(max(1e-12, v)) for v in post.values())
        max_ent = math.log(max(2, len(post)))
        conf = max(0.0, min(1.0, 1.0 - ent / max_ent))
        return base + [_logit(p_bayes), conf]

    def predict(self, obs: PlayerObservation) -> float:
        if obs.last_claim is None:
            return 0.0
        if (obs.hand_capacity_prior or 0.0) <= 0.0:
            return 1.0
        x = self._features(obs)
        return _sigmoid(sum(w * xi for w, xi in zip(self.weights, x)))

    def fit(
        self,
        samples: Sequence[tuple[PlayerObservation, int]],
        lr: float = 0.04,
        epochs: int = 25,
        l2: float = 5e-3,
    ) -> None:
        """Fit residual calibration anchored around exact Bayes logit (w[18] regularized toward 1.0)."""
        feats = [(self._features(obs), float(y)) for obs, y in samples]
        w = [0.0] * 20
        w[18] = 1.0
        for ep in range(epochs):
            step_lr = lr / (1.0 + 0.10 * ep)
            for x, y in feats:
                p = _sigmoid(sum(wi * xi for wi, xi in zip(w, x)))
                err = p - y
                for j in range(len(w)):
                    target = 1.0 if j == 18 else 0.0
                    reg_scale = (l2 * 2.0) if j == 18 else l2
                    w[j] -= step_lr * (err * x[j] + reg_scale * (w[j] - target))
            w[18] = max(0.75, min(1.35, w[18]))
        self.weights = w

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "contract": current_contract("adaptive_v4"),
                    "recency_decay": self.recency_decay,
                    "archetype_ids": [spec.archetype_id for spec in self.archetypes],
                    "weights": [round(float(w), 6) for w in self.weights],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> CalibratedPosteriorBelief:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        require_contract(data, "v4 calibrated posterior", method_version="adaptive_v4")
        model = cls(recency_decay=float(data.get("recency_decay", 0.90)))
        weights = [float(v) for v in data["weights"]]
        if len(weights) != 20 or not all(math.isfinite(w) for w in weights):
            raise ValueError("CalibratedPosteriorBelief weights must contain 20 finite floats.")
        model.weights = weights
        return model


def _sample_from_posterior(
    dist: dict[str, float], archetypes: Sequence[ArchetypeSpec], rng: random.Random
) -> ArchetypeSpec:
    u = rng.random()
    acc = 0.0
    for spec in archetypes:
        acc += dist[spec.archetype_id]
        if u <= acc:
            return spec
    return archetypes[-1]


def sample_posterior_rollout_state(
    obs: PlayerObservation,
    seat_specs: dict[int, ArchetypeSpec],
    seed: int,
    target_bluff_prob: float | None = None,
) -> LiarsBarEngine:
    """Exact O(1) constructive hidden-world sampler matching target posterior bluff probability.

    Preserves exact 20-card deck conservation and observer's known cards, while ensuring
    that the rollout world's `pending_play.is_bluff` frequency matches `target_bluff_prob`
    (respecting exact combinatorial impossibility when remaining truth < claimed_count).
    """
    rng = random.Random(seed)
    statuses = sorted(obs.players_public_status, key=lambda p: p.seat)
    sim = LiarsBarEngine(game_id="posterior_rollout", game_seed=_derive_seed(seed, "future"), record_truth=False)
    sim.alive = [p.alive for p in statuses]
    sim.shots_taken = [p.shots_taken for p in statuses]
    sim.fatal_chamber_slot = [
        rng.randint(p.shots_taken + 1, p.max_chambers) if p.alive else p.shots_taken
        for p in statuses
    ]
    deck = sim._build_full_deck()
    for card in obs.own_hand + obs.own_played_cards_this_round:
        deck.remove(card)
    rng.shuffle(deck)

    sim.hands = [[] for _ in statuses]
    sim.played_cards_this_round = [[] for _ in statuses]
    sim.round_id, sim.step_id, sim.target_rank = obs.round_id, obs.step_id, obs.target_rank
    sim.current_turn_seat = obs.observer_seat
    sim.current_round_history = list(obs.current_round_history)
    sim.past_public_actions = list(obs.past_public_actions)
    sim.past_rounds_history = list(obs.past_rounds_history)
    sim.pending_play = None

    for p in statuses:
        if p.alive and p.seat == obs.observer_seat:
            sim.hands[p.seat] = list(obs.own_hand)
            sim.played_cards_this_round[p.seat] = list(obs.own_played_cards_this_round)

    claimer_seat = obs.last_claim.claimer_seat if obs.last_claim is not None else None
    if claimer_seat is not None:
        claim = obs.last_claim
        assert claim is not None
        truth_pool = [c for c in deck if is_card_truthful(c, obs.target_rank)]
        false_pool = [c for c in deck if not is_card_truthful(c, obs.target_rank)]
        can_truth = len(truth_pool) >= claim.claimed_count
        can_bluff = len(false_pool) >= 1 and len(deck) >= claim.claimed_count

        if target_bluff_prob is None:
            cap = obs.hand_capacity_prior if obs.hand_capacity_prior is not None else 0.5
            claimer_shots = next(p.shots_taken for p in statuses if p.seat == claimer_seat)
            p_bluff = _archetype_bayes_bluff_prob(
                seat_specs[claimer_seat], claim.claimed_count, cap, claimer_shots
            )
        else:
            p_bluff = target_bluff_prob

        if can_truth and can_bluff:
            want_bluff = rng.random() < _clamp(p_bluff, eps=0.01)
        elif can_bluff:
            want_bluff = True
        else:
            want_bluff = False

        if not want_bluff:
            rng.shuffle(truth_pool)
            chosen_play = truth_pool[: claim.claimed_count]
            for c in chosen_play:
                deck.remove(c)
        else:
            rng.shuffle(false_pool)
            first_false = false_pool[0]
            deck.remove(first_false)
            rng.shuffle(deck)
            rest = [deck.pop() for _ in range(claim.claimed_count - 1)]
            chosen_play = [first_false] + rest

        rng.shuffle(deck)
        claimer_rem = [deck.pop() for _ in range(claim.claimer_remaining_cards)]
        already_before = max(0, sim.hand_size - claim.claimer_remaining_cards - claim.claimed_count)
        earlier_cards = [deck.pop() for _ in range(already_before)]
        sim.hands[claimer_seat] = list(sort_cards(claimer_rem))
        sim.played_cards_this_round[claimer_seat] = earlier_cards + list(sort_cards(chosen_play))
        sim.pending_play = PendingPlayTruth(
            round_id=claim.round_id,
            step_id=claim.step_id,
            claimer_seat=claim.claimer_seat,
            claimed_rank=claim.claimed_rank,
            actual_cards=sort_cards(chosen_play),
        )

    for p in statuses:
        if not p.alive or p.seat == obs.observer_seat or p.seat == claimer_seat:
            continue
        sim.hands[p.seat] = list(sort_cards([deck.pop() for _ in range(p.hand_count)]))
        sim.played_cards_this_round[p.seat] = [deck.pop() for _ in range(sim.hand_size - p.hand_count)]

    return sim


def _clone_sim(state: LiarsBarEngine) -> LiarsBarEngine:
    sim = copy.copy(state)
    for name in (
        "alive",
        "shots_taken",
        "fatal_chamber_slot",
        "current_round_history",
        "past_rounds_history",
        "past_public_actions",
    ):
        setattr(sim, name, list(getattr(state, name)))
    sim.hands = [list(hand) for hand in state.hands]
    sim.played_cards_this_round = [list(cards) for cards in state.played_cards_this_round]
    sim.step_truth_logs = []
    return sim


def _next_active_seat(obs: PlayerObservation) -> int | None:
    by_seat = {p.seat: p for p in obs.players_public_status}
    for offset in (1, 2, 3):
        s = (obs.observer_seat + offset) % 4
        p = by_seat[s]
        if p.alive and p.hand_count > 0:
            return s
    return None


class PosteriorAdaptiveBot(BaseAgent):
    """Alive-count-coupled, opponent-adapting Bayesian bot using only public observation history."""

    def __init__(
        self,
        name: str = "v4_posterior_adaptive",
        archetypes: Sequence[ArchetypeSpec] = V4_TRAIN_ARCHETYPES,
        seed: int = 42,
        recency_decay: float = 0.90,
        belief_fn: Callable[[PlayerObservation], float] | None = None,
        mix_play_style: bool = True,
    ) -> None:
        super().__init__(name=name, style_tag="posterior_adaptive")
        self.archetypes = tuple(archetypes)
        self.seed = seed
        self.recency_decay = recency_decay
        self.belief_fn = belief_fn
        self.mix_play_style = mix_play_style
        self._spec_by_id = {s.archetype_id: s for s in self.archetypes}

    def predict_bluff_prob(self, obs: PlayerObservation) -> float | None:
        if obs.last_claim is None:
            return None
        if self.belief_fn is not None:
            return float(self.belief_fn(obs))
        return posterior_bluff_probability(obs, archetypes=self.archetypes, recency_decay=self.recency_decay)

    def select_action(self, obs: PlayerObservation) -> Tuple[CandidateAction, AgentDecisionRecord]:
        rng = random.Random(_derive_seed(self.seed, obs.round_id, obs.step_id, obs.observer_seat))
        bluff_prob = self.predict_bluff_prob(obs)
        challenge_actions = [a for a in obs.legal_actions if a.action_type == "CHALLENGE"]
        truthful, pure_bluff, mixed_bluff = split_play_actions(obs)
        all_bluffs = pure_bluff + mixed_bluff

        num_alive = sum(1 for p in obs.players_public_status if p.alive)
        other_active = sum(
            1 for p in obs.players_public_status if p.alive and p.hand_count > 0 and p.seat != obs.observer_seat
        )
        own_lethal = obs.own_next_shot_lethal_prob
        posteriors = infer_opponent_posteriors(obs, archetypes=self.archetypes, recency_decay=self.recency_decay)
        prof = extract_count_conditioned_profile(obs)

        # Estimate next active opponent's count-conditioned voluntary challenge tendencies
        # combining Bayesian archetype posterior + direct Beta-Binomial shrinkage on public events.
        next_seat = _next_active_seat(obs)
        next_single_ch = 0.22
        next_multi_ch = 0.30
        if next_seat is not None:
            npost = posteriors[next_seat]
            prior_s = sum(npost[aid] * self._spec_by_id[aid].single_challenge_base for aid in npost)
            prior_m = sum(npost[aid] * self._spec_by_id[aid].multi_challenge_base for aid in npost)
            nprof = prof[str(next_seat)]
            s_opp, s_ch = nprof["single_challenge_opps"], nprof["single_challenges"]
            m_opp, m_ch = nprof["multi_challenge_opps"], nprof["multi_challenges"]
            next_single_ch = (s_ch + 2.0 * prior_s) / (s_opp + 2.0)
            next_multi_ch = (m_ch + 2.0 * prior_m) / (m_opp + 2.0)

        # Playing a bluff that empties our hand while <=1 other player holds cards triggers
        # a 100% forced challenge (0% survival); exclude such bluffs whenever CHALLENGE is legal.
        non_suicidal_bluffs = [
            a for a in all_bluffs
            if not (len(a.cards) == len(obs.own_hand) and other_active <= 1)
        ]

        if challenge_actions:
            if len(obs.legal_actions) == 1 or (not truthful and not non_suicidal_bluffs):
                return challenge_actions[0], self._record(obs, bluff_prob, challenge_actions[0])

            # Alive-count-coupled challenge discipline:
            # - In 3-4p games, challenging while holding truth cards is a public good (bystanders free-ride);
            #   only challenge with truth cards when bluff certainty is overwhelming or claimer emptied hand.
            # - In 2p heads-up (zero-sum) or with 0 truth cards, exploit inferred tells sharply.
            claimer_empty = (obs.last_claim is not None and obs.last_claim.claimer_remaining_cards == 0)
            if truthful:
                if num_alive >= 4:
                    cutoff = 0.92 if not claimer_empty else 0.80
                elif num_alive == 3:
                    cutoff = 0.86 if not claimer_empty else 0.72
                else:
                    base_cut = 0.58 if len(truthful) == 1 else 0.68
                    cutoff = base_cut + 0.14 * max(0.0, own_lethal - 1.0 / 6.0)
                    if claimer_empty:
                        cutoff -= 0.08
            else:
                bluff_risk = min(next_single_ch, next_multi_ch)
                cutoff = 0.45 - 0.45 * (bluff_risk - 0.20)
                if num_alive == 2:
                    cutoff -= 0.08
            cutoff = min(0.97, max(0.18, cutoff))
            if bluff_prob is not None and bluff_prob >= cutoff:
                return challenge_actions[0], self._record(obs, bluff_prob, challenge_actions[0])

        # Opponent-aware play selection:
        # - When playing truthful: bait whichever count-bucket the next seat hunts MORE!
        # - When forced to bluff: hide in whichever count-bucket the next seat hunts LESS!
        if truthful:
            single_t = [a for a in truthful if len(a.cards) == 1]
            multi_t = [a for a in truthful if len(a.cards) >= 2]
            if single_t and multi_t:
                if next_multi_ch > next_single_ch + 0.08:
                    chosen = rng.choice(multi_t)
                elif next_single_ch > next_multi_ch + 0.08:
                    chosen = rng.choice(single_t)
                elif own_lethal >= 0.40:
                    chosen = rng.choice(multi_t) if rng.random() < 0.70 else rng.choice(single_t)
                elif self.mix_play_style:
                    p_multi = 0.50 + max(-0.35, min(0.35, 1.8 * (next_multi_ch - next_single_ch)))
                    chosen = rng.choice(multi_t) if rng.random() < p_multi else rng.choice(single_t)
                else:
                    chosen = truthful[0]
            else:
                chosen = rng.choice(truthful)
            return chosen, self._record(obs, bluff_prob, chosen)

        bluff_pool = non_suicidal_bluffs or all_bluffs
        single_b = [a for a in bluff_pool if len(a.cards) == 1]
        multi_b = [a for a in bluff_pool if len(a.cards) >= 2]
        if single_b and multi_b:
            chosen = rng.choice(multi_b) if next_multi_ch + 0.04 < next_single_ch else rng.choice(single_b)
        else:
            chosen = rng.choice(bluff_pool) if bluff_pool else obs.legal_actions[0]
        return chosen, self._record(obs, bluff_prob, chosen)

    def _record(
        self, obs: PlayerObservation, belief: float | None, action: CandidateAction
    ) -> AgentDecisionRecord:
        return AgentDecisionRecord(
            belief_bluff_prob=belief,
            belief_source="bayesian_posterior" if obs.last_claim else "none",
            estimated_opponent_style="unknown",
            rationale=describe_action(obs, action),
            rationale_source="action_description",
            opponent_profile=public_opponent_profile(obs),
        )


def evaluate_posterior_teacher_values(
    obs: PlayerObservation,
    archetypes: Sequence[ArchetypeSpec] = V4_TRAIN_ARCHETYPES,
    num_rollouts: int = 16,
    rollout_seed: int = 1000,
    recency_decay: float = 0.90,
    belief_fn: Callable[[PlayerObservation], float] | None = None,
    candidate_actions: Sequence[CandidateAction] | None = None,
) -> dict[str, float]:
    """v4 Calibrated History-Posterior Rollout Teacher (strictly observation-only)."""
    posteriors = infer_opponent_posteriors(obs, archetypes=archetypes, recency_decay=recency_decay)
    target_bluff_prob = None
    if obs.last_claim is not None:
        target_bluff_prob = (
            float(belief_fn(obs))
            if belief_fn is not None
            else posterior_bluff_probability(obs, archetypes=archetypes, recency_decay=recency_decay)
        )
    candidates = list(obs.legal_actions if candidate_actions is None else candidate_actions)
    legal = {a.action_id: a for a in obs.legal_actions}
    if not candidates or any(legal.get(a.action_id) != a for a in candidates) or len({a.action_id for a in candidates}) != len(candidates):
        raise ValueError("Candidates must be distinct legal actions from this observation.")
    totals: dict[str, float] = {a.action_id: 0.0 for a in candidates}

    for idx in range(num_rollouts):
        traj_seed = _derive_seed(rollout_seed, "v4_traj", idx)
        pick_rng = random.Random(_derive_seed(traj_seed, "pick_archetypes"))
        seat_specs: dict[int, ArchetypeSpec] = {}
        agents: list[BaseAgent] = []
        for seat in range(4):
            if seat == obs.observer_seat:
                agents.append(
                    PosteriorAdaptiveBot(
                        name="v4_teacher_actor",
                        archetypes=archetypes,
                        seed=_derive_seed(traj_seed, "actor", seat),
                        recency_decay=recency_decay,
                        belief_fn=belief_fn,
                    )
                )
            else:
                spec = _sample_from_posterior(posteriors[seat], archetypes, pick_rng)
                seat_specs[seat] = spec
                agents.append(
                    ParameterizedArchetypeBot(spec, seed=_derive_seed(traj_seed, "opp", seat))
                )

        base_state = sample_posterior_rollout_state(
            obs, seat_specs, _derive_seed(traj_seed, "world"), target_bluff_prob=target_bluff_prob
        )
        for action in candidates:
            sim = _clone_sim(base_state)
            sim_agents = copy.deepcopy(agents)
            res = sim.step(action)
            penalized = None if res is None else (res.penalized_seat == obs.observer_seat)
            while not sim.is_game_over():
                turn = sim.current_turn_seat
                turn_obs = sim.get_observation(turn)
                act, rec = sim_agents[turn].select_action(turn_obs)
                res = sim.step(act, rec)
                if penalized is None and res is not None:
                    penalized = (res.penalized_seat == obs.observer_seat)
            reward = 0.65 * (sim.winner_seat == obs.observer_seat) + 0.35 * (not penalized)
            totals[action.action_id] += reward / num_rollouts

    return totals


def generate_adaptive_v4_replays(
    num_games: int,
    base_seed: int,
    archetypes: Sequence[ArchetypeSpec] = V4_TRAIN_ARCHETYPES,
    prefix: str = "train_v4",
) -> list[GameReplayBundle]:
    """Generate deterministic full replays from the symmetric v4 archetype pool."""
    if num_games < 1:
        raise ValueError("num_games must be positive.")
    replays: list[GameReplayBundle] = []
    for index in range(num_games):
        game_id = f"{prefix}_{base_seed}_game_{index:05d}"
        env_seed = _derive_seed(base_seed, "v4_env", index)
        lineup_rng = random.Random(_derive_seed(base_seed, "v4_lineup", index))
        agents = [
            ParameterizedArchetypeBot(
                lineup_rng.choice(archetypes),
                seed=_derive_seed(base_seed, "v4_policy", index, seat),
            )
            for seat in range(4)
        ]
        engine = LiarsBarEngine(game_id=game_id, game_seed=env_seed)
        while not engine.is_game_over():
            obs = engine.get_observation()
            act, rec = agents[obs.observer_seat].select_action(obs)
            engine.step(act, rec)
        replay = GameReplayBundle.from_engine(engine, [a.name for a in agents])
        verify_replay_determinism(replay, raise_on_error=True)
        replays.append(replay)
    return replays


def make_v4_action_sample(
    obs: PlayerObservation,
    action: CandidateAction,
    game_id: str,
    *,
    rollout_candidate_values: dict[str, float] | None = None,
    action_supervision_policy: dict[str, Any] | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Build a contract-verified adaptive_v4 action training sample."""
    from liars_agent.training.action_supervision import make_action_supervision

    completion = json.dumps({"chosen_action_id": action.action_id}, ensure_ascii=False)
    validate_action_decision(obs, completion)
    sample: dict[str, Any] = {
        "task": "action",
        "sample_id": f"{game_id}_r{obs.round_id}_s{obs.step_id}_action",
        "game_id": game_id,
        "round_id": obs.round_id,
        "step_id": obs.step_id,
        "actor_seat": obs.observer_seat,
        "ablation_mode": obs.ablation_mode,
        "observation": obs.to_dict(),
        "system": SYSTEM_PROMPT_V4,
        "prompt": build_user_prompt(obs, include_count_profile=True),
        "completion": completion,
        "contract": current_contract("adaptive_v4"),
        **metadata,
    }
    if rollout_candidate_values is not None:
        sample["rollout_candidate_values"] = rollout_candidate_values
        sample["action_supervision"] = make_action_supervision(
            sample,
            rollout_candidate_values,
            action_supervision_policy,
            source=str(metadata.get("source", "calibrated_posterior_teacher_v4")),
        )
    return sample


def extract_v4_belief_samples(
    replays: Sequence[GameReplayBundle | dict[str, Any]],
    target_samples: int,
    seed: int = 42,
    actor_seat: int | None = None,
) -> list[dict[str, Any]]:
    """Extract true engine-labelled binary belief samples with v4 count-conditioned prompts."""
    bundles = [r.to_dict() if isinstance(r, GameReplayBundle) else r for r in replays]
    rows: list[dict[str, Any]] = []
    if not bundles:
        return rows
    queues = []
    for bundle in bundles:
        points = [
            p for p in bundle["steps"]
            if p["pending_claim_was_bluff"] is not None
            and (actor_seat is None or p["actor_seat"] == actor_seat)
        ]
        random.Random(_derive_seed(seed, "v4_belief_points", bundle["game_id"])).shuffle(points)
        queues.append((bundle["game_id"], points))
    while len(rows) < target_samples and any(points for _, points in queues):
        for game_id, points in queues:
            if not points or len(rows) == target_samples:
                continue
            point = points.pop()
            obs = PlayerObservation.from_dict(point["observation"])
            if obs.last_claim is None or obs.ablation_mode != "full":
                raise ValueError("Belief source must be a full observation with a pending claim.")
            label = point["pending_claim_was_bluff"]
            if not isinstance(label, bool):
                raise ValueError("Belief labels must be actual boolean engine truth.")
            rows.append(
                {
                    "task": "belief",
                    "sample_id": f"{game_id}_r{obs.round_id}_s{obs.step_id}_belief",
                    "game_id": game_id,
                    "round_id": obs.round_id,
                    "step_id": obs.step_id,
                    "actor_seat": obs.observer_seat,
                    "ablation_mode": "full",
                    "observation": obs.to_dict(),
                    "system": BELIEF_SYSTEM_PROMPT_V4,
                    "prompt": build_bluff_user_prompt(obs, include_count_profile=True),
                    "completion": str(int(label)),
                    "label": int(label),
                    "source": "engine_pending_claim_truth",
                    "contract": current_contract("adaptive_v4"),
                }
            )
    return rows


def _is_history_critical_point(obs: PlayerObservation) -> bool:
    """Mid/late-round decision where opponent reveal history directly impacts action value."""
    if obs.round_id < 3 or obs.last_claim is None:
        return False
    claimer = obs.last_claim.claimer_seat
    has_claimer_reveal = any(r.claimer_seat == claimer for r in obs.past_rounds_history)
    if not has_claimer_reveal:
        return False
    own_truth = sum(1 for c in obs.own_hand if is_card_truthful(c, obs.target_rank))
    num_alive = sum(1 for p in obs.players_public_status if p.alive)
    return own_truth <= 1 or num_alive <= 2


def extract_v4_action_samples(
    replays: Sequence[GameReplayBundle | dict[str, Any]],
    target_samples: int,
    *,
    belief_fn: Callable[[PlayerObservation], float] | None = None,
    num_rollouts: int = 16,
    margin: float = 0.05,
    seed: int = 42,
    history_critical_ratio: float = 0.5,
    action_supervision_policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract v4 action samples using the calibrated history-posterior rollout teacher."""
    bundles = [r.to_dict() if isinstance(r, GameReplayBundle) else r for r in replays]
    samples: list[dict[str, Any]] = []
    if not bundles or target_samples <= 0:
        return samples
    for index, bundle in enumerate(bundles):
        quota = target_samples // len(bundles) + (index < target_samples % len(bundles))
        if quota <= 0:
            continue
        raw_points = [p for p in bundle["steps"] if len(p["observation"]["legal_actions"]) > 1]
        rng = random.Random(_derive_seed(seed, "v4_action_points", bundle["game_id"]))
        critical_points = []
        regular_points = []
        for point in raw_points:
            obs = PlayerObservation.from_dict(point["observation"])
            if _is_history_critical_point(obs):
                critical_points.append((point, obs))
            else:
                regular_points.append((point, obs))
        rng.shuffle(critical_points)
        rng.shuffle(regular_points)

        crit_quota = min(len(critical_points), int(round(quota * max(0.0, min(1.0, history_critical_ratio)))))
        ordered = critical_points[:crit_quota] + regular_points + critical_points[crit_quota:]

        accepted = 0
        for _, obs in ordered:
            if accepted >= quota:
                break
            values = evaluate_posterior_teacher_values(
                obs,
                num_rollouts=num_rollouts,
                rollout_seed=_derive_seed(seed, "v4_teacher", bundle["game_id"], obs.round_id, obs.step_id),
                belief_fn=belief_fn,
            )
            best = max(obs.legal_actions, key=lambda a: (values[a.action_id], a.action_type, a.cards))
            advantage = values[best.action_id] - sum(values.values()) / len(values)
            if advantage < margin:
                continue
            samples.append(
                make_v4_action_sample(
                    obs,
                    best,
                    bundle["game_id"],
                    rollout_candidate_values=values,
                    action_supervision_policy=action_supervision_policy,
                    rollout_best_value=values[best.action_id],
                    rollout_advantage=advantage,
                    candidate_count=len(values),
                    teacher="calibrated_posterior_continuation_v4",
                    history_critical=_is_history_critical_point(obs),
                )
            )
            accepted += 1
    return samples


def _build_legal_actions_for_hand(own_hand: tuple[str, ...]) -> tuple[CandidateAction, ...]:
    actions: list[CandidateAction] = [CandidateAction(action_id="A0", action_type="CHALLENGE")]
    aid = 1
    seen: set[tuple[str, ...]] = set()
    hand_list = list(sort_cards(own_hand))
    n = len(hand_list)
    for k in range(1, min(3, n) + 1):
        def _dfs(start: int, cur: list[str]) -> None:
            nonlocal aid
            if len(cur) == k:
                tup = tuple(cur)
                if tup not in seen:
                    seen.add(tup)
                    actions.append(CandidateAction(action_id=f"A{aid}", action_type="PLAY", cards=tup))
                    aid += 1
                return
            for i in range(start, n):
                cur.append(hand_list[i])
                _dfs(i + 1, cur)
                cur.pop()
        _dfs(0, [])
    return tuple(actions)


def build_v4_counterfactual_train_samples(
    num_pairs: int,
    *,
    belief_fn: Callable[[PlayerObservation], float] | None = None,
    seed: int = 42,
    num_rollouts: int = 16,
    action_supervision_policy: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build deterministic counterfactual history-contrast pairs strictly for the train split.

    Each pair holds the exact same current hand, target rank, and pending claim while
    flipping Seat 1's prior count-conditioned reveals between single-bluff/multi-truth
    and multi-bluff/single-truth, encouraging attention to cross-round public history.
    Returns `(samples, synthetic_train_game_ids)`.
    """
    if num_pairs <= 0:
        return [], []
    rng = random.Random(_derive_seed(seed, "v4_counterfactual_pairs"))
    samples: list[dict[str, Any]] = []
    game_ids: list[str] = []
    ranks = ("Q", "K", "A")

    for pair_idx in range(num_pairs):
        trank = ranks[pair_idx % 3]
        non_t = [r for r in ranks if r != trank]
        # Focus on critical 0-1 truth-card hands and 2-player / 3-player endgame states
        own_truth_cnt = pair_idx % 2
        hand_cards = [trank] * own_truth_cnt + [rng.choice(non_t) for _ in range(5 - own_truth_cnt)]
        own_hand = sort_cards(hand_cards)
        claimed_count = rng.choice((1, 2, 2, 3))
        num_alive = rng.choice((2, 2, 3))
        own_shots = rng.randint(0, 3)

        statuses = (
            PlayerPublicInfo(seat=0, alive=True, hand_count=5, shots_taken=own_shots),
            PlayerPublicInfo(seat=1, alive=True, hand_count=5 - claimed_count, shots_taken=rng.randint(1, 3)),
            PlayerPublicInfo(seat=2, alive=(num_alive >= 3), hand_count=5 if num_alive >= 3 else 0, shots_taken=2 if num_alive >= 3 else 6),
            PlayerPublicInfo(seat=3, alive=False, hand_count=0, shots_taken=6),
        )
        claim = ClaimEvent(
            round_id=6,
            step_id=2,
            claimer_seat=1,
            claimed_rank=trank,
            claimed_count=claimed_count,
            claimer_remaining_cards=5 - claimed_count,
        )
        cur_hist = (
            PublicEvent(6, 1, 1, "PLAY", claimed_count, trank, 5 - claimed_count),
        )
        legal = _build_legal_actions_for_hand(own_hand)
        past_actions = (
            PublicEvent(1, 1, 1, "PLAY", 2, "Q", 3),
            PublicEvent(2, 1, 1, "PLAY", 2, "K", 3),
            PublicEvent(3, 1, 1, "PLAY", 3, "A", 2),
            PublicEvent(4, 1, 1, "PLAY", 1, "Q", 4),
            PublicEvent(5, 1, 1, "PLAY", 1, "A", 4),
        )
        hist_multi_truth = (
            RoundResolution(1, "Q", 1, 0, 2, ("Q", "Q"), False, False, 0, 0, 1, False),
            RoundResolution(2, "K", 1, 0, 2, ("K", "JOKER"), False, False, 0, 1, 2, False),
            RoundResolution(3, "A", 1, 0, 3, ("A", "A", "A"), False, False, 0, 2, 3, False),
            RoundResolution(4, "Q", 1, 0, 1, ("K",), True, True, 1, 0, 1, False),
            RoundResolution(5, "A", 1, 0, 1, ("Q",), True, True, 1, 1, 2, False),
        )
        hist_multi_bluff = (
            RoundResolution(1, "Q", 1, 0, 2, ("K", "A"), True, True, 1, 0, 1, False),
            RoundResolution(2, "K", 1, 0, 2, ("Q", "A"), True, True, 1, 1, 2, False),
            RoundResolution(3, "A", 1, 0, 3, ("Q", "K", "K"), True, True, 1, 2, 3, False),
            RoundResolution(4, "Q", 1, 0, 1, ("Q",), False, False, 0, 0, 1, False),
            RoundResolution(5, "A", 1, 0, 1, ("A",), False, False, 0, 1, 2, False),
        )

        for tag, past_res in (("mt", hist_multi_truth), ("mb", hist_multi_bluff)):
            gid = f"cf_v4_{seed}_{pair_idx:04d}_{tag}"
            game_ids.append(gid)
            obs = PlayerObservation(
                round_id=6,
                step_id=2,
                observer_seat=0,
                target_rank=trank,
                own_hand=own_hand,
                own_played_cards_this_round=(),
                players_public_status=statuses,
                last_claim=claim,
                current_round_history=cur_hist,
                past_rounds_history=past_res,
                legal_actions=legal,
                past_public_actions=past_actions,
                ablation_mode="full",
            )
            values = evaluate_posterior_teacher_values(
                obs,
                num_rollouts=num_rollouts,
                rollout_seed=_derive_seed(seed, "cf_teacher", pair_idx, tag),
                belief_fn=belief_fn,
            )
            best = max(obs.legal_actions, key=lambda a: (values[a.action_id], a.action_type, a.cards))
            advantage = values[best.action_id] - sum(values.values()) / len(values)
            samples.append(
                make_v4_action_sample(
                    obs,
                    best,
                    gid,
                    rollout_candidate_values=values,
                    action_supervision_policy=action_supervision_policy,
                    rollout_best_value=values[best.action_id],
                    rollout_advantage=advantage,
                    candidate_count=len(values),
                    teacher="calibrated_posterior_continuation_v4",
                    source="v4_counterfactual_history_pair",
                    history_critical=True,
                )
            )
    return samples, game_ids

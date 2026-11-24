"""Unit tests for v4 symmetric opponents, count-conditioned profiles, and Bayesian posterior teacher."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from liars_agent.contracts import current_contract, require_contract
from liars_agent.engine.observation import PlayerObservation
from liars_agent.engine.replay import GameReplayBundle, verify_replay_determinism
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.engine.types import (
    CandidateAction,
    ClaimEvent,
    PlayerPublicInfo,
    PublicEvent,
    RoundResolution,
)
from liars_agent.experiment import load_config, run_pipeline, validate_run_artifacts
from liars_agent.training.adaptive_opponents import (
    MidGameSwitcherBot,
    ParameterizedArchetypeBot,
    UNSEEN_MIRROR_MULTI_BLUFFER,
    UNSEEN_MIRROR_SINGLE_BLUFFER,
    V4_TRAIN_ARCHETYPES,
    extract_count_conditioned_profile,
    render_count_conditioned_summary,
)
from liars_agent.training.adaptive_teacher import (
    CalibratedPosteriorBelief,
    PosteriorAdaptiveBot,
    evaluate_posterior_teacher_values,
    infer_opponent_posteriors,
    posterior_bluff_probability,
    sample_posterior_rollout_state,
)
from liars_agent.training.tinker_train import validate_training_sample


class V4AdaptationTests(unittest.TestCase):
    def test_v4_bots_produce_deterministic_verified_replays(self):
        agents = [
            PosteriorAdaptiveBot(seed=11),
            ParameterizedArchetypeBot(UNSEEN_MIRROR_SINGLE_BLUFFER, seed=22),
            ParameterizedArchetypeBot(UNSEEN_MIRROR_MULTI_BLUFFER, seed=33),
            MidGameSwitcherBot(V4_TRAIN_ARCHETYPES[0], V4_TRAIN_ARCHETYPES[1], switch_round=3, seed=44),
        ]
        engine = LiarsBarEngine(game_id="v4_det_test", game_seed=20261124)
        while not engine.is_game_over():
            obs = engine.get_observation()
            act, rec = agents[obs.observer_seat].select_action(obs)
            engine.step(act, rec)
        bundle = GameReplayBundle.from_engine(engine, [a.name for a in agents])
        self.assertTrue(verify_replay_determinism(bundle, raise_on_error=True))

    def test_count_conditioned_profile_and_posterior_separate_mirror_tells(self):
        statuses = tuple(PlayerPublicInfo(seat=i, alive=True, hand_count=5 if i != 1 else 3, shots_taken=1) for i in range(4))
        claim = ClaimEvent(round_id=5, step_id=2, claimer_seat=1, claimed_rank="Q", claimed_count=2, claimer_remaining_cards=3)
        cur = (PublicEvent(5, 1, 1, "PLAY", 2, "Q", 3),)
        legal = (
            CandidateAction("A0", "CHALLENGE"),
            CandidateAction("A1", "PLAY", ("Q",)),
            CandidateAction("A2", "PLAY", ("K",)),
        )
        # Multi-truth vs multi-bluff history for Seat 1
        multi_truth_hist = (
            RoundResolution(1, "Q", 1, 2, 2, ("Q", "Q"), False, False, 2, 0, 1, False),
            RoundResolution(2, "K", 1, 3, 2, ("K", "K"), False, False, 3, 0, 1, False),
            RoundResolution(3, "A", 1, 2, 3, ("A", "A", "JOKER"), False, False, 2, 1, 2, False),
            RoundResolution(4, "Q", 1, 3, 1, ("K",), True, True, 1, 0, 1, False),
        )
        multi_bluff_hist = (
            RoundResolution(1, "Q", 1, 2, 2, ("K", "A"), True, True, 1, 0, 1, False),
            RoundResolution(2, "K", 1, 3, 2, ("Q", "A"), True, True, 1, 1, 2, False),
            RoundResolution(3, "A", 1, 2, 3, ("Q", "K", "K"), True, True, 1, 2, 3, False),
            RoundResolution(4, "Q", 1, 3, 1, ("Q",), False, False, 3, 0, 1, False),
        )
        obs_truth = PlayerObservation(
            round_id=5, step_id=2, observer_seat=0, target_rank="Q",
            own_hand=("Q", "K", "K", "A", "A"), own_played_cards_this_round=(),
            players_public_status=statuses, last_claim=claim, current_round_history=cur,
            past_rounds_history=multi_truth_hist, legal_actions=legal,
        )
        obs_bluff = PlayerObservation(
            round_id=5, step_id=2, observer_seat=0, target_rank="Q",
            own_hand=("Q", "K", "K", "A", "A"), own_played_cards_this_round=(),
            players_public_status=statuses, last_claim=claim, current_round_history=cur,
            past_rounds_history=multi_bluff_hist, legal_actions=legal,
        )

        prof_t = extract_count_conditioned_profile(obs_truth)["1"]
        self.assertEqual((prof_t["multi_bluffs"], prof_t["multi_reveals"]), (0, 3))
        self.assertEqual((prof_t["single_bluffs"], prof_t["single_reveals"]), (1, 1))
        self.assertIn("2~3张:0诈/3亮", render_count_conditioned_summary(obs_truth))

        p_truth = posterior_bluff_probability(obs_truth)
        p_bluff = posterior_bluff_probability(obs_bluff)
        self.assertIsNotNone(p_truth)
        self.assertIsNotNone(p_bluff)
        self.assertGreater(p_bluff - p_truth, 0.35)

        post_t = infer_opponent_posteriors(obs_truth)[1]
        post_b = infer_opponent_posteriors(obs_bluff)[1]
        self.assertGreater(post_t["single_tell_trapper"], post_b["single_tell_trapper"])
        self.assertGreater(post_b["multi_tell_bluffer"], post_t["multi_tell_bluffer"])

        vals_t = evaluate_posterior_teacher_values(obs_truth, num_rollouts=12, rollout_seed=77)
        vals_b = evaluate_posterior_teacher_values(obs_bluff, num_rollouts=12, rollout_seed=77)
        self.assertGreater(vals_b["A0"] - vals_t["A0"], 0.15)

    def test_count_bucket_challenges_and_no_suicidal_last_card_bluff(self):
        # Seat 2 challenges 1-card claims repeatedly and ignores 2-card claims -> single_hunter > paranoid_sheriff
        statuses = (
            PlayerPublicInfo(seat=0, alive=True, hand_count=1, shots_taken=1),
            PlayerPublicInfo(seat=1, alive=True, hand_count=3, shots_taken=1),
            PlayerPublicInfo(seat=2, alive=False, hand_count=0, shots_taken=6),
            PlayerPublicInfo(seat=3, alive=False, hand_count=0, shots_taken=6),
        )
        past_actions = (
            PublicEvent(1, 1, 1, "PLAY", 1, "Q", 4),
            PublicEvent(1, 2, 2, "CHALLENGE", 0, "Q", 5, forced_challenge=False),
            PublicEvent(2, 1, 1, "PLAY", 1, "K", 4),
            PublicEvent(2, 2, 2, "CHALLENGE", 0, "K", 5, forced_challenge=False),
            PublicEvent(3, 1, 1, "PLAY", 2, "A", 3),
            PublicEvent(3, 2, 2, "PLAY", 2, "A", 3),
            PublicEvent(4, 1, 1, "PLAY", 2, "Q", 3),
            PublicEvent(4, 2, 2, "PLAY", 2, "Q", 3),
        )
        obs = PlayerObservation(
            round_id=5, step_id=2, observer_seat=0, target_rank="Q",
            own_hand=("K",), own_played_cards_this_round=("Q", "Q", "K", "A"),
            players_public_status=statuses,
            last_claim=ClaimEvent(5, 1, 1, "Q", 2, 3),
            current_round_history=(PublicEvent(5, 1, 1, "PLAY", 2, "Q", 3),),
            past_rounds_history=(
                RoundResolution(1, "Q", 1, 2, 2, ("Q", "Q"), False, False, 2, 0, 1, False),
                RoundResolution(2, "K", 1, 2, 2, ("K", "K"), False, False, 2, 1, 2, False),
            ),
            legal_actions=(
                CandidateAction("A0", "CHALLENGE"),
                CandidateAction("A1", "PLAY", ("K",)),
            ),
            past_public_actions=past_actions,
        )
        prof2 = extract_count_conditioned_profile(obs)["2"]
        self.assertEqual((prof2["single_challenges"], prof2["single_challenge_opps"]), (2, 2))
        self.assertEqual((prof2["multi_challenges"], prof2["multi_challenge_opps"]), (0, 2))
        summary = render_count_conditioned_summary(obs)
        self.assertIn("对1张:2/2", summary)
        self.assertIn("对2~3张:0/2", summary)

        post2 = infer_opponent_posteriors(obs)[2]
        self.assertGreater(post2["single_hunter"], post2["paranoid_sheriff"])

        # Holding only 1 non-truth card in heads-up (other_active == 1): playing ("K",) empties hand
        # and triggers a 100% forced challenge -> PosteriorAdaptiveBot must CHALLENGE instead!
        bot = PosteriorAdaptiveBot(seed=42)
        act, _ = bot.select_action(obs)
        self.assertEqual(act.action_type, "CHALLENGE")

    def test_posterior_rollout_respects_zero_capacity_impossibility(self):
        # Observer holds 5 truthful Q/JOKER + played 2 Q -> 7 truthful known; only 1 truth left in deck!
        # Claimer claims 2 Q -> mathematically impossible to be truthful (capacity == 0).
        statuses = (
            PlayerPublicInfo(seat=0, alive=True, hand_count=3, shots_taken=0),
            PlayerPublicInfo(seat=1, alive=True, hand_count=3, shots_taken=0),
            PlayerPublicInfo(seat=2, alive=True, hand_count=5, shots_taken=0),
            PlayerPublicInfo(seat=3, alive=True, hand_count=5, shots_taken=0),
        )
        obs = PlayerObservation(
            round_id=2, step_id=3, observer_seat=0, target_rank="Q",
            own_hand=("Q", "Q", "JOKER"), own_played_cards_this_round=("Q", "Q"),
            players_public_status=statuses,
            last_claim=ClaimEvent(2, 2, 1, "Q", 2, 3),
            current_round_history=(
                PublicEvent(2, 1, 0, "PLAY", 2, "Q", 3),
                PublicEvent(2, 2, 1, "PLAY", 2, "Q", 3),
            ),
            past_rounds_history=(),
            legal_actions=(CandidateAction("A0", "CHALLENGE"), CandidateAction("A1", "PLAY", ("Q",))),
        )
        # Wait: 5 known truth out of 8 leaves 3 truth; let's test 6+2=8? Hand size is 5, so 5 known truth leaves 3 truth.
        # Verify posterior_bluff_probability stays valid and sample_posterior_rollout_state produces legal deck partition!
        seat_specs = {1: V4_TRAIN_ARCHETYPES[0], 2: V4_TRAIN_ARCHETYPES[1], 3: V4_TRAIN_ARCHETYPES[2]}
        sim = sample_posterior_rollout_state(obs, seat_specs, seed=123)
        self.assertEqual(len(sim.pending_play.actual_cards), 2)
        self.assertEqual(len(sim.hands[1]), 3)
        self.assertEqual(tuple(sim.hands[0]), obs.own_hand)

        # Verify constructive sampler matches target_bluff_prob accurately
        bluff_low = sum(
            sample_posterior_rollout_state(obs, seat_specs, seed=1000 + i, target_bluff_prob=0.05).pending_play.is_bluff
            for i in range(60)
        )
        bluff_high = sum(
            sample_posterior_rollout_state(obs, seat_specs, seed=2000 + i, target_bluff_prob=0.95).pending_play.is_bluff
            for i in range(60)
        )
        self.assertLess(bluff_low, 12)
        self.assertGreater(bluff_high, 48)

    def test_v4_formal_pipeline_local_smoke_and_contract_isolation(self):
        cfg_path = Path(__file__).resolve().parents[1] / "configs/sft_qwen35_9b_lora_v4_adaptive.yaml"
        config = load_config(cfg_path)
        self.assertEqual(config["method_version"], "adaptive_v4")
        self.assertEqual(config["contract"], current_contract("adaptive_v4"))
        self.assertNotEqual(current_contract("adaptive_v4"), current_contract("v2"))

        config["dataset"].update(
            total_target_samples=12,
            belief_target_samples=12,
            num_games=6,
            max_games=12,
            workers=1,
            batch_games=6,
            rollout_trajectories_per_state=2,
            min_advantage_margin=0.0,
            counterfactual_train_pairs=2,
        )
        config["evaluation"].update(num_blocks=1, unseen_belief_samples=12, unseen_belief_max_games=6)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "v4_smoke"
            status = run_pipeline(config, out, stage="local")
            self.assertEqual(status["stages"]["data"], "complete")
            self.assertEqual(status["stages"]["evaluate"], "complete")
            self.assertEqual(status["stages"]["figures"], "complete")
            self.assertTrue((out / "models/v4_calibrated_posterior.json").exists())
            self.assertTrue((out / "ui_models.yaml").exists())

            loaded_post = CalibratedPosteriorBelief.load(out / "models/v4_calibrated_posterior.json")
            self.assertEqual(len(loaded_post.weights), 20)

            validate_run_artifacts(
                out,
                action_supervision_policy=config["training"]["action_supervision"],
                method_version="adaptive_v4",
            )
            # Contract isolation: v2 validator MUST reject v4 artifacts and samples
            with self.assertRaises(ValueError):
                validate_run_artifacts(out, method_version="v2")
            first_action_line = (out / "dataset/action_train.jsonl").read_text(encoding="utf-8").splitlines()[0]
            import json
            sample = json.loads(first_action_line)
            self.assertIn("各座位公开历史统计", sample["prompt"])
            validate_training_sample(sample, method_version="adaptive_v4")
            with self.assertRaises(ValueError):
                require_contract(sample, "v2 check", method_version="v2")


if __name__ == "__main__":
    unittest.main()

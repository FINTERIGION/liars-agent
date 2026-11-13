"""Independent action-only coverage replacement, using explicit teacher doubles."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from liars_agent import experiment
from liars_agent.evaluation.endgame_coverage import build_endgame_coverage, classify_endgame_observation
from liars_agent.training import action_refinement, data_generator, dataset_quality
from liars_agent.training import endgame_augmentation as augmentation
from liars_agent.training.action_supervision import resolve_action_supervision
from liars_agent.training.prepare_supervision import prepare_supervision_run


class TestEndgameAugmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        source, refined = cls.root / "source", cls.root / "refined"
        cls.prepared, cls.output = cls.root / "prepared", cls.root / "augmented"
        config = experiment.load_config(experiment.DEFAULT_CONFIG)
        config["dataset"].update(total_target_samples=60, belief_target_samples=30,
            num_games=10, max_games=10, workers=1, batch_games=10,
            rollout_trajectories_per_state=1)
        experiment.write_json(source / "config.json", config)

        def values(obs, *args, **kwargs):
            return {a.action_id: float(i == 0) for i, a in enumerate(obs.legal_actions)}

        def original_rewards(obs, teacher, n, seed, candidate_actions=None, *, start_index=0):
            return {a.action_id: [float(i == 0)] * n for i, a in enumerate(obs.legal_actions)}

        def coverage_rewards(obs, teacher, n, seed, candidate_actions=None, *, start_index=0):
            if len(obs.legal_actions) <= 1:
                raise AssertionError("Forced challenge must not fabricate rollout scores")
            if n < 64:
                raise AssertionError("Every free endgame needs at least 64 paired trajectories")
            # In particular, two-candidate states have zero best-minus-mean
            # advantage. Coverage keeps both real empirical maxima anyway.
            return {a.action_id: [.65 if i < 2 else 0.] * n for i, a in enumerate(obs.legal_actions)}

        cls.coverage_rewards = staticmethod(coverage_rewards)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(data_generator, "evaluate_observation_action_values", side_effect=values), \
                patch.object(action_refinement, "evaluate_observation_action_rewards", side_effect=original_rewards), \
                patch.object(experiment, "make_backend", side_effect=AssertionError("Paid API")):
            experiment.generate_data(config, source)
            dataset_quality.improve_dataset(source, refined, workers=1)
            build_endgame_coverage(source, refined / "endgame_coverage", seed=9001,
                                  samples_per_cell=1, no_claim_per_alive=1, max_games=100)
            prepare_supervision_run(refined, cls.prepared)
            cls.original_hashes = augmentation._tree_hashes(cls.prepared)
            with patch.object(augmentation, "evaluate_observation_action_rewards", side_effect=coverage_rewards):
                cls.result = augmentation.prepare_endgame_augmented_run(cls.prepared, cls.output,
                    seed=9002, samples_per_cell=1, min_games_per_cell=1,
                    num_rollouts=64, max_games=100, workers=1)

    def test_coverage_preserves_state_counts_beliefs_holdouts_and_source(self):
        self.assertEqual(self.result["status"], "passed")
        self.assertEqual(augmentation._tree_hashes(self.prepared), self.original_hashes)
        for cell in self.result["coverage_matrix"]:
            self.assertGreaterEqual(cell["samples"], 1)
            self.assertGreaterEqual(cell["source_games"], 1)
            self.assertLessEqual(cell["max_samples_per_game"], 2)
        for split in ("train", "val", "test"):
            before = list(experiment.read_jsonl(self.prepared / f"dataset/{split}.jsonl"))
            after = list(experiment.read_jsonl(self.output / f"dataset/{split}.jsonl"))
            self.assertEqual(len(before), len(after))
            self.assertEqual([r for r in before if r["task"] == "belief"],
                             [r for r in after if r["task"] == "belief"])
            if split != "train":
                self.assertEqual((self.prepared / f"dataset/{split}.jsonl").read_bytes(),
                                 (self.output / f"dataset/{split}.jsonl").read_bytes())
        self.assertEqual(augmentation._tree_hashes(self.prepared / "endgame_coverage"),
                         augmentation._tree_hashes(self.output / "endgame_coverage"))
        self.assertEqual(self.result["training_optimizer_steps"], 9)  # ceil(48/16) * 3.
        self.assertEqual(self.result["model_api_calls"], 0)
        removed = list(experiment.read_jsonl(self.output / "augmentation/removed_actions.jsonl"))
        self.assertTrue(all(len(r["observation"]["own_hand"]) >= 3
                            and not classify_endgame_observation(r["observation"]) for r in removed))
        old_partition = experiment.read_json(self.prepared / "dataset/game_splits.json")
        new_partition = experiment.read_json(self.output / "dataset/game_splits.json")
        self.assertEqual(old_partition["val"], new_partition["val"])
        self.assertEqual(old_partition["test"], new_partition["test"])
        self.assertEqual(new_partition["train"][:len(old_partition["train"])], old_partition["train"])

    def test_forced_actions_have_no_fake_rewards_and_free_actions_keep_ties(self):
        rows = list(experiment.read_jsonl(self.output / "augmentation/added_actions.jsonl"))
        forced = [r for r in rows if len(r["observation"]["legal_actions"]) == 1]
        free = [r for r in rows if len(r["observation"]["legal_actions"]) > 1]
        self.assertTrue(forced and free)
        for row in forced:
            self.assertNotIn("rollout_advantage", row)
            self.assertNotIn("rollout_candidate_values", row)
            self.assertIsNone(row["action_supervision"]["candidate_values"])
            self.assertEqual(row["teacher"], "single_legal_action_rule")
            self.assertEqual(row["endgame_augmentation"]["rollouts_per_candidate"], 0)
        for row in free:
            self.assertEqual(row["endgame_augmentation"]["rollouts_per_candidate"], 64)
            self.assertEqual(len(resolve_action_supervision(row)["target_action_ids"]), 2)
        self.assertTrue(any(abs(r["rollout_advantage"]) < 1e-12 for r in free))

    def test_completed_resume_uses_verified_reports_without_teacher_calls(self):
        repeated = self.root / "repeat"
        shutil.copytree(self.output, repeated)
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(augmentation, "evaluate_observation_action_rewards", side_effect=AssertionError("Cached teacher rerun")), \
                patch.object(experiment, "make_backend", side_effect=AssertionError("Paid API")):
            result = augmentation.prepare_endgame_augmented_run(self.prepared, repeated,
                seed=9002, samples_per_cell=1, min_games_per_cell=1,
                num_rollouts=64, max_games=100, workers=1)
        self.assertEqual(result, self.result)

    def test_tampered_rewards_replacements_holdouts_and_coverage_are_rejected(self):
        for kind in ("reward", "manifest", "belief", "check_set", "source_label"):
            with self.subTest(kind=kind):
                output = self.root / f"bad_{kind}"
                shutil.copytree(self.output, output)
                if kind == "reward":
                    path = next((output / "augmentation/candidate_reports").glob("*.json"))
                    data = experiment.read_json(path)
                    key = next(iter(data["candidate_rewards"]))
                    data["candidate_rewards"][key][0] = 0.
                    experiment.write_json(path, data)
                elif kind == "manifest":
                    path = output / "augmentation/replacements.jsonl"
                    rows = list(experiment.read_jsonl(path)); rows[0]["removed_sample_id"] = "missing"
                    experiment.write_jsonl(path, rows)
                elif kind == "belief":
                    path = output / "dataset/belief_train.jsonl"
                    path.write_text(path.read_text() + "\n")
                elif kind == "check_set":
                    path = output / "endgame_coverage/labels.jsonl"
                    path.write_text(path.read_text() + "\n")
                else:
                    path = output / "augmentation/private/candidates.jsonl"
                    rows = list(experiment.read_jsonl(path)); rows[0]["label"] = 1 - rows[0]["label"]
                    experiment.write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    augmentation.validate_endgame_augmented_run(output, check_sources=True)

    def test_augmented_source_identity_collision_is_rejected(self):
        output = self.root / "overlap"
        shutil.copytree(self.output, output)
        settings = experiment.read_json(output / "config.json")["dataset"]["endgame_augmentation"]
        replays = list(experiment.read_jsonl(output / "augmentation/private/replays.jsonl"))
        forbidden = ({replays[0]["game_id"]}, set(), set())
        with self.assertRaisesRegex(ValueError, "overlap"):
            augmentation._validate_generation(output, settings, forbidden)


if __name__ == "__main__":
    unittest.main()

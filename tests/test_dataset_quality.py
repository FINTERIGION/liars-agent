"""Refinement preserves partitions/belief truth and rejects corrupted targets."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from liars_agent import experiment
from liars_agent.training import action_refinement, dataset_quality as quality


class TestDatasetQuality(TestCase):
    def source(self, root):
        source = root / "source"
        config = experiment.load_config(experiment.DEFAULT_CONFIG)
        config["dataset"].update(total_target_samples=3, belief_target_samples=3, num_games=3,
            max_games=9, workers=1, batch_games=3, rollout_trajectories_per_state=1,
            min_advantage_margin=.05)
        experiment.write_json(source / "config.json", config)
        experiment.generate_data(config, source)
        return source

    @staticmethod
    def rewards(obs, teacher, num_rollouts, seed, candidate_actions=None, *, start_index=0):
        return {a.action_id: [1.0 if i == 0 else 0.0] * num_rollouts
                for i, a in enumerate(obs.legal_actions)}

    def test_complete_local_refinement_resume_and_prepaid_validation(self):
        with TemporaryDirectory() as tmp:
            source = self.source(Path(tmp)); output = Path(tmp) / "improved"
            with patch.object(action_refinement, "evaluate_observation_action_rewards", side_effect=self.rewards), \
                    patch.object(experiment, "make_backend", side_effect=AssertionError("Paid API")):
                result = quality.improve_dataset(source, output, workers=1)
            self.assertEqual(result["original_action_rows"], 3)
            self.assertEqual(quality.validate_refined_dataset(output, check_sources=True)["action_rows_verified"], 3)
            for name in ("train", "val", "test"):
                self.assertEqual((source / f"dataset/belief_{name}.jsonl").read_bytes(),
                                 (output / f"dataset/belief_{name}.jsonl").read_bytes())
            self.assertEqual(experiment.load_config(output / "config.json"), experiment.read_json(output / "config.json"))
            with patch.object(quality, "refine_action_observation", side_effect=AssertionError("Recomputed cached rollout")):
                self.assertEqual(quality.improve_dataset(source, output, workers=1), result)
            rows = list(experiment.read_jsonl(output / "dataset/train.jsonl"))
            row = next(r for r in rows if r["task"] == "action")
            row["completion"] = json.dumps({"chosen_action_id": row["observation"]["legal_actions"][-1]["action_id"]})
            experiment.write_jsonl(output / "dataset/train.jsonl", rows)
            with patch.object(experiment, "make_backend", side_effect=AssertionError("API before corruption detection")):
                with self.assertRaises(ValueError):
                    experiment.run_pipeline(experiment.load_config(output / "config.json"), output, "sft")

    def test_low_advantage_is_replaced_only_from_same_partition(self):
        with TemporaryDirectory() as tmp:
            source = self.source(Path(tmp)); output = Path(tmp) / "improved"
            original = [r for sp in quality.SPLITS for r in experiment.read_jsonl(source / f"dataset/action_{sp}.jsonl")]
            rejected_obs = [r["observation"] for r in original]
            def scores(obs, teacher, n, seed, candidate_actions=None, *, start_index=0):
                if obs.to_dict() in rejected_obs:
                    return {a.action_id: [0.0] * n for a in obs.legal_actions}
                return self.rewards(obs, teacher, n, seed, candidate_actions, start_index=start_index)
            with patch.object(action_refinement, "evaluate_observation_action_rewards", side_effect=scores):
                result = quality.improve_dataset(source, output, workers=1)
            self.assertEqual(result["replacements"], 3)
            self.assertEqual(result["rejected_candidates"], 3)
            self.assertEqual(quality.validate_refined_dataset(output)["action_rows_verified"], 3)
            new_ids = {r["sample_id"] for sp in quality.SPLITS for r in experiment.read_jsonl(output / f"dataset/action_{sp}.jsonl")}
            self.assertFalse(new_ids & {r["sample_id"] for r in original})
            self.assertEqual((source / "dataset/game_splits.json").read_bytes(),
                             (output / "dataset/game_splits.json").read_bytes())

    def test_identical_input_reuses_one_report_independent_of_old_label(self):
        with TemporaryDirectory() as tmp:
            source = self.source(Path(tmp))
            row = next(experiment.read_jsonl(source / "dataset/action_train.jsonl"))
            config = experiment.read_json(source / "config.json")
            job = (row, experiment.teacher_config(config), 12, True, str(Path(tmp) / "reports"), None)
            with patch.object(action_refinement, "evaluate_observation_action_rewards", side_effect=self.rewards):
                first = quality._evaluate_job(job)
            row = dict(row, completion='{"chosen_action_id":"A999"}', game_id="other_private_game")
            with patch.object(quality, "refine_action_observation", side_effect=AssertionError("Duplicate input not reused")):
                self.assertEqual(quality._evaluate_job((row, *job[1:])), first)

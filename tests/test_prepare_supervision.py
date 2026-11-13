"""Small local preparation runs preserve source data and bind supervision scores."""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from liars_agent import experiment
from liars_agent.agents.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from liars_agent.training import action_refinement, dataset_quality
from liars_agent.training.action_supervision import resolve_action_supervision
from liars_agent.training.prepare_supervision import prepare_supervision_run, validate_prepared_supervision_run


def hashes(directory):
    return {str(path.relative_to(directory)): dataset_quality.file_hash(path)
            for path in directory.rglob("*") if path.is_file()}


class TestPrepareSupervision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "source"
        cls.refined = cls.root / "refined"
        cls.prepared = cls.root / "prepared"
        config = experiment.load_config(experiment.DEFAULT_CONFIG)
        config["dataset"].update(total_target_samples=9, belief_target_samples=9, num_games=3,
            max_games=3, workers=1, batch_games=3, rollout_trajectories_per_state=1,
            min_advantage_margin=0.0)
        experiment.write_json(cls.source / "config.json", config)
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.generate_data(config, cls.source)
        groups = {}
        for split in ("train", "val", "test"):
            rows = list(experiment.read_jsonl(cls.source / f"dataset/action_{split}.jsonl"))
            if len(rows) != 3:
                raise AssertionError("Fixture requires exactly three action states per game partition.")
            for row, group in zip(rows, ("clear", "tied", "small_gap")):
                groups[dataset_quality.input_key(row)] = group

        def rewards(obs, teacher, num_rollouts, seed, candidate_actions=None, *, start_index=0):
            key = dataset_quality.input_key({"system": SYSTEM_PROMPT, "prompt": build_user_prompt(obs)})
            group = groups[key]
            result = {}
            for index, action in enumerate(obs.legal_actions):
                if group == "small_gap":
                    # Difference (.35*3 - .65)/8 = .05 triggers 32 then 64.
                    pattern = ([.35] * 3 + [0.0] * 5) if index == 0 else (
                        [.65] + [0.0] * 7 if index == 1 else [0.0] * 8)
                else:
                    value = 1.0 if index == 0 else 0.0
                    if group == "tied" and index < 2:
                        value = .65
                    pattern = [value] * 8
                result[action.action_id] = [pattern[i % 8] for i in range(start_index, start_index + num_rollouts)]
            return result

        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(action_refinement, "evaluate_observation_action_rewards", side_effect=rewards), \
                patch.object(experiment, "make_backend", side_effect=AssertionError("Paid API")):
            dataset_quality.improve_dataset(cls.source, cls.refined, workers=1)
            cls.source_hashes = hashes(cls.refined)
            cls.summary = prepare_supervision_run(cls.refined, cls.prepared)

    def test_preparation_preserves_sources_belief_partitions_and_correct_weights(self):
        self.assertEqual(hashes(self.refined), self.source_hashes)
        self.assertEqual((self.refined / "dataset/game_splits.json").read_bytes(),
                         (self.prepared / "dataset/game_splits.json").read_bytes())
        for split in ("train", "val", "test"):
            self.assertEqual((self.refined / f"dataset/belief_{split}.jsonl").read_bytes(),
                             (self.prepared / f"dataset/belief_{split}.jsonl").read_bytes())
            old = list(experiment.read_jsonl(self.refined / f"dataset/{split}.jsonl"))
            new = list(experiment.read_jsonl(self.prepared / f"dataset/{split}.jsonl"))
            self.assertEqual(old, [{k: v for k, v in row.items() if k != "action_supervision"} for row in new])
            self.assertTrue(all("action_supervision" not in row for row in new if row["task"] == "belief"))
            self.assertEqual(self.summary["samples_per_split"][split]["groups"],
                             {"clear": 1, "tied": 1, "small_gap": 1})
            self.assertEqual(self.summary["samples_per_split"][split]["target_set_sizes"], {1: 2, 2: 1})
            weights = sorted(resolve_action_supervision(row)["loss_weight"] for row in new if row["task"] == "action")
            self.assertEqual(weights, [.5, 1.0, 1.0])
        self.assertAlmostEqual(self.summary["training_action_raw_weight_mean"], 5 / 6)
        self.assertAlmostEqual(self.summary["training_action_weight_scale"], 6 / 5)
        self.assertEqual(self.summary["training_action_effective_total_weight"], 3)
        self.assertEqual(self.summary["training_action_answer_sequences"], 4)
        self.assertEqual(self.summary["model_api_calls"], 0)
        self.assertEqual(validate_prepared_supervision_run(self.prepared, check_sources=True)["num_samples"], 18)

    def test_preparation_resumes_partial_output_without_recomputing_teacher(self):
        resumed = self.root / "resumed"
        shutil.copytree(self.prepared, resumed)
        (resumed / "dataset/sft_all_samples.jsonl").unlink()
        (resumed / "quality/action_supervision_summary.json").unlink()
        experiment.write_json(resumed / "run_status.json", {"stages": {"data": "preparing_supervision"}})
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(dataset_quality, "refine_action_observation", side_effect=AssertionError("Teacher rerun")), \
                patch.object(experiment, "make_backend", side_effect=AssertionError("Paid API")):
            resumed_summary = prepare_supervision_run(self.refined, resumed)
        self.assertEqual(resumed_summary, self.summary)
        self.assertEqual(validate_prepared_supervision_run(resumed, check_sources=True)["status"], "passed")
        self.assertEqual(experiment.read_json(resumed / "run_status.json")["stages"]["data"], "complete")
        self.assertEqual(hashes(self.refined), self.source_hashes)

    def test_descriptor_report_and_merged_file_tampering_are_detected(self):
        for kind in ("descriptor", "candidate_report", "merged"):
            with self.subTest(kind=kind):
                output = self.root / f"corrupted_{kind}"
                shutil.copytree(self.prepared, output)
                if kind == "descriptor":
                    rows = list(experiment.read_jsonl(output / "dataset/train.jsonl"))
                    row = next(row for row in rows if row["task"] == "action")
                    row["action_supervision"]["loss_weight"] = .123
                    experiment.write_jsonl(output / "dataset/train.jsonl", rows)
                    experiment.write_jsonl(output / "dataset/action_train.jsonl", (r for r in rows if r["task"] == "action"))
                    merged = list(experiment.read_jsonl(output / "dataset/sft_all_samples.jsonl"))
                    merged = [row if r["sample_id"] == row["sample_id"] else r for r in merged]
                    experiment.write_jsonl(output / "dataset/sft_all_samples.jsonl", merged)
                    message = "targets/weights"
                elif kind == "candidate_report":
                    path = next((output / "quality/candidate_reports").glob("*.json"))
                    report = experiment.read_json(path)
                    key = next(iter(report["report"]["final"]["candidate_means"]))
                    report["report"]["final"]["candidate_means"][key] = .123
                    experiment.write_json(path, report)
                    message = "candidate reports"
                else:
                    path = output / "dataset/sft_all_samples.jsonl"
                    rows = list(experiment.read_jsonl(path))
                    experiment.write_jsonl(path, rows[:-1])
                    message = "merged file"
                with self.assertRaisesRegex(ValueError, message):
                    validate_prepared_supervision_run(output, check_sources=True)


if __name__ == "__main__":
    unittest.main()

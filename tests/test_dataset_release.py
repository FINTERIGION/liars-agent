"""Portable release checks use tiny synthetic files and no model/experiment calls."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from liars_agent import experiment
from liars_agent.contracts import current_contract
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.training.data_generator import make_action_sample
from liars_agent.training import dataset_release as release


class TestDatasetRelease(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # The extracted path deliberately differs from manifest.run_directory.
        self.run = self.root / "downloaded-elsewhere"
        self.run.mkdir()
        self.manifests = self.root / "repository" / "datasets"
        self.addCleanup(patch.stopall)
        patch.object(release, "RELEASE_MANIFEST_ROOT", self.manifests).start()
        self.release_id = "selfplay-v3-20261114"
        self.manifest_path = self.manifests / self.release_id / "manifest.json"
        self.config = experiment.load_config(experiment.DEFAULT_CONFIG)
        self.config["dataset"]["endgame_augmentation"] = {"source_run": "/no/such/historical/source"}
        experiment.write_json(self.run / "config.json", self.config)
        experiment.write_json(self.run / "dataset/data_summary.json", {"contract": current_contract(), "shortfall": 0})
        engine = LiarsBarEngine(game_seed=29)
        engine.step(engine.get_legal_actions()[0])
        obs = engine.get_observation()
        for split in ("train", "val", "test"):
            row = make_action_sample(obs, obs.legal_actions[0], split)
            experiment.write_jsonl(self.run / f"dataset/{split}.jsonl", [row])
        experiment.write_json(self.run / "dataset/game_splits.json", {name: [name] for name in ("train", "val", "test")})
        experiment.write_json(self.run / "run_status.json", {"stages": {}})
        experiment.write_json(self.run / "endgame_coverage/coverage_report.json", {"contract": current_contract()})
        experiment.write_json(self.run / "endgame_coverage/private/generation_metadata.json",
                              {"source_run": "/no/such/historical/source"})
        self.pin()

    def pin(self):
        files = []
        for path in sorted(self.run.rglob("*")):
            if not path.is_file() or path.name == release.RELEASE_MARKER:
                continue
            files.append({"path": path.relative_to(self.run).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size, "immutable": path.name != "run_status.json"})
        self.manifest = {"schema_version": release.RELEASE_SCHEMA_VERSION, "release_id": self.release_id,
            "contract": current_contract(), "run_directory": "original-run-directory",
            "files": files, "source_identity": {"game_ids": ["train", "val", "test"], "game_seeds": [11, 12, 13]}}
        self.write_manifest()

    def write_manifest(self):
        experiment.write_json(self.manifest_path, self.manifest)
        experiment.write_json(self.run / release.RELEASE_MARKER, {"release_id": self.release_id,
            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()})

    def test_renamed_run_validates_and_only_initial_full_check_binds_mutable_outputs(self):
        self.assertEqual(release.validate_dataset_release(self.run, required=True, verify_all=True), self.manifest)
        experiment.write_json(self.run / "run_status.json", {"stages": {"later_stage": "running"}})
        self.assertEqual(release.validate_dataset_release(self.run), self.manifest)
        with self.assertRaisesRegex(ValueError, "run_status.json"):
            release.validate_dataset_release(self.run, verify_all=True)

    def test_changed_missing_or_same_size_corrupt_input_fails(self):
        path = self.run / "dataset/train.jsonl"
        original = path.read_bytes()
        for data in (None, original + b" ", original.replace(b'"train"', b'"other"')):
            with self.subTest(data="missing" if data is None else len(data)):
                if data is None:
                    path.unlink()
                else:
                    path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, "dataset/train.jsonl"):
                    release.validate_dataset_release(self.run)
                path.write_bytes(original)

    def test_absent_marker_preserves_unpublished_runs_and_cli_requires_marker(self):
        (self.run / release.RELEASE_MARKER).unlink()
        self.assertIsNone(release.validate_dataset_release(self.run))
        with self.assertRaisesRegex(ValueError, "marker is missing"):
            release.validate_dataset_release(self.run, required=True)
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(release.main(["--run-dir", str(self.run)]), 1)
        self.assertIn("marker is missing", stderr.getvalue())

    def test_bad_marker_or_repository_manifest_cannot_be_replaced_by_packaged_hashes(self):
        marker = self.run / release.RELEASE_MARKER
        for change in ({"release_id": "other-release"}, {"release_id": "../../outside"},
                       {"manifest_sha256": "0" * 64}):
            with self.subTest(change=change):
                self.write_manifest()
                experiment.write_json(marker, {**experiment.read_json(marker), **change})
                with self.assertRaises(ValueError):
                    release.validate_dataset_release(self.run)
        self.write_manifest()
        self.manifest_path.write_bytes(self.manifest_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "repository-pinned manifest"):
            release.validate_dataset_release(self.run)
        # Rehashing a replacement manifest inside the run cannot change trust.
        fake = {**self.manifest, "files": []}
        experiment.write_json(self.run / "manifest.json", fake)
        experiment.write_json(marker, {"release_id": self.release_id,
            "manifest_sha256": hashlib.sha256((self.run / "manifest.json").read_bytes()).hexdigest()})
        with self.assertRaisesRegex(ValueError, "repository-pinned manifest"):
            release.validate_dataset_release(self.run)

    def test_contract_path_escape_and_symlink_escape_fail(self):
        valid = copy.deepcopy(self.manifest)
        self.manifest["contract"] = {}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "contract"):
            release.validate_dataset_release(self.run)
        for bad_path in ("../outside", "/outside", "C:/outside", "dataset/../outside", "dataset\\outside"):
            self.manifest = copy.deepcopy(valid)
            self.manifest["files"][0]["path"] = bad_path
            self.write_manifest()
            with self.subTest(path=bad_path), self.assertRaises(ValueError):
                release.validate_dataset_release(self.run)
        self.manifest = valid
        self.write_manifest()
        target = self.run / self.manifest["files"][0]["path"]
        external = self.root / "outside"
        external.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "escapes"):
            release.validate_dataset_release(self.run)

    def test_cli_full_verification_reports_file_count_without_backend(self):
        with patch("liars_agent.training.tinker_backend.TinkerBackend.__init__", side_effect=AssertionError("No model API")), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(release.main(["--run-dir", str(self.run)]), 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"status": "passed", "release_id": self.release_id,
            "verified_files": len(self.manifest["files"]), "verify_all": True})

    def test_portable_pipeline_retains_row_checks_and_skips_only_historical_sources(self):
        with patch("liars_agent.training.endgame_augmentation.validate_endgame_augmented_run",
                   side_effect=AssertionError("Historical source should not be read")), \
                patch.object(experiment, "make_backend", side_effect=RuntimeError("mock paid-entry reached")) as paid:
            with self.assertRaisesRegex(RuntimeError, "mock paid-entry reached"):
                experiment.run_pipeline(self.config, self.run, "preflight")
            paid.assert_called_once()
        self.assertFalse(Path(self.config["dataset"]["endgame_augmentation"]["source_run"]).exists())
        # Even a pinned file must still pass the existing sample schema.
        rows = list(experiment.read_jsonl(self.run / "dataset/train.jsonl"))
        rows[0]["completion"] = '{"chosen_action_id":"NOT_LEGAL"}'
        experiment.write_jsonl(self.run / "dataset/train.jsonl", rows)
        self.pin()
        with self.assertRaisesRegex(ValueError, "unavailable"):
            experiment.validate_run_artifacts(self.run)

    def test_portable_pipeline_keeps_partition_checks_and_unreleased_provenance_path(self):
        partition = experiment.read_json(self.run / "dataset/game_splits.json")
        partition["val"] = ["train"]
        experiment.write_json(self.run / "dataset/game_splits.json", partition)
        self.pin()
        with self.assertRaisesRegex(ValueError, "multiple dataset partitions"):
            experiment.validate_run_artifacts(self.run)
        experiment.write_json(self.run / "dataset/game_splits.json", {name: [name] for name in ("train", "val", "test")})
        (self.run / release.RELEASE_MARKER).unlink()
        with patch("liars_agent.training.endgame_augmentation.validate_endgame_augmented_run") as legacy:
            experiment.validate_run_artifacts(self.run)
        legacy.assert_called_once_with(self.run)

    def test_endgames_use_pinned_source_identity_before_evaluator(self):
        from liars_agent.evaluation import endgame_coverage, endgame_model
        model_result = {"models": {"rule_counter": {"contract": current_contract()}}, "dataset": {"fixture": True}}
        with patch.object(endgame_coverage, "_validate_endgame_coverage", return_value={"status": "complete"}) as frozen, \
                patch.object(endgame_coverage, "validate_endgame_coverage", side_effect=AssertionError("Historical source")), \
                patch.object(endgame_model, "evaluate_endgame_models", return_value=model_result) as evaluate, \
                patch.object(experiment, "llm_factory", side_effect=AssertionError("No model calls")):
            result = experiment.evaluate_endgames(self.config, self.run, None, ["rule_counter"], required=True)
        self.assertEqual(result["status"], "complete")
        frozen.assert_called_once_with(self.run / "endgame_coverage", ({"train", "val", "test"}, {11, 12, 13}))
        evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()

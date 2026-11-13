import json
from pathlib import Path
import tempfile
import unittest

from liars_agent.contracts import current_contract
from liars_agent.engine.replay import GameReplayBundle, save_replays_jsonl
from liars_agent.engine.state import LiarsBarEngine
from liars_agent.evaluation.endgame_coverage import build_endgame_coverage, validate_endgame_coverage


class EndgameCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.source, cls.output = root / "source", root / "coverage"
        (cls.source / "dataset/batches").mkdir(parents=True)
        (cls.source / "config.json").write_text(json.dumps({"contract": current_contract()}))
        (cls.source / "dataset/game_splits.json").write_text(json.dumps({
            "train": ["main_source_game"], "val": [], "test": [],
        }))
        engine = LiarsBarEngine(game_id="main_source_game", game_seed=17)
        while not engine.is_game_over():
            engine.step(engine.get_legal_actions()[0])
        save_replays_jsonl([GameReplayBundle.from_engine(engine)],
                           cls.source / "dataset/batches/replays_0000.jsonl")
        cls.report = build_endgame_coverage(cls.source, cls.output, seed=2026111301,
            samples_per_cell=2, max_games=100, no_claim_per_alive=1)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_reachable_cells_complete_and_two_player_free_empty_is_impossible(self):
        self.assertEqual(self.report["status"], "complete")
        possible = [cell for cell in self.report["coverage_matrix"] if cell["status"] != "unreachable"]
        self.assertEqual(len(possible), 16)
        for cell in possible:
            self.assertGreaterEqual(cell["samples"], 2)
            self.assertGreaterEqual(cell["source_games"], (cell["samples"] + 1) // 2)
        impossible = [cell for cell in self.report["coverage_matrix"] if cell["status"] == "unreachable"]
        self.assertEqual(len(impossible), 2)
        self.assertTrue(all(cell["scenario"] == "empty_claimer_free" and cell["alive_players"] == 2 for cell in impossible))
        self.assertTrue(self.report["all_saved_replays_verified"])
        self.assertEqual(self.report["overlapping_environment_seeds"], 0)
        self.assertFalse(self.report["natural_distribution"])

    def test_inputs_have_no_labels_or_environment_identity_and_no_claim_is_separate(self):
        ids = set()
        for filename in ("observations.jsonl", "no_pending_claim.jsonl"):
            for line in (self.output / filename).read_text().splitlines():
                row = json.loads(line)
                self.assertNotIn(row["sample_id"], ids)
                ids.add(row["sample_id"])
                for key in ("game_id", "game_seed", "label", "pending_claim_actual_cards", "ground_truth_hands"):
                    self.assertNotIn(key, row)
                    self.assertNotIn(key, row["observation"])
                self.assertEqual(row["legal_actions"], row["observation"]["legal_actions"])
                if filename == "no_pending_claim.jsonl":
                    self.assertIsNone(row["observation"]["last_claim"])
                    self.assertIsNone(row["belief_prompt"])
        labels = [json.loads(line) for line in (self.output / "labels.jsonl").read_text().splitlines()]
        self.assertEqual(sum(row["label"] is None for row in labels), 3)

    def test_label_or_prompt_mutation_is_rejected(self):
        for filename, key in (("labels.jsonl", "label"), ("observations.jsonl", "action_prompt")):
            path = self.output / filename
            original = path.read_text()
            records = [json.loads(line) for line in original.splitlines()]
            if key == "label":
                index = next(i for i, record in enumerate(records) if record[key] is not None)
                records[index][key] = 1 - records[index][key]
            else:
                records[0][key] = "A prompt that does not match the legal observation"
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            try:
                with self.subTest(field=key), self.assertRaises(ValueError):
                    validate_endgame_coverage(self.output, self.source)
            finally:
                path.write_text(original)

    def test_environment_seed_overlap_is_rejected_before_replay_verification(self):
        path = self.output / "private/replays.jsonl"
        original = path.read_text()
        records = [json.loads(line) for line in original.splitlines()]
        records[0]["game_seed"] = 17
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        try:
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_endgame_coverage(self.output, self.source)
        finally:
            path.write_text(original)

    def test_same_seed_reproduces_the_same_frozen_examples(self):
        output = Path(self.temp.name) / "repeat"
        repeated = build_endgame_coverage(self.source, output, seed=2026111301,
            samples_per_cell=2, max_games=100, no_claim_per_alive=1)
        self.assertEqual(repeated, self.report)
        for filename in ("observations.jsonl", "no_pending_claim.jsonl", "labels.jsonl", "private/sample_sources.jsonl"):
            self.assertEqual((output / filename).read_bytes(), (self.output / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()

"""A tiny transport regression for changes below checkpoint-match tolerance."""
from __future__ import annotations

import unittest

from liars_agent.training.trial_verification import (
    TrialVerificationError, score_trial_probe, verify_trial_reload_and_optimizer,
)
import test_trial_verification as fixtures


class TestTrialSensitivity(unittest.TestCase):
    def test_initial_weights_cannot_pass_when_update_only_exceeds_tiny_floor(self):
        # Reuse the small, explicit local transport and real SDK data classes.
        # No actual model, service, source dataset, or saved run is involved.
        case = fixtures.TestTrialVerification()
        case.setUp()
        self.addCleanup(case.doCleanups)
        client = case.service.training
        initial_weight = client.state["weight"]
        before = score_trial_probe(client, case.backend, case.data)
        control = score_trial_probe(client, case.backend, case.data)
        for rate in (1e-6, 2e-6, 3e-6):
            case.backend.run_training_batch(client, case.data, learning_rate=rate)
        saved = score_trial_probe(client, case.backend, case.data)
        path = client.save_state("tiny-trial").path
        # Deliberately save the initial, untrained weight while keeping the
        # optimizer fields. A merely nonzero change must not certify this load.
        case.service.saved_states[path]["weight"] = initial_weight
        with self.assertRaises(TrialVerificationError) as raised:
            verify_trial_reload_and_optimizer(case.backend, client, path, case.data,
                before, control, saved, case.data, 4e-6)
        report = raised.exception.partial_report
        self.assertEqual(report["status"], "not_passed")
        update = report["checks"]["observable_update"]
        self.assertGreater(update["update_delta"]["max_abs_delta"], 1e-6)
        self.assertTrue(update["update_delta"]["matched"])
        self.assertEqual(client.optim_calls, 3)
        self.assertFalse(case.service.resumed)


if __name__ == "__main__":
    unittest.main()

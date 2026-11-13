"""Offline stage integration, frozen probes, and explicit model-double wiring."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from liars_agent import experiment
from liars_agent.agents.rule_bots import CardCounterBot
from liars_agent.agents.base import estimate_heuristic_bluff_belief
from liars_agent.contracts import current_contract
from liars_agent.engine.types import AgentDecisionRecord


def small_config():
    config = experiment.load_config(experiment.DEFAULT_CONFIG)
    config['dataset'].update(total_target_samples=3, belief_target_samples=3, num_games=3,
        max_games=6, workers=1, batch_games=3, rollout_trajectories_per_state=1, min_advantage_margin=0)
    config['evaluation'].update(num_blocks=1, bootstrap_samples=100, unseen_belief_samples=3, unseen_belief_max_games=3)
    config['rs'].update(num_games=2, target_samples=2, belief_target_samples=2, max_games=6,
        candidates_per_state=3, rollouts_per_candidate=1, min_advantage_margin=0)
    return config


class TestExperimentIntegration(unittest.TestCase):
    def test_complete_local_pipeline_and_cached_resume_never_construct_backend(self):
        config = small_config()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(experiment, 'make_backend', side_effect=AssertionError('Local stage called API')):
                status = experiment.run_pipeline(config, output, stage='local')
                self.assertEqual(status['real_experiments'], 'pending')
                summary = experiment.read_json(output / 'dataset/data_summary.json')
                self.assertEqual((summary['actual_samples'], summary['actual_belief_samples'], summary['shortfall']), (3, 3, 0))
                self.assertEqual(summary['samples_per_split'], {'train': 2, 'val': 2, 'test': 2})
                self.assertEqual(summary['classifier_fit']['num_samples'], 1)
                partition = experiment.read_json(output / 'dataset/game_splits.json')
                self.assertEqual(len({gid for ids in partition.values() for gid in ids}), 3)
                for name, ids in partition.items():
                    rows = list(experiment.read_jsonl(output / f'dataset/{name}.jsonl'))
                    self.assertEqual({r['game_id'] for r in rows}, set(ids))
                    self.assertEqual({r['task'] for r in rows}, {'action', 'belief'})
                result = experiment.read_json(output / 'experiment_summary.json')
                self.assertEqual(set(result['models_evaluated']), {'small_classifier', 'rule_counter'})
                self.assertEqual(len(result['generated_figures']), 1)
                self.assertTrue(all(Path(p).is_file() for p in result['generated_figures']))
                for pool in ('seen', 'unseen'):
                    a = experiment.read_json(output / f'evaluation/belief_{pool}_small_classifier.json')
                    b = experiment.read_json(output / f'evaluation/belief_{pool}_rule_counter.json')
                    self.assertEqual([r['sample_id'] for r in a['predictions']], [r['sample_id'] for r in b['predictions']])
                with patch.object(experiment, 'generate_data', side_effect=AssertionError('Regenerated cached data')), \
                     patch('liars_agent.evaluation.duplicate_runner.run_duplicate_evaluation', side_effect=AssertionError('Replayed cached games')):
                    self.assertEqual(experiment.run_pipeline(config, output, stage='local'), status)

    def test_local_stage_rejects_real_models(self):
        for key in ('zero_shot', 'lora_sft', 'lora_rs'):
            with self.subTest(key=key), TemporaryDirectory() as directory:
                with patch.object(experiment, 'make_backend', side_effect=AssertionError('API')):
                    with self.assertRaises(ValueError):
                        experiment.run_pipeline(small_config(), Path(directory), 'local', [key])

    def test_changed_configuration_and_stale_contract_fail_before_backend(self):
        for kind in ('config', 'artifact', 'evaluation'):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                config = small_config(); output = Path(directory)
                if kind == 'config':
                    previous = copy.deepcopy(config); previous['seed'] += 1
                    experiment.write_json(output / 'config.json', previous)
                elif kind == 'artifact':
                    experiment.write_json(output / 'dataset/data_summary.json', {'shortfall': 0})
                else:
                    experiment.write_json(output / 'evaluation/unseen_zero_shot_full.json', {'contract': {}})
                with patch.object(experiment, 'make_backend', side_effect=AssertionError('API before compatibility check')):
                    with self.assertRaises(ValueError):
                        experiment.run_pipeline(config, output, 'sft')

    def test_invalid_supervision_fails_before_backend(self):
        with TemporaryDirectory() as directory:
            output = Path(directory); config = small_config()
            experiment.generate_data(config, output)
            path = output / 'dataset/train.jsonl'; rows = list(experiment.read_jsonl(path))
            row = next(r for r in rows if r['task'] == 'action')
            row['completion'] = '{"chosen_action_id":"NOT_LEGAL"}'
            experiment.write_jsonl(path, rows)
            with patch.object(experiment, 'make_backend', side_effect=AssertionError('API')):
                with self.assertRaises(ValueError):
                    experiment.run_pipeline(config, output, 'sft')

    def test_old_training_objective_fails_before_backend(self):
        for artifact in ('adapter_metadata.json', 'training_progress.json'):
            with self.subTest(artifact=artifact), TemporaryDirectory() as directory:
                output = Path(directory)
                experiment.write_json(output / 'models/sft' / artifact, {'contract': current_contract()})
                with patch.object(experiment, 'make_backend', side_effect=AssertionError('API')):
                    with self.assertRaisesRegex(ValueError, 'supervision objective'):
                        experiment.run_pipeline(small_config(), output, 'sft')

    def test_pipeline_keeps_tokenizer_checks_before_preflight(self):
        from liars_agent.training.tinker_backend import TinkerBackend
        from test_tinker_backend import SDK, Service, Tokenizer
        with TemporaryDirectory() as directory:
            output = Path(directory); config = small_config()
            config['training']['max_seq_length'] = 128
            experiment.generate_data(config, output)
            backend = TinkerBackend(budget_path=output / 'budget.jsonl', sdk=SDK, service_client=Service())
            backend._tokenizer = Tokenizer()
            with patch.object(experiment, 'make_backend', return_value=backend), \
                    patch.object(backend, 'preflight', side_effect=AssertionError('Premature service session')):
                with self.assertRaisesRegex(ValueError, 'exceeding max_seq_length'):
                    experiment.run_pipeline(config, output, 'sft')

    def test_rs_filters_fallback_without_distilling_probability_or_rationale(self):
        config = small_config(); counts = {'select': 0, 'sample': 0}
        class TestAgent(CardCounterBot):
            def select_action(self, obs):
                counts['select'] += 1
                return super().select_action(obs)
            def sample_candidates(self, obs, num_samples, include_belief=True):
                self.assert_not_used = include_belief
                counts['sample'] += 1
                bad = AgentDecisionRecord(was_fallback=True, rationale='INVALID_FALLBACK')
                valid = AgentDecisionRecord(belief_bluff_prob=.314, rationale='UNVERIFIED_RATIONALE')
                return [(obs.legal_actions[-1], bad)] + [(action, valid) for action in obs.legal_actions[:2]]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(experiment, 'llm_factory', return_value=lambda seed: TestAgent(name='test_model', seed=seed)):
                summary = experiment.generate_rs_data(config, output, None)
                self.assertEqual((summary['actual_samples'], summary['actual_belief_samples']), (2, 2))
                rows = list(experiment.read_jsonl(output / 'dataset/rs_train.jsonl'))
                actions = [r for r in rows if r['task'] == 'action']
                self.assertTrue(all(set(json.loads(r['completion'])) == {'chosen_action_id'} for r in actions))
                self.assertTrue(all(r['completion'] in ('0', '1') for r in rows if r['task'] == 'belief'))
                for replay_path in (output / 'dataset/rs_games').glob('*.replay.jsonl'):
                    replay = next(experiment.read_jsonl(replay_path))
                    self.assertEqual(set(replay['seat_agent_names']), {'test_model','honest_conservative','aggressive_bluffer','card_counter'})
                previous = copy.copy(counts)
                self.assertEqual(experiment.generate_rs_data(config, output, None), summary)
                self.assertEqual(counts, previous)

    def test_model_training_wiring_keeps_shared_validation_and_checkpoint_contract(self):
        config = small_config()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            experiment.generate_data(config, output)
            base_train = list(experiment.read_jsonl(output / 'dataset/train.jsonl'))
            rs = [dict(row, game_id='fresh_rs') for row in base_train]
            experiment.write_jsonl(output / 'dataset/rs_train.jsonl', rs)
            config['rs'].update(target_samples=1, belief_target_samples=1)
            metadata = {'contract': current_contract(), 'base_model': config['base_model'], 'state_path':'tinker://test/state',
                        'action_supervision_policy': config['training']['action_supervision']}
            experiment.write_json(output / 'models/sft/adapter_metadata.json', metadata)
            with patch('liars_agent.training.tinker_train.run_tinker_lora_sft', return_value={}) as train:
                experiment.train_model(config, output, object(), 'rs')
                self.assertEqual(train.call_args.kwargs['resume_metadata'], metadata)
                self.assertEqual(train.call_args.kwargs['resume_from'], metadata['state_path'])
                self.assertEqual({r['game_id'] for r in train.call_args.kwargs['splits']['train']}, {'fresh_rs'})
                self.assertEqual(train.call_args.kwargs['splits']['val'], list(experiment.read_jsonl(output / 'dataset/val.jsonl')))

    def test_all_figures_and_ablations_with_explicit_local_model_doubles(self):
        config = small_config()
        class ModelDouble(CardCounterBot):
            def predict_bluff_prob(self, obs):
                return estimate_heuristic_bluff_belief(obs)[0] if obs.last_claim else None
        with TemporaryDirectory() as directory:
            output = Path(directory); experiment.generate_data(config, output)
            for stage in ('sft', 'rs'):
                experiment.write_json(output / f'models/{stage}/adapter_metadata.json',
                    {'contract':current_contract(), 'backend':'explicit_test_double',
                     'action_supervision_policy': config['training']['action_supervision']})
            with patch.object(experiment, 'llm_factory', side_effect=lambda cfg,out,backend,key: lambda seed: ModelDouble(name=key,seed=seed)):
                summary = experiment.evaluate(config, output, None, list(experiment.MODEL_LABELS))
            self.assertEqual(len(summary['history_ablation']), 3)
            self.assertEqual(len(experiment.generate_figures(output)), 4)
            main = summary['results']['seen']['lora_sft']
            self.assertEqual(summary['history_ablation'][0]['win_rate_ci'], main['win_rate_ci'])
            self.assertTrue(all(r['win_rate_ci']['num_resamples'] == 100 for r in summary['history_ablation']))


if __name__ == '__main__':
    unittest.main()

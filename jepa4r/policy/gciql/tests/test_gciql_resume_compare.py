# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_gciql_resume_compare.py (verbatim except: imports / paths adapted to the package)
"""Full GCIQL checkpoint sidecar comparisons; no JAX or GPU needed."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from jepa4r.policy.gciql import gate_checks as gate


class GciqlSidecarTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = dict(kind='gciql')
        self.state = dict(binding=dict(config=dict(agent_config={'lr': 3e-4}, total_updates=1000000),
                            source_bindings={'train': 'frozen-data'}, code={'source.py': 'a'*64}),
                          rng=dict(python=random.Random(13).getstate(),
                                   numpy=np.random.RandomState(17).get_state(),
                                   torch=np.arange(16, dtype=np.uint8),
                                   cuda=[np.arange(12, dtype=np.uint8)]),
                          budget_total=1000000, network_step=3, adam_count=2,
                          events=dict(committed_updates=2, validation_count=0,
                                      next_stage='sample_train', last_validation=0,
                                      checkpoint_event_complete=True, next_update=3))
        for role in ('actual', 'reference'):
            folder = self.root / role
            folder.mkdir()
            for field in ('state', 'next_batch', 'next_ids'):
                path = folder / (field + '.npz')
                np.savez(path, complete_array=np.arange(8, dtype=np.float32 if field != 'next_ids' else np.int64))
                self.cfg[role + '_' + field] = dict(path=str(path), sha256=gate.sha(path))
            self.write_sidecar(role, copy.deepcopy(self.state))

    def write_sidecar(self, role, value):
        folder = self.root / role
        path = folder / 'run_state.pkl'
        path.write_bytes(pickle.dumps(value, protocol=4))
        agent = folder / 'params_2.pkl'
        agent.write_bytes(b'fixture agent transaction bytes')
        manifest = folder / 'manifest.json'
        manifest.write_text(json.dumps(dict(counter=2, sidecar_sha256=gate.sha(path), agent_sha256=gate.sha(agent))))
        self.cfg[role + '_sidecar'] = dict(path=str(path), sha256=gate.sha(path))
        self.cfg[role + '_checkpoint_manifest'] = dict(path=str(manifest), sha256=gate.sha(manifest))

    def rejected(self):
        try:
            result = gate.compare_resume(self.cfg)
        except ValueError:
            return
        self.assertFalse(result['passed'])

    def test_equal_full_sidecars_pass_and_record_inapplicable_state(self):
        result = gate.compare_resume(self.cfg)
        self.assertTrue(result['passed'])
        self.assertTrue(result['sidecar']['passed'])
        self.assertEqual(result['state_applicability']['scheduler'], 'not_applicable_constant_optax_adam')
        self.assertEqual(result['state_applicability']['amp_scaler'], 'not_applicable_jax_fp32')

    def test_each_host_rng_field_is_compared_even_when_next_batch_matches(self):
        for field in ('python', 'numpy', 'torch', 'cuda'):
            with self.subTest(field=field):
                state = copy.deepcopy(self.state)
                if field == 'python':
                    state['rng'][field] = random.Random(99).getstate()
                elif field == 'numpy':
                    state['rng'][field] = np.random.RandomState(99).get_state()
                elif field == 'torch':
                    state['rng'][field][7] += 1
                else:
                    state['rng'][field][0][7] += 1
                self.write_sidecar('actual', state)
                self.rejected()

    def test_missing_rng_and_event_or_counter_mismatch_are_rejected(self):
        values = []
        state = copy.deepcopy(self.state)
        del state['rng']['cuda']
        values.append(state)
        for key in ('next_update', 'validation_count'):
            state = copy.deepcopy(self.state)
            state['events'][key] += 1
            values.append(state)
        state = copy.deepcopy(self.state)
        state['adam_count'] = 1
        values.append(state)
        for value in values:
            self.write_sidecar('actual', value)
            self.rejected()

    def test_source_binding_and_optional_scheduler_state_cannot_be_dropped(self):
        state = copy.deepcopy(self.state)
        state['binding']['code']['source.py'] = 'b'*64
        self.write_sidecar('actual', state)
        self.rejected()
        state = copy.deepcopy(self.state)
        state['scheduler'] = {'unexpected_step': 2}
        self.write_sidecar('actual', state)
        self.rejected()

    def test_missing_sidecar_or_broken_transaction_is_rejected(self):
        saved = self.cfg.pop('actual_sidecar')
        self.rejected()
        self.cfg['actual_sidecar'] = saved
        path = self.root / 'actual/params_2.pkl'
        path.write_bytes(b'modified agent bytes')
        self.rejected()

    def test_next_ids_are_still_compared(self):
        path = self.root / 'actual/next_ids.npz'
        np.savez(path, complete_array=np.arange(8, dtype=np.int64) + 1)
        self.cfg['actual_next_ids']['sha256'] = gate.sha(path)
        self.rejected()

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires existing Torch to unpickle real RNG tensors')
    def test_real_stored_torch_rng_tensors_are_compared_without_cuda(self):
        import torch
        state = copy.deepcopy(self.state)
        state['rng']['torch'] = torch.get_rng_state().clone()
        state['rng']['cuda'] = [torch.get_rng_state().clone()]  # Stored CUDA payloads are CPU byte tensors too.
        self.write_sidecar('actual', state)
        self.write_sidecar('reference', state)
        self.assertTrue(gate.compare_resume(self.cfg)['passed'])
        state['rng']['cuda'][0][17] += 1
        self.write_sidecar('actual', state)
        self.rejected()


if __name__ == '__main__':
    unittest.main()

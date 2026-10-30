"""Only the new metadata closure boundary and count-guarded bridge are tested."""
import copy
import unittest
from metadata_backup import verify_metadata, MODEL
from parameter_bridge import load, ASSET


def normal():
    manifest = dict(actual_exit_code=0, checkpoint_sha256=MODEL,
        splits={'train': [None]*5000, 'val': [None]*500})
    seal = dict(actual_exit_code=0, train_rows=5005000, val_rows=500500, full_raw_ID_coverage=True)
    audit = dict(actual_exit_code=0, checkpoint_sha256=MODEL, episodes_verified=5500, raw_rows_verified=5505500,
        latent_dim=192, dtype='float32', all_new_latent_SHA_shape_dtype_finite=True, all_canonical_ID_and_terminal_next_contracts=True)
    receipt = dict(actual_exit_code=0, cube_count=4, data_id='cube-quadruple-play-v0', PCD_episodes=5500, PCD_rows=5505500,
        chain=dict(original_runtime_v1_actual_exit_code=1, original_GPU_gate_accepted=False, logical_completed_phases=8))
    exits = [dict(actual_exit_code=0, residual_owned_members=[]) for _ in range(8)]
    return [manifest, seal, audit, receipt, exits, ['1', '1', '0', '0', '0']]


class MetadataTests(unittest.TestCase):
    def test_normal_metadata_and_exact_reversible_parameters(self):
        verify_metadata(*normal())
        for kind in ('pack', 'copy'):
            module = load(kind)
            self.assertTrue(module.parameter_bridge_proof['reverse_source_equals_original'])
        self.assertEqual(ASSET, 'PCD_cache_closed_metadata')

    def test_missing_new_closure_refused(self):
        fixture = copy.deepcopy(normal()); fixture[4][-1]['actual_exit_code'] = None
        with self.assertRaises(AssertionError): verify_metadata(*fixture)


if __name__ == '__main__': unittest.main()

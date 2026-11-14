"""M2 target rows retain their own real-source contract with a shared encoder."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from jepa4r.policy.gciql import gate_checks as gate


class MatrixTargetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runtime = dict(model='lewm_rgb', cube_count=3, camera={'actual': 'target-camera'})
        self.raw = dict(current_visual=np.zeros((224, 224, 3), np.uint8),
                        goal_visual=np.ones((224, 224, 3), np.uint8),
                        current_state=np.arange(46, dtype=np.float32),
                        goal_state=np.arange(46, dtype=np.float32) + 1)
        self.case = dict(inputs=self.arrays('input.npz', self.raw), source_encoder_config={'explicit': True})
        snapshot = dict(images=np.stack([self.raw['current_visual'], self.raw['goal_visual']]),
                        states=np.stack([self.raw['current_state'], self.raw['goal_state']]),
                        raw_row_ids=np.asarray([0,1000],np.int64))
        source = self.arrays('source.npz', snapshot)
        code = self.root / 'actual_producer.py'; code.write_text('# frozen producer\n')
        producer = dict(actual_exit_code=0, cube_count=3, camera=self.runtime['camera'],
                        raw_inputs=source, raw_rows=[0,1000],source_binding={str(code): gate.sha(code)})
        self.case['input_selectors'] = {'current':{'raw_row_id':0},'goal':{'raw_row_id':1000}}
        self.descriptor = dict(schema='cube_matrix_target_inputs_v1', cube_count=3,
                               camera=self.runtime['camera'], inputs=self.case['inputs'],
                               producer_receipt=self.record('producer.json', producer), source_arrays=source,
                               rows={name: dict(key='images' if 'visual' in name else 'states',
                                                row=0 if name.startswith('current') else 1) for name in self.raw})
        self.case['input_provenance'] = self.record('provenance.json', self.descriptor)

    def tearDown(self): self.temporary.cleanup()

    def arrays(self, name, value):
        path = self.root / name; np.savez(path, **value)
        return dict(path=str(path), sha256=gate.sha(path))

    def record(self, name, value):
        path = self.root / name; path.write_text(json.dumps(value), encoding='utf8')
        return dict(path=str(path), sha256=gate.sha(path))

    def test_real_target_rows_and_source_are_checked(self):
        raw, evidence = gate._matrix_target_inputs(self.case, self.runtime)
        self.assertEqual(evidence['cube_count'], 3)
        self.assertEqual(evidence['provenance'], self.case['input_provenance'])
        for key in self.raw: np.testing.assert_array_equal(raw[key], self.raw[key])

    def test_shared_source_needs_target_provenance(self):
        del self.case['input_provenance']
        with self.assertRaisesRegex(ValueError, 'provenance'):
            gate._matrix_target_inputs(self.case, self.runtime)

    def test_rebound_valid_arrays_cannot_mix_visual_and_state_rows(self):
        # Both fields really exist and all pins are correct. Their physical row
        # association must still agree within each declared phase.
        raw=dict(self.raw,current_state=self.raw['goal_state'])
        inputs=self.arrays('mixed.npz',raw)
        descriptor=copy.deepcopy(self.descriptor);descriptor['inputs']=inputs
        descriptor['rows']['current_state']['row']=1
        case=dict(self.case,inputs=inputs,input_provenance=self.record('mixed.json',descriptor))
        with self.assertRaisesRegex(ValueError,'row|selector'):gate._matrix_target_inputs(case,self.runtime)

    def test_all_fields_same_goal_row_cannot_override_current_selector(self):
        raw=dict(self.raw,current_visual=self.raw['goal_visual'],current_state=self.raw['goal_state'])
        inputs=self.arrays('same-row.npz',raw);descriptor=copy.deepcopy(self.descriptor);descriptor['inputs']=inputs
        descriptor['rows']['current_state']['row']=descriptor['rows']['current_visual']['row']=1
        case=dict(self.case,inputs=inputs,input_provenance=self.record('same-row.json',descriptor))
        with self.assertRaisesRegex(ValueError,'row|selector'):gate._matrix_target_inputs(case,self.runtime)

    def test_compact_snapshot_index_is_resolved_via_canonical_row_ids(self):
        raw,evidence=gate._matrix_target_inputs(self.case,self.runtime)
        self.assertEqual(evidence['resolved_physical_rows'],{'current':0,'goal':1000})
        wrong=dict(self.case,input_selectors={'current':{'raw_row_id':0},'goal':{'raw_row_id':1}})
        with self.assertRaisesRegex(ValueError,'row|selector'):gate._matrix_target_inputs(wrong,self.runtime)

    def test_camera_cube_input_binding_and_row_swap_reject(self):
        for field, value in [('camera', {'actual': 'source-camera'}), ('cube_count', 1),
                             ('inputs', {'path': 'wrong', 'sha256': '0'*64}),
                             ('rows', {k: dict(v, row=1-v['row']) for k, v in self.descriptor['rows'].items()})]:
            with self.subTest(field=field):
                descriptor = dict(self.descriptor, **{field: value})
                case = dict(self.case, input_provenance=self.record('bad.json', descriptor))
                with self.assertRaises(ValueError): gate._matrix_target_inputs(case, self.runtime)

    def test_wrong_N_state_or_nonfinite_rejects_before_forward(self):
        for name, value in [('current_state', np.zeros(28, np.float32)),
                            ('goal_state', np.full(46, np.nan, np.float32)),
                            ('current_visual', np.zeros((224,224,3), np.float32))]:
            with self.subTest(name=name):
                case = dict(self.case, inputs=self.arrays('wrong.npz', dict(self.raw, **{name:value})))
                with self.assertRaises(ValueError): gate._matrix_target_inputs(case, self.runtime)

    def test_legacy_no_source_pin_preserves_unwitnessed_input_path(self):
        case = {'inputs': self.case['inputs']}
        raw, evidence = gate._matrix_target_inputs(case, self.runtime)
        self.assertIsNone(evidence['provenance'])
        self.assertEqual(raw['goal_state'].shape, (46,))

    def test_failed_producer_or_changed_code_reject(self):
        producer = json.loads(Path(self.descriptor['producer_receipt']['path']).read_text())
        producer['actual_exit_code'] = 1
        descriptor = dict(self.descriptor, producer_receipt=self.record('failed.json', producer))
        case = dict(self.case, input_provenance=self.record('failed-provenance.json', descriptor))
        with self.assertRaises(ValueError): gate._matrix_target_inputs(case, self.runtime)
        (self.root / 'actual_producer.py').write_text('# changed\n')
        with self.assertRaises(ValueError): gate._matrix_target_inputs(self.case, self.runtime)


if __name__ == '__main__': unittest.main()

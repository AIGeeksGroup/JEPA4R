# Moved from OGBench-cube@4fbb0c6:tests/gates/test_records.py (verbatim except: module names / paths of the moved code adapted)
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
try:
    from jepa4r.policy.gciql.gates import records
except ImportError:
    records = None


def metadata():
    return {'gate': 'M1', 'case_id': 'cpu-forward', 'source_revision': 'baseline-commit',
            'producer_sha256': '1'*64, 'input_sha256': '2'*64, 'config_sha256': '3'*64,
            'weight_sha256': '4'*64, 'precision': 'fp32', 'seed': 0,
            'counts': {'actual': 2, 'expected': 2}, 'actual_exit_code': 0,
            'started_at': '2026-11-11T00:00:00+00:00', 'finished_at': '2026-11-11T00:00:01+00:00'}


class RecordsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(records, "Immutable gate record component is not implemented")
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_roundtrip_and_refuse_overwrite(self):
        path = self.root/'case'; arrays = {'encoding': np.arange(4, dtype=np.float32)}
        records.write_record(path, metadata(), arrays)
        record, loaded = records.load_record(path)
        self.assertTrue(records.validate_record(record, loaded)['passed'])
        np.testing.assert_array_equal(loaded['encoding'], arrays['encoding'])
        with self.assertRaises(FileExistsError): records.write_record(path, metadata(), arrays)

    def test_empty_record_or_malformed_counts_not_accepted(self):
        path = self.root/'empty'; records.write_record(path, metadata(), {})
        record, loaded = records.load_record(path)
        self.assertFalse(records.validate_record(record, loaded)['passed'])
        record['counts'] = None
        self.assertFalse(records.validate_record(record, loaded)['passed'])

    def test_tamper_array_file_or_array_metadata_rejected(self):
        path = self.root/'case'; records.write_record(path, metadata(), {'x': np.ones(2)})
        with (path/'arrays.npz').open('ab') as stream: stream.write(b'tampered')
        with self.assertRaises(ValueError): records.load_record(path)
        path = self.root/'case2'; records.write_record(path, metadata(), {'x': np.ones(2)})
        obj = json.loads((path/'record.json').read_text()); obj['arrays']['x']['sha256'] = '0'*64
        (path/'record.json').write_text(json.dumps(obj))
        with self.assertRaises(ValueError): records.load_record(path)

    def test_nonfinite_arrays_retained_but_not_accepted(self):
        path = self.root/'nan'; records.write_record(path, metadata(), {'x': np.array([np.nan])})
        record, loaded = records.load_record(path)
        self.assertFalse(records.validate_record(record, loaded)['passed'])

    def test_nonzero_exit_and_count_mismatch_preserved_but_not_accepted(self):
        for index, change in enumerate([{'actual_exit_code': 1}, {'counts': {'actual': 1, 'expected': 2}}]):
            m = metadata(); m.update(change); path = self.root/str(index)
            records.write_record(path, m, {'x': np.ones(2)})
            record, loaded = records.load_record(path)
            self.assertFalse(records.validate_record(record, loaded)['passed'])

    def test_missing_binding_invalid_digest_or_wrong_expected_revision(self):
        m = metadata(); del m['input_sha256']
        with self.assertRaises(ValueError): records.write_record(self.root/'missing', m, {'x': np.ones(2)})
        m = metadata(); m['producer_sha256'] = 'abcd'
        with self.assertRaises(ValueError): records.write_record(self.root/'badsha', m, {'x': np.ones(2)})
        for role in ['actual', 'reference']:records.write_record(self.root/role, metadata(), {'x': np.ones(2)})
        result = records.compare_records(self.root/'actual', self.root/'reference', expected_actual={'source_revision': 'wrong', 'producer_sha256': '1'*64}, expected_reference={'source_revision': 'baseline-commit', 'producer_sha256': '1'*64})
        self.assertFalse(result['passed'])

    def test_trusted_source_and_producer_expectations_are_required(self):
        for role in ['actual', 'reference']: records.write_record(self.root/role, metadata(), {'x': np.ones(2)})
        trusted = {'source_revision': 'baseline-commit', 'producer_sha256': '1'*64}
        for incomplete in [{}, {'source_revision': 'baseline-commit'}, {'producer_sha256': '1'*64}]:
            for role in ['actual', 'reference']:
                expected = {'expected_actual': dict(trusted), 'expected_reference': dict(trusted)}
                expected['expected_' + role] = incomplete
                self.assertFalse(records.compare_records(self.root/'actual', self.root/'reference', **expected)['passed'])
        bad = dict(trusted, producer_sha256='9'*64)
        self.assertFalse(records.compare_records(self.root/'actual', self.root/'reference', expected_actual=bad, expected_reference=trusted)['passed'])

    def test_different_producers_expected_but_semantic_bindings_must_match(self):
        a = metadata(); a['source_revision'] = 'main-commit'; a['producer_sha256'] = '5'*64
        records.write_record(self.root/'actual', a, {'x': np.ones(2)})
        records.write_record(self.root/'reference', metadata(), {'x': np.ones(2)})
        options = {'expected_actual': {'source_revision': 'main-commit', 'producer_sha256': '5'*64}, 'expected_reference': {'source_revision': 'baseline-commit', 'producer_sha256': '1'*64}}
        self.assertTrue(records.compare_records(self.root/'actual', self.root/'reference', **options)['passed'])
        a['input_sha256'] = '6'*64
        records.write_record(self.root/'wrong-input', a, {'x': np.ones(2)})
        self.assertFalse(records.compare_records(self.root/'wrong-input', self.root/'reference', **options)['passed'])

if __name__ == '__main__': unittest.main()

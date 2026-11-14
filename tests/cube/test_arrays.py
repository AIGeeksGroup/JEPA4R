import hashlib
import unittest
import numpy as np
try:
    from jepa4r.policy.gciql.gates import arrays as gates
except ImportError:
    gates = None


class ArraysTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gates, "Shared array comparator is not implemented")

    def test_identity_uses_exact_contract_and_c_order(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int16)
        raw = b"int16(2, 2)" + b"\x01\x00\x02\x00\x03\x00\x04\x00"
        self.assertEqual(gates.array_identity(a), hashlib.sha256(raw).hexdigest())
        self.assertEqual(gates.array_identity(a), gates.array_identity(np.asfortranarray(a)))
        self.assertNotEqual(gates.array_identity(a), gates.array_identity(a.reshape(4)))
        self.assertNotEqual(gates.array_identity(a), gates.array_identity(a.astype(np.int32)))
        with self.assertRaises(ValueError):
            gates.array_identity(np.array([object()]))

    def test_all_elements_and_inclusive_tolerance(self):
        reference = np.array([100., 0., 1.], dtype=np.float64)
        actual = np.array([100., 1e-6, 1.], dtype=np.float64)
        self.assertTrue(gates.compare_array(actual, reference)['passed'])
        actual[1] = np.nextafter(1e-6, np.inf)
        result = gates.compare_array(actual, reference)
        self.assertFalse(result['passed'])
        self.assertEqual(result['out_of_tolerance'], 1)
        self.assertEqual(result['max_abs_error'], actual[1])
        self.assertFalse(gates.compare_array(np.array([1., 2.]), np.array([2., 1.]))['passed'])

    def test_exact_reload_zero_and_large_integer_comparison(self):
        a = np.array([1., -2.], dtype=np.float32)
        self.assertTrue(gates.compare_array(a, a.copy(), exact=True)['passed'])
        b = a.copy(); b[0] = np.nextafter(b[0], np.float32(np.inf))
        self.assertFalse(gates.compare_array(b, a, exact=True)['passed'])
        a = np.array([2**63 + 1], dtype=np.uint64)
        b = np.array([2**63 + 2], dtype=np.uint64)
        self.assertFalse(gates.compare_array(a, b, exact=True)['passed'])

    def test_reject_shape_dtype_nonfinite_and_bad_tolerance(self):
        a = np.ones(2, dtype=np.float32)
        for bad in [
            a.reshape(1, 2),
            a.astype(np.float64),
            np.array([np.nan, 1.0], dtype=np.float32),
            np.array([np.inf, 1.0], dtype=np.float32),
        ]:
            self.assertFalse(gates.compare_array(bad, a)['passed'])
        with self.assertRaises(ValueError): gates.compare_array(a, a, atol=-1)
        with self.assertRaises(ValueError): gates.compare_array(a, a, rtol=float('nan'))
        with self.assertRaises(ValueError): gates.compare_array(a, a, rtol=float('inf'))

    def test_full_gradient_maps_require_all_names(self):
        ref = {'layer1': np.array([1., 2.]), 'layer2': np.array([-1.])}
        self.assertTrue(gates.compare_array_maps(ref, ref)['passed'])
        self.assertFalse(gates.compare_array_maps({'layer1': ref['layer1']}, ref)['passed'])
        wrong = dict(ref, layer1=np.array([2., 1.]))
        self.assertFalse(gates.compare_array_maps(wrong, ref)['passed'])

    def test_delta_direction_magnitude_and_initial_state(self):
        start = {'a': np.zeros(2), 'b': np.zeros(1)}
        end = {'a': np.array([1., 0.]), 'b': np.array([1.])}
        good = gates.compare_parameter_updates(start, end, start, end)
        self.assertTrue(good['passed']); self.assertEqual(good['relative_l2'], 0.)
        swapped = {'a': np.array([0., 1.]), 'b': np.array([1.])}
        self.assertFalse(gates.compare_parameter_updates(start, swapped, start, end)['passed'])
        twice = {k: v * 2 for k, v in end.items()}
        self.assertFalse(gates.compare_parameter_updates(start, twice, start, end)['passed'])
        self.assertFalse(gates.compare_parameter_updates(start, start, start, start)['passed'])
        different_start = {k: v + 3 for k, v in start.items()}
        different_end = {k: v + 3 for k, v in end.items()}
        self.assertFalse(gates.compare_parameter_updates(different_start, different_end, start, end)['passed'])

    def test_empty_gradient_pack_does_not_pass(self):
        self.assertFalse(gates.compare_array_maps({}, {})['passed'])

    def test_nested_optimizer_rng_and_batch_identity(self):
        ref = {
            'optimizer': {'step': 2, 'momentum': np.array([1.0, 2.0])},
            'rng': (3, np.array([1, 9], dtype=np.uint32)),
            'next_batch': [4, 7],
        }
        self.assertTrue(gates.compare_nested(ref, ref)['passed'])
        for key, value in [
            ('optimizer', {'step': 3, 'momentum': np.array([1.0, 2.0])}),
            ('rng', (3, np.array([1, 8], dtype=np.uint32))),
            ('next_batch', [7, 4]),
        ]:
            bad = dict(ref); bad[key] = value
            self.assertFalse(gates.compare_nested(bad, ref)['passed'])
        self.assertFalse(gates.compare_nested({1: 'state'}, {'1': 'state'})['passed'])
        self.assertFalse(gates.compare_nested((1, 2), [1, 2])['passed'])
        self.assertFalse(gates.compare_nested({'lr': float('nan')}, {'lr': float('nan')})['passed'])

if __name__ == '__main__': unittest.main()

"""CPU contract checks; --source-only substitutes only JAX tree/FrozenDict plumbing."""
import argparse
import ast
import dataclasses
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest import mock

import numpy as np


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--impl', type=Path, default=Path(__file__).parent / 'source_reference/datasets.py')
PARSER.add_argument('--source-only', action='store_true')
ARGS, TEST_ARGS = PARSER.parse_known_args()
SOURCE_SHA = '52c373922a87e1ba8ad8b3b96c3474d5e6385e2ea2c7206930f7a8a315816609'


def load_official():
    source = ARGS.impl.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    if not ARGS.source_only:
        spec = importlib.util.spec_from_file_location('official_datasets_for_test', ARGS.impl)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    # Execute the exact source AST for these classes, without importing absent
    # local JAX/Flax. This does NOT establish real JAX/FrozenDict integration.
    def tree_map(fn, tree):
        if isinstance(tree, dict):
            return {key: tree_map(fn, tree[key]) for key in sorted(tree)}
        return fn(tree)

    def tree_leaves(tree):
        if isinstance(tree, dict):
            return [leaf for key in sorted(tree) for leaf in tree_leaves(tree[key])]
        return [tree]

    class FlatFrozenDict:
        def __init__(self, data):
            self._dict = dict(data)

        def __getitem__(self, key):
            return self._dict[key]

        def __contains__(self, key):
            return key in self._dict

    namespace = dict(np=np, dataclasses=dataclasses, Any=object,
                     FrozenDict=FlatFrozenDict,
                     jax=types.SimpleNamespace(tree_util=types.SimpleNamespace(
                         tree_map=tree_map, tree_leaves=tree_leaves)))
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in ('get_size', 'Dataset', 'GCDataset')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ARGS.impl), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


OFFICIAL = load_official()


def fixture():
    count = 521
    observations = np.arange(count * 46, dtype=np.float32).reshape(count, 46) / np.float32(31)
    actions = np.arange(count * 5, dtype=np.float32).reshape(count, 5) / np.float32(71)
    terminals = np.zeros(count, np.float32)
    terminals[[4, 101, 207, 520]] = 1
    valids = 1 - terminals
    cfg = dict(discount=.99, value_p_curgoal=.2, value_p_trajgoal=.5,
               value_p_randomgoal=.3, value_geom_sample=True,
               actor_p_curgoal=0., actor_p_trajgoal=1., actor_p_randomgoal=0.,
               actor_geom_sample=False, gc_negative=True, p_aug=0., frame_stack=None)
    return OFFICIAL.GCDataset(OFFICIAL.Dataset.create(observations=observations,
        actions=actions, terminals=terminals, valids=valids), cfg)


def same_rng(test, left, right):
    test.assertEqual(left[0], right[0])
    test.assertEqual(left[2:], right[2:])
    np.testing.assert_array_equal(left[1], right[1])


def same_batch(test, expected, actual):
    test.assertEqual(list(expected), list(actual))
    for key in expected:
        test.assertEqual(expected[key].shape, actual[key].shape, key)
        test.assertEqual(expected[key].dtype, actual[key].dtype, key)
        test.assertEqual(expected[key].tobytes(), actual[key].tobytes(), key)


class OrderedInputContract(unittest.TestCase):
    def make_stream(self, data, mode='baseline'):
        from ordered_input import OrderedInput
        return OrderedInput(data, mode=mode, batch_size=1024)

    def test_default_is_original_sampler(self):
        data = fixture()
        np.random.seed(17)
        expected = data.sample(1024)
        expected_rng = np.random.get_state()
        np.random.seed(17)
        with self.make_stream(data) as stream:
            actual = stream.sample(1024)
            self.assertFalse(stream.pending)
            same_batch(self, expected, actual)
            same_rng(self, expected_rng, np.random.get_state())

    def test_all_modes_sequence_validation_save_resume_and_end(self):
        data = fixture()
        np.random.seed(887)
        expected = []
        for block in (2, 3, 1):
            train = []
            for _ in range(block):
                train.append((data.sample(1024), np.random.get_state()))
            validation = data.sample(1024)  # Official main leaves evaluation=False.
            after_validation = np.random.get_state()
            external_seed = np.random.randint(2**31)
            expected.append((train, validation, after_validation, external_seed, np.random.get_state()))
        final_rng = np.random.get_state()
        next_batch = data.sample(1024)
        for mode in ('baseline', 'prefetch1', 'workers2', 'workers4'):
            with self.subTest(mode=mode):
                np.random.seed(887)
                with self.make_stream(data, mode) as stream:
                    for train, validation, after_validation, seed, after_external in expected:
                        for batch, rng in train:
                            same_batch(self, batch, stream.sample())
                            same_rng(self, rng, stream.committed_rng_state)
                        stream.barrier()  # save/validation/evaluation boundary
                        self.assertFalse(stream.pending)
                        same_rng(self, train[-1][1], np.random.get_state())
                        same_batch(self, validation, data.sample(1024))
                        same_rng(self, after_validation, np.random.get_state())
                        self.assertEqual(seed, np.random.randint(2**31))
                        same_rng(self, after_external, np.random.get_state())
                    stream.barrier()
                    same_rng(self, final_rng, np.random.get_state())
                same_batch(self, next_batch, data.sample(1024))

    def test_known_final_batch_does_not_leave_lookahead(self):
        data = fixture()
        np.random.seed(73)
        expected = data.sample(1024)
        expected_rng = np.random.get_state()
        np.random.seed(73)
        with self.make_stream(data, 'workers4') as stream:
            same_batch(self, expected, stream.sample(prefetch_next=False))
            self.assertFalse(stream.pending)
            same_rng(self, expected_rng, np.random.get_state())

    def test_close_discards_unconsumed_future_and_restores_rng(self):
        data = fixture()
        np.random.seed(19)
        data.sample(1024)
        expected_rng = np.random.get_state()
        np.random.seed(19)
        stream = self.make_stream(data, 'workers2')
        stream.sample()
        self.assertTrue(stream.pending)
        stream.close()
        same_rng(self, expected_rng, np.random.get_state())
        self.assertFalse(stream.pending)
        with self.assertRaises(RuntimeError):
            stream.sample()

    def test_reject_unsupported_protocol_and_unbarriered_external_rng(self):
        data = fixture()
        for key, value in (('p_aug', .1), ('frame_stack', 2)):
            old = data.config[key]
            data.config[key] = value
            with self.assertRaises(ValueError):
                self.make_stream(data, 'workers2')
            data.config[key] = old
        stream = self.make_stream(data, 'prefetch1')
        stream.sample()
        np.random.rand()
        with self.assertRaises(RuntimeError):
            stream.barrier()
        with self.assertRaises(RuntimeError):
            stream.close()

    def test_read_workers_are_real_and_sampling_stays_on_caller(self):
        import ordered_input
        data = fixture()
        for count in (2, 4):
            np.random.seed(51)
            expected = data.sample(1024)
            np.random.seed(51)
            threads, rng_threads = [], []
            gate = threading.Barrier(count)
            original_read = ordered_input._Gather.read
            original_randint = np.random.randint

            def tracked_read(request):
                if threading.current_thread().name.startswith('ordered-rows'):
                    threads.append(threading.get_ident())
                    gate.wait(timeout=5)
                return original_read(request)

            def tracked_randint(*args, **kwargs):
                rng_threads.append(threading.get_ident())
                return original_randint(*args, **kwargs)

            with mock.patch.object(ordered_input._Gather, 'read', tracked_read), \
                    mock.patch.object(np.random, 'randint', tracked_randint):
                with self.make_stream(data, f'workers{count}') as stream:
                    same_batch(self, expected, stream.sample(prefetch_next=False))
            self.assertEqual(len(threads), 4)
            self.assertEqual(len(set(threads)), count)
            self.assertEqual(set(rng_threads), {threading.get_ident()})

    def test_read_failure_propagates_and_restores_rng(self):
        import ordered_input
        data = fixture()
        np.random.seed(197)
        before = np.random.get_state()
        with self.make_stream(data, 'workers2') as stream:
            with mock.patch.object(ordered_input._Gather, 'read', side_effect=OSError('read failed')):
                with self.assertRaisesRegex(OSError, 'read failed'):
                    stream.sample()
            same_rng(self, before, np.random.get_state())
            self.assertFalse(stream.pending)

    def test_failed_lookahead_does_not_skip_unreturned_current_batch(self):
        data = fixture()
        np.random.seed(373)
        expected = data.sample(1024)
        np.random.seed(373)
        before = np.random.get_state()
        with self.make_stream(data, 'workers2') as stream:
            original_sample = stream._deferred.sample
            calls = 0

            def fail_second_plan(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    np.random.rand()
                    raise OSError('planning failed')
                return original_sample(*args, **kwargs)

            with mock.patch.object(stream._deferred, 'sample', fail_second_plan):
                with self.assertRaisesRegex(OSError, 'planning failed'):
                    stream.sample()
            same_rng(self, before, np.random.get_state())
            same_rng(self, before, stream.committed_rng_state)
            self.assertFalse(stream.pending)
            same_batch(self, expected, stream.sample(prefetch_next=False))


if __name__ == '__main__':
    print('TEST_BACKEND=' + ('source_only_no_real_jax_flax' if ARGS.source_only else 'real_official_import'))
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])

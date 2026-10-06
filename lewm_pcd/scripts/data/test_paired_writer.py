"""Test the existing paired writer without loading optional simulator packages."""

import ast
from contextlib import ExitStack
from pathlib import Path
import unittest

import numpy as np

from collection_protocol import episode_record, validate_fixed_paired_schema


def paired_writer_class():
    source = Path(__file__).with_name('collect_cube_pcd.py').read_text(encoding='utf-8')
    node = next(
        item for item in ast.parse(source).body
        if isinstance(item, ast.ClassDef) and item.name == 'PairedWriter'
    )
    namespace = {
        'ExitStack': ExitStack, 'episode_record': episode_record,
        'PIX_COLS': ('pixels', 'render_time'),
        'PCD_COLS': ('pcd_', 'pixels/', 'cam/'),
        'validate_fixed_paired_schema': validate_fixed_paired_schema,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<PairedWriter>', 'exec'), namespace)
    return namespace['PairedWriter']


class RecordingWriter:
    def __init__(self, *, open_error=False, write_error=False):
        self.open_error, self.write_error = open_error, write_error
        self.closed = False
        self.episodes = []

    def __enter__(self):
        if self.open_error:
            raise OSError('open failed')
        return self

    def __exit__(self, *exc):
        self.closed = True

    def write_episode(self, episode):
        if self.write_error:
            raise OSError('write failed')
        self.episodes.append(episode)


class PairedWriterTests(unittest.TestCase):
    def episode(self):
        return {
            'step_idx': np.arange(3),
            'action': np.r_[np.zeros((2, 5)), np.full((1, 5), np.nan)],
            'qpos': np.zeros((3, 13)), 'qvel': np.zeros((3, 12)),
            'collection_reset_seed': np.full(3, 23),
            'privileged/target_block_pos': np.tile([0.5, 0.2, 0.02], (3, 1)),
            'privileged/target_block_yaw': np.zeros((3, 1)),
            'terminated': np.zeros(3, dtype=bool),
            'truncated': np.array([False, False, True]),
            'success': np.zeros(3, dtype=bool),
            'pixels': np.zeros((3, 4, 4, 3), dtype=np.uint8),
            'pcd_xyz': np.zeros((3, 10, 3), dtype=np.float32),
            'pcd_rgb': np.zeros((3, 10, 3), dtype=np.uint8),
        }

    @staticmethod
    def fixed_kwargs():
        return {
            'fixed_schema': True,
            'expected_action_dim': 5,
            'expected_rgb_shape': (4, 4, 3),
            'expected_pcd_shape': (10, 3),
        }

    def test_second_open_failure_closes_first_file(self):
        pix, pcd = RecordingWriter(), RecordingWriter(open_error=True)
        with self.assertRaisesRegex(OSError, 'open failed'):
            with paired_writer_class()(pix, pcd):
                pass
        self.assertTrue(pix.closed)

    def test_validation_happens_before_either_modality_write(self):
        writer_type = paired_writer_class()
        self.assertIn('expected_actions', writer_type.__init__.__code__.co_varnames)
        pix, pcd = RecordingWriter(), RecordingWriter()
        episode = {**self.episode(), 'pcd_rgb': np.zeros((2, 10, 3))}
        with writer_type(pix, pcd, expected_actions=2, **self.fixed_kwargs()) as writer:
            with self.assertRaisesRegex(ValueError, 'shape|length'):
                writer.write_episodes([episode])
        self.assertEqual(pix.episodes, [])
        self.assertEqual(pcd.episodes, [])

    def test_pcd_write_failure_is_not_recorded_as_paired_success(self):
        writer_type = paired_writer_class()
        self.assertIn('expected_actions', writer_type.__init__.__code__.co_varnames)
        pix, pcd = RecordingWriter(), RecordingWriter(write_error=True)
        accepted = []
        with self.assertRaisesRegex(OSError, 'write failed'):
            with writer_type(pix, pcd, expected_actions=2, on_episode=accepted.append, **self.fixed_kwargs()) as writer:
                writer.write_episodes([self.episode()])
        self.assertEqual(writer.pix_episodes_written, 1)
        self.assertEqual(writer.pcd_episodes_written, 0)
        self.assertEqual(accepted, [])
        self.assertTrue(pix.closed and pcd.closed)

    def test_both_writes_preserve_shared_columns_and_lengths(self):
        writer_type = paired_writer_class()
        self.assertIn('expected_actions', writer_type.__init__.__code__.co_varnames)
        pix, pcd = RecordingWriter(), RecordingWriter()
        accepted = []
        with writer_type(pix, pcd, expected_actions=2, on_episode=accepted.append, **self.fixed_kwargs()) as writer:
            writer.write_episodes([self.episode()])
        self.assertNotIn('pcd_xyz', pix.episodes[0])
        self.assertNotIn('pixels', pcd.episodes[0])
        for key in ('qpos', 'qvel', 'action', 'step_idx', 'collection_reset_seed'):
            np.testing.assert_array_equal(pix.episodes[0][key], pcd.episodes[0][key])
        self.assertEqual(accepted[0]['frames'], 3)
        self.assertEqual(writer.pix_episodes_written, writer.pcd_episodes_written)

    def test_fixed_validation_rejects_missing_modality_before_either_write(self):
        writer_type = paired_writer_class()
        pix, pcd = RecordingWriter(), RecordingWriter()
        episode = self.episode()
        del episode['pcd_rgb']
        with writer_type(pix, pcd, expected_actions=2, **self.fixed_kwargs()) as writer:
            with self.assertRaisesRegex(ValueError, 'pcd_rgb'):
                writer.write_episodes([episode])
        self.assertEqual(pix.episodes, [])
        self.assertEqual(pcd.episodes, [])

    def test_default_nonfixed_writer_preserves_permissive_behavior(self):
        writer_type = paired_writer_class()
        pix, pcd = RecordingWriter(), RecordingWriter()
        episode = self.episode()
        del episode['pixels'], episode['pcd_xyz'], episode['pcd_rgb']
        with writer_type(pix, pcd) as writer:
            writer.write_episodes([episode])
        self.assertEqual(writer.pix_episodes_written, 1)
        self.assertEqual(writer.pcd_episodes_written, 1)


if __name__ == '__main__':
    unittest.main()

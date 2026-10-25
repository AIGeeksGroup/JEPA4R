"""Production profile and rejection checks; no fixture can enter the reader."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class _Column:
    def __init__(self, shape, dtype, values=None):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.is_virtual = True
        self.values = values

    def __getitem__(self, index):
        return self.values[index]


class _View(dict):
    """Metadata-only HDF5 interface for original producer-validator tests."""
    pass


class ContractTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).parents[1] / 'contracts.py'
        self.assertTrue(path.is_file(), 'shared data-contract implementation is missing')
        return importlib.import_module('bench.ogbench.contracts')

    def test_collected_profiles_preserve_row_and_dimension_contracts(self):
        module = self.module()
        expected = {'single200': (1, 201, 200, 200, 28),
                    'double500': (2, 501, 500, 500, 37),
                    'triple_front600': (3, 1001, 1000, 600, 46),
                    'quadruple600': (4, 601, 600, 600, 55)}
        for name, values in expected.items():
            with self.subTest(name=name):
                profile = module.get_profile(name)
                self.assertEqual(tuple(profile[k] for k in
                    ('cube_count', 'raw_rows', 'action_rows', 'view_rows', 'state_dim')), values)
                self.assertEqual(profile['source'], 'self_collected')
                self.assertEqual((profile['train_episodes'], profile['validation_episodes']), (10000, 1000))
                self.assertEqual(profile['action_dim'], 5)
                self.assertEqual(profile['pixel_shape'], [224, 224, 3])
                self.assertEqual(len(profile['manifest_sha256']), 64)
                self.assertEqual(len(profile['train_sha256']), 64)

    def test_profile_is_independent_and_unknown_names_reject(self):
        module = self.module()
        item = module.get_profile('double500')
        item['camera']['position'][0] = 0
        self.assertEqual(module.get_profile('double500')['camera']['position'][0], 1.287)
        for name in ('double', 'cube2', 'official', '', {'name': 'double500'}):
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.get_profile(name)

    def test_exact_registered_profile_dict_is_accepted_but_modified_dict_rejects(self):
        module = self.module()
        registered = module.get_profile('double500')
        try:
            actual = module.get_profile(registered)
        except ValueError:
            self.fail('exact registered profile dict must resolve to its frozen identity')
        self.assertEqual(actual, registered)
        registered['view_rows'] = 600
        with self.assertRaisesRegex(ValueError, 'registered|profile'):
            module.get_profile(registered)

    def test_official_data_have_separate_metadata_and_no_custom_reader(self):
        module = self.module()
        quotas = {1: (1000, 100), 2: (1000, 100), 3: (3000, 300), 4: (5000, 500)}
        for count in range(1, 5):
            profile = module.get_profile('official' + str(count))
            self.assertEqual(profile['source'], 'official')
            self.assertEqual(profile['cube_count'], count)
            self.assertEqual(profile['raw_rows'], 1001)
            self.assertEqual(profile.get('transitions_per_episode'), 1000)
            self.assertEqual((profile['train_episodes'], profile['validation_episodes']), quotas[count])
            self.assertIsNone(profile['action_rows'])
            self.assertIsNone(profile['view_rows'])
            self.assertFalse(profile['reader_supported'])
            self.assertIn('terminal', profile['row_semantics'])
            self.assertEqual(len(profile['metadata_evidence']['sha256']), 64)
            self.assertIsNone(profile['train_view'])
            with self.assertRaisesRegex(ValueError, 'official'):
                module.validate_view('unavailable.h5', profile['name'], 'train')

    def test_wrong_split_name_or_digest_reject_before_hdf5_import(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '2d_train_action_rows500.h5'
            path.write_bytes(b'not a sealed production view')
            with self.assertRaisesRegex(ValueError, 'split'):
                module.validate_view(path, 'double500', 'test')
            with self.assertRaisesRegex(ValueError, 'identity|name'):
                module.validate_view(path, 'single200', 'train')
            with self.assertRaisesRegex(ValueError, 'sha256|checksum'):
                module.validate_view(path, 'double500', 'train')
            with self.assertRaisesRegex(ValueError, 'sha256|checksum'):
                module.validate_view(path, 'double500', 'train', expected_sha256='0' * 64)

    def test_manifest_checksum_is_mandatory(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data_views').mkdir()
            manifest = root / 'data_views/training_view_index.json'
            manifest.write_text(json.dumps({'actual_exit_code': 0}))
            with self.assertRaisesRegex(ValueError, 'manifest'):
                module.validate_manifest(root, 'double500')

    def test_original_validators_accept_schema_and_reject_provenance_mutation(self):
        module = self.module()
        import hashlib
        for name in ('single200', 'double500', 'triple_front600', 'quadruple600'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                profile = module.get_profile(name)
                rows, episodes = profile['view_rows'], profile['train_episodes']
                view = _View(pixels=_Column((rows * episodes, 224, 224, 3), 'uint8'),
                             action=_Column((rows * episodes, 5), 'float32'),
                             ep_len=_Column((episodes,), 'int64', np.full(episodes, rows)),
                             ep_offset=_Column((episodes,), 'int64', np.arange(episodes) * rows))
                path = Path(directory) / profile['train_view']
                path.write_bytes(b'metadata-only validator unit test')
                view.filename = str(path)
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                item = dict(path=str(path), sha256=sha, episodes=episodes,
                            rows=rows * episodes, keys=['pixels', 'action'],
                            split='train', modality='2d')
                manifest = dict(views=[item], canonical_index={'sha256': 'canonical'},
                                independent_seal={'sha256': 'seal'},
                                sources={'train': {'sha256': 'train'},
                                         'validation': {'sha256': 'validation'}})
                if name == 'single200':
                    view.attrs = dict(view_recipe='single_cube_H201_action_bearing_rows200_v1',
                                      canonical_index_sha256='canonical', source_train_sha256='train',
                                      source_validation_sha256='validation', excluded_terminal_action_row=200)
                    mutation = ('excluded_terminal_action_row', 201)
                elif name == 'double500':
                    view.attrs = dict(view_recipe='all_tasks_H500_action_bearing_rows500_v1',
                                      canonical_index_sha256='canonical', independent_seal_sha256='seal',
                                      raw_terminal_observation_row500='preserved_in_raw_archive')
                    mutation = ('independent_seal_sha256', 'wrong')
                elif name == 'quadruple600':
                    view.attrs = dict(view_recipe='quadruple_H600_action_bearing_rows600_v1',
                                      env_type='quadruple', physical_cube_count=4, horizon_actions=600,
                                      canonical_index_sha256='canonical', independent_seal_sha256='seal',
                                      raw_terminal_observation_row600='preserved_in_raw_archive')
                    mutation = ('physical_cube_count', 3)
                else:
                    source = dict(env_type='triple', physical_cube_count=3, horizon_actions=1000,
                                  raw_rows_per_episode=1001, observation_dim=46, qpos_dim=35,
                                  qvel_dim=32, action_dim=5, train_episodes=10000, validation_episodes=1000)
                    consumer = dict(source, horizon_actions=600, raw_rows_per_episode=601)
                    parent = dict(source_freeze_sha256='source', canonical_index_sha256='canonical',
                                  independent_seal_sha256='seal')
                    recipe = dict(schema='triple_front600_v1', scope='production',
                                  source_contract=source, contract=consumer, parent=parent)
                    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True,
                                                         separators=(',', ':')).encode()).hexdigest()
                    manifest.update(stage='passed', actual_exit_code=0, recipe=recipe,
                                    consumer_recipe_sha256=identity)
                    view.attrs = dict(prefix_view_schema='triple_front600_v1',
                                      source_triple_contract=json.dumps(source), triple_contract=json.dumps(consumer),
                                      training_scope='production', env_type='triple', physical_cube_count=3,
                                      horizon_actions=600, consumer_manifest='training_view_index.json',
                                      consumer_recipe_sha256=identity,
                                      **{'parent_' + key: value for key, value in parent.items()})
                    mutation = ('training_scope', 'fixture')
                (Path(directory) / 'training_view_index.json').write_text(json.dumps(manifest))
                module._validate_open_file(view, profile, sha)
                key, bad_value = mutation
                view.attrs[key] = bad_value
                with self.assertRaises(ValueError):
                    module._validate_open_file(view, profile, sha)


if __name__ == '__main__':
    unittest.main()

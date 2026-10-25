import copy
import ast
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace

try:
    from bench.ogbench.downstream import bind_recipe, build_command, validate_cache_protocol
except ImportError:
    bind_recipe = build_command = None


def recipe(cubes=2, method='lewm_rgb'):
    family, dim = {'state': ('state', 0), 'lewm_rgb': ('lewm', 192),
                   'lewm_pcd': ('lewm', 192), 'pldm': ('pldm', 512),
                   'lejepa': ('lejepa', 512)}[method]
    return dict(protocol='cube_all_distance_v1', cube_count=cubes,
                modality='state' if method == 'state' else 'pcd' if method == 'lewm_pcd' else 'rgb',
                input_contract=dict(current_dim=19+9*cubes if not dim else dim+19,
                                    goal_dim=3*cubes if not dim else dim+3*cubes,
                                    latent_dim=dim, zero_padding=False),
                encoder_binding=dict(family=family, latent_dim=dim,
                                     checkpoint_sha256=None if not dim else 'a'*64,
                                     preprocessing=dict(input_dtype='float32' if method=='lewm_pcd' else 'uint8', layout='XYZRGB1200' if method=='lewm_pcd' else 'HWC224', device='cpu', source_sha256='b'*64,center=[.425,0.,.2],scale=4.,use_rgb=True)),
                native_protocol=dict(evaluator='ogbench_native_v1', max_episode_steps=500,
                                     camera=None if not dim else dict(source_name='front_pixels' if cubes==1 else 'front', render_name='front_pixels', position=[1.053,-.014,.639] if cubes==1 else [1.287,0.,.509], fov=37. if cubes==1 else 45.,local_x_up_degrees=0. if cubes==1 else 5., pullback_m=0., width=224,height=224,aa_samples=4,pointcloud_crop_box=[[-.25,-.5,-.02],[.75,.5,.9]]),
                                     encoding_protocol='fresh_current_and_goal_each_policy_call_2N'))


class DownstreamContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(bind_recipe, 'common downstream binding has not been implemented')

    def test_true_widths_for_all_cubes_and_encoders(self):
        for cubes in (1, 2, 3, 4):
            for method in ('state', 'lewm_rgb', 'lewm_pcd', 'pldm', 'lejepa'):
                cfg = recipe(cubes, method)
                profile = {'cube_count': cubes, 'camera': {'source_name':'front_pixels', 'render_name':'front_pixels'}} if cubes==1 else {'cube_count': cubes}
                try:
                    bound = bind_recipe(profile, method, cfg)
                except ValueError as error:
                    self.fail('Valid declared camera was rejected: '+str(error))
                self.assertEqual(bound['input_contract'], cfg['input_contract'])
                self.assertFalse(bound['input_contract']['zero_padding'])
                self.assertEqual(bound['state_dim'], 19+9*cubes)
        self.assertEqual(bind_recipe({'cube_count': 1, 'camera': {'source_name':'front_pixels', 'render_name':'front_pixels'}}, 'lejepa', recipe(1, 'lejepa'))['latent_dim'], 512)

    def test_single_uses_declared_identity_camera_without_extra_pitch(self):
        from bench.ogbench.contracts import get_profile
        profile = get_profile('single200')
        cfg = recipe(1, 'lejepa')
        cfg['native_protocol']['max_episode_steps'] = profile['max_episode_steps']
        try:
            bound = bind_recipe(profile, 'lejepa', cfg)
        except ValueError as error:
            self.fail('Single identity camera was rejected: '+str(error))
        self.assertEqual(bound['native_protocol']['camera']['source_name'], 'front_pixels')
        self.assertEqual(bound['native_protocol']['camera']['local_x_up_degrees'], 0.)
        for key, value in (('source_name','front'), ('local_x_up_degrees',5.), ('pullback_m',.1)):
            broken = copy.deepcopy(cfg)
            broken['native_protocol']['camera'][key] = value
            with self.assertRaises(ValueError):
                bind_recipe(profile, 'lejepa', broken)
        broken_profile = copy.deepcopy(profile)
        broken_profile['camera']['source_name'] = 'front'
        with self.assertRaises(ValueError):
            bind_recipe(broken_profile, 'lejepa', cfg)

    def test_invalid_family_width_and_padding_rejected(self):
        cases = []
        for path, value in ((('encoder_binding', 'family'), 'lejepa'),
                            (('encoder_binding', 'latent_dim'), 192),
                            (('input_contract', 'zero_padding'), True),
                            (('input_contract', 'current_dim'), 211)):
            cfg = recipe(3, 'pldm')
            cfg[path[0]][path[1]] = value
            cases.append(cfg)
        for cfg in cases:
            with self.assertRaises(ValueError):
                bind_recipe({'cube_count': 3}, 'pldm', cfg)

    def test_unbound_checkpoint_and_protocol_rejected(self):
        cfg = recipe()
        cfg['encoder_binding']['checkpoint_sha256'] = 'latest'
        with self.assertRaises(ValueError):
            bind_recipe({'cube_count': 2}, 'lewm_rgb', cfg)
        cfg = recipe()
        cfg['protocol'] = 'official_ogbench'
        with self.assertRaises(ValueError):
            bind_recipe({'cube_count': 2}, 'lewm_rgb', cfg)
        with self.assertRaises(ValueError):
            bind_recipe({'cube_count': 4}, 'lewm_rgb', recipe(3))

    def test_native_requires_explicit_horizon_and_camera(self):
        for key in ('max_episode_steps', 'camera', 'encoding_protocol'):
            cfg = recipe()
            del cfg['native_protocol'][key]
            with self.assertRaises(ValueError):
                bind_recipe({'cube_count': 2}, 'lewm_rgb', cfg)
        for camera in ({}, {'position': [1.287, 0., .509]}):
            cfg = recipe()
            cfg['native_protocol']['camera'] = camera
            with self.assertRaises(ValueError):
                bind_recipe({'cube_count': 2}, 'lewm_rgb', cfg)
        cfg = recipe(2, 'lewm_pcd')
        del cfg['native_protocol']['camera']['pointcloud_crop_box']
        with self.assertRaises(ValueError):
            bind_recipe({'cube_count': 2}, 'lewm_pcd', cfg)

    def test_cache_camera_and_preprocessing_are_bound(self):
        cfg = bind_recipe({'cube_count': 2}, 'lewm_rgb', recipe())
        manifest = dict(protocol=dict(encoder_family='lewm', latent_dim=192,
            camera=copy.deepcopy(cfg['native_protocol']['camera']), pixel_preprocess_device='cpu',
            source_pins={'original_preprocessing.py': 'b'*64}))
        validate_cache_protocol(manifest, cfg)
        for key, value in (('camera', {}), ('pixel_preprocess_device', 'cuda'), ('source_pins', {})):
            broken = copy.deepcopy(manifest)
            broken['protocol'][key] = value
            with self.assertRaises(ValueError):
                validate_cache_protocol(broken, cfg)
        for key in ('source_name', 'render_name'):
            for missing in (False, True):
                broken = copy.deepcopy(manifest)
                if missing:
                    del broken['protocol']['camera'][key]
                else:
                    broken['protocol']['camera'][key] = 'other_camera'
                with self.assertRaises(ValueError):
                    validate_cache_protocol(broken, cfg)

    def test_preprocessing_device_matches_actual_family_path(self):
        for method in ('lejepa', 'lewm_rgb', 'lewm_pcd'):
            cfg = recipe(2, method)
            cfg['encoder_binding']['preprocessing']['device'] = 'cuda'
            with self.assertRaises(ValueError):
                bind_recipe({'cube_count': 2}, method, cfg)
        cfg = recipe(2, 'pldm')
        cfg['encoder_binding']['preprocessing']['device'] = 'cuda'
        bound = bind_recipe({'cube_count': 2}, 'pldm', cfg)
        self.assertEqual(bound['encoder_binding']['preprocessing']['device'], 'cuda')

    def test_command_is_common_module_and_rejects_unsupported_state_native(self):
        from bench.ogbench.contracts import profile_names
        name = profile_names()[0]
        command = build_command(name, 'pldm', 'gciql', '/data', '/out', '/recipe.json', python='/python')
        self.assertEqual(command[:3], ['/python', '-m', 'bench.ogbench.downstream.runtime'])
        self.assertIn('/recipe.json', command)
        with self.assertRaises(ValueError):
            build_command(name, 'state', 'native-sr', '/data', '/out', '/recipe.json')


class InputProjectionTests(unittest.TestCase):
    def test_default_192_is_original_projection_without_padding(self):
        import numpy as np
        from bench.ogbench.downstream.inputs import CubeInputs
        source = Path(__file__).parents[1]/'downstream/vendor/cube_inputs.py'
        spec = importlib.util.spec_from_file_location('_legacy_cube_projection', source)
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for cubes in (1, 2, 3, 4):
            states = np.arange(2*(19+9*cubes), dtype=np.float32).reshape(2, -1)
            features = np.arange(384, dtype=np.float32).reshape(2, 192)
            new, old = CubeInputs(cubes), module.CubeInputs(cubes)
            np.testing.assert_array_equal(new.current(features, states), old.current(features, states))
            np.testing.assert_array_equal(new.goal(features, states), old.goal(features, states))
            self.assertEqual(new.current(features, states).shape, (2, 211))
            self.assertEqual(new.goal(features, states).shape, (2, 192+3*cubes))

    def test_512_width_and_wrong_state_rejected(self):
        import numpy as np
        from bench.ogbench.downstream.inputs import CubeInputs
        adapter = CubeInputs(3, 512)
        states = np.ones((2, 46), np.float32)
        latent = np.ones((2, 512), np.float32)
        self.assertEqual(adapter.current(latent, states).shape, (2, 531))
        self.assertEqual(adapter.goal(latent, states).shape, (2, 521))
        with self.assertRaises(ValueError):
            adapter.current(latent, states[:, :-1])
        with self.assertRaises(ValueError):
            adapter.current(np.zeros((2, 192), np.float32), states)


class OriginalMathTests(unittest.TestCase):
    def test_source_manifest_paths_are_portable(self):
        from bench.ogbench.downstream import _source
        self.assertTrue(callable(getattr(_source, 'source_path', None)))
        self.assertEqual(_source.source_path('source/ogbench/impls/utils/evaluation.py'),
                         _source.VENDOR/'source/ogbench/impls/utils/evaluation.py')
        for key in ('source\\ogbench\\impls\\utils\\evaluation.py', '../gciql.py', '/tmp/file.py'):
            with self.assertRaises(ValueError):
                _source.source_path(key)
        manifest = _source.verify_sources()
        self.assertIn('source/ogbench/impls/utils/evaluation.py', manifest['files'])
        self.assertFalse(any('\\' in key for key in manifest['files']))

    def test_backend_and_rng_environment_are_explicit(self):
        from bench.ogbench.downstream import runtime
        self.assertTrue(callable(getattr(runtime, 'runtime_environment', None)))
        for action in ('train', 'gate'):
            env = runtime.runtime_environment('gciql', action)
            self.assertEqual(env['JAX_PLATFORMS'], 'cuda,cpu')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
            self.assertEqual(env['JAX_THREEFRY_PARTITIONABLE'], 'false')
            self.assertEqual(env['XLA_FLAGS'], '--xla_gpu_autotune_level=0')
        self.assertEqual(runtime.runtime_environment('gciql', 'reload')['CUDA_VISIBLE_DEVICES'], '')
        native = runtime.runtime_environment('native-sr')
        self.assertEqual(native['JAX_PLATFORMS'], 'cpu')
        self.assertEqual(native['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(native['MUJOCO_GL'], 'egl')

    def test_source_closure_hashes_and_algorithm_ast_are_preserved(self):
        from bench.ogbench.downstream._source import verify_sources
        self.assertTrue(verify_sources()['original_bytes_preserved'])
        root = Path(__file__).parents[1]/'downstream'
        pairs = (
            ('trainer.py', 'vendor/trainer_runtime.py', {'run', 'initial_agent', 'get_batch', 'fingerprint'}),
            ('sampler.py', 'vendor/legacy_distance_inputs.py', {'sample', 'sample_goals', 'object_xyz', 'create_agent'}),
            ('native_core.py', 'vendor/visual_sr.py', {'main', 'run_episode', 'load_agent', 'sample_actions',
                'reset', 'capture', 'step', 'current_raw'}),
        )
        def selected(path, names):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.walk(ast.parse(path.read_text()))
                    if isinstance(node, ast.FunctionDef) and node.name in names}
        for current, original, names in pairs:
            self.assertEqual(selected(root/current, names), selected(root/original, names), current)


@unittest.skipUnless(importlib.util.find_spec('jax') and importlib.util.find_spec('flax'),
                     'bounded numerical sampler parity needs existing JAX/Flax CPU runtime')
class SamplerNumericalTests(unittest.TestCase):
    """CPU sampling only: no agent, optimizer, simulator, or encoder is created."""
    @staticmethod
    def inputs(cubes, dim=192):
        import numpy as np
        from bench.ogbench.downstream._source import install_impls
        install_impls()
        from utils.datasets import Dataset
        states = np.arange(12*(19+9*cubes), dtype=np.float32).reshape(12, -1)/100.
        terminals = np.zeros(12, np.float32)
        terminals[[5, 11]] = 1.
        valids = 1.-terminals
        actions = np.arange(60, dtype=np.float32).reshape(12, 5)/100.
        raw = Dataset.create(observations=states, actions=actions, terminals=terminals, valids=valids)
        features = np.arange(12*dim, dtype=np.float32).reshape(12, dim)/1000.
        features.setflags(write=False)
        config = dict(discount=.99, value_p_curgoal=.2, value_p_trajgoal=.5,
            value_p_randomgoal=.3, value_geom_sample=True, actor_p_curgoal=0.,
            actor_p_trajgoal=1., actor_p_randomgoal=0., actor_geom_sample=False,
            gc_negative=True, p_aug=0., frame_stack=None)
        return raw, features, config

    def test_original192_cube23_same_rng_goals_rewards_and_arrays_exact(self):
        import numpy as np
        import sys
        from bench.ogbench.downstream._source import VENDOR, load_source
        from bench.ogbench.downstream.sampler import DistanceInputs
        sys.path.insert(0, str(VENDOR))
        legacy = load_source('_bounded_original_distance_inputs', 'legacy_distance_inputs.py')
        ids = np.array([0, 4, 6, 10])
        for cubes in (2, 3):
            raw, features, cfg = self.inputs(cubes)
            for mode in ('state', 'rgb', 'pcd'):
                kwargs = dict(cubes=cubes, mode=mode, workers=0,
                              features=None if mode == 'state' else features)
                old, new = legacy.DistanceInputs(raw, cfg, **kwargs), DistanceInputs(raw, cfg, **kwargs)
                try:
                    np.random.seed(913)
                    expected = old.sample(4, idxs=ids)
                    expected_rng = np.random.get_state()
                    np.random.seed(913)
                    actual = new.sample(4, idxs=ids)
                    actual_rng = np.random.get_state()
                    self.assertEqual(set(actual), set(expected))
                    for key in expected:
                        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f'{cubes}/{mode}/{key}')
                    for key in old.last_ids:
                        np.testing.assert_array_equal(new.last_ids[key], old.last_ids[key], err_msg=key)
                    self.assertEqual(actual_rng[0], expected_rng[0])
                    np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
                    self.assertEqual(actual_rng[2:], expected_rng[2:])
                finally:
                    old.close()
                    new.close()

    def test_cube14_actual512_roles_and_terminal_boundary(self):
        import numpy as np
        from bench.ogbench.downstream.sampler import DistanceInputs, object_xyz
        ids = np.array([0, 4, 6, 10])
        for cubes in (1, 4):
            raw, features, cfg = self.inputs(cubes, 512)
            sampler = DistanceInputs(raw, cfg, cubes=cubes, mode='rgb', features=features, latent_dim=512)
            try:
                np.random.seed(914)
                batch = sampler.sample(4, idxs=ids)
                self.assertEqual(batch['observations'].shape, (4, 531))
                self.assertEqual(batch['actor_goals'].shape, (4, 512+3*cubes))
                np.testing.assert_array_equal(batch['observations'][:, :512], features[ids])
                np.testing.assert_array_equal(batch['observations'][:, 512:], raw['observations'][ids, :19])
                goal_ids = sampler.last_ids['value_goal']
                expected_xyz = object_xyz(raw['observations'][goal_ids], cubes).reshape(4, -1)
                np.testing.assert_array_equal(batch['value_goals'][:, :512], features[goal_ids])
                np.testing.assert_array_equal(batch['value_goals'][:, 512:], expected_xyz)
                success = np.all(np.linalg.norm(object_xyz(raw['observations'][ids], cubes)-
                    object_xyz(raw['observations'][goal_ids], cubes), axis=-1) <= .4, axis=-1)
                np.testing.assert_array_equal(batch['rewards'], success.astype(batch['rewards'].dtype)-1.)
                np.testing.assert_array_equal(batch['masks'], 1.-success.astype(batch['masks'].dtype))
                for invalid in (5, 11, -1):
                    with self.assertRaises(ValueError):
                        sampler.sample(1, idxs=np.array([invalid]))
            finally:
                sampler.close()


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'camera fixture needs existing MuJoCo; no episode/render')
class CameraFixtureTests(unittest.TestCase):
    def test_single_identity_and_preserved_multi_camera_paths(self):
        import mujoco
        import numpy as np
        from bench.ogbench.downstream import native_core
        groups = native_core.GROUPS
        try:
            for cubes in (1, 2, 3, 4):
                single = cubes == 1
                source_position = [1.053,-.014,.639] if single else [1.287,0.,.509]
                source_name = 'front_pixels' if single else 'front'
                fov = 37. if single else 45.
                source_xml = f'<camera name="{source_name}" pos="{source_position[0]} {source_position[1]} {source_position[2]}" quat=".7 .2 .3 .6" fovy="{fov}"/>'
                target_xml = '' if single else '<camera name="front_pixels" pos="0 0 0" quat="1 0 0 0" fovy="45"/>'
                model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+source_xml+target_xml+'</worldbody></mujoco>')
                source = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, source_name)
                target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'front_pixels')
                initial_position, initial_quaternion = model.cam_pos[source].copy(), model.cam_quat[source].copy()
                rotation, expected_quaternion = np.empty(4), np.empty(4)
                mujoco.mju_axisAngle2Quat(rotation, np.array([1.,0.,0.]), np.deg2rad(0. if single else 5.))
                mujoco.mju_mulQuat(expected_quaternion, initial_quaternion, rotation)
                pullback = .1 if cubes == 4 else 0.
                w,x,y,z = expected_quaternion
                local_z = np.array([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)])
                expected_position = initial_position+pullback*local_z
                camera = dict(source_name=source_name,render_name='front_pixels',position=expected_position.tolist(),
                    source_position=source_position,fov=fov,local_x_up_degrees=0. if single else 5.,pullback_m=pullback,
                    width=224,height=224,aa_samples=4)
                native_core.GROUPS = {'fixture': dict(mode='rgb',camera=camera)}
                native_core.configure_camera(SimpleNamespace(_model=model), 'fixture')
                np.testing.assert_array_equal(model.cam_quat[target], expected_quaternion)
                np.testing.assert_array_equal(model.cam_pos[target], expected_position)
                self.assertEqual(float(model.cam_fovy[target]), fov)
                if single:
                    np.testing.assert_array_equal(model.cam_quat[target], initial_quaternion)
                    np.testing.assert_array_equal(model.cam_pos[target], initial_position)
        finally:
            native_core.GROUPS = groups


if __name__ == '__main__':
    unittest.main()

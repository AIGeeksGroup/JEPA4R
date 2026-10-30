"""CPU/source/hook contracts only: no optimizer, encoder or Native execution."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parents[1] / 'inventory/downstream_reference'
ENTRY = HERE / 'functional.py'


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.core = None
        if ENTRY.exists():
            spec = importlib.util.spec_from_file_location('_functional_under_test', ENTRY)
            self.core = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.core)
            self.core.ROOT = REFERENCE

    def available(self):
        self.assertIsNotNone(self.core, 'Missing PLDM1/4 functional bootstrap')
        return self.core

    def modules(self):
        return self.available().bootstrap()

    def config(self, modules, n=1, purpose='formal'):
        path = REFERENCE/'cube23_distance_gciql_v1/five_group_extension_v1/configs/G08_cube3_rgb.json'
        template = json.loads(path.read_text())
        delivery = dict(group_id=f'pldm_cube{n}', cube_count=n, latent_dim=512, purpose=purpose,
            data_binding={'data_id':f'cube-{n}-fixture'}, feature_loader_arguments={},
            encoder_binding={'family':'pldm','latent_dim':512}, camera={}, cache_audit={})
        return modules['prepare'].make_config(template, delivery)

    def test_bootstrap_exists(self):
        self.available()

    def test_recipe_survives_native_JSON_roundtrip_exactly(self):
        proof = self.available().recipe()
        self.assertEqual(json.loads(json.dumps(proof)),proof)

    def test_legacy_sources_exact_and_overlay_counted(self):
        core = self.available()
        for name in core.MODULES:
            path = REFERENCE/'foundation_visual_gciql_v1/functional'/f'{name}.py'
            original = path.read_bytes().decode()
            self.assertEqual(core.effective_source(name, pldm=False), original)
            changed = core.effective_source(name)
            for before, after in reversed(core.EDITS.get(name, ())):
                self.assertEqual(changed.count(after), 1)
                changed = changed.replace(after, before, 1)
            self.assertEqual(changed, original)

    def test_bootstrap_restores_aliases_and_cuda_environment(self):
        core = self.available()
        poisons = {name:types.ModuleType('_unrelated_'+name) for name in core.MODULES}
        with patch.dict(sys.modules, poisons), patch.dict(os.environ, {'JAX_PLATFORMS':'cuda','CUDA_VISIBLE_DEVICES':'7','MUJOCO_GL':'test'}):
            before_path = list(sys.path)
            modules = core.bootstrap()
            self.assertEqual(set(modules), set(core.MODULES))
            for name, module in poisons.items():
                self.assertIs(sys.modules[name], module)
            self.assertEqual(sys.path, before_path)
            self.assertEqual(os.environ['JAX_PLATFORMS'], 'cuda')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '7')
            self.assertEqual(os.environ['MUJOCO_GL'], 'test')

    def test_config_shapes_fresh_budget_and_real_commands(self):
        modules = self.modules()
        for n, goal, cap in [(1,515,200),(4,524,1000)]:
            for purpose in ('formal','pilot'):
                cfg = self.config(modules,n,purpose)
                modules['runtime'].check_config(cfg)
                self.assertEqual(cfg['input_contract'], dict(current_dim=531,goal_dim=goal,latent_dim=512,zero_padding=False))
                self.assertEqual((cfg['seed'],cfg['batch_size'],cfg['total_updates'],cfg['resume_from']), (0,1024,1000000,None))
                self.assertEqual(cfg['native_max_episode_steps'],cap)
                self.assertEqual(len(cfg['checkpoint_steps']),11)
                commands = modules['prepare'].commands(Path('config.json'),cfg)
                self.assertEqual('train' in commands,purpose=='formal')
                self.assertEqual('accept' in commands,purpose=='formal')
                for command in commands.values():
                    self.assertEqual(command[0],str(ENTRY))
                    self.assertIn(command[1],('runtime','native','audit'))
                self.assertEqual(commands['native_audit'][1],'audit')
                self.assertNotIn('--gate-smoke',commands['native_audit'])

    def test_real_projection_excludes_current_object_GT(self):
        core = self.available()
        import numpy as np
        spec = importlib.util.spec_from_file_location('_projection_fixture',REFERENCE/'cube_inputs.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for n in (1,4):
            contract = module.CubeInputs(n,512)
            states = np.arange(2*(19+9*n),dtype=np.float32).reshape(2,-1)
            z = np.ones((2,512),np.float32)
            current, goal = contract.current(z,states), contract.goal(z,states)
            self.assertEqual(current.shape,(2,531))
            self.assertEqual(goal.shape,(2,512+3*n))
            np.testing.assert_array_equal(current[:,512:],states[:,:19])
            np.testing.assert_array_equal(goal[:,512:],np.concatenate([states[:,19+9*i:22+9*i] for i in range(n)],axis=-1))
        inputs = core.bootstrap()['inputs']
        _, proof = inputs.overlay_source((REFERENCE/'cube23_distance_gciql_v1/distance_inputs.py').read_bytes(),512)
        self.assertTrue(proof['sampler_reward_factory_AST_exact'])

    def test_runtime_hook_and_trace_wrap_without_agent_or_sampling(self):
        core = self.available()
        modules = core.bootstrap()
        cfg = self.config(modules)
        runtime = types.SimpleNamespace(get_batch=lambda *a:None)
        preserved = types.SimpleNamespace(install=lambda *a:runtime,
            ORIGINAL_PREPARE=lambda *a:(None,types.SimpleNamespace(),{'runtime':{},'code':{}}))
        boundary = types.ModuleType('gciql_boundary')
        fake_inputs = types.SimpleNamespace(VisualInputs=object,load_cached_features=lambda **kw:None,width_proof={'AST':True})
        with patch.dict(sys.modules,{'state_runtime':preserved,'gciql_boundary':boundary}), patch.object(modules['runtime'],'load_inputs',return_value=fake_inputs), patch.object(core,'cache_pins',return_value={'/bound/plan.json':'f'*64}):
            installed, edge = modules['runtime'].install(cfg)
            self.assertIs(edge.prepare,installed.prepare)
            self.assertIs(edge.get_batch,installed.get_batch)
            _,_,binding = installed.prepare(cfg)
            self.assertEqual(binding['code'][str(ENTRY)],core.sha(ENTRY))
            self.assertIn('functional_overlay',binding['runtime'])
            self.assertEqual(binding['code']['/bound/plan.json'],'f'*64)

    def test_native_delayed_import_uses_new_runtime(self):
        modules = self.modules()
        cfg = self.config(modules)
        class ReachedNewRuntime(Exception):
            pass
        with patch.object(modules['runtime'],'install',side_effect=ReachedNewRuntime):
            with self.assertRaises(ReachedNewRuntime):
                modules['native_sr'].install_core(cfg,Path('config.json'))

    def test_cache_provenance_requires_full_closure_and_correct_cube(self):
        core = self.available()
        cfg = self.config(core.bootstrap())
        cfg['encoder_binding'].update(checkpoint_sha256='a'*64,spec_path='/spec.json',spec_sha256='b'*64,
            adapter={'path':'/adapter.py','sha256':'c'*64})
        cfg['cache_plan_binding'] = {'path':'/plan.json','sha256':'d'*64}
        cfg['delivery_binding'] = {'path':'/delivery.json','sha256':'e'*64}
        cfg['cache_audit'] = {'path':'/cache_audit.json','sha256':'f'*64}
        plan = dict(cube_count=1,encoder_family='pldm',latent_dim=512,checkpoint_sha256='a'*64,
            encoder_spec='/spec.json',encoder_spec_sha256='b'*64,encoder_adapter='/adapter.py',encoder_adapter_sha256='c'*64,
            inference_precision='fp32',encoder_preprocessing_device='cpu',source_pins={'asset/code.py':'9'*64},split_episodes={'train':1000,'val':100})
        delivery = dict(purpose='formal',cube_count=1,encoder_binding=cfg['encoder_binding'],
            feature_loader_arguments=cfg['feature_loader_arguments'],camera=cfg['camera'],cache_audit=cfg['cache_audit'])
        control = core.ROOT/'pldm_cube14_embeddings_v1/control/full'
        closure = {'actual_exit_code':0,'completed_phases':6,'training_updates':0,'SR_episodes':0}
        values = {'/plan.json':plan,'/delivery.json':delivery,'/cache_audit.json':{'actual_exit_code':0,'episodes_verified':1100},
            str(control/'runtime/closure.json'):closure}
        values = {Path(key).as_posix():value for key,value in values.items()}
        def text(path, *a, **kw):
            return json.dumps(values[path.as_posix()]) if path.as_posix() in values else '0\n'
        def digest(path):
            return {'/plan.json':'d'*64,'/delivery.json':'e'*64,'/cache_audit.json':'f'*64,
                    (core.ROOT/'asset/code.py').as_posix():'9'*64}.get(Path(path).as_posix(),'0'*64)
        with patch.object(Path,'read_text',autospec=True,side_effect=text),patch.object(core,'sha',side_effect=digest):
            pins = core.cache_pins(cfg)
            self.assertEqual(pins[str(core.ROOT/'asset/code.py')],'9'*64)
            self.assertIn(str(ENTRY),pins)
            closure['actual_exit_code'] = 1
            with self.assertRaises(AssertionError):core.cache_pins(cfg)
            closure['actual_exit_code'] = 0
            plan['cube_count'] = 4
            with self.assertRaises(AssertionError):core.cache_pins(cfg)

    def test_native_delayed_import_uses_same_new_source_binding(self):
        modules = self.modules()
        cfg = self.config(modules)
        class ReachedNewBinding(Exception):
            pass
        with patch.object(Path,'read_text',return_value=json.dumps(cfg)), patch.object(modules['audit_native'],'source_binding',side_effect=ReachedNewBinding), patch.object(sys,'argv',['functional.py','--config','cfg.json','--mode','tiny','--output','not-written']):
            with self.assertRaises(ReachedNewBinding):
                modules['native_sr'].main()

    def test_cli_forwards_original_argparse_without_running_work(self):
        core = self.available()
        for action,name in [('runtime','runtime'),('native','native_sr'),('audit','audit_native')]:
            observed = []
            modules = {name:types.SimpleNamespace(main=lambda:observed.append(sys.argv[1:]))}
            remaining = ['--config','cfg.json'] if action!='runtime' else ['verify-hooks','--config','cfg.json']
            with patch.object(core,'bootstrap',return_value=modules), patch.object(sys,'argv',['test']):
                core.main([action,*remaining])
            self.assertEqual(observed,[remaining])

    def test_pldm_config_rejects_other_groups_and_object_GT_width(self):
        modules = self.modules()
        for key,value in [('cube_count',2),('group_id','other'),('native_max_episode_steps',999)]:
            cfg = self.config(modules)
            cfg[key] = value
            with self.assertRaises(AssertionError):
                modules['runtime'].check_config(cfg)
        cfg = self.config(modules)
        cfg['input_contract']['current_dim'] = 540
        with self.assertRaises(AssertionError):
            modules['runtime'].check_config(cfg)


if __name__ == '__main__':
    unittest.main()

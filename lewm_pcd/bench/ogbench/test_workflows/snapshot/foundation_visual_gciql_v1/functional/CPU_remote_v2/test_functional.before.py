import ast
import importlib.util
from pathlib import Path
import sys
import types
from dataclasses import dataclass
import unittest
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class FunctionalTests(unittest.TestCase):
    def test_all_eight_shapes_reward_terminal_ids_and_rng(self):
        from inputs import load_inputs
        @dataclass
        class FakeGCDataset:
            dataset: object
            config: object
            def __post_init__(self):self.size=len(self.dataset['observations'])
            def sample_goals(self,ids):return np.random.randint(0,self.size,len(ids))
            def sample(self,batch_size,idxs=None,evaluation=False):
                v,a=self.sample_goals(idxs),self.sample_goals(idxs)
                return dict(observations=self.dataset['observations'][idxs],next_observations=self.dataset['observations'][idxs+1],
                    value_goals=self.dataset['observations'][v],actor_goals=self.dataset['observations'][a],
                    rewards=np.zeros(batch_size,np.float32),masks=np.ones(batch_size,np.float32),actions=np.zeros((batch_size,5),np.float32))
        module=types.ModuleType('utils.datasets');module.GCDataset=FakeGCDataset
        cube_path=HERE.parents[1]/'cube_downstream_101732Test_20261017_v1'
        sys.path.insert(0,str(cube_path))
        source=HERE.parents[1]/'cube23_gciql_distance_matched_101732Test_20261018_v1/distance_inputs.py'
        with patch.dict(sys.modules,{'utils.datasets':module}):
            for dim in (768,1024):
                inputs=load_inputs(source,dim)
                for n in (1,2,3,4):
                    states=np.zeros((1002,19+9*n),np.float32)
                    states[:,19]=np.arange(1002,dtype=np.float32)/1000
                    states.setflags(write=False)
                    features=np.ones((1002,dim),np.float32);features.setflags(write=False)
                    ds=dict(observations=states,valids=np.r_[np.ones(1000),0,1])
                    config=dict(frame_stack=None,p_aug=0)
                    sampler=inputs.VisualInputs(ds,config,cubes=n,mode='rgb',features=features,workers=0)
                    ids=np.array([0,999])
                    np.random.seed(27);reference=FakeGCDataset(ds,config).sample(2,ids);rng=np.random.get_state()
                    np.random.seed(27);batch=sampler.sample(2,ids)
                    self.assertTrue(np.array_equal(np.random.get_state()[1],rng[1]))
                    self.assertEqual(np.random.get_state()[2:],rng[2:])
                    np.testing.assert_array_equal(sampler.last_ids['next'],[1,1000])
                    success=np.all(np.linalg.norm(inputs.object_xyz(reference['observations'],n)-inputs.object_xyz(reference['value_goals'],n),axis=-1)<=.4,axis=-1)
                    np.testing.assert_array_equal(batch['rewards'],success.astype(np.float32)-1)
                    self.assertEqual(batch['observations'].shape,(2,dim+19))
                    self.assertEqual(batch['actor_goals'].shape,(2,dim+3*n))
                    np.testing.assert_array_equal(batch['observations'][:,dim:],states[ids,:19])
                    with self.assertRaises(ValueError):sampler.sample(1,np.array([1000]))
                    sampler.close()

    def test_width_overlay_preserves_sampler_reward_and_rng(self):
        from inputs import overlay_source
        source = (HERE.parents[1]/'cube23_gciql_distance_matched_101732Test_20261018_v1/distance_inputs.py').read_bytes()
        for dim in (768, 1024):
            result, proof = overlay_source(source, dim)
            self.assertTrue(proof['sampler_reward_factory_AST_exact'])
            self.assertIn('self.cubes not in (1,2,3,4)', result)
            self.assertIn(f'self.current_dim,self.goal_dim={dim+19},{dim}+3*self.cubes', result)
            tree = ast.parse(result)
            sample = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'sample')
            self.assertIn('np.all', ast.unparse(sample))

    def test_contract_and_command_paths(self):
        from prepare import make_config, commands
        for cubes in (1,2,3,4):
            for dim in (768,1024):
                delivery = dict(group_id=f'F{dim}_cube{cubes}',cube_count=cubes,latent_dim=dim,
                    encoder_binding={},data_binding={},feature_loader_arguments={},camera={},cache_audit={})
                cfg = make_config({'agent_config':{}}, delivery)
                self.assertEqual(cfg['input_contract']['current_dim'],dim+19)
                self.assertEqual(cfg['input_contract']['goal_dim'],dim+3*cubes)
                self.assertEqual(len(cfg['checkpoint_steps']),11)
                self.assertEqual(cfg['native_max_episode_steps'],{1:200,2:500,3:1000,4:1000}[cubes])
                stages=commands('/case/config.json',cfg)
                self.assertEqual(list(stages),['gate','reload','train','accept','native'])
                self.assertIn('/functional/runtime.py',stages['train'][0].replace('\\','/'))
                self.assertNotIn('native', ' '.join(stages['train']))
                delivery['purpose']='pilot'
                pilot=make_config({'agent_config':{}},delivery)
                pilot_stages=commands('/case/pilot.json',pilot)
                self.assertEqual(list(pilot_stages),['gate','reload','native'])
                self.assertIn('--gate-smoke',pilot_stages['native'])
                self.assertIn('pilot',pilot['execution_root'])


if __name__ == '__main__':
    unittest.main()

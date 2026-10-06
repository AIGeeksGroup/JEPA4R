import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from ogbench_il_rl.evaluation import official_state_sr as sr


class ContractTests(unittest.TestCase):
    def test_seed_partition(self):
        seeds=[sr.episode_seed(t,i) for t in range(1,6) for i in range(50)]
        self.assertEqual(len(set(seeds)),250)
        self.assertNotIn(970900000,seeds)
        self.assertEqual(seeds[0],970100000)
        self.assertEqual(seeds[-1],970500049)

    def test_no_early_checkpoint_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError): sr.require_final(Path(directory))

    def test_resume_requires_identity_and_unchanged_artifacts(self):
        sr.configure_cpu()
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); attempt=folder/'attempt_0001'; attempt.mkdir()
            (attempt/'reset.json').write_text('{"physical_reference":1}')
            (attempt/'steps.jsonl').write_text('{"step":1}\n')
            result={'status':'passed','protocol_sha256':'test_protocol','seed':970100000,'task':1,'index':0,
                    'diagnostic_only':False,'attempt':'attempt_0001','reset_sha256':sr.sha(attempt/'reset.json'),
                    'steps_sha256':sr.sha(attempt/'steps.jsonl')}
            sr.write(folder/'result.json',result)
            self.assertEqual(sr.run_episode(None,{},1,0,970100000,folder,'test_protocol'),result)
            with self.assertRaises(AssertionError): sr.run_episode(None,{},1,0,970100001,folder,'test_protocol')
            (attempt/'steps.jsonl').write_text('{"step":999}\n')
            with self.assertRaises(AssertionError): sr.run_episode(None,{},1,0,970100000,folder,'test_protocol')

    def test_original_evaluation_records_original_clip_and_reset_only_actions(self):
        sr.configure_cpu()
        import gymnasium as gym
        import numpy as np
        from utils.evaluation import evaluate
        class FakeEnv(gym.Env):
            def __init__(self):
                self.spec=SimpleNamespace(id='cube-single-v0',max_episode_steps=200)
                self._model=SimpleNamespace(cam_pos=np.zeros((1,3)),geom_rgba=np.ones((1,4)))
                self._data=SimpleNamespace(qpos=np.zeros(21),qvel=np.zeros(20),ctrl=np.zeros(7),act=np.zeros(0),
                    mocap_pos=np.zeros((1,3)),mocap_quat=np.ones((1,4)),qacc_warmstart=np.zeros(20),eq_active=np.zeros(0),time=0.)
                self._mode='task'; self._ob_type='states'; self.count=0
            def compute_observation(self):
                ob=np.zeros(28); ob[0]=self._data.qpos[0]; return ob
            def reset(self,seed=None,options=None):
                super().reset(seed=seed); self._data.qpos[:]=0; self.count=0
                for _ in range(2): self.step(np.ones(5)*.1)
                goal=self.compute_observation().copy()
                self._data.qpos[:]=0; self.count=0
                return self.compute_observation(),{'goal':goal}
            def step(self,action):
                self.count+=1; self._data.qpos[0]+=1; self._data.ctrl[:5]=action; self._data.time+=.05
                return self.compute_observation(),0.,self.count>=2,False,{'success':0.}
        class FakeAgent:
            def sample_actions(self,observations,goals,seed,temperature):
                assert temperature==0 and observations.shape==goals.shape==(28,)
                return np.array([2.,-2.,.1,0.,0.])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            env=sr.recorder_class()(FakeEnv(),970100000,path)
            stats,trajs,_=evaluate(FakeAgent(),env,task_id=1,config={'discrete':False},num_eval_episodes=1,num_video_episodes=0,eval_temperature=0)
            self.assertEqual(env.count,2); env.close()
            reset=json.loads((path/'reset.json').read_text())
            self.assertEqual(reset['requested_seed'],reset['actual_seed'])
            self.assertEqual(len(reset['official_internal_goal_initialization']),2)
            self.assertEqual(reset['extra_policy_warmup_steps'],0)
            rows=[json.loads(x) for x in (path/'steps.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['sent_action']['values'],[1.,-1.,.1,0.,0.])
            self.assertEqual(rows[0]['post_state']['ctrl']['values'][:5],[1.,-1.,.1,0.,0.])
            self.assertNotEqual(reset['initial_physical_state']['sha256'],reset['goal_physical_state']['sha256'])


if __name__=='__main__': unittest.main()

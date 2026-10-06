"""CPU-only interface tests; no simulated rollout or benchmark claim."""
import unittest,sys
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from ogbench_il_rl.evaluation.native_single_frame import ReplayPairWrapper,assert_identity_protocol,SingleFrameAgent,SafeGoalRenderer,array_record

def snapshot(u):
    return {'qpos':array_record(u._data.qpos),'qvel':array_record(u._data.qvel),'ctrl':array_record(u._data.ctrl),'mocap_pos':array_record(u._data.mocap_pos),'time':u._data.time}

class Env:
    def __init__(self):
        self.unwrapped=self;self.spec=SimpleNamespace(max_episode_steps=200);self._data=SimpleNamespace(qpos=np.zeros(21),qvel=np.zeros(20),ctrl=np.zeros(7),mocap_pos=np.zeros((1,3)),time=0.)
        self.np_random_seed=0;self.np_random=np.random.default_rng(0);self.current=0;self.internal=False;self.sent=[]
    def compute_observation(self):return np.full(28,self.current,dtype=np.float64)
    def reset(self,seed=None,options=None):
        self.np_random_seed=seed;self.np_random=np.random.default_rng(seed);self.internal=True
        self.current=10;self.step(np.full(5,.9,np.float32));self.step(np.full(5,.8,np.float32));goal=self.compute_observation()
        self.current=0;self._data.qpos[:]=0;self._data.qvel[:]=0;self.internal=False
        return self.compute_observation(),{'goal':goal}
    def step(self,action):
        self.sent.append(np.array(action));self._data.ctrl[:5]=action;self._data.time+=.04
        if self.internal:self._data.qpos[:]=self.current
        else:self.current+=1
        return self.compute_observation(),0.,False,self.current>=200,{'success':False}

class Actor:
    def reset(self):self.calls=0
    def set_goal_rgb(self,image):self.goal=image

class Replay(unittest.TestCase):
    def reference(self):
        env=Env();sent=[np.full(5,.1,np.float32),np.full(5,-.2,np.float32)];step=env.step;n=[0];observations=[];observe=env.compute_observation
        def hooked(a):i=n[0];n[0]+=1;return step(sent[i])
        def rec():v=observe();observations.append((v,snapshot(env)));return v
        env.step=hooked;env.compute_observation=rec;obs,info=env.reset(seed=123,options={'task_id':1,'render_goal':False})
        goal=[p for v,p in observations if np.array_equal(v,info['goal'])][-1]
        return {'requested_seed':123,'actual_seed':123,'observation':array_record(obs),'goal28':array_record(info['goal']),'initial_physical_state':snapshot(env),'goal_physical_state':goal,
         'model_colors_camera_lights':{},'rng_state':env.np_random.bit_generator.state,'official_internal_goal_initialization':[{'action':array_record(a)} for a in sent],'extra_policy_warmup_steps':0,'environment_max_episode_steps':200},sent
    def test_actual_box_replay_and_zero_policy_warmup_budget(self):
        ref,sent=self.reference();env=Env();original=env.step;actor=Actor();wrapper=ReplayPairWrapper(env,ref,actor,lambda u,g:np.zeros((224,224,3),np.uint8),snapshot_fn=snapshot,model_snapshot_fn=lambda u:{})
        wrapper.reset(options={'task_id':1,'render_goal':False});self.assertEqual(env.step,original)
        np.testing.assert_array_equal(env.sent[:2],sent);self.assertEqual(wrapper.count,0);self.assertEqual(wrapper.policy_warmup_steps,0)
        for i in range(200):wrapper.step(np.zeros(5,np.float32))
        self.assertEqual(wrapper.count,200)
        with self.assertRaises(ValueError):wrapper.step(np.zeros(5,np.float32))
    def test_goal_renderer_current_mutation_is_rejected_and_original_methods_restored(self):
        ref,_=self.reference();env=Env();original=env.step
        def bad(u,g):u._data.qpos[0]=999;return np.zeros((224,224,3),np.uint8)
        wrapper=ReplayPairWrapper(env,ref,Actor(),bad,snapshot_fn=snapshot,model_snapshot_fn=lambda u:{})
        with self.assertRaises(ValueError):wrapper.reset(options={'task_id':1,'render_goal':False})
        self.assertEqual(env.step,original)
    def test_independent_goal_data_renderer_keeps_current_state_and_requires_full_present_fields(self):
        env=Env();env._model=SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4)))
        goal=snapshot(env);goal['qpos']=array_record(np.full(21,10.));goal['time']=.08
        independent=[]
        def factory(model):
            data=SimpleNamespace(qpos=np.zeros(21),qvel=np.zeros(20),ctrl=np.zeros(7),mocap_pos=np.zeros((1,3)),time=0.);independent.append(data);return data
        class Renderer:
            def __init__(self,model,**kw):pass
            def update_scene(self,data,camera):
                self.data=data;self.assert_camera=camera
                if data is env._data or camera!='front_pixels':raise AssertionError('Goal render did not use independent real goal data')
            def render(self):return np.full((224,224,3),int(self.data.qpos[0]),np.uint8)
            def close(self):pass
        renderer=SafeGoalRenderer(data_factory=factory,forward=lambda m,d:None,renderer_factory=Renderer,snapshot_fn=snapshot,model_snapshot_fn=lambda u:{})
        before=snapshot(env)
        with patch.dict(sys.modules,{'mujoco':SimpleNamespace()}):
            image=renderer(env,goal)
            self.assertEqual(image[0,0,0],10);self.assertEqual(before,snapshot(env));self.assertIsNot(independent[0],env._data)
            bad={k:v for k,v in goal.items() if k!='ctrl'}
            with self.assertRaises(ValueError):renderer(env,bad)
    def test_only_explicit_identity_single_frame_protocol_accepted(self):
        p={'action_domain':'official_npz_normalized_box_identity','action_normalization':'identity','policy_history':1,'policy_warmup_steps':0,'evaluation':{'temperature':0,'clip':[-1,1],'gaussian_noise':None}}
        assert_identity_protocol(p)
        for key,value in [('policy_history',3),('policy_warmup_steps',2),('action_normalization','zscore')]:
            wrong={**p,key:value}
            with self.assertRaises(ValueError):assert_identity_protocol(wrong)

try:import torch
except ImportError:torch=None

@unittest.skipIf(torch is None,'Torch required only for CPU actor interface fixture')
class Agent(unittest.TestCase):
    def test_first_step_calls_real_mean_and_no_unscale_clip_or_feature_cache(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=[]
            def get_action(self,inputs):self.calls.append({k:v.clone() for k,v in inputs.items()});return torch.tensor([[2.,-.5,0.,.25,-2.]])
        model=Model();frame=iter([np.full((224,224,3),x,np.uint8) for x in [10,20]])
        agent=SingleFrameAgent(model,lambda:next(frame),lambda rgb:torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float(),device='cpu',precision='fp32')
        agent.set_goal_rgb(np.zeros((224,224,3),np.uint8))
        a=agent.sample_actions(np.zeros(28),np.zeros(28),temperature=0,seed=None);b=agent.sample_actions(np.zeros(28),np.zeros(28),temperature=0,seed=None)
        np.testing.assert_array_equal(a,[2.,-.5,0.,.25,-2.]);np.testing.assert_array_equal(b,a)
        self.assertEqual(len(model.calls),2);self.assertEqual(agent.model_calls,2);self.assertEqual(agent.policy_warmup_steps,0)
        self.assertEqual(model.calls[0]['pixels'].shape,(1,3,224,224));self.assertEqual(model.calls[0]['pixels'][0,0,0,0],10);self.assertEqual(model.calls[1]['pixels'][0,0,0,0],20)

if __name__=='__main__':unittest.main()

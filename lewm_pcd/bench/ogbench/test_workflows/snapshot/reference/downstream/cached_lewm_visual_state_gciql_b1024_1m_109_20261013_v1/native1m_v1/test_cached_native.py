"""Narrow CPU-only 195-goal/default/nonmutation tests; no real renderer or WM."""
import ast,hashlib,json,pickle,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

P=Path(__file__).resolve().parent
W=Path('/root/autodl-tmp/workspace/users/lewm')
LOCKED=W/'coordination/cached_lewm_state_gciql_xyz_109_20261012_v1/native_v1/locked_dependencies/native_single_frame_adapter.py'
if not LOCKED.exists():
    LOCKED=P.parents[1]/'cached_lewm_state_gciql_xyz_109_20261012_v1/native_v1/locked_dependencies/native_single_frame_adapter.py'

def extract(path,names,namespace=None):
    tree=ast.parse(path.read_text());body=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert len(body)==len(names)
    scope=dict(np=np,hashlib=hashlib,json=json,pickle=pickle,Path=Path,sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
    if namespace:scope.update(namespace)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),scope)
    return scope

native=extract(P/'cached_state_xyz_native.py',{'CurrentStateXYZActor','VisualGoalStateActor','bind_completed'})

class VisualGoalContract(unittest.TestCase):
    def test_each_step_two_encoders_raw_goal_only_and_old_default(self):
        goal28=np.arange(28,dtype=np.float64)/10;calls=[];encoded=[];currents=[];seed=object()
        class Agent:
            def sample_actions(self,**kwargs):calls.append(kwargs);return np.zeros((1,5),np.float32)
        def current():currents.append(1);return np.ones((1,192),np.float32),{}
        def encode(raw):encoded.append(raw.copy());return np.arange(192,dtype=np.float32)[None],{'source':'paired_goal_fixture'}
        actor=native['VisualGoalStateActor'](Agent(),current,goal28,encode)
        raw=np.full((224,224,3),17,np.uint8);actor.set_goal_rgb(raw)
        initial=actor.raw_goal.tobytes();self.assertIsNone(actor.goal);self.assertEqual(len(encoded),0)
        for _ in range(4):
            actor.sample_actions(np.arange(28,dtype=np.float64),goal28,seed,0)
            self.assertIsNone(actor.goal)
        self.assertEqual((actor.model_calls,actor.encoder_calls,actor.goal_encoder_calls),(4,4,4))
        self.assertEqual(len(encoded),4);self.assertEqual(len(currents),4)
        self.assertFalse(actor.raw_goal.flags.writeable);self.assertEqual(actor.raw_goal.tobytes(),initial)
        for call in calls:
            self.assertIs(call['seed'],seed);self.assertEqual(call['temperature'],0)
            self.assertEqual(call['observations'].shape,(1,220));self.assertEqual(call['goals'].shape,(1,195))
            np.testing.assert_array_equal(call['goals'][0,192:],goal28[19:22].astype(np.float32))
        self.assertEqual(len({id(call['goals']) for call in calls}),4)
        with self.assertRaises(ValueError):actor.raw_goal[0,0,0]=99
        with self.assertRaises(AssertionError):actor.set_goal_rgb(raw)
        original=extract(P/'original_bytes/cached_state_xyz_native.py',{'CurrentStateXYZActor'})['CurrentStateXYZActor']
        for cls in (original,native['CurrentStateXYZActor']):
            old=cls(Agent(),current,goal28,'cache');old.set_goal_rgb(goal28[19:22].astype(np.float32))
            old.sample_actions(goal28,goal28,seed,0);self.assertEqual(calls[-1]['goals'].shape,(1,3))

    def test_locked_goal_renderer_independent_data_and_current_nonmutation(self):
        self.assertEqual(hashlib.sha256(LOCKED.read_bytes()).hexdigest(),'fcbc9a13cc38b6ad6ad08bc61b8b463dbd35a76d6511c928e3a94a35c1d699f9')
        helper=extract(LOCKED,{'array_record','default_snapshot','default_model_snapshot','SafeGoalRenderer'})
        shapes={'qpos':(21,),'qvel':(20,),'ctrl':(7,),'act':(0,),'mocap_pos':(1,3),'mocap_quat':(1,4),'qacc_warmstart':(20,),
                'eq_active':(3,),'qfrc_applied':(20,),'xfrc_applied':(26,6),'userdata':(0,),'plugin_state':(0,)}
        def data():return SimpleNamespace(**{k:np.zeros(v,np.float64) for k,v in shapes.items()},time=0.)
        live=data();goal=data();goal.qpos[:]=7.;goal.ctrl[:]=3.;goal.time=1.25
        goal_state={k:helper['array_record'](getattr(goal,k)) for k in shapes};goal_state['time']=goal.time
        u=SimpleNamespace(_data=live,_model=SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4))),np_random=np.random.default_rng(17),private={'mode':'task'})
        snap=lambda u:{**{k:getattr(u._data,k).tolist() for k in shapes},'time':u._data.time,'rng':json.loads(json.dumps(u.np_random.bit_generator.state)),'private':dict(u.private)}
        before=snap(u);captured=[];closed=[]
        class Renderer:
            def __init__(self,*args,**kwargs):pass
            def update_scene(self,d,camera):self.data=d;self.assert_camera=camera
            def render(self):return np.full((224,224,3),int(self.data.qpos[0]),np.uint8)
            def close(self):closed.append(1)
        def transform(u,d,rgb):captured.append((d,rgb.copy()));return rgb
        renderer=helper['SafeGoalRenderer'](data_factory=lambda model:data(),forward=lambda model,d:None,
            renderer_factory=Renderer,snapshot_fn=snap,model_snapshot_fn=lambda u:{'AA':4},frame_transform=transform)
        with patch.dict(sys.modules,{'mujoco':SimpleNamespace()}):image=renderer(u,goal_state)
        self.assertIs(u._data,live);self.assertEqual(snap(u),before);self.assertIsNot(captured[0][0],live)
        self.assertEqual(captured[0][0].time,1.25);self.assertEqual(len(closed),1)
        np.testing.assert_array_equal(image,np.full((224,224,3),7,np.uint8))

    def test_protected_runner_actor_current_body_and_gate2_rejected_as1m(self):
        def definitions(path):return {x.name:ast.dump(x) for x in ast.walk(ast.parse(path.read_text())) if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
        before=definitions(P/'original_bytes/official_state_sr.py');after=definitions(P/'official_state_sr.py')
        for name in before.keys()-{'main'}:self.assertEqual(before[name],after[name],name)
        before=definitions(P/'original_bytes/cached_state_xyz_native.py');after=definitions(P/'cached_state_xyz_native.py')
        for name in ('CurrentStateXYZActor','current_latent','load_factory','prepare_reference_replay'):self.assertEqual(before[name],after[name],name)
        self.assertEqual(subprocess.check_output([sys.executable,str(P/'original_bytes/official_state_sr.py'),'--stage','describe']),
                         subprocess.check_output([sys.executable,str(P/'official_state_sr.py'),'--stage','describe']))
        with tempfile.TemporaryDirectory() as folder:
            train=Path(folder);snapshot=train/'checkpoints/update_001000000';snapshot.mkdir(parents=True)
            cfg=dict(total_updates=1000000,batch_size=1024,seed=0,input_profile='cached',goal_mode='latent_object_xyz',reward_mode='current_object_xyz',visual_mode='cache',cache={'modality':'pcd'})
            binding=dict(config=cfg,agent_config=dict(agent_name='gciql',actor_loss='ddpgbc',alpha=1,encoder=None))
            (train/'resolved.json').write_text(json.dumps(dict(total_updates=1000000,binding=binding)))
            (snapshot/'manifest.json').write_text(json.dumps(dict(counter=2)))
            config=dict(train_root=str(train),selected_updates=1000000,goal_mode='latent_object_xyz',visual_mode='cache',modality='pcd',source_sha256={},required_zero_exits=[])
            path=train/'fixture.json';path.write_text(json.dumps(config))
            with self.assertRaises(AssertionError):native['bind_completed'](path,train)

if __name__=='__main__':unittest.main()

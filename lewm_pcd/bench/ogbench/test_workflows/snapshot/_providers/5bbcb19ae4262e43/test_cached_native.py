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

    def test_zero_visual_goal195_without_current_or_goal_encoder(self):
        zero = extract(P/'cached_state_xyz_native.py', {'CurrentStateXYZActor', 'ZeroVisualGoalStateActor'})
        calls=[]; seed=object(); goal28=np.arange(28,dtype=np.float64)/10
        class Agent:
            def sample_actions(self, **kwargs): calls.append(kwargs); return np.zeros((1,5),np.float32)
        def forbidden(): raise AssertionError('Zero control must not call encoder or renderer')
        actor=zero['ZeroVisualGoalStateActor'](Agent(),forbidden,goal28,'zero192')
        actor.set_goal_rgb(goal28[19:22].astype(np.float32))
        for _ in range(4):actor.sample_actions(np.arange(28,dtype=np.float64),goal28,seed,0)
        self.assertEqual((actor.model_calls,actor.encoder_calls,actor.goal_encoder_calls),(4,0,0))
        for call in calls:
            self.assertIs(call['seed'],seed);self.assertEqual(call['temperature'],0)
            self.assertEqual(call['observations'].shape,(1,220));self.assertEqual(call['goals'].shape,(1,195))
            self.assertEqual(np.count_nonzero(call['observations'][:,:192]),0);self.assertEqual(np.count_nonzero(call['goals'][:,:192]),0)
            np.testing.assert_array_equal(call['observations'][0,192:],np.arange(28,dtype=np.float32))
            np.testing.assert_array_equal(call['goals'][0,192:],goal28[19:22].astype(np.float32))

    def test_zero_replay_glue_no_renderer_no_frozen_and_owned_close(self):
        scope=extract(P/'cached_state_xyz_native.py', {'CurrentStateXYZActor','ZeroVisualGoalStateActor','VisualGoalStateActor','make_replay_environment','annotate_reset'})
        goal=np.arange(28,dtype=np.float64)/10;closed=[];calls=[]
        def forbidden(*args,**kwargs):raise AssertionError('Zero path attempted renderer or encoder')
        class Agent:
            def sample_actions(self,**kwargs):calls.append(kwargs);return np.zeros((1,5),np.float32)
        class Replay:
            def __init__(self,env,reference,actor,renderer,**kwargs):
                self.env,self.reference,self.actor,self.renderer,self.on_reset=env,reference,actor,renderer,kwargs['on_reset'];self.count=0
            def reset(self):
                xyz=self.renderer(self.env.unwrapped,self.reference['goal_physical_state']);self.actor.set_goal_rgb(xyz)
                self.on_reset(dict(goal_rgb_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),strict_initial_and_goal_match=True,policy_warmup_steps=0))
            def step(self,action):self.count+=1
            def close(self):closed.append('paired')
        def owned_close(paired,extra):
            paired.close()
            if extra:extra()
            closed.append('owned_done')
        with tempfile.TemporaryDirectory() as directory:
            attempt=Path(directory);reference=dict(goal28=dict(values=goal.tolist(),dtype=str(goal.dtype)),goal_physical_state={'fixture':'paired_truegoal'},model_colors_camera_lights={})
            ref=attempt/'reference.json';ref.write_text(json.dumps(reference))
            appearance=SimpleNamespace(receipt={'scope':'fake only'})
            legacy=SimpleNamespace(physical_snapshot=forbidden,model_snapshot=forbidden,write=lambda path,value:Path(path).write_text(json.dumps(value)))
            modules=dict(legacy=legacy,replay=SimpleNamespace(ReplayPairWrapper=Replay,SafeGoalRenderer=forbidden),bridge=SimpleNamespace(),
                frame=SimpleNamespace(NativePCDFrame=forbidden,array_sha=lambda a:hashlib.sha256(np.asarray(a).tobytes()).hexdigest()),
                appearance=SimpleNamespace(AppearanceAllowlist=lambda *args:appearance),native_runner=SimpleNamespace(close_owned_environment=owned_close))
            runtime=SimpleNamespace(agent=Agent(),frozen=None,modules=modules,physics_runtime={'fixture':'CPU'},config=dict(visual_mode='zero192',goal_mode='latent_object_xyz',modality='pcd',rgb_recipe='fixture'))
            env=SimpleNamespace(unwrapped=SimpleNamespace(_visualize_info=False,render=forbidden))
            binding={'reference_manifest':{'1:0':dict(reset_path=str(ref),reset_file_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),seed=970100000)}}
            with patch.dict(sys.modules,{'gymnasium':SimpleNamespace(Wrapper=lambda env:SimpleNamespace()),'mujoco':SimpleNamespace(),'torch':SimpleNamespace()}):
                proxy,actor=scope['make_replay_environment'](env,runtime,binding,1,0,attempt,None)
                proxy.reset()
                for _ in range(4):proxy.step(actor.sample_actions(goal,goal,object(),0))
                proxy.close();proxy.close()
            case=json.loads((attempt/'input_case.json').read_text());pair=json.loads((attempt/'pairing.json').read_text())
            self.assertEqual((case['model_calls'],case['current_encoder_calls'],case['goal_encoder_calls'],case['total_encoder_calls']),(4,0,0,0))
            self.assertEqual(case['policy_input_shapes'],[[1,220],[1,195]]);self.assertFalse(case['frozen_state_check_applicable']);self.assertFalse(case['goal_visual_rendered'])
            self.assertFalse(pair['goal_visual_rendered']);self.assertTrue(pair['zero_visual_channels']);self.assertEqual(pair['goal_XYZ'],goal[19:22].astype(np.float32).tolist())
            self.assertFalse((attempt/'goal_source.json').exists());self.assertEqual(closed,['paired','owned_done'])
            rows=[json.loads(x) for x in (attempt/'perception.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),4)
            for row in rows:self.assertEqual(row['encoder_forward_calls'],0);self.assertTrue(row['zero_visual_channels']);self.assertFalse(row['current_frame']['rendered'])

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
        for name in ('current_latent','load_factory','prepare_reference_replay'):self.assertEqual(before[name],after[name],name)  # Explicit actor profile is covered numerically below.
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



class RL3ProfileTests(unittest.TestCase):
    def test_profile_current_and_next_contract_without_object_leak(self):
        methods=extract(P/'cached_state_xyz_native.py', {'CurrentStateXYZActor','ZeroVisualGoalStateActor','VisualGoalStateActor'})
        calls=[]
        class Agent:
            def sample_actions(self,**kwargs):calls.append(kwargs);return np.zeros((1,5),np.float32)
        goal28=np.arange(28,dtype=np.float32)
        state=np.arange(28,dtype=np.float32)+100
        latent=np.arange(192,dtype=np.float32)[None]
        for profile,dim in [('full28',220),('robot_proprio',211)]:
            actor=methods['ZeroVisualGoalStateActor'](Agent(),None,goal28,'zero192',state_profile=profile)
            actor.set_goal_rgb(goal28[19:22].copy())
            actor.sample_actions(state,goal28,temperature=0)
            c=calls[-1];self.assertEqual(c['observations'].shape,(1,dim));self.assertEqual(c['goals'].shape,(1,195))
            np.testing.assert_array_equal(c['observations'][0,192:],state[:dim-192])
            self.assertFalse(c['observations'][:,:192].any());self.assertFalse(c['goals'][:,:192].any())
            np.testing.assert_array_equal(c['goals'][0,-3:],goal28[19:22])
        actor=methods['VisualGoalStateActor'](Agent(),lambda:(latent.copy(),{}),goal28,lambda raw:(latent.copy(),{}),state_profile='robot_proprio')
        actor.set_goal_rgb(np.ones((224,224,3),np.uint8))
        actor.sample_actions(state,goal28,temperature=0);first=calls[-1]
        changed=state.copy();changed[19:28]+=12345
        actor.sample_actions(changed,goal28,temperature=0);second=calls[-1]
        np.testing.assert_array_equal(first['observations'],second['observations']);np.testing.assert_array_equal(first['goals'],second['goals'])
        self.assertEqual(actor.encoder_calls,2);self.assertEqual(actor.goal_encoder_calls,2);self.assertIsNone(actor.goal)
        self.assertEqual(first['observations'].shape,(1,211));self.assertEqual(first['goals'].shape,(1,195))
        with self.assertRaises(AssertionError):methods['CurrentStateXYZActor'](Agent(),None,goal28,'zero192',state_profile='typo')

    def test_profile_binding_rejects_full28_parent_and_gate2_as1m(self):
        with tempfile.TemporaryDirectory() as folder:
            train=Path(folder);snapshot=train/'checkpoints/update_001000000';snapshot.mkdir(parents=True)
            cfg=dict(total_updates=1000000,batch_size=1024,seed=0,input_profile='cached',goal_mode='latent_object_xyz',reward_mode='current_object_xyz',visual_mode='cache',cache={'modality':'pcd'})
            binding=dict(config=cfg,agent_config=dict(agent_name='gciql',actor_loss='ddpgbc',alpha=1,encoder=None))
            (train/'resolved.json').write_text(json.dumps(dict(total_updates=1000000,binding=binding)))
            (snapshot/'manifest.json').write_text(json.dumps(dict(counter=2)))
            config=dict(train_root=str(train),selected_updates=1000000,goal_mode='latent_object_xyz',visual_mode='cache',modality='pcd',state_profile='robot_proprio',source_sha256={},required_zero_exits=[])
            path=train/'fixture.json';path.write_text(json.dumps(config))
            with self.assertRaises(AssertionError):native['bind_completed'](path,train)
            cfg['state_profile']='robot_proprio'
            (train/'resolved.json').write_text(json.dumps(dict(total_updates=1000000,binding=binding)))
            with self.assertRaises(AssertionError):native['bind_completed'](path,train)

if __name__=='__main__':unittest.main()

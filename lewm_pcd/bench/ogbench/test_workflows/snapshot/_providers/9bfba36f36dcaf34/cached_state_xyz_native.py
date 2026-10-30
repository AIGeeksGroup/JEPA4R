"""Current220 with explicit XYZ3 or visual195 goals around the accepted GCIQL/native modules."""
import hashlib,importlib.util,json,pickle,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

BASE_NATIVE=Path('/root/autodl-tmp/workspace/users/lewm/coordination/pcd_frozen_lewm_gciql_100k_sr_109_20261010_v1/pcd_gciql_native.py')
BASE_NATIVE_SHA='36fa1e22928a4782c6a90328eae49544bfd55b4d6f51b118783ca7b17bfa8968'
assert hashlib.sha256(BASE_NATIVE.read_bytes()).hexdigest()==BASE_NATIVE_SHA
spec=importlib.util.spec_from_file_location('accepted_native_glue',BASE_NATIVE)
accepted=importlib.util.module_from_spec(spec);spec.loader.exec_module(accepted)
sha,alias,source_modules=accepted.sha,accepted.alias,accepted.source_modules


def bind_completed(config_path,train):
    config=json.loads(Path(config_path).read_text());train=Path(train).resolve()
    assert str(train)==config['train_root']
    for path,h in config['source_sha256'].items():assert sha(path)==h,path
    for path in config['required_zero_exits']:assert Path(path).read_text().strip()=='0',path
    resolved=json.loads((train/'resolved.json').read_text());binding=resolved['binding'];cfg=binding['config']
    selected=config.get('selected_updates',100000)
    assert config.get('goal_mode','object_xyz') in ('object_xyz','latent_object_xyz')
    visual_goal=config.get('goal_mode','object_xyz')=='latent_object_xyz'
    assert selected in ((1000000,) if visual_goal else (100000,500000)) and resolved['total_updates']==selected
    assert cfg['total_updates']==(1000000 if visual_goal else 100000)
    assert cfg['batch_size']==(1024 if visual_goal else 128) and cfg['seed']==0
    assert cfg['input_profile']=='cached' and cfg['goal_mode']==config.get('goal_mode','object_xyz') and cfg['reward_mode']=='current_object_xyz'
    assert cfg['visual_mode']==config['visual_mode'] and cfg['cache']['modality']==config['modality']
    state_profile=cfg.get('state_profile','full28');assert state_profile in ('full28','robot_proprio')
    assert config.get('state_profile','full28')==state_profile
    current_dim=192+(28 if state_profile=='full28' else 19)
    assert binding['agent_config']['agent_name']=='gciql' and binding['agent_config']['actor_loss']=='ddpgbc'
    assert binding['agent_config']['alpha']==1 and binding['agent_config']['encoder'] is None
    snapshot=train/f'checkpoints/update_{selected:09d}';manifest=json.loads((snapshot/'manifest.json').read_text())
    checkpoint=snapshot/f'params_{selected}.pkl'
    assert manifest['counter']==selected and sha(checkpoint)==manifest['agent_sha256']==config['agent_sha256']
    assert sha(snapshot/'run_state.pkl')==manifest['sidecar_sha256']==config['sidecar_sha256']
    with (snapshot/'run_state.pkl').open('rb') as f:side=pickle.load(f)
    assert json.loads(json.dumps(side['binding']))==binding and side['network_step']==selected+1 and side['adam_count']==selected
    assert side['budget_total']==selected
    audit=json.loads(Path(config['final_audit']).read_text());probe=json.loads(Path(config['final_probe']).read_text())
    assert audit['status']=='passed_CPU_final_acceptance' and audit['exit_code']==0 and audit['counter']==selected
    assert audit[f'original_GCIQL_fresh_factory_{current_dim}_goal195_full153_leaves_exact' if visual_goal else f'original_GCIQL_fresh_factory_{current_dim}_goal3_full153_leaves_exact']
    assert probe['status']=='passed_CPU_fresh_cached_action_probe' and probe['exit_code']==0 and probe['full153_saved_state_exact']
    assert probe['checkpoint_sha256']==config['agent_sha256']
    assert probe['restored_network_step']==selected+1 and probe['adam_count']==selected
    if selected==500000:
        parent=config['parent_transaction'];p=Path(parent['path'])
        assert resolved['initial_counter']==100000 and resolved['resumed_from']==str(p) and train!=p.parent.parent
        assert audit['segment_start_counter']==100000 and audit['parent_transaction']==parent
        assert parent['counter']==100000 and sha(p/'params_100000.pkl')==parent['agent_sha256']
        assert sha(p/'run_state.pkl')==parent['sidecar_sha256']
    if visual_goal:
        assert resolved['initial_counter']==0 and resolved['resumed_from'] is None and cfg['visual_mode'] in ('cache','zero192')
        assert probe['current_next_shape']==[4,current_dim] and probe['goal_shape']==[4,195] and probe['action_shape']==[4,5]
    for field in ('manifest','seal','terminal_manifest'):assert sha(cfg['cache'][field])==cfg['cache'][field+'_sha256']
    cache=json.loads(Path(cfg['cache']['manifest']).read_text())
    plan=dict(config=config,training_binding=binding,snapshot_manifest=manifest,cache_manifest=cache,
        environment_kwargs=config['environment_kwargs'],config_path=str(Path(config_path).resolve()),config_sha256=sha(config_path),
        resolved_sha256=sha(train/'resolved.json'),final_audit_sha256=sha(config['final_audit']),final_probe_sha256=sha(config['final_probe']),
        runtime_policy_input='current[visual192,state28],goal[visual192,XYZ3]' if visual_goal else 'current[visual192,state28],goalXYZ3',runtime_encoder_forwards_per_policy_step=0 if cfg['visual_mode']=='zero192' else 1,
        semantics='exactlast1M cached visual-state GCIQL; current-distance4cm negative reward; online current and paired goal encoders each policy step; no encoder-output reuse' if visual_goal else f'exactlast{selected//1000}K cached input GCIQL; current-distance4cm negative reward; online current encoder only; goal original28 projected19:22')
    if visual_goal:plan.update(runtime_encoder_forwards_per_policy_step=2,runtime_current_encoder_forwards_per_policy_step=1,runtime_goal_encoder_forwards_per_policy_step=1)
    if visual_goal and cfg['visual_mode']=='zero192':
        plan.update(runtime_encoder_forwards_per_policy_step=0,runtime_current_encoder_forwards_per_policy_step=0,runtime_goal_encoder_forwards_per_policy_step=0,
            runtime_policy_input='current[zeros192,state28],goal[zeros192,XYZ3]',semantics='exactlast1M matched zero-visual GCIQL; current-distance4cm negative reward; no encoder or visual rendering')
    if state_profile=='robot_proprio':plan.update(state_profile=state_profile,current_dims=current_dim,network_state_indices=list(range(19)),excluded_object_state_indices=list(range(19,28)),runtime_policy_input='current[visual192,robot_proprio19],goal[visual192,XYZ3]')
    return checkpoint,{'seed':0,'agent':binding['agent_config']},plan


def prepare_reference_replay(root,base,source,config):
    return source_modules(config)['state_adapter'].prepare_reference_replay(root,Path(config['reference_helper_root']),source)


def load_factory(checkpoint,flags,plan):
    import jax,torch,flax.serialization
    config=plan['config'];modules=source_modules(config);bridge=modules['bridge']
    source=json.loads(Path(config['wm_source_config']).read_text())
    official=Path(source['workspace'])/source['official'];pcd=Path(source['workspace'])/source['repo']/'lewm_pcd'
    sys.path[:0]=[str(official/'impls'),str(official),str(pcd),str(pcd/'third_party/stable-worldmodel')]
    from utils.flax_utils import restore_agent
    assert jax.default_backend()=='cpu'
    old_path=list(sys.path);sys.path.insert(0,config['training_code_root'])
    try:training=alias('cached_native_training',Path(config['training_code_root'])/'train.py',config['source_sha256'][config['training_code_root']+'/train.py'])
    finally:sys.path[:]=old_path
    training_config=plan['training_binding']['config'];cfg=dict(flags['agent'])
    cfg['actor_hidden_dims']=tuple(cfg['actor_hidden_dims']);cfg['value_hidden_dims']=tuple(cfg['value_hidden_dims'])
    template=training.create_agent(training_config,cfg)
    selected=config.get('selected_updates',100000)
    agent=restore_agent(template,str(checkpoint.parent),selected)
    with checkpoint.open('rb') as f:saved=pickle.load(f)['agent']
    actual=flax.serialization.to_state_dict(agent)
    aa,ta=jax.tree_util.tree_flatten(saved);bb,tb=jax.tree_util.tree_flatten(actual)
    assert ta==tb and len(bb)==153
    assert all(np.asarray(x).dtype==np.asarray(y).dtype and np.asarray(x).shape==np.asarray(y).shape and np.isfinite(y).all() and np.array_equal(x,y) for x,y in zip(aa,bb))
    assert int(agent.network.step)==selected+1 and int(agent.network.opt_state[0].count)==selected
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    cache=plan['cache_manifest'];frozen=None;snapshot=None
    if config['visual_mode']=='cache':
        precision=cache['precision'];assert precision['precision']=='fp32' and precision['torch_float32_matmul_precision']=='highest'
        assert precision['cuda_matmul_allow_tf32'] is False and precision['cudnn_allow_tf32']==torch.backends.cudnn.allow_tf32
        modality=config['modality'];weight=Path(source['workspace'])/source['routes'][modality]['checkpoint']
        assert sha(weight)==cache['source_checkpoint_sha256']
        assert sha(pcd/'utils.py')==cache['preprocess_source_sha256']
        frozen=bridge.FrozenLatents(source,modality,reader=None,device='cuda')
        assert modules['hash_helper'].module_hash(frozen.encoder)==cache['encoder_hashes']['trainer_numpy_dtype']
        assert all(not v.requires_grad for v in frozen.world_model.parameters())
        assert not frozen.world_model.training
        snapshot=frozen.snapshot()
    physics=modules['native_runner'].configure_numpy_physics_runtime()
    return SimpleNamespace(agent=agent,frozen=frozen,frozen_snapshot=snapshot,modules=modules,config=config,
        physics_runtime=physics,encoder_source_binding=cache['encoder_hashes']),cfg


class CurrentStateXYZActor:
    """Shared replay lifecycle; legacy set_goal_rgb method receives XYZ3 only."""
    def __init__(self,agent,current_latent,carrier_goal,visual_mode,on_action=None,state_profile='full28'):
        assert visual_mode in ('cache','zero192')
        assert state_profile in ('full28','robot_proprio')
        self.state_profile=state_profile;self.state_dim=28 if state_profile=='full28' else 19
        self.agent,self.current_latent,self.carrier_goal=agent,current_latent,np.array(carrier_goal,copy=True)
        self.visual_mode,self.on_action=visual_mode,on_action;self.reset()
    def reset(self):self.goal=None;self.model_calls=0;self.encoder_calls=0
    def set_goal_rgb(self,xyz):
        xyz=np.asarray(xyz);assert xyz.shape==(3,) and xyz.dtype==np.float32 and np.isfinite(xyz).all()
        np.testing.assert_array_equal(xyz,self.carrier_goal[19:22].astype(np.float32))
        self.goal=xyz.copy()[None]  # Numeric target only; no goal image or encoder.
    def sample_actions(self,observations,goals=None,seed=None,temperature=1.0):
        assert temperature==0 and self.goal is not None
        state=np.asarray(observations);assert state.shape==(28,) and np.isfinite(state).all()
        assert np.array_equal(goals,self.carrier_goal)
        if self.visual_mode=='zero192':latent=np.zeros((1,192),np.float32);receipt={'visual_mode':'zero192','rendered':False}
        else:
            latent,receipt=self.current_latent();self.encoder_calls+=1
        assert latent.shape==(1,192) and latent.dtype==np.float32 and np.isfinite(latent).all()
        current=np.concatenate((latent,state[:self.state_dim].astype(np.float32)[None]),axis=-1)
        action=np.asarray(self.agent.sample_actions(observations=current,goals=self.goal,seed=seed,temperature=temperature))
        assert action.shape==(1,5) and action.dtype==np.float32 and np.isfinite(action).all() and np.abs(action).max()<=1
        self.model_calls+=1
        if self.on_action:self.on_action(self,current,self.goal,action[0],receipt)
        return action[0]


class ZeroVisualGoalStateActor(CurrentStateXYZActor):
    """Same original actor path and220/195 capacity; both visual blocks are zeros."""
    def reset(self):
        super().reset();self.goal_encoder_calls=0
    def set_goal_rgb(self,xyz):
        super().set_goal_rgb(xyz)
        self.goal=np.concatenate((np.zeros((1,192),np.float32),self.goal),axis=-1)


class VisualGoalStateActor(CurrentStateXYZActor):
    """Retain raw paired goal only; recompute goal features before every action."""
    def __init__(self,agent,current_latent,carrier_goal,goal_encoder,on_action=None,goal_source=None,state_profile='full28'):
        self.goal_encoder=goal_encoder;self.goal_source=goal_source if goal_source is not None else {}
        super().__init__(agent,current_latent,carrier_goal,'cache',on_action,state_profile)
    def reset(self):
        super().reset();self.goal_encoder_calls=0;self.raw_goal=None;self.goal_identity=None;self.goal_source_identity=None
    def set_goal_rgb(self,raw_goal):
        assert self.raw_goal is None and self.goal_encoder_calls==0
        raw_goal=np.asarray(raw_goal)
        assert ((raw_goal.shape==(224,224,3) and raw_goal.dtype==np.uint8) or (raw_goal.shape==(1200,6) and raw_goal.dtype==np.float32)) and np.isfinite(raw_goal).all()
        self.raw_goal=raw_goal.copy();self.raw_goal.flags.writeable=False
        xyz=self.carrier_goal[19:22].astype(np.float32)[None]
        self.goal_source_identity=dict(goal_frame=dict(self.goal_source),raw_goal_sha256=hashlib.sha256(self.raw_goal.tobytes()).hexdigest(),
            raw_goal_shape=list(self.raw_goal.shape),raw_goal_dtype=str(self.raw_goal.dtype),
            goal_XYZ=xyz[0].tolist(),goal_XYZ_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),
            semantics='fixed raw paired goal and XYZ3 only; target encoder executes on every policy call')
    def sample_actions(self,observations,goals=None,seed=None,temperature=1.0):
        assert temperature==0 and self.raw_goal is not None and np.array_equal(goals,self.carrier_goal)
        assert not self.raw_goal.flags.writeable and hashlib.sha256(self.raw_goal.tobytes()).hexdigest()==self.goal_source_identity['raw_goal_sha256']
        latent,receipt=self.goal_encoder(self.raw_goal)
        assert latent.shape==(1,192) and latent.dtype==np.float32 and np.isfinite(latent).all()
        self.goal=np.concatenate((latent,self.carrier_goal[19:22].astype(np.float32)[None]),axis=-1)
        self.goal_encoder_calls+=1
        self.goal_identity=dict(receipt,goal_latent_sha256=hashlib.sha256(latent.tobytes()).hexdigest(),
            goal_input_sha256=hashlib.sha256(self.goal.tobytes()).hexdigest(),goal_input_shape=[1,195])
        try:return super().sample_actions(observations,goals,seed,temperature)
        finally:self.goal=None  # Never reuse an encoder output on the next policy call.


def make_replay_environment(env,runtime,binding,task,index,attempt,base):
    import gymnasium as gym,mujoco,torch
    modules=runtime.modules;legacy=modules['legacy'];replay=modules['replay'];bridge=modules['bridge'];config=runtime.config
    record=binding['reference_manifest'][f'{task}:{index}'];reference_path=Path(record['reset_path'])
    assert sha(reference_path)==record['reset_file_sha256'];reference=json.loads(reference_path.read_text())
    assert not env.unwrapped._visualize_info
    visual_goal=config.get('goal_mode','object_xyz')=='latent_object_xyz'
    render_goal=visual_goal and config['visual_mode']=='cache'
    frame=None;perception=(attempt/'perception.jsonl').open('x',buffering=1)
    try:
        if config['visual_mode']=='cache' and config['modality']=='pcd':
            frame=modules['frame'].NativePCDFrame(config['pcd_core'],config['pcd_provider'],record['seed'])
        def current_latent():
            u=env.unwrapped;before=(legacy.physical_snapshot(u),legacy.model_snapshot(u),json.loads(json.dumps(u.np_random.bit_generator.state)))
            assert int(u._model.vis.quality.offsamples)==4
            rgb=u.render(camera='front_pixels');assert rgb.shape==(224,224,3) and rgb.dtype==np.uint8
            if frame is not None:
                cloud=frame.capture(u,u._data,rgb,'current');replay.SingleFrameAgent.validate_pcd(cloud)
                rows={'pcd_xyz':cloud[:,:3].copy()[None],'pcd_rgb':cloud[:,3:].astype(np.uint8)[None]};receipt=frame.last_receipts['current']
            else:
                rows={'pixels':rgb.copy()[None]};receipt={'phase':'current','shape':[224,224,3],'RGB_AA4_sha256':modules['frame'].array_sha(rgb),
                    'source_pose_sha256':{k:modules['frame'].array_sha(getattr(u._data,k)) for k in ['qpos','qvel','ctrl','mocap_pos','mocap_quat']},'source_time':float(u._data.time)}
            assert before==(legacy.physical_snapshot(u),legacy.model_snapshot(u),u.np_random.bit_generator.state)
            inputs=bridge.preprocess_rows(runtime.frozen.preprocessor,rows,runtime.frozen.input_key).to(runtime.frozen.device)
            with torch.inference_mode():latent=runtime.frozen.encoder(inputs).logits
            return latent.cpu().numpy(),receipt
        def on_action(actor,current,goal,action,receipt):
            xyz=goal[:,-3:]
            row={'model_call':actor.model_calls,'encoder_forward_calls':actor.encoder_calls,'goal_encoder_calls':0,
                'current_frame':receipt,'current_input_sha256':modules['frame'].array_sha(current),'goal_XYZ':xyz[0].tolist(),
                'goal_XYZ_sha256':modules['frame'].array_sha(xyz),'sampled_action':action.tolist(),'sampled_action_already_GCIQL_clipped':True,
                'official_evaluate_clip_follows':True,'policy_input_shapes':[[1,current.shape[1]],[1,3]],'precision':'float32',
                'visual_mode':config['visual_mode'],'runtime_feature_cache':False}
            if visual_goal and not render_goal:
                assert not np.any(current[:,:192]) and not np.any(goal[:,:192])
                row.update(current_encoder_calls=0,goal_encoder_calls=0,encoder_forward_calls=0,policy_input_shapes=[[1,192+actor.state_dim],[1,195]],
                    zero_visual_channels=True,goal_visual_rendered=False,goal_input_sha256=modules['frame'].array_sha(goal),
                    offline_feature_cache=False,current_feature_reuse=False,goal_feature_reuse=False)
            elif visual_goal:
                assert hashlib.sha256(goal.tobytes()).hexdigest()==actor.goal_identity['goal_input_sha256']
                row.update(encoder_forward_calls=actor.encoder_calls+actor.goal_encoder_calls,current_encoder_calls=actor.encoder_calls,
                    goal_encoder_calls=actor.goal_encoder_calls,policy_input_shapes=[[1,192+actor.state_dim],[1,195]],
                    offline_feature_cache=False,current_feature_reuse=False,goal_feature_reuse=False,
                    goal_raw_sha256=actor.goal_source_identity['raw_goal_sha256'],goal_identity=actor.goal_identity)
            if actor.state_profile=='robot_proprio':row.update(state_profile='robot_proprio',network_state_indices=list(range(19)),excluded_object_state_indices=list(range(19,28)))
            perception.write(json.dumps(row,allow_nan=False)+'\n')
        g=reference['goal28'];goal28=np.asarray(g['values'],dtype=np.dtype(g['dtype']))
        goal_capture={}
        def goal_frame_transform(u,data,rgb):
            if frame is not None:
                raw=frame.capture(u,data,rgb,'goal');goal_capture.update(frame.last_receipts['goal'])
            else:
                raw=rgb
                goal_capture.update(phase='goal',shape=[224,224,3],RGB_AA4_sha256=modules['frame'].array_sha(rgb),
                    source_pose_sha256={k:modules['frame'].array_sha(getattr(data,k)) for k in ['qpos','qvel','ctrl','mocap_pos','mocap_quat']},source_time=float(data.time))
            return raw
        def encode_goal(raw):
            if frame is not None:
                replay.SingleFrameAgent.validate_pcd(raw)
                rows={'pcd_xyz':raw[:,:3].copy()[None],'pcd_rgb':raw[:,3:].astype(np.uint8)[None]}
            else:
                assert raw.shape==(224,224,3) and raw.dtype==np.uint8
                rows={'pixels':raw.copy()[None]}
            inputs=bridge.preprocess_rows(runtime.frozen.preprocessor,rows,runtime.frozen.input_key).to(runtime.frozen.device)
            with torch.inference_mode():latent=runtime.frozen.encoder(inputs).logits
            return latent.cpu().numpy(),dict(goal_frame=dict(goal_capture),raw_goal_sha256=modules['frame'].array_sha(raw))
        if visual_goal and not render_goal:
            actor=ZeroVisualGoalStateActor(runtime.agent,None,goal28,'zero192',on_action,state_profile=config.get('state_profile','full28'))
            goal_renderer=lambda u,state:goal28[19:22].astype(np.float32)
        elif visual_goal:
            actor=VisualGoalStateActor(runtime.agent,current_latent,goal28,encode_goal,on_action,goal_capture,state_profile=config.get('state_profile','full28'))
            goal_renderer=replay.SafeGoalRenderer(frame_transform=goal_frame_transform)
        else:
            actor=CurrentStateXYZActor(runtime.agent,current_latent if config['visual_mode']=='cache' else None,goal28,config['visual_mode'],on_action,state_profile=config.get('state_profile','full28'))
            goal_renderer=lambda u,state:goal28[19:22].astype(np.float32)
        appearance=modules['appearance'].AppearanceAllowlist(reference['model_colors_camera_lights'],config['rgb_recipe'],legacy.model_snapshot,
            lambda model,gid:mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid))
        def on_reset(receipt):
            receipt=dict(receipt);receipt['goal_visual_raw_sha256' if render_goal else 'goal_XYZ_bytes_sha256']=receipt.pop('goal_rgb_sha256')
            receipt.update(reference_reset_file_sha256=record['reset_file_sha256'],goal_mode='latent_object_xyz' if visual_goal else 'object_xyz',goal_visual_rendered=render_goal,
                whole_model_visual_exact_claim=False,goal_XYZ=goal28[19:22].astype(np.float32).tolist())
            if visual_goal and not render_goal:
                receipt['zero_visual_channels']=True
            elif visual_goal:
                assert actor.goal_encoder_calls==0 and actor.goal is None and actor.goal_source_identity is not None
                receipt['goal_source']=actor.goal_source_identity
                legacy.write(attempt/'goal_source.json',actor.goal_source_identity)
            legacy.write(attempt/'pairing.json',receipt);legacy.write(attempt/'appearance.json',appearance.receipt)
        paired=replay.ReplayPairWrapper(env,reference,actor,goal_renderer,
            snapshot_fn=legacy.physical_snapshot,model_snapshot_fn=appearance,on_reset=on_reset)
        proxy=gym.Wrapper(env);proxy.reset=paired.reset;proxy.step=paired.step;closed=False
        def close():
            nonlocal closed
            if closed:return
            closed=True
            try:modules['native_runner'].close_owned_environment(paired,frame.close if frame is not None else None)
            finally:perception.close()
            if runtime.frozen is not None:runtime.frozen.check_frozen(runtime.frozen_snapshot)
            assert paired.count==actor.model_calls and actor.encoder_calls==(actor.model_calls if config['visual_mode']=='cache' else 0)
            if render_goal:assert actor.goal_encoder_calls==actor.model_calls and actor.goal is None
            elif visual_goal:assert actor.goal_encoder_calls==0 and actor.goal.shape==(1,195) and not np.any(actor.goal[:,:192])
            case=dict(policy_steps=paired.count,model_calls=actor.model_calls,
                current_encoder_calls=actor.encoder_calls,goal_encoder_calls=actor.goal_encoder_calls if visual_goal else 0,visual_mode=config['visual_mode'],
                frozen_full_state_exact=runtime.frozen is not None,frozen_state_check_applicable=runtime.frozen is not None,
                appearance_sha256=sha(attempt/'appearance.json'),perception_sha256=sha(attempt/'perception.jsonl'),
                physical_numpy_runtime=runtime.physics_runtime,whole_model_visual_exact_claim=False)
            if visual_goal and not render_goal:case.update(total_encoder_calls=0,zero_visual_channels=True,goal_visual_rendered=False,
                policy_input_shapes=[[1,192+actor.state_dim],[1,195]],offline_feature_cache=False,current_feature_reuse=False,goal_feature_reuse=False)
            elif visual_goal:case.update(total_encoder_calls=actor.encoder_calls+actor.goal_encoder_calls,goal_source_sha256=sha(attempt/'goal_source.json'),
                policy_input_shapes=[[1,192+actor.state_dim],[1,195]],offline_feature_cache=False,current_feature_reuse=False,goal_feature_reuse=False)
            if actor.state_profile=='robot_proprio':case.update(state_profile='robot_proprio',network_state_indices=list(range(19)),excluded_object_state_indices=list(range(19,28)))
            legacy.write(attempt/'input_case.json',case)
        proxy.close=close;return proxy,actor
    except BaseException:
        if frame is not None:frame.close()
        perception.close();raise


def annotate_reset(attempt):
    reset=json.loads((attempt/'reset.json').read_text());pair=json.loads((attempt/'pairing.json').read_text());case=json.loads((attempt/'input_case.json').read_text())
    visual_goal=pair['goal_mode']=='latent_object_xyz'
    assert pair['strict_initial_and_goal_match'] and pair['policy_warmup_steps']==0 and case['goal_encoder_calls']==(case['policy_steps'] if visual_goal and case['visual_mode']=='cache' else 0)
    reset.update(pairing_limitation='exact reference physics/private/RNG with declared markeralpha visual exception; modelgoal is real paired visual192+XYZ3' if visual_goal else 'exact reference physics/private/RNG with declared markeralpha visual exception; modelgoal isXYZ3 only',
        reference_replay=pair,whole_model_visual_exact_claim=False,input_case=case)
    if visual_goal and case['visual_mode']=='zero192':reset['pairing_limitation']='exact reference physics/private/RNG with declared marker-alpha exception; modelgoal is zeros192 plus true goalXYZ3; no visual rendering or encoder'
    tmp=attempt/'reset.json.tmp';tmp.write_text(json.dumps(reset,allow_nan=False));tmp.replace(attempt/'reset.json')
    return sha(attempt/'pairing.json')

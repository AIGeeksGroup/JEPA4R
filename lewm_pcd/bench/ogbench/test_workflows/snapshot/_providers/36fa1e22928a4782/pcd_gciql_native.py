"""Thin frozen-LeWM/Flax actor and sealed native replay/provider glue; no evaluation loop."""
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import numpy as np


def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''): h.update(block)
    return h.hexdigest()


def alias(name,path,expected):
    assert sha(path)==expected,str(path)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_binding(config,resolved,manifest,audit,reload):
    binding=resolved['binding']; training=binding['config']; agent=binding['agent_config']
    assert resolved['total_updates']==training['total_updates']==manifest['counter']==100000
    assert training['batch_size']==agent['batch_size']==128 and training['seed']==0
    for key,value in {'agent_name':'gciql','alpha':1.0,'actor_loss':'ddpgbc','encoder':None,'frame_stack':None,
                      'p_aug':0.,'lr':.0003,'actor_hidden_dims':[512,512,512],'value_hidden_dims':[512,512,512]}.items():
        assert agent[key]==value,key
    assert manifest['agent_sha256']==config['agent_sha256'] and manifest['sidecar_sha256']==config['sidecar_sha256']
    assert audit['exit_code']==0 and audit['completed_updates']==100000 and audit['batch_size']==128
    assert audit['network_step']==100001 and audit['adam_count']==100000
    assert reload['status']=='passed' and reload['exit_code']==0 and reload['full_state_leaf_count']==153
    assert reload['full_saved_state_exact'] and reload['network_step']==100001 and reload['adam_count']==100000
    assert reload['source_dependency_binding']==binding
    assert reload['wm_before']==reload['wm_after']==binding['frozen_full_state']==config['frozen_full_state']
    assert training['precision']=={'dtype':'float32','matmul_precision':'highest','cuda_matmul_allow_tf32':False,'cudnn_policy':'unchanged'}


def bind_completed(config_path,train):
    config=json.loads(Path(config_path).read_text()); train=Path(train).resolve()
    assert str(train)==config['train_root']
    for path,expected in config['source_sha256'].items(): assert sha(path)==expected,path
    for path in config['required_zero_exits']: assert Path(path).read_text().strip()=='0',path
    snapshot=train/'checkpoints/update_000100000'
    manifest=json.loads((snapshot/'manifest.json').read_text())
    checkpoint=snapshot/'params_100000.pkl'
    assert sha(checkpoint)==config['agent_sha256'] and sha(snapshot/'run_state.pkl')==config['sidecar_sha256']
    resolved=json.loads((train/'resolved.json').read_text())
    audit=json.loads(Path(config['final_audit']).read_text()); reload=json.loads(Path(config['final_reload']).read_text())
    validate_binding(config,resolved,manifest,audit,reload)
    with (snapshot/'run_state.pkl').open('rb') as f: sidecar=pickle.load(f)
    assert json.loads(json.dumps(sidecar['binding'],allow_nan=False))==resolved['binding'] and sidecar['budget_total']==100000
    assert sidecar['network_step']==100001 and sidecar['adam_count']==100000
    status=json.loads((train/'status.json').read_text())
    assert status['exit_code']==0 and status['completed_updates']==100000 and status['target_updates']==100000
    plan=dict(config=config,training_binding=resolved['binding'],snapshot_manifest=manifest,
        environment_kwargs=config['environment_kwargs'],config_path=str(Path(config_path).resolve()),
        config_sha256=sha(config_path),resolved_sha256=sha(train/'resolved.json'),
        final_audit_sha256=sha(config['final_audit']),final_reload_sha256=sha(config['final_reload']),
        semantics='exact-last100K B128; frozen fixed5-trained PCD LeWM; original Flax GCIQL; no training evaluation RNG is fabricated')
    return checkpoint,{'seed':0,'agent':resolved['binding']['agent_config']},plan


def source_modules(config):
    return {name:alias('pcd_native_'+name,record['path'],record['sha256']) for name,record in config['modules'].items()}


def prepare_reference_replay(root,base,source):
    config=json.loads(Path(__file__).with_name('config.json').read_text())
    modules=source_modules(config)
    return modules['state_adapter'].prepare_reference_replay(root,base,source)


def load_factory(checkpoint,flags,plan):
    import torch
    import jax
    import flax.serialization
    config=plan['config']; modules=source_modules(config)
    bridge=modules['bridge']
    source=json.loads(Path(config['wm_source_config']).read_text())
    official=Path(source['workspace'])/source['official']; pcd=Path(source['workspace'])/source['repo']/'lewm_pcd'
    # Source locks are reused from accepted training reports; no full NPZ/shard sweep or ReadPool.
    sys.path[:0]=[str(official/'impls'),str(official),str(pcd),str(pcd/'third_party/stable-worldmodel')]
    from agents.gciql import GCIQLAgent
    from utils.flax_utils import restore_agent
    assert jax.default_backend()=='cpu' and Path(sys.modules['agents.gciql'].__file__).resolve()==official/'impls/agents/gciql.py'
    cfg=dict(flags['agent']); cfg['actor_hidden_dims']=tuple(cfg['actor_hidden_dims']); cfg['value_hidden_dims']=tuple(cfg['value_hidden_dims'])
    template=GCIQLAgent.create(flags['seed'],np.zeros((1,192),np.float32),np.zeros((1,5),np.float32),cfg)
    agent=restore_agent(template,str(checkpoint.parent),100000)
    with checkpoint.open('rb') as f: saved=pickle.load(f)['agent']
    actual=flax.serialization.to_state_dict(agent)
    a,ta=jax.tree_util.tree_flatten(saved); b,tb=jax.tree_util.tree_flatten(actual)
    assert ta==tb and len(b)==153
    assert all(np.asarray(x).dtype==np.asarray(y).dtype and np.asarray(x).shape==np.asarray(y).shape
               and np.isfinite(y).all() and np.array_equal(x,y) for x,y in zip(a,b))
    assert int(agent.network.step)==100001 and int(agent.network.opt_state[0].count)==100000
    torch.set_num_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False
    assert torch.backends.cudnn.allow_tf32==plan['training_binding']['runtime']['cudnn_tf32']
    assert torch.cuda.is_available()
    frozen=bridge.FrozenLatents(source,'pcd',reader=None,device='cuda')
    assert sorted({str(v.dtype) for v in frozen.world_model.state_dict().values() if torch.is_floating_point(v)})==['torch.float32']
    # Reuse the training hash helper without constructing its samplers or eight workers.
    old_path=list(sys.path); sys.path.insert(0,config['training_code_root'])
    try: training=alias('pcd_native_training_helpers',Path(config['training_code_root'])/'train.py',config['source_sha256'][config['training_code_root']+'/train.py'])
    finally: sys.path[:]=old_path
    assert training.frozen_hash(frozen)==config['frozen_full_state']
    training.frozen_guard(frozen)
    physics=modules['native_runner'].configure_numpy_physics_runtime()
    return SimpleNamespace(agent=agent,frozen=frozen,modules=modules,config=config,training=training,
        original_agent_state=actual,frozen_snapshot=frozen.snapshot(),physics_runtime=physics),cfg


class FlaxFrozenPCDActor:
    """State28 is an environment carrier; only real encoded current192/goal192 reach Flax."""
    def __init__(self,agent,current,transform,encoder,validate_pcd,carrier_goal,on_action=None):
        self.agent,self.current,self.transform,self.encoder=agent,current,transform,encoder
        self.validate_pcd=validate_pcd;self.carrier_goal=np.array(carrier_goal,copy=True);self.on_action=on_action;self.reset()
    def reset(self):
        self.model_calls=0;self.encoder_calls=0;self.goal=None
    def set_goal_rgb(self,cloud):
        self.validate_pcd(cloud)
        self.goal=self.transform(np.array(cloud,copy=True))  # raw/preprocessed goal only; no latent cache.
    def sample_actions(self,observations,goals=None,seed=None,temperature=1.0):
        import torch
        assert temperature==0 and self.goal is not None
        assert np.asarray(observations).shape==(28,) and np.isfinite(observations).all()
        assert np.array_equal(goals,self.carrier_goal)
        cloud=np.asarray(self.current());self.validate_pcd(cloud);current=self.transform(cloud.copy())
        assert current.shape==self.goal.shape==(1,1200,6) and current.dtype==self.goal.dtype==torch.float32
        with torch.inference_mode():
            z=self.encoder(current).logits;zg=self.encoder(self.goal).logits
        self.encoder_calls+=2
        assert z.shape==zg.shape==(1,192) and z.dtype==zg.dtype==torch.float32
        assert torch.isfinite(z).all() and torch.isfinite(zg).all()
        action=np.asarray(self.agent.sample_actions(observations=z.cpu().numpy(),goals=zg.cpu().numpy(),seed=seed,temperature=temperature))
        assert action.shape==(1,5) and action.dtype==np.float32 and np.isfinite(action).all() and np.abs(action).max()<=1
        self.model_calls+=1
        if self.on_action: self.on_action(self,cloud,z,zg,action[0])
        return action[0]


def make_replay_environment(env,runtime,binding,task,index,attempt,base):
    import gymnasium as gym
    import mujoco
    modules=runtime.modules; legacy=modules['legacy']; replay=modules['replay']; bridge=modules['bridge']
    record=binding['reference_manifest'][f'{task}:{index}']; reference_path=Path(record['reset_path'])
    assert sha(reference_path)==record['reset_file_sha256']; reference=json.loads(reference_path.read_text())
    assert not env.unwrapped._visualize_info if hasattr(env.unwrapped,'_visualize_info') else True
    frame=modules['frame'].NativePCDFrame(runtime.config['pcd_core'],runtime.config['pcd_provider'],record['seed'])
    perception=(attempt/'perception.jsonl').open('x',buffering=1)
    try:
        def current():
            u=env.unwrapped;before=(legacy.physical_snapshot(u),legacy.model_snapshot(u),json.loads(json.dumps(u.np_random.bit_generator.state)))
            cloud=frame.capture(u,u._data,u.render(camera='front_pixels'),'current')
            assert before==(legacy.physical_snapshot(u),legacy.model_snapshot(u),u.np_random.bit_generator.state)
            return cloud
        def transform(cloud):
            rows={'pcd_xyz':cloud[:,:3].copy()[None],'pcd_rgb':cloud[:,3:].astype(np.uint8)[None]}
            return bridge.preprocess_rows(runtime.frozen.preprocessor,rows,runtime.frozen.input_key).to(runtime.frozen.device)
        def action_receipt(actor,cloud,z,zg,action):
            runtime.training.frozen_guard(runtime.frozen)
            perception.write(json.dumps({'model_call':actor.model_calls,'encoder_forward_calls':actor.encoder_calls,
                'current_frame':frame.last_receipts['current'],'goal_frame':frame.last_receipts['goal'],
                'current_latent_sha256':modules['frame'].array_sha(z.cpu().numpy()),
                'goal_latent_sha256':modules['frame'].array_sha(zg.cpu().numpy()),'sampled_action':action.tolist(),
                'sampled_action_already_GCIQL_clipped':True,'official_evaluate_clip_follows':True,
                'no_encoder_feature_cache':True,'policy_input_shapes':[[1,192],[1,192]],'precision':'float32'},allow_nan=False)+'\n')
        goal_record=reference['goal28'];carrier_goal=np.asarray(goal_record['values'],dtype=np.dtype(goal_record['dtype']))
        actor=FlaxFrozenPCDActor(runtime.agent,current,transform,runtime.frozen.encoder,replay.SingleFrameAgent.validate_pcd,carrier_goal,action_receipt)
        appearance=modules['appearance'].AppearanceAllowlist(reference['model_colors_camera_lights'],runtime.config['rgb_recipe'],legacy.model_snapshot,
            lambda model,gid:mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid))
        renderer=replay.SafeGoalRenderer(frame_transform=lambda u,data,rgb:frame.capture(u,data,rgb,'goal'))
        def on_reset(receipt):
            receipt=dict(receipt);receipt['goal_pcd_sha256']=receipt.pop('goal_rgb_sha256')
            receipt.update(reference_reset_file_sha256=record['reset_file_sha256'],observation_modality='pcd',
                PCD_goal_frame=frame.last_receipts['goal'],whole_model_visual_exact_claim=False)
            legacy.write(attempt/'pairing.json',receipt);legacy.write(attempt/'appearance.json',appearance.receipt)
        paired=replay.ReplayPairWrapper(env,reference,actor,renderer,snapshot_fn=legacy.physical_snapshot,model_snapshot_fn=appearance,on_reset=on_reset)
        proxy=gym.Wrapper(env);proxy.reset=paired.reset;proxy.step=paired.step;closed=False
        def close():
            nonlocal closed
            if closed:return
            closed=True
            try:
                modules['native_runner'].close_owned_environment(paired,frame.close)
            finally: perception.close()
            runtime.frozen.check_frozen(runtime.frozen_snapshot)
            assert runtime.training.frozen_hash(runtime.frozen)==runtime.config['frozen_full_state']
            assert paired.count==actor.model_calls and actor.encoder_calls==2*actor.model_calls
            legacy.write(attempt/'pcd_case.json',dict(policy_steps=paired.count,model_calls=actor.model_calls,
                encoder_forward_calls=actor.encoder_calls,encoder_forward_calls_per_policy_step=2,frozen_full_state_exact=True,
                appearance_sha256=sha(attempt/'appearance.json'),perception_sha256=sha(attempt/'perception.jsonl'),
                physical_numpy_runtime=runtime.physics_runtime,whole_model_visual_exact_claim=False))
        proxy.close=close
        return proxy,actor
    except BaseException:
        frame.close();perception.close();raise


def annotate_reset(attempt):
    reset=json.loads((attempt/'reset.json').read_text());pair=json.loads((attempt/'pairing.json').read_text());case=json.loads((attempt/'pcd_case.json').read_text())
    assert pair['strict_initial_and_goal_match'] and pair['policy_warmup_steps']==0 and case['frozen_full_state_exact']
    reset.update(pairing_limitation='exact reference physics/private/RNG with the declared sealed nonphysical marker-alpha visual exception',
        reference_replay=pair,whole_model_visual_exact_claim=False,pcd_case=case)
    tmp=attempt/'reset.json.tmp';tmp.write_text(json.dumps(reset,allow_nan=False));tmp.replace(attempt/'reset.json')
    return sha(attempt/'pairing.json')

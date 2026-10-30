"""Create one immutable native binding from completed cached training CPU evidence."""
import argparse,hashlib,json
from pathlib import Path
W=Path('/root/autodl-tmp/workspace/users/lewm');Q=W/'coordination/cached_lewm_state_gciql_xyz_109_20261012_v1'
B=W/'ogbench-official-single-play-20261006';N=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    cfg=json.loads(a.training_config.read_text());R=a.train_root or Path(cfg['output']);selected=a.selected_updates
    assert cfg['goal_mode'] in ('object_xyz','latent_object_xyz')
    visual_goal=cfg['goal_mode']=='latent_object_xyz'
    state_profile=cfg.get('state_profile','full28');assert state_profile in ('full28','robot_proprio')
    current_dim=192+(28 if state_profile=='full28' else 19)
    training_code=N.parent/'core_v1' if visual_goal else Q/'core_v1'
    resolved=json.loads((R/'resolved.json').read_text())
    assert cfg==resolved['binding']['config'] and cfg['input_profile']=='cached'
    assert selected in ((1000000,) if visual_goal else (100000,500000)) and resolved['total_updates']==selected
    assert cfg['total_updates']==(1000000 if visual_goal else 100000)
    audit=json.loads(a.final_audit.read_text());probe=json.loads(a.final_probe.read_text())
    assert audit['status']=='passed_CPU_final_acceptance' and audit['counter']==selected and audit['exit_code']==0
    assert probe['status']=='passed_CPU_fresh_cached_action_probe' and probe['full153_saved_state_exact'] and probe['exit_code']==0
    old_path=W/'coordination/pcd_frozen_lewm_gciql_100k_sr_109_20261010_v1/config.json';old=json.loads(old_path.read_text())
    modules=old['modules'];code=B/'official_lewm_gcbc_20261006';locked=Q/'native_v1/locked_dependencies'
    for key,name in [('legacy','official_state_sr.py'),('replay','native_single_frame_adapter.py')]:
        assert sha(locked/name)==modules[key]['sha256']
        modules[key]['path']=str(locked/name)
    assert sha(locked/'native_runtime_helpers.py')=='a83bcf181b00d57fe5b41ef4e61c42dded0561b1ec9d4dc2c7a1c64dfdb8bc3a'
    previous_runtime=modules['native_runner']['path']
    modules['native_runner']={'path':str(locked/'native_runtime_helpers.py'),'sha256':sha(locked/'native_runtime_helpers.py')}
    modules['hash_helper']={'path':str(code/'trainer.py'),'sha256':sha(code/'trainer.py')}
    sources={path:h for path,h in old['source_sha256'].items() if not any(s in path for s in ['pcd_frozen_lewm_gciql_100k_b128','/resolved.json'])}
    for path in (B/'native_single_frame_adapter.py',B/'official_state_sr.py'):sources.pop(str(path),None)
    sources.pop(previous_runtime,None)
    sources.update({str(W/path):h for path,h in resolved['binding']['source_locks'].items() if path.endswith(('.py','.json','.yaml','.sh'))})
    for record in modules.values():sources[record['path']]=record['sha256']
    extra=[code/'dataset.py',a.training_config,R/'resolved.json',a.final_audit,a.final_probe,
        W/'coordination/pcd_frozen_lewm_gciql_100k_sr_109_20261010_v1/pcd_gciql_native.py']
    if visual_goal:extra += [Path(cfg['runtime_profile'])]
    extra += [training_code/x for x in (resolved['binding']['code'] if visual_goal else ['train.py','reader.py','gciql.py'])]
    extra += [N/x for x in ['official_state_sr.py','cached_state_xyz_native.py','test_cached_native.py','run_stage.sh','prepare_binding.py']]
    for path in extra:sources[str(path)]=sha(path)
    snap=R/f'checkpoints/update_{selected:09d}';manifest=json.loads((snap/'manifest.json').read_text())
    assert manifest['counter']==selected and sha(snap/f'params_{selected}.pkl')==manifest['agent_sha256']==probe['checkpoint_sha256']
    assert sha(snap/'run_state.pkl')==manifest['sidecar_sha256']
    exits=[R/'trainer_exit_code',a.launcher/'child_exit_code',a.launcher/'driver_exit_code',a.outer_exit]
    if selected==500000:
        assert a.train_root is not None and a.audit_exit is not None and a.probe_exit is not None
        parent=audit['parent_transaction'];p=Path(parent['path'])
        assert resolved['initial_counter']==100000 and resolved['resumed_from']==str(p) and R!=Path(cfg['output'])
        assert audit['segment_start_counter']==parent['counter']==100000
        assert sha(p/'params_100000.pkl')==parent['agent_sha256'] and sha(p/'run_state.pkl')==parent['sidecar_sha256']
        exits += [a.audit_exit,a.probe_exit]
    if visual_goal:
        assert cfg['batch_size']==1024 and cfg['visual_mode'] in ('cache','zero192')
        assert resolved['initial_counter']==0 and resolved['resumed_from'] is None
        assert audit[f'original_GCIQL_fresh_factory_{current_dim}_goal195_full153_leaves_exact']
        assert probe['current_next_shape']==[4,current_dim] and probe['goal_shape']==[4,195] and probe['action_shape']==[4,5]
        assert probe['restored_network_step']==1000001 and probe['adam_count']==1000000
        assert a.audit_exit is not None and a.probe_exit is not None
        exits += [a.audit_exit,a.probe_exit]
    for path in exits:assert path.read_text().strip()=='0'
    value=dict(train_root=str(R),selected_updates=selected,training_code_root=str(training_code),final_audit=str(a.final_audit),final_probe=str(a.final_probe),
        wm_source_config=cfg['source_config'],agent_sha256=manifest['agent_sha256'],sidecar_sha256=manifest['sidecar_sha256'],
        visual_mode=cfg['visual_mode'],modality=cfg['cache']['modality'],modules=modules,source_sha256=sources,reference_helper_root=str(locked),
        required_zero_exits=list(map(str,exits)),environment_kwargs=old['environment_kwargs'],pcd_core=old['pcd_core'],
        pcd_provider=old['pcd_provider'],rgb_recipe=old['rgb_recipe'],runtime_profile=cfg['runtime_profile'],runtime_profile_sha256=cfg['runtime_profile_sha256'],
        native_output=str(a.native_output),reference_replay=str(B/'official_state_final_sr_seed0_v1/metadata_snapshot_v1'),
        task_order=[1,2,3,4,5],episodes_per_task=50,max_policy_steps=200,seed_base=970000000,extra_warmup=0,
        acceptance='CPU/source prepared only; actual root-authorized tiny and250 remain pending')
    if selected==500000:value['parent_transaction']=parent
    if visual_goal:value.update(goal_mode='latent_object_xyz',policy_kind='cached-visual-state-xyz',
        current_dims=current_dim,goal_dims=195,goal_encoding='real paired reference visual encoder every policy step; retain raw/preprocessed goal only, no encoder-output reuse')
    if visual_goal and cfg['visual_mode']=='zero192':value.update(zero_visual_channels=True,goal_encoding='zeros192 plus true paired goalXYZ3; no goal rendering or encoder')
    if state_profile=='robot_proprio':value.update(state_profile=state_profile,network_state_indices=list(range(19)),excluded_object_state_indices=list(range(19,28)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps({'binding':str(a.output),'sha256':sha(a.output),'source_files':len(sources)}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ['training-config','final-audit','final-probe','launcher','outer-exit','native-output','output']:p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--train-root',type=Path);p.add_argument('--selected-updates',type=int,default=100000)
    p.add_argument('--audit-exit',type=Path);p.add_argument('--probe-exit',type=Path)
    main(p.parse_args())

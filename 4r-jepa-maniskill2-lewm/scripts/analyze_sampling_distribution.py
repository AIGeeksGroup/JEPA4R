"""Read-only CPU audit of a committed Lance trajectory snapshot.

No simulator, model training or collection changes are made. Last-row actions
are excluded from paired state/action statistics because upstream collect
rotates each episode's incoming actions into departure actions with wraparound.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa


def numeric(column):
    values=column.combine_chunks()
    if pa.types.is_fixed_size_list(values.type):
        return values.values.to_numpy(zero_copy_only=False).reshape(len(values),values.type.list_size)
    return values.to_numpy(zero_copy_only=False)


def quantiles(values):
    x=np.asarray(values).reshape(-1)
    x=x[np.isfinite(x)]
    if not len(x):return None
    return {name:float(value) for name,value in zip(['min','p05','p25','median','p75','p95','p99','max'],np.quantile(x,[0,.05,.25,.5,.75,.95,.99,1]))}


def rate(values):
    x=np.asarray(values,dtype=bool)
    count=int(x.sum()); n=int(x.size); p=count/n
    result={'count':count,'denominator':n,'fraction':p,'percent':100*p}
    if x.ndim > 1:
        # Adjacent frames within an episode are correlated, so an IID
        # binomial interval with frame count as n would be misleading.
        result['uncertainty']='descriptive frame share; no IID frame confidence interval'
        return result
    z=1.96; denominator=1+z*z/n
    centre=(p+z*z/(2*n))/denominator
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    result['wilson95_percent']=[100*(centre-half),100*(centre+half)]
    return result


def histogram(x, edges):
    counts,edges=np.histogram(x,bins=edges)
    return [{'left':float(a),'right':float(b),'count':int(n)} for a,b,n in zip(edges[:-1],edges[1:],counts)]


def analyze(dataset, output, snapshot_version=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    dataset=Path(dataset).resolve()
    table=lancedb.connect(str(dataset.parent)).open_table(dataset.stem)
    version=table.version if snapshot_version is None else snapshot_version
    table.checkout(version)
    schema=table.schema
    requested=['episode_idx','step_idx','sim_state','cube_pos','seed','action','observation','rl_to_frames','object_height_above_table','tcp_object_distance','object_goal_distance','object_displacement','reward']
    missing=[k for k in requested if k not in schema.names]
    if missing:raise ValueError(f'Missing numeric audit columns: {missing}')
    count=table.count_rows()
    arrow=table.search().select(requested).limit(count).to_arrow()
    a={name:numeric(arrow[name]) for name in requested}
    ids=a['episode_idx'].reshape(-1).astype(np.int64); steps=a['step_idx'].reshape(-1).astype(np.int64)
    order=np.lexsort((steps,ids)); a={k:v[order] for k,v in a.items()};ids=ids[order];steps=steps[order]
    ep_ids,offsets,lengths=np.unique(ids,return_index=True,return_counts=True)
    complete=(lengths==200)
    for j,(start,length) in enumerate(zip(offsets,lengths)):
        complete[j] &= np.array_equal(steps[start:start+length],np.arange(200)) if length==200 else False
    keep=np.repeat(complete,lengths)
    a={k:v[keep] for k,v in a.items()}; ids=ids[keep];steps=steps[keep]
    ep_ids=ep_ids[complete]; E=len(ep_ids); H=200;N=E*H
    if E==0:raise ValueError('No complete 200-frame episodes')
    matrix=lambda key:a[key].reshape(E,H,-1)
    state=matrix('sim_state');obs=matrix('observation');pcube=matrix('cube_pos')
    assert state.shape[-1]==73 and obs.shape[-1]==30
    qpos=state[:,:,52:61];qvel=state[:,:,61:70];goal=state[:,:,70:73]
    cube=state[:,:,13:16]
    assertions={'qpos_vs_observation_max_abs':float(np.max(np.abs(qpos-obs[:,:,:9]))),'qvel_vs_observation_max_abs':float(np.max(np.abs(qvel-obs[:,:,9:18]))),'cube_vs_column_max_abs':float(np.max(np.abs(cube-pcube)))}
    assert max(assertions.values())<1e-6, assertions
    dist=np.linalg.norm(cube-goal,axis=-1)
    stored_dist=matrix('object_goal_distance')[:,:,0]
    assertions['goal_distance_vs_stored_max_abs']=float(np.max(np.abs(dist-stored_dist)))
    static=np.max(np.abs(qvel[:,:,:7]),axis=-1)<=.2
    at_goal=dist<=.025
    native_success=at_goal & static
    rewards=matrix('reward')[:,:,0]
    reward_success=np.isclose(rewards,1,rtol=0,atol=1e-7)
    assertions['success_predicate_vs_reward1_disagreements']=int(np.count_nonzero(native_success!=reward_success))
    heights=matrix('object_height_above_table')[:,:,0]
    tcp_dist=matrix('tcp_object_distance')[:,:,0]
    displacement=matrix('object_displacement')[:,:,0]
    elevated=heights>=.05
    elevated_near_tcp=elevated & (tcp_dist<=.06)
    goal_changed=np.linalg.norm(np.diff(goal,axis=1),axis=-1)>1e-6
    goal_changes=goal_changed.sum(axis=1)
    speed=np.linalg.norm(np.diff(cube,axis=1),axis=-1)/.05
    height_velocity=np.diff(heights,axis=1)/.05
    omega=qvel[:,:,:7]
    first_lift=np.where(elevated.any(axis=1),elevated.argmax(axis=1),-1)
    first_goal=np.where(at_goal.any(axis=1),at_goal.argmax(axis=1),-1)
    first_success=np.where(native_success.any(axis=1),native_success.argmax(axis=1),-1)
    returned_after_lift=np.zeros(E,dtype=bool);near_ground_before_first_success=np.zeros(E,dtype=bool)
    lift_attempts=np.zeros(E,dtype=np.int64)
    hold_run=[]; static_cube_fraction=[]; path_len=[];lift_smoothness=[]
    per_episode=[]
    for i in range(E):
        # Count separated episodes of object elevation, with hysteresis.
        flying=False; seen_lift=False; successful_so_far=False; attempts=0;current_hold=0;longest_hold=0
        for t in range(H):
            successful_so_far |= bool(native_success[i,t])
            if not flying and heights[i,t]>=.05:
                flying=True;seen_lift=True;attempts+=1
            if flying and heights[i,t]<=.02:
                returned_after_lift[i]=True
                if not successful_so_far:near_ground_before_first_success[i]=True
                flying=False
            if native_success[i,t]:current_hold+=1;longest_hold=max(longest_hold,current_hold)
            else:current_hold=0
        lift_attempts[i]=attempts;hold_run.append(longest_hold)
        static_cube_fraction.append(float(np.mean(speed[i]<.005)))
        path_len.append(float(np.linalg.norm(np.diff(cube[i],axis=0),axis=-1).sum()))
        t=first_lift[i]
        smooth=None
        if t>=0:
            segment=height_velocity[i,max(0,t-8):min(H-1,t+8)]
            smooth=float(np.mean(segment>=-.01))
            lift_smoothness.append(smooth)
        row={'episode_idx':int(ep_ids[i]),'seed':int(matrix('seed')[i,0,0]),'any_recomputed_native_success':bool(native_success[i].any()),'last_frame_native_success':bool(native_success[i,-1]),'any_at_goal':bool(at_goal[i].any()),'any_elevated_5cm':bool(elevated[i].any()),'elevated_while_tcp_near':bool(elevated_near_tcp[i].any()),'returns_to_table_after_lift':bool(returned_after_lift[i]),'returns_to_table_before_first_success':bool(near_ground_before_first_success[i]),'elevation_excursions':int(attempts),'goal_changes':int(goal_changes[i]),'first_elevation_seconds':None if t<0 else float((t+1)*.05),'first_success_seconds':None if first_success[i]<0 else float((first_success[i]+1)*.05),'cube_max_height_above_table_m':float(heights[i].max()),'cube_path_length_m':path_len[-1],'stationary_cube_transition_fraction':static_cube_fraction[-1],'max_success_run_seconds':longest_hold*.05,'initial_cube_world_xyz':cube[i,0].tolist(),'initial_goal_world_xyz':goal[i,0].tolist(),'mean_episode_return_per_stored_frame':float(rewards[i].mean()),'near_first_lift_nondescending_fraction':smooth}
        per_episode.append(row)
    action=matrix('action')[:,:-1,:]
    transforms=matrix('rl_to_frames').reshape(E,H,2,4,4)[:,:-1]
    R=np.swapaxes(transforms[:,:,1,:3,:3],-1,-2)
    translation_base=np.einsum('...ij,...j->...i',R,action[...,:3])*.1
    rotation_base=np.einsum('...ij,...j->...i',R,action[...,3:6])
    # Native controller caps rotation-vector norm at one, then scales by .1rad.
    clipped_norm=np.maximum(1,np.linalg.norm(action[...,3:6],axis=-1,keepdims=True))
    commanded_rotation=rotation_base/clipped_norm*.1
    energy=np.sum(commanded_rotation**2,axis=(0,1))
    rotation_fraction=energy/energy.sum()
    rotation_saturated=np.max(np.abs(action[...,3:6]),axis=-1)>=1-1e-6
    xyp=np.linalg.norm(commanded_rotation[...,:2],axis=-1)
    channels=[]
    for j in range(7):
        flat=action[...,j].ravel()
        channels.append({'channel':j,'mean':float(flat.mean()),'std':float(flat.std()),'abs_ge_099_fraction':float(np.mean(np.abs(flat)>=.99)),'quantiles':quantiles(flat)})
    unclip=commanded_rotation[~rotation_saturated]
    worldstatebounds={'cube_xyz':{'min':cube.min(axis=(0,1)).tolist(),'max':cube.max(axis=(0,1)).tolist()},'goal_xyz':{'min':goal.min(axis=(0,1)).tolist(),'max':goal.max(axis=(0,1)).tolist()}}
    valid_key_count=len(np.unique(np.column_stack([ids,steps]),axis=0))
    field_validity={key:{'shape':list(value.shape),'dtype':str(value.dtype),'nonfinite_values':int(np.size(value)-np.isfinite(value).sum())} for key,value in a.items()}
    quarters=[]
    for number,index in enumerate(np.array_split(np.arange(E),min(4,E)),1):
        quarters.append({'quarter':number,'episodes':len(index),'success':rate(native_success[index].any(axis=1)),'elevation':rate(elevated[index].any(axis=1)),'return_before_success':rate(near_ground_before_first_success[index]),'mean_goal_changes':float(goal_changes[index].mean())})
    summary={'snapshot':{'dataset':str(dataset),'lance_version':version,'snapshot_utc':datetime.now(timezone.utc).isoformat(),'committed_rows':count,'complete_episodes_analyzed':E,'states_analyzed':N,'paired_state_actions_analyzed':E*(H-1),'excluded_incomplete_episodes':int((~complete).sum()),'episode_frames':H,'dt_seconds':.05,'original_columns':schema.names},'integrity':{'duplicate_episode_step_keys':N-valid_key_count,'fields':field_validity,'state_layout_cross_checks':assertions,'missing_native_boolean_columns':[k for k in ['object_grasped','success','is_obj_placed','is_robot_static'] if k not in schema.names]},'episode_rates':{'any_recomputed_native_success':rate(native_success.any(axis=1)),'final_frame_native_success':rate(native_success[:,-1]),'ever_at_goal':rate(at_goal.any(axis=1)),'ever_elevated_5cm':rate(elevated.any(axis=1)),'ever_elevated_while_tcp_near':rate(elevated_near_tcp.any(axis=1)),'never_elevated_5cm':rate(~elevated.any(axis=1)),'returns_to_table_after_lift':rate(returned_after_lift),'returns_to_table_before_first_success':rate(near_ground_before_first_success),'multiple_elevation_excursions':rate(lift_attempts>=2)},'frame_rates':{'at_goal':rate(at_goal),'arm_static_native_threshold':rate(static),'recomputed_native_success':rate(native_success),'elevated_5cm':rate(elevated),'tcp_within_6cm':rate(tcp_dist<=.06),'cube_stationary_under_5mm_per_second':rate(speed<.005)},'trajectory_quantiles':{'first_lift_seconds':quantiles((first_lift[first_lift>=0]+1)*.05),'first_success_seconds':quantiles((first_success[first_success>=0]+1)*.05),'max_cube_height_m':quantiles(heights.max(axis=1)),'goal_changes_per_episode':quantiles(goal_changes),'elevation_excursions':quantiles(lift_attempts),'cube_path_length_m':quantiles(path_len),'max_success_run_seconds':quantiles(np.asarray(hold_run)*.05),'near_first_lift_nondescending_fraction':quantiles(lift_smoothness),'cube_speed_m_per_s':quantiles(speed),'action_step_difference_norm':quantiles(np.linalg.norm(np.diff(action,axis=1),axis=-1))},'actions':{'channels':channels,'world_aligned_rotation_energy_fraction_xyz':rotation_fraction.tolist(),'rotation_component_saturated_fraction':float(rotation_saturated.mean()),'non_saturated_roll_pitch_max_rad':float(np.max(np.abs(unclip[:,:2]))) if len(unclip) else None,'rotation_non_yaw_norm_quantiles_rad':quantiles(xyp),'translation_command_base_xyz_quantiles_m':[quantiles(translation_base[...,i]) for i in range(3)]},'state_space':worldstatebounds,'episode_order_quartiles':quarters,'histograms':{'first_lift_seconds':histogram((first_lift[first_lift>=0]+1)*.05,np.linspace(0,10,21)),'maximum_cube_height_m':histogram(heights.max(axis=1),np.linspace(0,.7,15)),'initial_goal_height_m':histogram(goal[:,0,2],np.linspace(0,.55,12)),'elevation_excursions':histogram(lift_attempts,np.arange(-.5,float(lift_attempts.max())+1.5,1))},'definitions':{'recomputed_native_success':'saved cube-goal distance<=.025m AND max absolute saved seven arm joint velocities<=.2 rad/s; cross-check reward==1','grasp':'unknown: contact impulses were not recorded; elevation/TCP proximity are not proof of native grasp','table_return':'after reaching5cm above table, falls/lowers to<=2cm; may be intentional release, not automatically failure','first_lift_smoothness':'fraction of vertical transitions>=-.01m/s in16-transition neighborhood; descriptive only','action_alignment':'states are s1..s200; corresponding departure actions are a1..a199 then wraparound a0; final row excluded from paired statistics','coordinates':'sim_state/cube_pos are physical world; actual pointcloud/observation under obs_frame=world use base frame via upstream wrapper; Panda base rotation is identity','uncertainty':'episode Wilson intervals assume approximately independent resets; descriptive snapshot, not a validation of trained world-model generalization'}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    (output/'episodes.json').write_text(json.dumps(per_episode,ensure_ascii=False)+'\n')
    np.savez_compressed(output/'numeric_episode_snapshot.npz',episode_ids=ep_ids,heights=heights,goal_distance=dist,tcp_distance=tcp_dist,recomputed_native_success=native_success,cube_world_xyz=cube,goal_world_xyz=goal,action=action,world_rotation_command=commanded_rotation)
    print(json.dumps({'episodes':E,'states':N,'lance_version':version,'native_success_episode_percent':summary['episode_rates']['any_recomputed_native_success']['percent'],'final_success_percent':summary['episode_rates']['final_frame_native_success']['percent'],'table_return_before_success_percent':summary['episode_rates']['returns_to_table_before_first_success']['percent'],'rotation_yaw_energy_percent':100*rotation_fraction[2],'output':str(output)},ensure_ascii=False,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--snapshot-version',type=int)
    args=parser.parse_args();analyze(args.dataset,args.output,args.snapshot_version)

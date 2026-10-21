"""Geometry-feedback candidate; unqualified, no old Adroit/DexPolicy weights.

Actual RH56 thumb/index/middle tip span and grasp centre guide approach/closing.
Only real object/hand contacts permit lifting. No attachment constraint, demo
action table, time-to-action lookup or seed filtering is used. Pen's wrist-axis
feedback is a candidate and requires a physically qualified held reset first.
"""
import numpy as np


class RM75RH56ScriptedPolicy:
    qualification='SCRIPTED_UNQUALIFIED'
    phase_codes={'reach':0,'align':1,'grasp':2,'lift':3,'carry':4,'hold':5,'release':6,'retreat':7}

    def __init__(self,seed=None,step_distance=.015,task=None,orientation_step=.08,carry_tilt_rad=0.,stable_grasp=False,top_down_approach=False,carry_step_distance=None,tracked_approach=False,thumb_first=False,approach_quaternion=None,ball_relative_approach=False,held_rotation_step=None):
        if task not in (None,'pen','relocate'):raise ValueError('unsupported candidate task')
        self.task=task;self.thumb_first=bool(thumb_first)
        self.approach_quaternion=None
        if approach_quaternion is not None:
            q=np.asarray(approach_quaternion,dtype=np.float64)
            if q.shape!=(4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q),1.,atol=1e-4,rtol=0):raise ValueError('approach quaternion must be finite unit WXYZ4')
            if task=='pen' or self.thumb_first or tracked_approach:raise ValueError('approach quaternion requires relocate without thumb_first/tracked_approach')
            self.approach_quaternion=q.copy()/np.linalg.norm(q)
        self.ball_relative_approach=bool(ball_relative_approach)
        if self.ball_relative_approach and (self.approach_quaternion is None or not top_down_approach):raise ValueError('ball-relative approach requires reference quaternion and top-down approach')
        self.orientation_step=float(orientation_step)
        self.held_rotation_step=None if held_rotation_step is None else float(held_rotation_step)
        if self.held_rotation_step is not None and (not np.isfinite(self.held_rotation_step) or self.held_rotation_step<0):raise ValueError('held rotation step must be finite nonnegative or None; zero requests target directly')
        self.carry_tilt_rad=float(carry_tilt_rad)
        self.stable_grasp=bool(stable_grasp);self.top_down_approach=bool(top_down_approach);self.tracked_approach=bool(tracked_approach)
        if not np.isfinite(self.carry_tilt_rad):raise ValueError('carry tilt must be finite')
        if not 0<self.orientation_step<=.25:raise ValueError('invalid candidate orientation step')
        self.seed=seed;self.rng=np.random.default_rng(seed);self.env=None;self._memory=[]
        self.carry_step_distance=None if carry_step_distance is None else float(carry_step_distance)
        if self.carry_step_distance is not None and (not np.isfinite(self.carry_step_distance) or self.carry_step_distance<=0):raise ValueError('carry step distance must be positive finite or None')
        self.step_distance=float(step_distance);self.type='rm75_rh56_scripted_candidate'
        self.provenance={'kind':self.qualification,'trained_checkpoint':None,'pen_reorientation':'UNQUALIFIED',
            'signal_source':'simulation_ground_truth_and_actual_contacts','protocol':'our_candidate_13d_not_dexpolicy_schema',
            'approach_quaternion':None if self.approach_quaternion is None else self.approach_quaternion.tolist(),'approach_pose_gate_rad':.05,
            'thumb_first':self.thumb_first,'thumb_first_waiting_position':'thumb_side_xy_plus_quarter_radius_z; centre_after_confirmation_or_timeout',
            'thumb_first_force_n':.1,'thumb_first_contact_frames':2,'thumb_first_wait_limit_steps':12,
            'carry_tilt_rad':self.carry_tilt_rad,'carry_tilt_frame':'robot_base_x','stable_grasp':self.stable_grasp,
            'top_down_approach':self.top_down_approach,'approach_radius_m':[.35,.60],'approach_max_rad':np.pi/4,'down_cone_rad':np.pi/3,
            'carry_step_distance':self.carry_step_distance,'carry_step_phase':'entry_and_output_carry_only','tracked_approach':self.tracked_approach,
            'tracked_gate_modes':[(1.,1.),(.5,.5),(.25,.25),(0.,1.),(0.,0.)],'tracked_ik_tolerance_source':'env.ik_tolerance; legacy fallback=(.002,.02)'}
        if self.ball_relative_approach:self.provenance['ball_relative_approach']='base-to-ball XY bearing; preserve reference tilt; reference +Z projection defines original bearing'
        if self.held_rotation_step is not None:self.provenance.update(held_rotation_step_rad=self.held_rotation_step,held_rotation_from='lift; zero requests existing bounded target without policy interpolation')

    def _approach_target(self,f):
        if not self.ball_relative_approach:return self.approach_quaternion
        w,x,y,z=self.approach_quaternion;obj=f['object_pos']
        # Left yaw about base Z: follow the ball while retaining downward tilt.
        angle=np.arctan2(-obj[0],obj[1])-np.arctan2(-2*(x*z+w*y),2*(y*z-w*x))
        return self._quat_multiply(np.array([np.cos(angle/2),0.,0.,np.sin(angle/2)]),self.approach_quaternion)

    def set_env(self,env):
        for wrapped in env.envs:
            if wrapped.unwrapped.action_space.shape!=(13,):raise ValueError('candidate policy requires 13D action')
            if self.task is not None and wrapped.unwrapped.task!=self.task:raise ValueError('configured candidate task differs from environment')
            if self.approach_quaternion is not None and wrapped.unwrapped.task!='relocate':raise ValueError('approach quaternion requires relocate environment')
        self.env=env;self._memory=[{'phase':'reach','grip':None} for _ in env.envs]

    @staticmethod
    def _contact_support(env):
        if callable(getattr(env,'contact_support',None)):return env.contact_support()
        return {'available':False,'preloaded':False,'thumb_force':0.,'finger_force':0.,'friction_capacity':0.}

    def _closing_targets(self,env,f,memory,aperture,support):
        low,high=env.hand_limits[:,0],env.hand_limits[:,1]
        previous=memory['grip'] if memory.get('grip') is not None else f.get('hand_command',f['hand_qpos'])
        # A slow actual joint must not make the position command recede, as
        # happened in seed76030 before the first unqualified lift.
        hands=np.maximum(np.asarray(previous),f['hand_qpos']).copy()
        if not support['preloaded']:
            increment=np.clip((aperture-float(f['object_width']))*6,.03,.15)
            hands[:5]+=increment*(high-low)[:5]
        hands=np.clip(hands,low,high);hands[-1]=low[-1]+.65*(high[-1]-low[-1])
        memory['grip']=hands.copy();memory['support']=support
        return hands

    def _thumb_first_targets(self,env,f,memory,opened,aperture,closing_ok):
        low,high=env.hand_limits[:,0],env.hand_limits[:,1];hands=opened.copy()
        previous=memory.get('grip');previous=f.get('hand_command',f['hand_qpos']) if previous is None else previous
        hands[4]=max(opened[4],previous[4],f['hand_qpos'][4]);stage=memory['thumb_first_stage_code']
        if stage<2:
            hands[:4]=low[:4]
            if closing_ok:hands[4]+=np.clip((aperture-float(f['object_width']))*6,.03,.15)*(high[4]-low[4])
            if closing_ok or stage==1:
                memory['thumb_first_wait_steps']+=1;support=self._contact_support(env)
                loaded=bool(support['available'] and support.get('thumb_force',0.)>.1 and int(support.get('finger_mask',0))&1)
                memory['thumb_first_contact_frames']=memory['thumb_first_contact_frames']+1 if loaded else 0
                stage=2 if memory['thumb_first_contact_frames']>=2 else 3 if memory['thumb_first_wait_steps']>=12 else 1
                memory['thumb_first_stage_code']=stage
                if stage>=2:hands[:4]=opened[:4]
        hands=np.clip(hands,low,high);hands[5]=opened[5];memory['grip']=hands.copy()
        return hands,stage>=2  # contact confirmation/timeout still requires the original align geometry gate

    @staticmethod
    def _quat_multiply(a,b):
        return np.r_[a[0]*b[0]-np.dot(a[1:],b[1:]),a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]

    @staticmethod
    def _quat_step(current,target,angle):
        current=np.asarray(current,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
        dot=float(np.dot(current,target))
        if dot<0:target=-target;dot=-dot
        half=float(np.arccos(np.clip(dot,-1.,1.)))
        if half<1e-8:return target.copy()
        fraction=min(1.,angle/(2*half))
        result=(np.sin((1-fraction)*half)*current+np.sin(fraction*half)*target)/np.sin(half)
        return result/np.linalg.norm(result)

    def _pen_hold(self,env,f,memory):
        current=np.asarray(f['object_axis']);target=np.asarray(f['target_axis'])
        axis=np.cross(current,target);sine=np.linalg.norm(axis);cosine=np.clip(np.dot(current,target),-1,1)
        if sine<1e-8:
            if cosine>=0:delta=np.array([1.,0,0,0])
            else:
                basis=np.eye(3)[np.argmin(np.abs(current))];axis=np.cross(current,basis);axis/=np.linalg.norm(axis)
                delta=np.r_[np.cos(self.orientation_step/2),axis*np.sin(self.orientation_step/2)]
        else:
            angle=min(np.arctan2(sine,cosine),self.orientation_step)
            delta=np.r_[np.cos(angle/2),axis/sine*np.sin(angle/2)]
        quaternion=self._quat_multiply(delta,f['ee_quat']);quaternion/=np.linalg.norm(quaternion)
        offset=f['object_pos']-f['ee_pos'];v=delta[1:]
        rotated=offset+2*np.cross(v,np.cross(v,offset)+delta[0]*offset)
        desired=f['target_pos']-rotated
        distance=np.linalg.norm(desired-f['ee_pos'])
        if distance>self.step_distance:desired=f['ee_pos']+(desired-f['ee_pos'])*self.step_distance/distance
        memory['phase']='hold';hands=f.get('hand_command',f['hand_qpos'])
        return env.encode_action(desired,quaternion,hands)

    def _tracked_gate(self,env,action,memory):
        f=env.feedback();actual=f['ee_pos'];actual_quat=f['ee_quat']
        planned,planned_quat,hands=env.decode_action(action.astype(np.float64))
        angle=2*np.arccos(np.clip(abs(np.dot(actual_quat,planned_quat)),-1.,1.))
        position_tolerance,rotation_tolerance=getattr(env,'ik_tolerance',(.002,.02))
        saved=env._ik_errors.copy();memory['tracked_gate_used']=1
        try:
            for alpha,rotation_scale in ((1.,1.),(.5,.5),(.25,.25),(0.,1.),(0.,0.)):
                candidate=action.copy() if alpha==1. else env.encode_action(actual+alpha*(planned-actual),self._quat_step(actual_quat,planned_quat,rotation_scale*angle),hands).astype(np.float32)
                candidate[7:]=action[7:]
                position,quaternion,_=env.decode_action(candidate.astype(np.float64))
                joints=env._solve_ik(position,quaternion);residual=env._ik_errors.copy()
                memory['tracked_gate_residual']=residual;memory['tracked_gate_joint_delta_max']=float(np.max(np.abs(joints-f['arm_qpos'])))
                if np.isfinite(residual).all() and residual[0]<=position_tolerance and residual[1]<=rotation_tolerance:
                    memory['tracked_gate_alpha']=alpha;memory['tracked_gate_rotation_scale']=rotation_scale;memory['approach_quat_cmd']=quaternion.copy()
                    return candidate
        finally:env._ik_errors=saved
        raise RuntimeError('tracked approach: no converged decoded IK target before step')

    def _oracle(self,env,memory,tracked=False):
        f=env.feedback();ee=f['ee_pos'];obj=f['object_pos'];target=f['target_pos'];tips=f['tip_positions']
        if env.task=='pen' and f['object_contact_count']>=2 and f['object_world_pos'][2]>=.075:
            return self._pen_hold(env,f,memory)
        opposite=(tips[1]+tips[2])/2;centre=(tips[0]+opposite)/2;aperture=np.linalg.norm(tips[0]-opposite)
        low,high=env.hand_limits[:,0],env.hand_limits[:,1]
        opened=low+.1*(high-low);opened[-1]=low[-1]+.65*(high[-1]-low[-1])
        actual=np.asarray(f['hand_qpos']);phase=memory['phase'];entry_phase=phase;hands=opened.copy()
        thumb_mode=self.thumb_first and env.task=='relocate'
        pose_mode=self.approach_quaternion is not None and env.task=='relocate'
        if thumb_mode:
            for key in ('thumb_first_stage_code','thumb_first_wait_steps','thumb_first_contact_frames'):memory.setdefault(key,0)
        hold_stable=self.stable_grasp and env.task=='relocate'
        approach=self.top_down_approach and env.task=='relocate'
        if phase in ('lift','carry','hold') and f['object_contact_count']<2 and np.linalg.norm(obj-centre)>.10:
            phase=memory['phase']='reach';memory['grip']=None
            memory.pop('grasp_offset',None);memory.pop('grasp_quaternion',None);memory['preload_count']=0
            memory.pop('carry_quat_cmd',None);memory.pop('carry_quat_target',None)
        direction_ok=not approach or (1-2*np.sum(f['ee_quat'][1:3]**2)<=-.5 and centre[2]<ee[2])
        closing_ok=direction_ok and (not approach or centre[2]>=obj[2])
        support=self._contact_support(env) if approach else None
        if approach and phase=='grasp' and not closing_ok and not support['preloaded']:
            phase=memory['phase']='align';memory['grip']=None;memory['preload_count']=0
        if approach and 'approach_quaternion' not in memory:memory['approach_quaternion']=f['ee_quat'].copy()
        memory['reach_xy_error']=float(np.linalg.norm(centre[:2]-obj[:2]))
        memory['reach_above_height']=float(centre[2]-obj[2])
        if phase=='reach':
            if thumb_mode:
                hands[:4]=low[:4]
                for key in ('thumb_first_stage_code','thumb_first_wait_steps','thumb_first_contact_frames'):memory[key]=0
            desired=ee+(obj+np.array([0.,0.,.10])-centre)
            if np.linalg.norm(centre[:2]-obj[:2])<.02 and centre[2]>obj[2]+.05 and (not pose_mode or memory['approach_pose_ready']):phase='align'
        elif phase=='align':
            desired=ee+(obj-centre);thumb_ready=True
            if thumb_mode:hands,thumb_ready=self._thumb_first_targets(env,f,memory,opened,aperture,closing_ok)
            if thumb_mode and memory['thumb_first_stage_code']==1:
                thumb_axis=tips[0,:2]-opposite[:2];axis_length=np.linalg.norm(thumb_axis)
                if axis_length>1e-8:
                    radius=float(f['object_width'])*.5
                    thumb_goal=obj+radius*np.r_[thumb_axis/axis_length,.25]
                    desired=ee+(thumb_goal-tips[0])
            if approach and support['preloaded'] and thumb_ready and not pose_mode:hands=self._closing_targets(env,f,memory,aperture,support)
            if thumb_ready and np.linalg.norm(centre-obj)<.018 and (closing_ok or (approach and support['preloaded'] and direction_ok)):phase='grasp'
        elif phase=='grasp':
            desired=ee+(obj-centre)
            support=self._contact_support(env);hands=self._closing_targets(env,f,memory,aperture,support)
            offset=obj-ee;previous=memory.get('grasp_offset')
            stable=previous is not None and np.linalg.norm(offset-previous)<.003
            memory['preload_count']=memory.get('preload_count',0)+1 if support['preloaded'] and stable else 0
            memory['grasp_offset']=offset.copy()
            if memory['preload_count']>=5 and direction_ok:  # .25s of loaded opposition at20Hz
                phase='lift'
                if env.task=='relocate':memory['grasp_quaternion']=f['ee_quat'].copy()
        elif phase=='lift':
            hands=self._closing_targets(env,f,memory,aperture,self._contact_support(env))
            desired=ee+np.array([0.,0.,max(.18-f['object_world_pos'][2],0.)])
            if f['object_world_pos'][2]>.13 and f['object_contact_count']>=2:phase='carry'
        elif phase=='carry':
            hands=self._closing_targets(env,f,memory,aperture,self._contact_support(env));desired=ee+(target-obj)
            if np.linalg.norm(target-obj)<.06:phase='hold'
        elif phase=='hold':
            hands=self._closing_targets(env,f,memory,aperture,self._contact_support(env));desired=ee+(target-obj)
            if env.task=='relocate' and int(f['goal_frames'])>25 and not hold_stable:phase='release'
        elif phase=='release':
            desired=ee.copy()
            if f['object_contact_count']==0:phase='retreat'
        else:desired=ee+np.array([0.,0.,.08])
        memory['phase']=phase
        delta=desired-ee;length=np.linalg.norm(delta)
        limit=self.carry_step_distance if self.carry_step_distance is not None and env.task=='relocate' and entry_phase=='carry' and phase=='carry' else self.step_distance
        if length>limit:desired=ee+delta*limit/length
        quaternion=memory.get('grasp_quaternion',f['ee_quat']) if env.task=='relocate' and phase in ('lift','carry','hold') else f['ee_quat']
        if hold_stable and phase=='hold':quaternion=memory.get('carry_quat_cmd',quaternion)
        rotation_phase=phase in ('carry','hold') or (self.held_rotation_step is not None and phase=='lift')
        if env.task=='relocate' and self.carry_tilt_rad!=0. and rotation_phase and (not hold_stable or phase!='hold'):
            if 'carry_quat_target' not in memory:
                angle=self.carry_tilt_rad
                if approach:
                    down_angle=np.arccos(np.clip(-(1-2*np.sum(quaternion[1:3]**2)),-1.,1.))
                    angle=np.sign(angle)*min(abs(angle),max(0.,np.pi/3-down_angle))
                tilt=np.array([np.cos(angle/2),np.sin(angle/2),0.,0.])
                memory['carry_quat_target']=self._quat_multiply(tilt,quaternion)
            step=self.orientation_step if self.held_rotation_step is None else self.held_rotation_step
            memory['carry_quat_cmd']=memory['carry_quat_target'].copy() if step==0 else self._quat_step(memory.get('carry_quat_cmd',quaternion),memory['carry_quat_target'],step)
            quaternion=memory['carry_quat_cmd']
        if pose_mode and phase in ('reach','align','grasp'):
            quaternion=self._quat_step(f['ee_quat'],self._approach_target(f),self.orientation_step)
        elif tracked:
            if 'tracked_initial_radius' not in memory:memory['tracked_initial_radius']=float(np.linalg.norm(centre[:2]));memory.setdefault('approach_quaternion',f['ee_quat'].copy())
            goal=float(np.linalg.norm(obj[:2]))
            angle=np.pi/4*np.clip((goal-.35)/.25,0.,1.);memory['tracked_ref_angle_rad']=float(angle)
            axis=np.array([obj[1],-obj[0],0.]);norm=np.linalg.norm(axis);axis=axis/norm if norm>1e-8 else np.array([1.,0.,0.])
            reference=self._quat_multiply(np.r_[np.cos(angle/2),axis*np.sin(angle/2)],memory['approach_quaternion'])
            quaternion=self._quat_step(f['ee_quat'],reference,self.orientation_step)
        elif approach and phase in ('reach','align','grasp'):
            predicted=centre+(desired-ee);rho=np.linalg.norm(predicted[:2])
            angle=np.pi/4*np.clip((rho-.35)/.25,0.,1.)
            if angle>0 or not closing_ok:
                axis=np.array([obj[1],-obj[0],0.]);norm=np.linalg.norm(axis);axis=axis/norm if norm>1e-8 else np.array([1.,0.,0.])
                target_quat=self._quat_multiply(np.r_[np.cos(angle/2),axis*np.sin(angle/2)],memory['approach_quaternion'])
                quaternion=self._quat_step(memory.get('approach_quat_cmd',memory['approach_quaternion']),target_quat,self.orientation_step)
            memory['approach_quat_cmd']=quaternion.copy()
        return env.encode_action(desired,quaternion,hands)

    def get_action(self,info_dict,**kwargs):
        if self.env is None:raise RuntimeError('set_env before get_action')
        actions=np.empty((self.env.num_envs,13),dtype=np.float32)
        for index,wrapped in enumerate(self.env.envs):
            env=wrapped.unwrapped
            if 'native_step_idx' in info_dict and int(info_dict['native_step_idx'][index][0])==0:
                self._memory[index]={'phase':'reach','grip':None}
                if (self.top_down_approach or self.tracked_approach) and env.task=='relocate':
                    initial=env.feedback();self._memory[index]['approach_quaternion']=initial['ee_quat'].copy()
                    if self.tracked_approach:self._memory[index]['tracked_initial_radius']=float(np.linalg.norm(((initial['tip_positions'][0]+(initial['tip_positions'][1]+initial['tip_positions'][2])/2)/2)[:2]))
            memory=self._memory[index];memory.update(tracked_gate_used=0,tracked_gate_alpha=-1.,tracked_gate_rotation_scale=-1.,tracked_gate_residual=np.array([-1.,-1.]),tracked_gate_joint_delta_max=-1.,tracked_ref_angle_rad=-1.)
            if self.approach_quaternion is not None and env.task=='relocate':
                feedback=env.feedback();target=self._approach_target(feedback);actual_quaternion=feedback['ee_quat'];error=2*np.arccos(np.clip(abs(np.dot(actual_quaternion,target)),0.,1.))
                memory.update(approach_pose_error_rad=float(error),approach_pose_ready=bool(error<=.05))
                if self.ball_relative_approach:memory['approach_target_quaternion']=target.copy()
            random=bool(info_dict.get('rand_action',np.zeros((self.env.num_envs,1)))[index][0])
            tracked=self.tracked_approach and env.task=='relocate' and not random and float(info_dict.get('noise_std',np.zeros((self.env.num_envs,1)))[index][0])==0. and memory['phase'] in ('reach','align','grasp')
            if random:
                action=self.rng.uniform(-1,1,13);action[3:7]=self.rng.normal(size=4)
            else:
                action=self._oracle(env,self._memory[index],tracked=tracked)
                noise=float(info_dict.get('noise_std',np.zeros((self.env.num_envs,1)))[index][0])
                action=action+self.rng.normal(0,noise,13)
            # Quaternion must remain on its manifold. Position/hand noise uses
            # the original raw -> noise -> clip rule; projection is an explicit
            # task-necessary change from scalar-joint Adroit actions.
            norm=np.linalg.norm(action[3:7])
            if not np.isfinite(action).all() or norm<1e-8:raise ValueError('invalid candidate policy action')
            action[3:7]/=norm;actions[index]=np.clip(action,-1,1)
            if tracked:actions[index]=self._tracked_gate(env,actions[index],memory)
        return actions

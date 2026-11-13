"""Camera-only pose adjustment for the current paired Cube protocol."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import MethodType

EFFECTIVE_CAMERAS={}

def camera_position(position,rotation,distance):
    import numpy as np
    if distance not in (0.,.10):raise ValueError('Only disabled or +0.10m local-Z pullback is authorized')
    return np.asarray(position,dtype=np.float64).copy()+distance*np.asarray(rotation,dtype=np.float64).reshape(3,3)[:,2]

def install(root,distance):
    if distance==0.:return False
    import numpy as np
    import mujoco
    import quadruple_collection as quad
    import collection_protocol as protocol
    original_factory=quad.make_quadruple_wrapper
    original_manifest=protocol.CollectionManifest
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def factory(*args,**kwargs):
        wrapped=original_factory(*args,**kwargs)
        capture=wrapped._capture
        def effective_capture(self,info):
            if self.steps==0:
                u=self.unwrapped
                before={key:getattr(u._data,key).copy() for key in ('qpos','qvel','ctrl','act','mocap_pos','mocap_quat')}
                rng_before=copy.deepcopy(u.np_random.bit_generator.state)
                pos,rotation,fovy=self.camera
                if not np.array_equal(pos,[1.287,0.,.509]) or fovy!=45.:raise ValueError('Reset must establish original front+5 baseline before pullback')
                if self.seed in EFFECTIVE_CAMERAS:raise ValueError('Reset seed reused in this bounded collection')
                candidate=camera_position(pos,rotation,distance)
                quat=u._model.cam_quat[self.cid].copy()
                u._model.cam_pos[self.cid]=candidate;mujoco.mj_camlight(u._model,u._data)
                if not np.array_equal(u._data.cam_xmat[self.cid],rotation) or not np.array_equal(u._model.cam_quat[self.cid],quat):raise ValueError('Pullback changed camera orientation')
                if any(not np.array_equal(before[k],getattr(u._data,k)) for k in before) or u.np_random.bit_generator.state!=rng_before:raise ValueError('Camera-only override changed physical state or RNG')
                self.camera=(u._data.cam_xpos[self.cid].copy(),u._data.cam_xmat[self.cid].copy(),float(u._model.cam_fovy[self.cid]))
                EFFECTIVE_CAMERAS[self.seed]=dict(baseline_position=pos.tolist(),effective_position=self.camera[0].tolist(),effective_rotation=self.camera[1].reshape(3,3).tolist(),effective_quaternion=quat.tolist(),FOV45=fovy,camera_pitch_up_deg=5.,camera_local_Z_pullback_m=distance,physical_fields_and_environment_RNG_unchanged=True, joint_layout=[dict(name=mujoco.mj_id2name(u._model,mujoco.mjtObj.mjOBJ_JOINT,j),type=int(u._model.jnt_type[j]),qposadr=int(u._model.jnt_qposadr[j]),dofadr=int(u._model.jnt_dofadr[j])) for j in range(u._model.njnt)])
            return capture(info)
        wrapped._capture=MethodType(effective_capture,wrapped)
        return wrapped
    class EffectiveManifest(original_manifest):
        def __init__(self,paths,config):
            config=dict(config)
            config['base_camera_reference']=config.get('camera')
            config['camera']='effective original front+5deg, same orientation/FOV45, local+Z 0.10m pullback for both RGB and PCD'
            config['camera_override']=dict(enabled=True,local_Z_pullback_m=distance,adapter_source=str(Path(__file__).resolve()),adapter_sha256=source_sha,original_ROOT_sources_unchanged=True)
            super().__init__(paths,config)
        def add_episode(self,record):
            record=dict(record)
            record['effective_camera']=EFFECTIVE_CAMERAS[record['reset_seed']]
            return super().add_episode(record)
    quad.make_quadruple_wrapper=factory
    protocol.CollectionManifest=EffectiveManifest
    return True

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--base-root',type=Path,required=True)
    ap.add_argument('--camera-pullback-m',type=float,default=0.)
    args,passthrough=ap.parse_known_args()
    if passthrough[:1]==['--']:passthrough=passthrough[1:]
    root=args.base_root.resolve()
    if args.camera_pullback_m not in (0.,.1):raise ValueError('Unapproved pullback')
    sys.path[:0]=[str(root/'code/lewm_pcd'),str(root/'code/lewm_pcd/scripts/data'),str(root/'code/lewm_pcd/third_party/stable-worldmodel'),str(root/'third_party/ogbench'),str(root/'third_party/stable-pretraining')]
    install(root,args.camera_pullback_m)
    original=root/'code/lewm_pcd/scripts/data/collect_cube_pcd.py'
    sys.argv=[str(original),*passthrough]
    runpy.run_path(str(original),run_name='__main__')

if __name__=='__main__':main()

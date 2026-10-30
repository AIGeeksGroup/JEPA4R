"""One official reset/zero action and front+5 RGB/depth; no policy or SR."""
import json
from pathlib import Path
import time
import numpy as np
import mujoco
import gymnasium as gym
import ogbench

ROOT=Path(__file__).resolve().parent


def main():
    env=gym.make('cube-triple-v0',render_mode='rgb_array',width=224,height=224)
    renderer=None
    try:
        obs,info=env.reset(seed=1732001)
        assert obs.shape==(46,) and info['goal'].shape==(46,)
        assert env.action_space.shape==(5,) and env.spec.max_episode_steps==1000
        base=env.unwrapped
        model,data=base._model,base._data
        camera=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'front')
        assert camera>=0
        initial=model.cam_quat[camera].copy()
        rotation=np.empty(4,np.float64)
        mujoco.mju_axisAngle2Quat(rotation,np.array([1.,0.,0.]),np.deg2rad(5.))
        up=np.empty(4,np.float64)
        mujoco.mju_mulQuat(up,initial,rotation)
        qpos,qvel=data.qpos.copy(),data.qvel.copy()
        model.cam_quat[camera]=up
        mujoco.mj_forward(model,data)
        np.testing.assert_array_equal(data.qpos,qpos)
        np.testing.assert_array_equal(data.qvel,qvel)
        rgb=base.render(camera='front')
        assert rgb.shape==(224,224,3) and rgb.dtype==np.uint8
        renderer=mujoco.Renderer(model,height=224,width=224)
        renderer.enable_depth_rendering();renderer.update_scene(data,camera='front')
        depth=renderer.render()
        assert depth.shape==(224,224) and np.isfinite(depth).all()
        np.savez_compressed(ROOT/'smoke/renderer.npz',rgb=rgb,depth=depth)
        env.step(np.zeros(5,np.float32))
        (ROOT/'smoke/renderer_report.json').write_text(json.dumps(dict(actual_exit_code=0,
            completed=time.time(),state_dim=46,action_dim=5,max_episode_steps=1000,
            reset_count=1,zero_action_steps=1,policy_calls=0,SR_episodes=0,
            camera='front',front_local_X_up_degrees=5,pullback=0,
            camera_position=model.cam_pos[camera].tolist(),FOV=float(model.cam_fovy[camera]),
            original_quaternion=initial.tolist(),actual_quaternion=up.tolist(),
            RGB_shape=list(rgb.shape),depth_finite=True),indent=2))
    finally:
        if renderer is not None:renderer.close()
        env.close()


if __name__=='__main__':main()

"""The verified original official opaque Cube scene setup."""
def setup():
    import numpy as np
    import ogbench
    import mujoco
    from ogbench.manipspace.envs.cube_env import CubeEnv
    env=ogbench.make_env_and_datasets('cube-single-play-v0',env_only=True,ob_type='states',width=224,height=224,
          mode='data_collection',visualize_info=False,terminate_at_goal=False,max_episode_steps=1001)
    env.reset(seed=0); u=env.unwrapped
    if u.__class__ is not CubeEnv or u._ob_type!='states' or u._visualize_info: raise ValueError('Nonofficial/visual environment')
    if int(u._model.vis.quality.offsamples)!=4: raise ValueError('Primary RGB AA4 contract changed')
    u._model.vis.quality.offsamples=0
    renderer=mujoco.Renderer(u._model,height=224,width=224)
    u._model.vis.quality.offsamples=4
    cid=mujoco.mj_name2id(u._model,mujoco.mjtObj.mjOBJ_CAMERA,'front_pixels')
    f=112/np.tan(np.deg2rad(u._model.cam_fovy[cid])/2)
    foreground=np.zeros(u._model.ngeom,bool); cube=np.zeros(u._model.ngeom,bool)
    for g in range(u._model.ngeom):
        body=int(u._model.geom_bodyid[g]); rb=int(u._model.body_rootid[body])
        if body and u._model.body_mocapid[rb]<0:
            foreground[g]=True; cube[g]='object' in (mujoco.mj_id2name(u._model,mujoco.mjtObj.mjOBJ_BODY,rb) or '')
    return env,u,renderer,cid,float(f),foreground,cube

"""Feasibility probe for point-cloud collection on swm/OGBCube-v0.

Checks, using ONLY the env's existing mujoco.Renderer via CustomMuJoCoEnv.render(...):
  1. segmentation render -> per-class pixel share (background / robot / cube / target marker)
  2. depth render semantics (meters?) and back-projection to world frame
  3. cube-point centroid vs privileged/block_0_pos (validates intrinsics + extrinsics)
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'TMP' / 'pydeps')); sys.path.insert(0, str(HERE.parents[1] / 'third_party' / 'stable-worldmodel'))
import numpy as np, mujoco
import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy

np.random.seed(0)
world = swm.World('swm/OGBCube-v0', num_envs=1, max_episode_steps=200, image_shape=[224, 224],
                  env_type='single', ob_type='pixels', multiview=False, width=224, height=224,
                  visualize_info=False, terminate_at_goal=False, mode='data_collection')
world.set_policy(ExpertPolicy(policy_type='plan_oracle'))
world.reset(seed=3072)
env = world.envs.envs[0].unwrapped
model, data = env._model, env._data
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'front_pixels')
H = W = 224
fovy = float(model.cam_fovy[cam_id]); f = 0.5 * H / np.tan(np.deg2rad(fovy) / 2)
print(f'camera front_pixels id={cam_id} fovy={fovy:.1f}deg  focal={f:.1f}px')

# classify every geom: static background (body 0) / cube (object bodies) / target marker (mocap) / robot
def geom_class(g):
    b = model.geom_bodyid[g]
    if b == 0: return 'background'
    root = model.body_rootid[b]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or ''
    if model.body_mocapid[root] >= 0: return 'target_marker'
    if 'object' in name: return 'cube'
    return 'robot'
gclass = np.array([geom_class(g) for g in range(model.ngeom)])
print('geoms per class:', {c: int((gclass == c).sum()) for c in np.unique(gclass)})

def probe(tag):
    rgb = env.render(camera='front_pixels')
    depth = env.render(camera='front_pixels', depth=True)
    seg = env.render(camera='front_pixels', segmentation=True)[..., 0]  # geom id, -1 = nothing
    cls = np.full(seg.shape, 'none', dtype=object); ok = seg >= 0; cls[ok] = gclass[seg[ok]]
    n = H * W
    share = {c: int((cls == c).sum()) for c in ('background', 'robot', 'cube', 'target_marker', 'none')}
    print(f'[{tag}] rgb {rgb.shape} {rgb.dtype} | depth {depth.shape} {depth.dtype} min={depth.min():.3f} max={depth.max():.3f} m')
    print(f'[{tag}] pixel share:', {c: f'{v} ({100*v/n:.2f}%)' for c, v in share.items()})
    print(f'[{tag}] uniform 1200 pts -> cube {1200*share["cube"]/n:.1f}, robot {1200*share["robot"]/n:.0f}; '
          f'foreground-only 1200 pts -> cube {1200*share["cube"]/max(1,share["cube"]+share["robot"]):.0f}')
    # back-project the whole image to world frame
    v, u = np.mgrid[0:H, 0:W]
    z = depth
    x_c = (u + 0.5 - W / 2) / f * z; y_c = -(v + 0.5 - H / 2) / f * z; z_c = -z
    p_c = np.stack([x_c, y_c, z_c], -1).reshape(-1, 3)
    R = data.cam_xmat[cam_id].reshape(3, 3); t = data.cam_xpos[cam_id]
    p_w = p_c @ R.T + t
    cube_pts = p_w[(cls == 'cube').reshape(-1)]
    bg_pts = p_w[(cls == 'background').reshape(-1)]
    cube_pos = data.joint('object_joint_0').qpos[:3]
    if len(cube_pts):
        print(f'[{tag}] cube pts centroid {cube_pts.mean(0).round(3)} vs true block pos {cube_pos.round(3)} '
              f'| err {np.linalg.norm(cube_pts.mean(0) - cube_pos)*100:.1f} cm (visible-face offset expected ~1cm)')
    print(f'[{tag}] background z: median {np.median(bg_pts[:,2]):.3f} m, robot z range '
          f'{p_w[(cls=="robot").reshape(-1)][:,2].min():.2f}..{p_w[(cls=="robot").reshape(-1)][:,2].max():.2f} m')
    return rgb, depth, seg

t0 = time.time(); probe('reset')
for _ in range(60):
    a = world.policy.get_action(world.infos); _, _, _, _, world.infos = world.envs.step(a)
probe('step60')
t0 = time.time(); n = 50
for _ in range(n): env.render(camera='front_pixels'); env.render(camera='front_pixels', depth=True); env.render(camera='front_pixels', segmentation=True)
print(f'rgb+depth+seg render cost: {1000*(time.time()-t0)/n:.1f} ms / frame')

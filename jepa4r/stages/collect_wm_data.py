"""Stage ``collect_wm_data`` -- paired image / point-cloud trajectories for world-model training.

Paper: Sec. 4.1 (one depth camera -> 1,200-point xyzrgb clouds), Sec. 5.2 and supp. sec:supp-data
(mixtures, episode counts and lengths -> ``configs/env/*.yaml: wm_data``).

in : cfg.env (simulator, camera, pointcloud, wm_data)
out: data/wm/<env>/ in the jepa4r.data.storage layout. Images and point clouds record the SAME simulator
     rollouts row by row, so states, actions, splits and starts coincide for the 2D and 3D models.
     wm_data.recipe == "mixture"       -> episodes drawn from wm_data.mixture (jepa4r.data.behaviors)
     wm_data.recipe == "ogbench_noisy" -> OGBench's noisy recipe and official scale (Puzzle)
uses: jepa4r.envs.make_env, jepa4r.data.behaviors, jepa4r.data.pointcloud.process, jepa4r.data.storage
"""


def run(cfg, out_dir):
    raise NotImplementedError("TODO(integrate): paired rollouts (rgb + point cloud) for cfg.env.wm_data")

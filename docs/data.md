# Data

All data are collected in simulation. Images and point clouds are rendered from the same fixed eye-to-hand camera
in the same rollouts, so states, actions, splits and starts coincide for the 2D and 3D models.

## Point Clouds

`jepa4r/data/pointcloud.py` turns one depth image into one point cloud. Every place that produces point clouds
(data collection, policy data, online policy evaluation) calls this one function. Settings: `pointcloud` in
`configs/env/*.yaml`.

| Step | Setting |
|---|---|
| Lift the depth image to world-frame points with the camera intrinsics and extrinsics | `camera` |
| Crop to the workspace box | `crop_box` |
| Remove the table, floor, static fixtures and invisible goal markers using the renderer's segmentation; on Adroit Relocate keep the visible goal marker | `drop`, `keep_extra` |
| Merge points within 5 mm voxels | `voxel_size: 0.005` |
| Sample N = 1,200 points | `num_points`, `sampler` |
| Normalize with a fixed affine map (on Cube shared with the real arm) | `affine` |

Each point stores xyz and rgb, so an observation is a `[1200, 6]` array. Images are `224 × 224` RGB.

## World-Model Data

`jepa4r-run collect_wm_data env=<task>` writes `data/wm/<task>/`. Settings: `wm_data` in `configs/env/*.yaml`.

| Task | Episodes | Steps | Behaviors (proportions) |
|---|---|---|---|
| Cube, 1 / 2 / 3 / 4 cubes | 10,000 | 200 / 500 / 1,000 / 600 | low-noise expert, high-noise expert, expert-then-random, random (0.2, 0.4, 0.2, 0.2) |
| Puzzle 3×3 / 4×4 / 4×5 | 1,000 / 1,000 / 3,000 | 1,001 | OGBench's noisy recipe |
| Scene, MetaWorld (per task), Adroit Relocate | 10,000 | 200 | noisy expert, expert interrupted by random actions, random (0.5, 0.3, 0.2) |

World models read clips of four frames spaced five environment steps apart; the action block between consecutive
frames concatenates the five actions (`jepa4r/data/clips.py`).

## Policy Data

`jepa4r-run prepare_policy_data env=<task>` writes `data/policy/<task>/`. Settings: `policy_data` in
`configs/env/*.yaml`.

| Task | Source | Learner inputs |
|---|---|---|
| Cube | OGBench's official play data | latent, robot state; goal: target state of the objects |
| Puzzle | OGBench's official play data | latent, button lights; goal: target state of the lights |
| Scene | 25 demonstrations per evaluation task | latent (+ proprioception) |
| MetaWorld | 25 demonstrations per task | latent (+ proprioception) |
| Adroit Relocate | 100 demonstrations | latent (+ proprioception) |

## Storage Format

To be documented with the integrated collection code (`jepa4r/data/storage.py`). Required per step: `rgb`
`[224, 224, 3]` uint8, `pointcloud` `[1200, 6]` float32, `action`, episode id and step index, and the
train / validation split; optional: simulator state, proprioception, goal state and segmentation ids.

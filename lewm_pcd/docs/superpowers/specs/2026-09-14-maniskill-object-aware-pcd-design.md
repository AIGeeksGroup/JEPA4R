# ManiSkill PickCube Object-Aware Point Cloud Design

**Date:** 2026-09-14
**Status:** Approved sampling allocation; implementation and replay not started

## 1. Goal

Create a new, versioned PickCube point-cloud dataset in which every model input
contains exactly 1,200 unique points. The allocation uses a 25% Cube target as
a hard lower bound rather than requiring every frame to contain exactly 25% Cube:

| Group | Normal target | Constraint |
|---|---:|---:|
| Cube | 300 | at least 300 (at least 25%) |
| Robot points nearest the TCP | 300 | exactly 300 (25%) |
| Remaining robot body | 480 | fills the ordinary residual |
| Local table context | 120 | at most 120 (at most 10%) |

Under normal visibility the counts remain `[300, 300, 480, 120]`. Extra Cube
points may replace unavailable residual context, so the Cube share may exceed 25%.

The existing uniformly sampled dataset remains unchanged and serves as the
`uniform_v1` baseline. The new dataset is named `object_aware_v2`.

## 2. Why a New Dataset Is Required

The current ManiSkill adapter renders one 128 x 128 camera and uniformly samples
1,200 points from the cropped scene. A strict segmentation replay of CEM episode
970 found:

- 3 to 29 raw visible Cube points per frame, mean 19.6;
- 0 to 7 Cube points after sampling, mean 2.3;
- 7 of 50 model inputs contained no Cube point;
- the first frame contained 9,320 table points among 10,938 valid candidates;
- the goal sphere contributed zero sensor points because ManiSkill hides it.

This makes the latent representation primarily describe the table and robot,
while Cube motion has little influence on the planning cost.

## 3. Alternatives Considered

### 3.1 Single-camera quota with replacement

Repeat the small number of visible Cube points until the quota is filled. This is
fast, but repeated points add almost no information to a point-wise MLP followed
by max pooling. Rejected for the formal dataset.

### 3.2 Multi-view segmented sampling

Render several real RGB-D views, identify entities using simulator segmentation,
voxel-deduplicate each semantic pool, and sample without replacement. This keeps
the input camera-derived and is closest to the six-view OGBench PCD pipeline.
Selected.

### 3.3 Privileged mesh-surface generation

Generate dense Cube points directly from the known pose and cube mesh. This
guarantees coverage but exposes occluded geometry and weakens comparison with a
camera-based baseline. Rejected.

## 4. Rendering Protocol

Replay the unchanged official `PickCube-v1` states and actions. Only observation
rendering changes; physics, reset seeds, actions, episode IDs, and splits do not.

Use six fixed cameras looking at `[-0.1, 0.0, 0.1]` metres:

| Camera | Eye position (m) | Resolution |
|---|---|---:|
| `base_camera` | `[0.30, 0.00, 0.60]` | 128 x 128 |
| `front_left` | `[0.25, -0.35, 0.45]` | 256 x 256 |
| `front_right` | `[0.25, 0.35, 0.45]` | 256 x 256 |
| `rear_left` | `[-0.35, -0.35, 0.45]` | 256 x 256 |
| `rear_right` | `[-0.35, 0.35, 0.45]` | 256 x 256 |
| `top` | `[0.00, 0.00, 0.80]` | 256 x 256 |

All cameras use a 90 degree vertical field of view, 0.01 m near plane, and
100 m far plane. If episode 970 contains any frame with fewer than 300 unique
Cube voxels at 2.5 mm, repeat the smoke at 384 x 384 for the five added cameras.
If that still fails, stop before full replay and revise the protocol.

Only the original 128 x 128 base RGB and depth images are persisted. Extra views
exist only during replay, preventing a large storage increase.

## 5. Semantic Pools

Use ManiSkill segmentation IDs rather than RGB thresholds:

- Cube: `env.unwrapped.cube.per_scene_id`;
- Robot: all IDs from `env.unwrapped.agent.robot.links`;
- Table: `env.unwrapped.table_scene.table.per_scene_id`;
- Goal: `env.unwrapped.goal_site.per_scene_id`, always excluded;
- Ground and any unrecognized actor: excluded.

Transform every camera point into the world frame, apply the existing workspace
bounds, and voxel-deduplicate at 2.5 mm before quota selection. A voxel contributes
at most one point to a semantic pool.

The local table pool contains only table points whose XY distance is at most
0.20 m from either the Cube center or the TCP. Distant table points are discarded
before quota selection.

## 6. Quota Selection

For episode `e` and frame `t`, seed a private NumPy generator with
`3072 + e * 1_000_003 + t`. Do not modify NumPy global RNG or the environment RNG.

This follows the OGBench PCD wrapper's `cube_fraction` idea: reserve
`round(0.25 * 1200) = 300` Cube points before filling the remainder. The local
ManiSkill protocol is stricter than the reference wrapper: it never samples with
replacement and rejects a frame with fewer than 300 unique Cube voxels.

1. Select the mandatory 300 Cube voxels uniformly without replacement.
2. Sort robot voxels by Euclidean distance to the current TCP position and select
   the nearest mandatory 300 as the near-gripper group.
3. Select up to 120 voxels uniformly without replacement from the local table pool.
4. Fill the remaining slots from the remaining robot pool, uniformly without
   replacement. With 120 table points this normally selects 480 robot-body points.
5. If that residual robot pool is insufficient, fill the shortage with additional
   unused Cube voxels. This is the only normal path by which Cube exceeds 25%.
6. Concatenate the groups and apply one seeded permutation so group order is not
   visible to the encoder.

Exact duplicate points and replacement sampling are forbidden. A frame fails when
the Cube or near-TCP minimum is unavailable, or when all permitted unique residual
pools still cannot produce 1,200 points. The episode is then not published and the
runner records the failing episode, frame, pool sizes, and camera protocol.

## 7. Stored Contract

The model-visible tensors remain unchanged:

```text
pcd_xyz: float32 [frames, 1200, 3]
pcd_rgb: uint8   [frames, 1200, 3]
```

Add one audit-only tensor that is not passed to the model:

```text
pcd_group: uint8 [frames, 1200]
1 = Cube, 2 = near-TCP robot, 3 = remaining robot, 4 = local table
```

Keep the existing RGB, depth, proprioception, Cube pose, goal position, success,
action, episode index, and step index columns. The merged HDF5 records:

```text
schema_version = 2
sampling_protocol = object_aware_v2
voxel_size = 0.0025
nominal_group_counts = [300, 300, 480, 120]
cube_min_points = 300
camera_protocol = six_view_pickcube_v1
```

## 8. Paths and Preservation

Do not overwrite or delete `derived/`, its merged HDF5, checkpoints, logs, or CEM
results. Write the new artifacts under:

```text
/root/autodl-tmp/workspace/users/lewm/datasets/maniskill/pickcube/derived/object_aware_v2/
  episodes/
  validation_summary_1000.json
  pickcube_object_aware_v2.h5
  manifest.json

/root/autodl-tmp/workspace/users/lewm/datasets/maniskill/pickcube/logs/
  object_aware_v2_replay_1000.log
  object_aware_v2_replay_1000.status
```

Episode shards use temporary files plus atomic rename. A resumable run skips only
shards that pass schema and quota validation.

The manifest records the camera configuration, workspace bounds, voxel size,
quota counts, group codes, deterministic seed formula, source trajectory hashes,
per-frame minimum pool sizes, output hashes, code commit, and creation time.

## 9. Training and CEM Compatibility

Normalization remains the current affine XYZ transform and `rgb / 255`, so the
encoder interface stays `1200 x 6` and the PCD architecture does not change.

Changing the sampling distribution invalidates direct comparison with a checkpoint
trained on `uniform_v1`. Train a fresh baseline on `object_aware_v2`. CEM must use
the same six-view renderer, semantic sampler, quotas, voxel size, and deterministic
seed rule for live current observations. Its goal point cloud remains the final
demonstration frame sampled by the same protocol.

This data task does not start training or CEM automatically.

## 10. Verification Gates

Before full replay, episode 970 must pass all of the following:

- every frame has shape `1200 x 3` for XYZ and RGB;
- every frame contains at least 300 Cube points, exactly 300 near-TCP robot points,
  at most 120 local-table points, and exactly 1,200 total points;
- the normal case remains `[300, 300, 480, 120]`, while any extra Cube points and
  the pool shortage that caused them are recorded;
- every selected group contains unique 2.5 mm voxels and groups do not overlap;
- Cube points carry the Cube segmentation ID;
- both robot groups carry robot-link IDs;
- table points carry the table actor ID;
- the near-TCP robot group is no farther from TCP than the remaining-robot
  selection boundary;
- no point is NaN or outside workspace bounds;
- running the same frame twice produces byte-identical outputs;
- replayed Cube pose and proprioception match the existing source episode;
- the generated HTML visibly shows dense Cube and near-gripper coverage;
- existing ManiSkill regression tests remain green.

Only after the smoke passes may the detached 1,000-episode replay start. Completion
requires a `COMPLETE` status marker, 1,000 validated shards, a merged HDF5, dataset
statistics, a manifest, and SHA-256 hashes. The smoke report also records wall time,
peak GPU memory, peak host memory, and projected full output size. A PID alone is
never completion proof.

## 11. Resource Policy

Use the existing ManiSkill replay environment and only an idle GPU. Keep code,
data, logs, and markers under the persistent workspace/data disk. Preserve a
recoverable per-episode log so a long replay can resume after disconnection.

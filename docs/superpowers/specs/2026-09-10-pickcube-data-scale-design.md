# PickCube Data-Scale Experiment Design

## Objective

Test whether the failed LiftCube model-to-simulator ranking is primarily caused by insufficient trajectory coverage. The experiment expands manipulation dynamics data without changing the JEPA architecture or using task-specific heads inside CEM.

## Dataset choice

Use the official ManiSkill2 `PickCube-v0` motion-planning dataset:

- 1,000 successful Panda trajectories;
- the same rigid-cube approach, grasp, lift, and transport structure as LiftCube;
- raw states and actions can be replayed into the existing `pointcloud_robot_seg` observation mode and `pd_ee_delta_pose` control mode;
- small enough to replay and audit locally.

Do not initially mix StackCube or PegInsertionSide. Their multi-object or different-contact geometry would make this experiment unable to isolate data scale.

## Data flow

1. Preserve the downloaded raw H5 and JSON under a new immutable source directory and record SHA-256 hashes.
2. Replay two episodes first through ManiSkill2 using `pointcloud_robot_seg` and `pd_ee_delta_pose`.
3. Verify observation/action lengths, success, object pose, TCP pose, gripper qpos `[7:9]`, RGB-D point cloud, and robot segmentation.
4. Replay all 1,000 episodes only after the pilot passes.
5. Convert replay output into the existing compact NPZ contract with 1,024 sampled scene points per frame and a deterministic 900/100 train/validation split.
6. Compute proprioception statistics from the training split only.

## Training stages

Stage A trains on PickCube with the existing latent prediction, wrong-action ranking, proprioception, object-position, and SIGReg losses. Contact supervision is disabled because the downloaded trajectories do not contain exact cube-robot contact labels.

Stage B initializes from Stage A and fine-tunes on the existing LiftCube compact dataset with the verified contact sidecar. This restores the task-specific contact signal without discarding the larger approach-and-grasp coverage.

Auxiliary heads are training supervision and diagnostics. The decisive CEM audit still compares candidates using only scene latent and proprioception unless a separately named task-aware planner is evaluated.

## Gates

- The replay compatibility fix must have a regression test.
- The two-episode replay must finish successfully and pass schema checks before full replay.
- The compact converter must be tested on a synthetic replay fixture before processing the full dataset.
- A short GPU forward/loss/backward run must pass before the formal run.
- Training must record correct-action, zero-action, shuffled-action, and persistence errors. Shuffled-action degradation below 10 percent is a stop signal.
- Lower training loss or accurate auxiliary heads alone do not establish control quality. Final evidence requires frozen-probe diagnostics and the same-state model-versus-simulator ranking audit.

## Artifact policy

All raw data, compact data, configurations, checkpoints, logs, status files, and hashes use new paths. Existing LiftCube assets and checkpoints are never overwritten. GPU ownership is checked immediately before each GPU job.

# LiftCube Proprioception and Goal-Aware Planning Design

**Date:** 2026-09-08
**Status:** Approved architecture (option 1: independent proprio token)
**Base commit:** `51a1baba694e1f7007ad624b7e3b66be5bd1fbdf`
**Feature branch:** `codex/liftcube-proprio-goal`
**Existing worktree:** `/workspace/users/lewm/projects/4d-jepa-architecture/.worktrees/liftcube-sonata-ptv3`

## Goal and evidence boundary

Fix two structural problems in the LiftCube pipeline:

1. The scene-only world model is non-Markov because robot geometry is removed from the point cloud and no robot proprioception is provided.
2. The latent-CEM objective treats a teleported cube point cloud as the sole goal and therefore does not directly express lifting, holding, or stability.

The implementation must make these quantities available and test their data flow. Passing tests, obtaining finite losses, or reducing a predicted planner cost will establish only that the new code path works. It will not establish grasping or control success. Only real simulator rollouts evaluated by peak, hold, and stability criteria can support that claim.

## Scope

This change includes:

- a new non-destructive compact RGB-plus-proprioception dataset;
- train-only proprioception statistics and explicit manifest provenance;
- an opt-in proprioception branch in the world model;
- next-proprioception and object-position auxiliary prediction;
- autoregressive proprioception state in multi-step rollout;
- a structured LiftCube `GoalSpec` and height/hold/stability-centered CEM cost;
- focused tests, full regression, syntax/whitespace checks, one-batch GPU training smoke, one renderer/CEM cost smoke, and one new bounded training run of at most 2,500 steps;
- separate commits for data, model, planner, and tests/verification support.

This change excludes robot mesh or URDF point generation, changes to the frozen Sonata checkpoint, overwriting existing compact data, resuming an old optimizer, and a formal ten-seed CEM evaluation.

## Repository and branch safety

The source checkout is already a linked Git worktree. It remains the isolated workspace; no nested worktree is created. Before feature changes:

- `codex/liftcube-before-proprio-goal-2026-09-08` points to the unchanged base commit;
- the existing linked worktree switches to `codex/liftcube-proprio-goal`;
- the old branch, compact dataset, checkpoint, run directories, and CEM results remain unchanged.

Generated `__pycache__`, failed temporary files, incomplete data outputs, and incomplete smoke outputs are removed after diagnosis. Long-running training logs and checkpoints are stored under `/workspace/users/lewm/runs/4d-jepa-architecture/`, never inside `/nix` or a virtual environment on `/workspace`.

## Data preparation

### Source and destination

The preparation script reads:

- source compact episodes from `/workspace/datasets/maniskill2/v0.5.3/compact_rgb/LiftCube-v0`;
- raw replay arrays from `/workspace/datasets/maniskill2/v0.5.3/replayed/LiftCube-v0/count_0100/trajectory.pointcloud_robot_seg.pd_ee_delta_pose.h5`;
- replay metadata from the adjacent JSON file.

It writes a new root only:

`/workspace/datasets/maniskill2/v0.5.3/compact_rgb_proprio/LiftCube-v0`

The destination must be absent or empty for publication. Each episode is written to a temporary file, reopened and compared field-by-field, then atomically renamed. The manifest is published only after every episode passes validation. A verification mode reopens an existing output without replacing it.

### Episode fields

Each output NPZ preserves every source field byte-for-byte:

- `xyz`
- `rgb`
- `action`
- `object_pos`
- `goal_pos`
- `success`

It adds raw `float32` replay fields:

- `qpos`: `[F, 9]`
- `qvel`: `[F, 9]`
- `tcp_pose`: `[F, 7]`
- `base_pose`: `[F, 7]`

For each `traj_<episode_id>`, validation requires:

- observation-aligned `xyz`, `rgb`, `object_pos`, `qpos`, `qvel`, `tcp_pose`, and `base_pose` all have length `F`;
- the static task target `goal_pos` has shape `[3]` and action-aligned `success` has length `F - 1`, matching the existing compact schema;
- `action` and raw replay `actions` both have length `F - 1`, shape `[F - 1, 7]`, and identical values;
- proprioception dimensions are exactly 9, 9, 7, and 7;
- all numeric arrays are finite and contain no NaN or infinity;
- regenerated output fields equal their in-memory source arrays after reopening;
- an episode file contains data from exactly one matching replay group.

### Manifest contract

The new manifest copies the exact ordered `episode_ids`, `train_episode_ids`, and `val_episode_ids` from the existing compact RGB manifest. It rejects any change from 100 total, 90 train, and 10 validation episodes.

It additionally records:

- `robot_uid: panda`
- `control_mode: pd_ee_delta_pose`
- `action_dim: 7`
- `qpos_dim: 9`
- `qvel_dim: 9`
- `tcp_pose_dim: 7`
- `base_pose_dim: 7`
- `proprio_dim: 25`
- `proprio_layout`: `qpos[0:9]`, `qvel[9:18]`, `tcp_pose[18:25]`
- raw replay HDF5/JSON paths, file size, modification time, and SHA-256 where practical;
- source compact path and source manifest SHA-256;
- per-episode frame and action counts;
- normalization split, ordered training episode IDs, sample count, mean, std, and std floor.

The script verifies the replay JSON control mode. It verifies `robot_uid=panda` against the runtime LiftCube environment/agent configuration before publication because the replay JSON does not itself contain a robot UID field.

### Train-only normalization

The 25-dimensional feature is exactly:

`concat(qpos[0:9], qvel[0:9], tcp_pose[0:7])`

Mean and population standard deviation are accumulated only from complete episodes listed in `train_episode_ids`. Standard deviation is floored at `1e-6`. Validation episodes are not opened by the statistics function. Raw arrays remain raw in NPZ files. Normalization occurs in the model through checkpointed buffers so offline training and online planning use one implementation.

No quaternion sign canonicalization is introduced in data preparation. Raw `tcp_pose` is preserved. Preparation records quaternion norm and adjacent-frame sign diagnostics; a sign discontinuity is reported as a data diagnostic rather than silently rewriting pose values.

## Dataset and training batch contract

`ManiSkillSequenceDataset` keeps its current default behavior for legacy callers. With `include_proprio=False`, it returns `(points, actions)` exactly as before.

With `include_proprio=True`, it returns a mapping containing:

- `points`: `[T + 1, K, C]`
- `actions`: `[T * skip_frame, 7]`
- `proprio`: `[T + 1, 25]`
- `object_pos`: `[T + 1, 3]`

All four tensors use the same sparse frame indices:

`start + [0, skip_frame, ..., T * skip_frame]`

Actions retain every ordered atomic control action from `start` through `start + T * skip_frame - 1`. Dataset indices are still constructed within one episode, so no window crosses an episode boundary. Proprioception and object positions are not resampled or independently indexed.

The new training config sets `include_rgb=true`, `include_proprio=true`, and points at `compact_rgb_proprio`. The runner copies the manifest proprioception mean/std into the model config before constructing the model. Those exact values are written to `config.json`, the copied run manifest, and every checkpoint.

## Model architecture

### Configuration and legacy compatibility

A focused `models/proprio.py` module defines:

- a structured proprioception configuration;
- normalization buffers;
- `ProprioEncoder`, a two-layer MLP from 25 dimensions to `D`;
- next-proprioception and object-position heads.

The default configuration disables the branch. When disabled, no new parameters are instantiated, existing forward signatures remain valid, and old checkpoints load strictly with their stored legacy configuration. The new configuration enables all new modules and requires proprioception input. Loading a legacy state dict into the enabled architecture with `strict=False` is forbidden.

The new 2,500-step run starts a fresh optimizer and fresh predictor/proprioception/auxiliary heads while retaining the configured frozen Sonata backbone checkpoint. The old LiftCube world-model checkpoint is a comparison baseline, not a resume or hidden partial initialization source.

### Robot token and transition conditioning

The raw 25-dimensional input is normalized by stored train-only buffers. Each frame becomes one robot token:

`robot_token_t = ProprioEncoder(normalized_proprio_t) + panda_type_embedding`

For transition `t -> t+1`, the current predictor already receives the causal temporal scene features and the ten ordered action tokens for that transition. Each predictor layer now performs cross-attention with:

`key_value_t = concat(robot_token_t[None, :], action_tokens_t[0:10])`

Scene tokens remain the queries. The robot token is therefore available in every predictor layer rather than being concatenated once and discarded. Predictor temporal attention remains causal, so a transition can also carry information from earlier conditioned scene features.

No future robot token is an input. During teacher-forced training, sources are `proprio[:, 0:T]` and targets are `proprio[:, 1:T+1]`.

### Auxiliary predictions

The predictor exposes its per-transition decoder features before the scene regression head. Mean pooling across the `K` scene-token dimension feeds two independent heads:

- `next_proprio_head`: `[B, T, D] -> [B, T, 25]` in normalized proprioception coordinates, accompanied by an explicit denormalized raw-unit view;
- `object_position_head`: `[B, T, D] -> [B, T, 3]` in raw world-frame meters.

The heads are derived from action- and robot-conditioned transition features, not from target scene latents. Consequently planner-time predictions are available without future observations.

### Loss and metrics

The enabled model uses:

`L = L_latent + lambda_rank * L_action_rank + lambda_proprio * L_next_proprio + lambda_object * L_object_pos + 0.09 * L_SIGReg`

Both `lambda_proprio` and `lambda_object` are non-negative configuration fields. Existing flow and ranking behavior remains unchanged.

`L_next_proprio` is MSE in normalized feature space. Logging additionally denormalizes predictions and reports raw-unit MSE for:

- qpos dimensions 0:9;
- qvel dimensions 9:18;
- TCP pose dimensions 18:25.

`L_object_pos` is raw-meter xyz MSE. Logging reports both complete xyz MSE and z-only MSE. Total, latent, ranking, SIGReg, proprioception, object xyz, object z, qpos, qvel, and TCP metrics are included in train/preflight/validation records.

## Multi-step rollout

The rollout state contains parallel histories:

- scene latent history: `[B, T, K, D]`;
- normalized proprioception history: `[B, T, 25]`;
- ordered atomic action history: `[B, (T - 1) * skip_frame, 7]`.

For rollout step `h`:

1. Append candidate action chunk `h` to the action history.
2. Predict the next scene latent, next normalized proprioception, and next object position.
3. Emit the scene latent, normalized proprioception, denormalized raw proprioception, and raw object position. The normalized field is the only one used for recurrence; the raw field is the only one used for qpos/qvel/TCP metrics and planner geometry.
4. Shift the scene history and append predicted scene latent.
5. Shift the proprioception history and append predicted next proprioception.
6. Shift the action history by exactly `skip_frame` atomic actions.

Only the initial history comes from real observations. Step two and later cannot receive ground-truth future proprioception. At the next real MPC replan boundary, the actually observed latest scene and proprioception replace the predicted boundary state, while candidate rollouts again remain fully autoregressive.

The legacy latent-only rollout entry point remains usable for disabled-proprioception models. The enabled planner uses an explicit structured rollout result with `scene_latent`, `proprio_normalized`, raw `proprio`, and raw `object_pos` rather than hidden mutable model state.

## Online proprioception extraction

The evaluator extracts `qpos`, `qvel`, and `tcp_pose` from the same wrapped observation used to build the scene cloud. It concatenates them in the manifest layout and validates shape `[25]` and finiteness. `base_pose` may be recorded for provenance but is not part of the default model feature.

Warm-up produces three aligned boundary observations at control steps 0, 10, and 20, together with all twenty neutral atomic actions between them. After executing one planned ten-action chunk, both the point cloud and proprioception are recaptured at that boundary.

## GoalSpec and planner cost

### GoalSpec

The new planner path replaces the teleported-cube goal as the primary objective with a validated `GoalSpec` containing:

- `initial_cube_height`
- `lift_threshold`, resolved from the environment and normally about 0.22 m
- `hold_control_steps`, default 10
- `height_tolerance`, default 0.01 m
- optional demonstrated goal latent
- all scalar cost weights
- approach radius used by early-close shaping

It does not contain a demonstrated robot qpos target. `build_goal_cloud` remains only as an explicitly legacy helper for old latent-only tests and checkpoints; the enabled proprioception planner does not call it.

### Cost terms

Let `h[-2]` and `h[-1]` be the predicted object z coordinates at the final two world-model nodes. A node interval is one ten-action chunk, so requiring both nodes above threshold represents at least ten continuous control steps between them. A hold-aware configuration requires horizon at least two.

The default terms are:

`C_height = relu(lift_threshold - h[-1])^2`

`C_hold = mean(relu(lift_threshold - h[-2:])^2)`

`delta_h = h[-1] - h[-2]`

`C_stability = relu(abs(delta_h) - height_tolerance)^2 + relu(-delta_h)^2`

`C_latent` is the existing normalized latent trajectory cost only when a demonstrated goal latent is supplied.

For each pre-lift predicted node, `C_approach` is squared predicted TCP-to-object distance. `C_early_close` multiplies premature close amount by the squared distance beyond a configured approach radius. Both are shaping terms; they do not define real success.

The combined candidate cost is:

`C = w_height*C_height + w_hold*C_hold + w_stability*C_stability + w_latent*C_latent + w_approach*C_approach + w_early_close*C_early_close`

Initial defaults are `w_height=1.0`, `w_hold=1.0`, `w_stability=0.5`, `w_latent=0.05`, `w_approach=0.1`, and `w_early_close=0.1`. Every value is configuration-driven and serialized to result JSON. Calibration remains an empirical follow-up; finite smoke results do not validate these weights.

Any NaN/inf prediction produces infinite candidate cost. Empty rollouts, shape mismatches, non-finite GoalSpec values, negative weights, hold horizon below two, or a non-positive model microbatch size raise explicit errors. CEM keeps population 128 and model microbatch size 32.

## Evaluator boundary

Predicted planner quantities and physical evaluator quantities remain separate in output JSON:

- planner records each weighted and unweighted cost component and predicted object/TCP traces;
- evaluator retains initial, maximum, delta, peak step, final height, complete height trace, peak, hold, stability, early termination, environment success, and physical success fields;
- predicted success never overwrites physical success;
- a brief threshold crossing followed by a fall cannot satisfy the physical hold criterion.

## Test-first implementation requirements

Every production slice begins with a focused failing test that is run and observed to fail for the intended missing behavior. Coverage includes:

1. compact proprioception field lengths, shapes, finiteness, source equality, and episode isolation;
2. train-only normalization, including a validation-only outlier that must not change mean/std;
3. exact preservation of the ordered 90/10 episode split;
4. aligned dataset outputs for points, actions, proprioception, and object position;
5. different predictions for identical scene/actions with different proprioception;
6. non-zero finite gradients through both proprioception conditioning and action conditioning;
7. step two of recursive rollout using predicted rather than ground-truth future proprioception;
8. qpos/qvel/TCP and object xyz/z loss and metric separation;
9. below-threshold height penalty;
10. one high node followed by a low node failing hold;
11. exactly two consecutive chunk-boundary nodes above threshold satisfying a ten-control-step hold target;
12. stable holding costing less than a transient upward spike or downward fall;
13. approach and early-close shaping active only in their intended phase;
14. NaN, empty trajectory, invalid horizon, and early terminated evaluator paths;
15. existing latent CEM, RGB, dataset, action-conditioning, wrapper, and checkpoint tests remaining compatible.

## Verification and training sequence

Run in this order:

1. dedicated data preparation and dataset tests;
2. dedicated model/proprioception/loss/rollout tests;
3. dedicated GoalSpec/cost/evaluator tests;
4. complete project `pytest` in `/nix/lewm/4d-jepa-worldmodel`;
5. Python syntax compilation and `git diff --check`;
6. prepare and verify all 100 episodes in the new destination, then inspect the manifest and representative train/validation episodes;
7. one real batch forward/loss/backward preflight in `/nix/lewm/4d-jepa-worldmodel` with finite gradients and recorded shapes;
8. one renderer, online proprioception extraction, autoregressive rollout, and CEM-cost smoke in `/nix/lewm/4d-jepa-mpc` using only a few candidates/iterations/steps;
9. start one new run under `/workspace/users/lewm/runs/4d-jepa-architecture/`, capped at 2,500 steps, with a unique run name and no resume;
10. monitor finite loss, auxiliary metrics, validation checkpoints, process status, and GPU ownership; stop on NaN, repeated runtime failure, or incompatible resource use;
11. do not launch the formal ten-seed CEM evaluation without a later explicit request.

Before claiming completion, re-run focused tests, full regression, syntax checks, `git diff --check`, check the final branch/status/log, and report all unverified assumptions. Training completion by itself is not control success.

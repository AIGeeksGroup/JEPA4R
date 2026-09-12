# LiftCube Frozen Latent Probe Design

**Date:** 2026-09-09

## Goal

Determine whether the frozen LiftCube checkpoint encodes task-relevant physical
state, and whether its predictor preserves that information across one sampled
transition. The diagnostic must distinguish representation failure from
predictor failure without changing the world-model weights.

## Non-goals

- Do not train or fine-tune the world model.
- Do not change CEM costs or run a new controller.
- Do not treat probe accuracy as physical-control success.
- Do not move the 5.3 GB replay HDF5 from A800 to `lzh-24`.

## Time and representation contract

For one dataset window, sampled frame ids are

```text
start + [0, 1, ..., T] * skip_frame
```

The first `T` frames are context. With `T=3` and `skip_frame=10`, an example is
`[49, 59, 69] -> 79`; actions `69:79` drive the last transition. Therefore the
predictor target is the next *sampled* frame, ten compact action steps later,
not the immediately adjacent compact frame.

Record three frozen representation boundaries:

1. `true_spatial_next`: the aggregator latent of the real target frame,
   `latent1_all[:, T]`.
2. `temporal_current`: the last causal temporal feature computed from context
   through the current frame.
3. `predicted_spatial_next`: the predictor output for the target frame,
   `pred_all[:, -1]`.

For the direct predictor test, train a probe only on `true_spatial_next` from
training episodes and apply that exact fitted probe to both
`true_spatial_next` and `predicted_spatial_next` on validation episodes. Do not
refit on predicted latents, because that could hide predictor distortion.

Because proprioception is injected separately, also report a full-state input
formed from pooled scene latent plus normalized current proprioception. Failure
of scene-only features to decode TCP or gripper state is not itself a model
failure if the full predictor state decodes them.

## Physical labels

All labels are aligned to the same compact frame id as the corresponding true
latent.

| Target | Source | Probe type |
|---|---|---|
| Object height | `object_pos[..., 2]` | linear regression |
| TCP position | `tcp_pose[..., :3]` | three-output linear regression |
| Gripper gap | `abs(qpos[-2]) + abs(qpos[-1])` | linear regression |
| Gripper closed | gap `<= 0.02 m` | weighted linear logistic classifier |
| Cube-robot contact | exact SAPIEN contact pair after the aligned control step | weighted linear logistic classifier |

Generic cube contact is not a valid contact label because the cube normally
touches the support surface. Only cube-to-robot-link contact is positive.

## Data generation

### A800: contact sidecar

Use the existing replay HDF5 and its matching JSON under
`/workspace/datasets/maniskill2/v0.5.3/replayed/LiftCube-v0/count_0100/`.
Replay the 100 episodes in the existing MPC environment and write a compact
sidecar containing `episode_id`, compact `frame_id`, cube-robot contact, and
alignment evidence. Before accepting an episode, compare replayed object and
TCP positions to the compact episode at sampled checkpoints. A mismatch fails
that episode instead of silently shifting labels.

The sidecar, command, environment versions, hashes, and log are retained under
`/workspace/users/lewm/runs/liftcube_contact_labels_20260909_v1/`. Transfer only
the small validated sidecar to `lzh-24`.

### lzh-24: latent extraction and probes

Use the frozen checkpoint
`/public/home/lzh_24/project/JEPA-4R/runs/group1_skip10_sonata_proprio_goal_batch8_2500/best.pt`
and the fixed 90/10 episode split in
`data/compact_rgb_proprio/LiftCube-v0/manifest.json`.

Extract deterministic windows for the train and validation episode sets. Fit
probe parameters only on training episodes. Model parameters remain in eval
mode with gradients disabled. Normalization statistics and any threshold
selection use training episodes only.

## Probe inputs

Scene token tensors are pooled using the fixed mean used by the checkpoint's
existing auxiliary heads. No learned pooling is introduced. Evaluate:

- scene-only true next latent;
- scene-only predicted next latent using the same true-latent probe;
- temporal current latent with its own train-only probe;
- scene latent plus normalized proprioception for robot-state targets.

The existing `next_proprio_head` and `object_position_head` are evaluated
directly as separate checkpoint heads, not counted as newly fitted probes.

## Metrics and interpretation

Continuous targets report sample count, target standard deviation, RMSE, MAE,
R-squared, and the constant-mean baseline. Binary targets report positive and
negative counts, AUROC, average precision, balanced accuracy, and the
majority/constant baseline. Report results per target and representation.

Interpretation rules:

- Poor true-latent probe: representation lacks linearly accessible state.
- Good true-latent probe but poor predicted-latent transfer: predictor damages
  or fails to propagate the state information.
- Good scene-plus-proprio result but poor scene-only TCP/gripper result:
  expected modality separation, not representation collapse.
- Good probes but poor model-vs-simulator ranking: latent Euclidean geometry or
  planning objective remains misaligned.
- Probe accuracy never overrides simulator success, contact, height, or ranking
  evidence.

## Outputs

Persist the final run under
`/public/home/lzh_24/project/JEPA-4R/runs/liftcube_frozen_probe_20260909_v1/`:

- `summary.json` with checkpoint/data/sidecar hashes and all metrics;
- `predictions.npz` with validation labels and probe predictions;
- `contact_sidecar.npz` copied from A800;
- `command.txt` and `run.log`;
- a concise human-readable `REPORT.md`.

## Tests and acceptance criteria

Write tests before implementation for:

1. exact `T`, target, action-span, and `skip_frame` indexing;
2. episode-disjoint train/validation fitting;
3. recovery of known synthetic continuous and binary signals;
4. one fitted true-latent probe being reused unchanged on predicted latents;
5. contact-sidecar episode/frame alignment and hard failure on mismatch;
6. scene-only versus scene-plus-proprio feature dimensions;
7. frozen model parameters receiving no gradients;
8. JSON/NPZ output schema and finite metrics where mathematically defined.

Acceptance requires focused tests, the existing relevant test suite, strict
checkpoint loading, deterministic repeated probe metrics, and a completed real
run on all eligible validation windows. Any missing contact class or replay
alignment failure must be reported explicitly rather than converted to a score.

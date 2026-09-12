# LiftCube Sonata PTv3 Latent Diagnostics Design

**Date:** 2026-09-06
**Status:** Approved in conversation; CEM explicitly deferred

## Goal

Replace only the spatial point-cloud encoder in the current LiftCube skip-10 world-model pipeline with an authentic pretrained Point Transformer V3, train a short comparable candidate on the existing 90 training trajectories, and determine whether the pretrained representation improves latent quality and action-conditioned prediction. Do not run CEM in this phase.

The retained pipeline is:

`point clouds -> spatial encoder -> temporal encoder -> ordered 10-action encoder -> cross-attention predictor -> latent/ranking/SIGReg losses`

The existing scratch PTv3 checkpoint remains an untouched comparison baseline.

## Isolation and provenance

- Baseline source commit: `567d0d7`.
- Recoverable backup branch: `codex/liftcube-before-sonata-ptv3-20260906` at `567d0d7`.
- Development branch: `codex/liftcube-sonata-ptv3`.
- Development worktree: `.worktrees/liftcube-sonata-ptv3/`.
- Pretrained source: Meta/Pointcept Sonata Small (`facebook/sonata`, `sonata_small.pth`) pinned to Hugging Face revision `fe3c5c914663b83aa01643ea15a389a1e2ff2772`.
- Expected weight SHA-256: `ee5dae277dde1d164499579a6b17d0bd38df4e4286b0f1ba21a82f8cc14270d1`.
- The checkpoint is stored outside Git under `/workspace/users/lewm/checkpoints/sonata/`; training never performs an implicit network download.
- The existing replay source is `/workspace/datasets/maniskill2/v0.5.3/replayed/LiftCube-v0/count_0100/trajectory.pointcloud_robot_seg.pd_ee_delta_pose.h5`.
- RGB-enhanced compact data is written to a new non-overwriting directory, `/workspace/datasets/maniskill2/v0.5.3/compact_rgb/LiftCube-v0`.
- Record the upstream code revision, checkpoint hash, checkpoint configuration, load report, trainable/frozen parameter counts, and feature layout in each run manifest.

Sonata Small is selected for the first experiment because it is the official lower-cost pretrained PTv3 checkpoint (about 155 MB versus about 434 MB for the base checkpoint). With only 90 training trajectories, the first test should spend capacity on evaluating transfer rather than full-backbone adaptation.

## Spatial encoder integration

Add a selectable aggregator backend while keeping the current scratch backend and its checkpoint format unchanged:

- `scratch_ptv3`: current implementation and behavior.
- `sonata_ptv3`: official Sonata Mode-2 encoder-only PTv3 plus a trainable projection to the existing latent width `D=64`.

The Sonata path must satisfy all of the following:

1. Construct the exact architecture from the checkpoint configuration and load its state dict strictly. A partial matching-key load is a hard failure.
2. Disable FlashAttention only through the architecture's supported configuration switch. Replacing parameter-free scatter reductions with equivalent native PyTorch reductions is allowed when needed by the existing environment; parameter names and tensor shapes must remain checkpoint-compatible.
3. Preserve the checkpoint input width. Inspect the checkpoint configuration before implementing the feature builder. The replay source contains point-aligned `xyzw [F,32768,4]`, `rgb [F,32768,3] uint8`, and `robot_seg [F,32768,1]`; extract RGB with the exact validity mask and deterministic sample indices used to create the existing XYZ. The RGB-enhanced compact episode stores the original fields plus `rgb [F,1024,3] uint8`. Assert bitwise RGB reproducibility and exact equality between regenerated and existing compact XYZ before publishing an episode.
4. Normalize RGB exactly as Sonata does (`float32(rgb) / 255`). For a 9-channel checkpoint input, construct `feat = [centered_xyz, normalized_rgb, zero_normal]`. Normals remain explicit zeros in the first experiment because they are not present in the replay data; do not synthesize noisy normals or silently slice/randomly initialize the pretrained input stem.
5. Use centered XYZ for the Sonata feature input while retaining the original world-frame XYZ as returned token positions. This keeps the pretrained feature convention separate from geometric matching and later control coordinates.
6. Follow Sonata's published encoder-only feature lifting procedure to map hierarchical features back to the input voxel/point resolution, then apply `LayerNorm -> Linear -> LayerNorm` to obtain `[B,K,64]` tokens.
7. Preserve deterministic output point order and the existing `[B,K,3]` positions contract expected by the temporal model, loss diagnostics, and rollout evaluator.

## Freeze policy

The first candidate freezes the entire pretrained PTv3 backbone:

- backbone parameters have `requires_grad=False`;
- the frozen backbone remains in evaluation mode during world-model training;
- its forward pass runs without autograd;
- only the feature projection, temporal encoder, action encoder, predictor, and existing loss-related trainable parameters update.

This is the safest comparison for 90 trajectories and reduces GPU memory. Full fine-tuning is not part of this phase.

A later last-stage unfreeze is justified only if diagnostics show both:

1. the raw pretrained latent has healthy variance/effective rank, and
2. its matched temporal response to LiftCube motion is materially weaker than the scratch baseline.

If temporal response is healthy but correct/shuffled action separation remains weak, the bottleneck is downstream action-conditioned dynamics and unfreezing PTv3 is not justified.

## Training experiment

Use a single comparable short run:

- Task: `LiftCube-v0`.
- Data: the same existing 90 training and 10 validation episode split, materialized under the new RGB-enhanced compact root with unchanged XYZ/actions/labels and point-aligned RGB.
- Sampling: cross-episode shuffled batches.
- Batch size: 8, subject to a GPU-3 peak-memory smoke gate.
- Context: `T=3`, `skip_frame=10`, all ordered 10 atomic actions retained per sparse transition.
- Steps: 2500.
- Checkpoints: every 500 steps.
- Action injection: current ordered action-token encoder and predictor cross-attention.
- Loss: current latent prediction + correct-vs-shuffled ranking + `0.09 * SIGReg`.
- `flow_weight=0`; motion weighting is not used for optimization.
- Device: GPU 3 only. Never stop or displace external GPU processes.

Run a 2-step overfit/finite-gradient gate and a short memory/time smoke before the detached 2500-step run. If batch 8 does not safely fit alongside external GPU-3 usage, wait for capacity rather than changing the comparison batch or taking another GPU without authorization.

## Latent diagnostics

Diagnostics are computed for both the existing scratch best checkpoint and every 500-step Sonata checkpoint on identical validation windows.

### Representation health

Report separately for raw Sonata features, projected spatial tokens, temporal context latents, and predicted latents:

- per-channel standard deviation distribution;
- token norm distribution;
- covariance effective rank / participation ratio;
- mean off-diagonal cosine similarity and its quantiles;
- fraction of near-constant channels;
- finite-value checks.

These metrics detect collapse or severe anisotropy; low prediction loss alone is not sufficient.

### Temporal and geometric signal

Because point arrays are independently sampled across frames, do not compare equal array indices. Build mutual-nearest geometric matches from returned world-frame positions and report:

- matched latent displacement from frame `t` to `t+10`;
- displacement on moving versus approximately static regions, used only as a diagnostic;
- correlation between geometric displacement magnitude and latent displacement magnitude;
- persistence error in the same matched representation.

### Action use

For horizons 1, 2, and 4, report:

- correct-action MSE;
- zero-action MSE;
- shuffled-action MSE;
- repeat-last-action MSE;
- shuffled relative degradation versus correct;
- prediction delta `||pred_correct - pred_shuffled||`;
- correct/shuffled predicted-latent cosine similarity;
- sampled gradient norm of the prediction with respect to action tokens.

The primary action-use gate remains: shuffled-action MSE must be at least 10% worse than correct-action MSE. Apparent separation is rejected when correct MSE degrades materially, persistence is better, or prediction delta grows without improved target alignment.

### Checkpoint interpretation

- A checkpoint is action-valid only if it passes the 10% shuffled gate and remains better than persistence at the relevant horizon.
- Among action-valid checkpoints, select the lowest correct-action validation MSE.
- If none is action-valid, report the lowest-correct-MSE checkpoint and the largest-gap checkpoint separately. Do not label either a success.

## Verification

Implementation follows test-first development. Required gates before training:

- strict pretrained checkpoint load and hash verification;
- RGB extraction reproduces existing XYZ exactly and applies the same mask/index to RGB;
- RGB-enhanced loader preserves episode split, frame/action alignment, dtype, and point correspondence;
- frozen/eval/no-grad behavior test;
- output shape/order/position contract tests;
- raw-to-input-resolution feature lifting test;
- scratch backend backward-compatibility test;
- focused latent-diagnostics tests, including independent point resampling;
- full `pytest` regression;
- Python syntax compilation and `git diff --check`;
- CUDA forward and 2-step training smoke on GPU 3 with recorded peak memory.

## Outputs and stopping condition

Use a new non-overwriting run directory under `/workspace/users/lewm/runs/4d-jepa-architecture/`. Persist:

- resolved config and provenance manifest;
- flushed training metrics JSONL;
- checkpoints every 500 steps;
- per-checkpoint latent diagnostics JSON;
- final scratch-versus-Sonata comparison JSON and concise report;
- PID, status marker, and log for recovery.

Once the detached training process is stable and durable logging/checkpoint behavior is confirmed, exit instead of continuously polling. CEM and environment success-rate evaluation remain explicitly out of scope until the latent diagnostics are reviewed.

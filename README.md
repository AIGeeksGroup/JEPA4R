# JEPA4R

JEPA4R is an action-conditioned world model for point-cloud dynamics. It encodes
each observation with a Point Transformer V3 backbone, models temporal context,
and predicts future spatial latents conditioned on ordered robot actions.

This release contains the upstream world-model training and offline evaluation
pipeline. Online control and MPC are intentionally kept out of the core package.

## Architecture

```text
point clouds [B,T+1,K,C]
    -> SpatialAggregator (PTv3)       -> spatial latents [B,T+1,K,D]
    -> TemporalEncoder                -> context latents [B,T,K,D]
                                      +
actions [B,T*skip_frame,A]
    -> ActionEncoder                  -> ordered action tokens
                                      |
                                      v
    -> Predictor (cross_attention)    -> predicted latents [B,T,K,D]
    -> LeWMLoss                       -> latent + flow + ranking + SIGReg
```

The recommended LiftCube configuration uses `skip_frame=10`. Every sparse
latent transition therefore consumes ten ordered atomic actions. The actions
are not averaged: `ActionEncoder` retains their order, adds positional encoding,
and exposes them to every predictor block through `cross_attention`.

The training objective is

```text
L_pred = (L_latent + lambda_flow * L_flow) / (1 + lambda_flow)
L = L_pred + lambda_rank * L_ranking + lambda_reg * L_SIGReg
```

`L_ranking` compares predictions under correct and mismatched actions. The
recommended action-token configuration sets motion weighting to zero and uses
the ranking loss to make action identity directly measurable.

## Repository Layout

```text
models/                    point ordering, PTv3 aggregation, temporal encoder,
                           action encoder, predictor, and WorldModel API
losses/lewm.py             latent, motion-weighted, ranking, and SIGReg losses
data/maniskill.py          compact trajectory dataset and cross-episode batches
configs/default.yaml       synthetic smoke configuration
configs/liftcube_baseline.yaml
configs/liftcube_skip10_action_tokens.yaml
scripts/train_liftcube.py
scripts/eval_liftcube_rollout.py
tests/
```

## Installation

JEPA4R targets Python 3.11 and CUDA GPUs. Dependencies are pinned because the
PyTorch and spconv wheel combinations are CUDA-specific.

```bash
# CUDA 12.1, used by the A800 training environment
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

# CUDA 12.8 / Blackwell
make install REQ=requirements-cu128.txt
```

The world-model environment intentionally excludes ManiSkill2, SAPIEN, and
Vulkan. Those packages belong to a separate downstream control environment.

## Data Contract

The compact LiftCube dataset is expected at a configurable root:

```text
<data-root>/LiftCube-v0/
    manifest.json
    episode_0000.npz
    ...
```

Each episode contains:

- `xyz`: `[frames, points, 3]`
- `action`: `[frames-1, 7]`

`manifest.json` defines 90 training and 10 validation episode IDs. Dataset
windows never cross episode boundaries. `CrossEpisodeBatchSampler` also keeps
the samples within a batch on distinct episodes when enough episodes remain.

## Training

Run the synthetic data-to-model smoke loop:

```bash
python train.py
```

Run a non-updating LiftCube preflight before a real experiment:

```bash
python scripts/train_liftcube.py \
  --config configs/liftcube_skip10_action_tokens.yaml \
  --preflight \
  train.data_root=/path/to/compact
```

Start training with explicit persistent paths:

```bash
python scripts/train_liftcube.py \
  --config configs/liftcube_skip10_action_tokens.yaml \
  train.data_root=/path/to/compact \
  train.run_root=/path/to/runs
```

The trainer writes a resolved config and dataset manifest, JSONL
metrics, `best.pt`, and `last.pt`. Existing non-empty run directories are not
overwritten unless `--resume` is supplied.

## Offline Rollout Evaluation

```bash
python scripts/eval_liftcube_rollout.py \
  --checkpoint /path/to/best.pt \
  --data-root /path/to/compact/LiftCube-v0 \
  --horizons 1 2 4 \
  --output /path/to/rollout_metrics.json
```

The evaluator compares correct, zero, shuffled, reversed, and repeat-last
actions against a latent persistence baseline. Its primary action-control flag
requires shuffled-action MSE to be at least 10 percent worse than correct-action
MSE; prediction delta alone is not treated as evidence of control.

## Public API

```python
from models import WorldModel, WorldModelConfig

model = WorldModel(WorldModelConfig.from_yaml("configs/default.yaml"))
result = model(points, actions, env="synthetic", negative_actions=wrong_actions)

latents, context = model.encode(points[:, :-1])
future = model.rollout(context_points, future_actions, env="synthetic")
```

The core tensor contracts are:

- `forward`: points `[B,T+1,K,C]`, actions `[B,T*skip_frame,A]`
- `encode`: points `[B,T,K,C]`
- `rollout`: points `[B,T,K,C]`, actions `[B,H*skip_frame,A]`

## Tests

```bash
python -m pytest -q
```

The suite covers point ordering and aggregation, temporal/action conditioning,
loss terms, compact trajectory windows, cross-episode batching, checkpoint
training behavior, and deterministic multi-horizon rollout metrics.

## Scope and Attribution

JEPA4R is research code. The spatial backbone is adapted from Pointcept's PTv3,
the action/predictor design is informed by HMA, and SIGReg follows LeWM. The
source files document the exact pieces retained and the local modifications.

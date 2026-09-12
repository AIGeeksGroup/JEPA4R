# LiftCube Sonata CEM Runtime Design

**Date:** 2026-09-07

## Goal

Run a recoverable ten-seed LiftCube simulation evaluation on A800 using the
frozen Sonata/PTV3 `best.pt`, while preserving the existing population-128 CEM
search and the separation between world-model training and downstream MPC
environments.

## Verified starting state

- A800 worktree: `.worktrees/liftcube-sonata-ptv3`, commit `a24386d`.
- Frozen checkpoint SHA-256:
  `38ca936f63af3fa552d0d8201876f5290f4a627f5e3e81e05d96963af4660b32`.
- Sonata backbone checkpoint already exists at
  `/workspace/users/lewm/checkpoints/sonata/sonata_small.pth` with the expected
  SHA-256 `ee5dae277dde1d164499579a6b17d0bd38df4e4286b0f1ba21a82f8cc14270d1`.
- GPU 2 was the only fully idle A800 GPU at preflight time; occupancy must be
  checked again immediately before launch.

## Runtime isolation

Keep `/nix/lewm/4d-jepa-worldmodel` unchanged. Extend only the reproducible MPC
contract in `requirements-mpc.txt` and `environments/mpc/pyproject.toml` with
the exact Sonata inference dependencies already pinned upstream:

- `torch-scatter` CUDA 12.1 wheel,
- Sonata at revision `18c09ff8d713494f78a8213792262b910977a65d`,
- `huggingface-hub==1.29.0`.

Retain `numpy==1.23.5` for ManiSkill2/SAPIEN compatibility. The MPC setup must
not install the root project as an editable distribution, because its training
metadata pins NumPy 1.26.4 and makes `pip check` fail. Evaluators import the
checked-out repository directly. The existing MPC environment may be updated
incrementally; a full rebuild is unnecessary unless exact import, CUDA, and
renderer checks fail.

The environment checker keeps its legacy default checkpoint hash for backwards
compatibility and accepts an explicit expected SHA-256 for the Sonata run. It
must verify the new runtime packages, strict model loading, CUDA tensor work,
and the renderer-backed one-step point-cloud observation.

## CEM micro-batching

Add `model_batch_size=32` to `CEMConfig`. Candidate sampling and global elite
selection remain population 128. Only world-model rollout calls are split into
at most 32 candidates and their costs are concatenated before selecting the 16
global elites. The final one-plan cost path remains unchanged.

Expose the value as `--model-batch-size` in the evaluator and record it in the
existing CEM provenance. Reject non-positive values. This avoids the
`128 * 512 = 65536` temporal-attention CUDA grid limit without weakening CEM's
search population.

## Recoverable ten-seed run

Add a Sonata-specific runner rather than modifying the old baseline runner.
It uses:

- seeds `4100` through `4109`,
- population 128, 5 iterations, 16 elites,
- model micro-batch 32,
- horizon 2, chunk size 10, maximum 200 environment steps,
- the frozen Sonata `best.pt`,
- physical GPU 2 only after a fresh availability check.

The runner writes an independent run directory plus PID, log, and atomic
`RUNNING`/`COMPLETE`/`FAILED` status. It refuses to overwrite an existing run
or create a duplicate process. Each episode is fsynced to `episodes.jsonl`, so
partial progress survives interruption.

## Verification and interpretation

Use test-first changes for:

1. population-128 rollout micro-batching with no model forward above 32,
2. the MPC dependency and checker contract,
3. the exact Sonata runner settings and non-overwrite guards.

Then run focused tests, the full pytest suite, Python/shell syntax checks, and
`git diff --check`. Before the formal run, require exact package versions,
`pip check`, strict checkpoint load, a CUDA tensor calculation, and a real
renderer smoke test.

Report physical success rate, cube-height trajectories, predicted CEM cost
versus zero-action cost, and the validation demonstrated/zero/persistence cost.
Finite costs or action sensitivity alone are not counted as successful control.

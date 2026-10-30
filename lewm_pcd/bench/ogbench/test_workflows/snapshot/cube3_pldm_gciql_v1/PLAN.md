# Cube3 PLDM encoder → GCIQL → Native SR

Authorized by the user's 2026-10-21 request. Host: `103032Test`. All new outputs live in `cube3_pldm_gciql_v1`; prior LeWM, state and Quad assets remain immutable.

The accepted asset is `cube3_pldm_impala512_epoch10_20261021_v1`. Its native frozen image embedding is 512 dimensions. This downstream experiment uses the PLDM encoder with the existing GCIQL policy; it does not reproduce the author's online PLDM planner.

Contracts:

- Official `cube-triple-play-v0`: 3,000 training and 300 validation episodes, 1,000 actions plus the true terminal observation. Encode all 3,303,300 canonical raw rows. Reuse the accepted qpos/qvel arrays; do not use custom600 training views or existing LeWM features.
- Camera must match the PLDM training source: original front pose plus local-X +5°, no pullback, FOV45, opaque robot, RGB224 AA4/full field of view. Static qpos/qvel rendering is not a claim of contact/dynamics replay.
- Frozen `encoder_py312.FrozenPLDM`: RGB uint8 BCHW224 → FP32/255 → antialiased resize65 → native Impala512. No ImageNet normalization, history duplication, PCA, zero padding or precision change.
- Current/next input: embedding512 + proprio19 = 531. Separate value/actor goals use their original sampled raw IDs: embedding512 + objectXYZ9 = 521. Raw current object truth is only available to the original distance-reward calculation.
- Preserve existing Cube3 distance-GCIQL protocol: all three cubes within 4cm, reward0/−1, mask1−success; fresh seed0, B1024, 1M updates, Adam3e-4, hidden512×3/LN, alpha1, discount.99, expectile.9, tau.005; four same-batch lookup threads and async100 metrics. Validate every5K. Retain step1 and every100K: 11 checkpoints.
- Final1M Native SR: same existing Cube3 tasks1–5 ×50 unique seeds, max1000 policy steps, original action/control mapping, real goal-then-current encoder calls each policy call, Torch1/BLAS4 and official success. No encoder-feature cache for new live frames.

Minimal implementation boundaries: a new explicit encoder/family adapter; native feature-width forwarding at the existing cache/input boundaries; a new role/config/output namespace; thin serial orchestration. Existing sampler, optimizer, losses, model hidden layers, renderer and environment dynamics remain frozen.

Sequential release: source/contract freeze → real-data GPU preprocessing/encoding pilot and terminal-boundary checks → full cache and independent CPU audit → original B1024 two-update gate and independent restore → fresh1M and full CPU acceptance → Native smoke/final250 and independent results audit → closed evidence/final-policy backup. A failed phase stops the owned relay and preserves its actual exit and logs. No duplicate producers or automatic hyperparameter fallback.

Only new case files have a writer. Root owns release/dispatch/status, cache owner owns encoding adapter, training owner owns GCIQL/Native adapters, independent reviewer checks frozen boundaries. Long stages retain argv, source/config hashes, progress, resources, logs and actual exit codes. After stable dispatch, use one ETA-based future check rather than fixed polling.

Native512 enlarges the GCIQL input layer relative to LeWM192. The hidden architecture and training hyperparameters stay unchanged; equal final weights or SR are not assumed.

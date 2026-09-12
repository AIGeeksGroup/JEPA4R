# LiftCube Frozen Latent Probe Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build and run a deterministic frozen-linear-probe diagnostic that separates information present in real scene latents from information lost by the one-step predictor, using exact LiftCube physical labels and simulator-derived cube-robot contact.

**Architecture:** Keep the world model frozen and add three small diagnostic layers around it. A pure NumPy probe module fits train-only ridge/logistic linear models and computes baselines and metrics. A LiftCube evaluator extracts aligned real/predicted/temporal features from the existing checkpoint and reuses the real-latent probe unchanged on predicted latents. A standalone ManiSkill replay script generates a frame-indexed contact sidecar on A800 and rejects misaligned trajectories.

**Tech Stack:** Python 3.11, NumPy 1.26, SciPy, PyTorch 2.5, pytest, ManiSkill2/SAPIEN for contact replay.

---

### Task 1: Add deterministic linear-probe primitives

**Files:**
- Create: `planning/frozen_probe.py`
- Test: `tests/test_frozen_probe.py`

**Step 1: Write the failing tests**

Add tests for:

- `fit_ridge_probe(X, y, alpha)` recovering a known multi-output linear signal;
- `fit_logistic_probe(X, y, l2, max_iter)` separating a known binary signal and rejecting a one-class training target;
- `continuous_metrics` reporting `n`, target standard deviation, RMSE, MAE, R-squared, and train-mean baseline RMSE;
- `binary_metrics` reporting class counts, AUROC, average precision, balanced accuracy, and train-majority baseline;
- fitted probe objects producing identical coefficients when repeated with the same inputs.

**Step 2: Run tests to verify they fail**

Run: `/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python -m pytest tests/test_frozen_probe.py -q`

Expected: import failure because `planning.frozen_probe` does not yet exist.

**Step 3: Write minimal implementation**

Implement immutable `StandardizedLinearProbe` with stored train mean/scale, coefficient, intercept, and `predict`. Implement ridge using an intercept-preserving standardized closed-form solve. Implement deterministic weighted logistic regression with SciPy L-BFGS and explicit class validation. Implement metrics without choosing thresholds from validation data; classification uses probability `>= 0.5`.

**Step 4: Run focused tests**

Run: `/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python -m pytest tests/test_frozen_probe.py -q`

Expected: all pass.

**Step 5: Commit**

Commit only `planning/frozen_probe.py` and `tests/test_frozen_probe.py` with message `feat: add deterministic frozen probe primitives`.

### Task 2: Add aligned LiftCube feature/label extraction

**Files:**
- Create: `scripts/eval_liftcube_frozen_probe.py`
- Test: `tests/test_liftcube_frozen_probe.py`

**Step 1: Write the failing tests**

Use synthetic compact episodes and a tiny fake world model to test:

- `ProbeWindowDataset` uses `frame_ids = start + arange(T+1)*skip_frame` and last actions `start:(start+T*skip_frame)`;
- labels come from the exact target frame `start + T*skip_frame`;
- train and validation episode ids remain disjoint and no validation samples enter fitting;
- `pool_scene_tokens` is a fixed token mean and `append_current_proprio` adds exactly 25 normalized dimensions;
- `extract_frozen_features` returns `true_spatial_next`, `predicted_spatial_next`, `temporal_current`, current normalized proprio, labels, episode ids, and frame ids;
- extraction runs under inference mode, leaves all world-model parameters unchanged, and creates no parameter gradients.

**Step 2: Run tests to verify they fail**

Run: `/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python -m pytest tests/test_liftcube_frozen_probe.py -q`

Expected: import failure because the evaluator does not yet exist.

**Step 3: Implement dataset and extraction**

Reuse `ManiSkillSequenceDataset` sampling semantics and strict checkpoint loading. Add an index-preserving `ProbeWindowDataset`, deterministic data loaders, and `extract_frozen_features`. Call the model once per batch with real points/actions/proprio/object positions, then read:

- `output["target"]` for the real target scene latent;
- `output["pred"]` for the predicted target scene latent;
- `output["latent2"]` for the last causal temporal state;
- target-frame physical arrays for object height, TCP xyz, gripper gap/closed;
- `output["object_pos"][:, -1]` and `output["next_proprio"][:, -1]` for existing-head evaluation.

Pool only with `.mean(axis=token_axis)`. Store current normalized proprio at sampled context frame `T-1`, not target proprio, so the full-state probe cannot leak future state.

**Step 4: Run focused tests**

Run: `/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python -m pytest tests/test_liftcube_frozen_probe.py -q`

Expected: all pass.

**Step 5: Commit**

Commit only evaluator and its test with message `feat: extract frozen LiftCube probe features`.

### Task 3: Add contact-sidecar contract and complete evaluator outputs

**Files:**
- Modify: `scripts/eval_liftcube_frozen_probe.py`
- Modify: `tests/test_liftcube_frozen_probe.py`

**Step 1: Write failing contract tests**

Add tests that:

- load sidecar arrays `episode_id`, `frame_id`, `cube_robot_contact`, `object_pos_error_m`, and `tcp_pos_error_m`;
- join contact labels by exact `(episode_id, frame_id)` and reject duplicates, missing keys, non-binary labels, or alignment errors above the configured tolerance;
- fit every probe only on train features;
- train one real-spatial probe per target and use the same serialized coefficients for both validation real and predicted spatial inputs;
- emit a versioned `summary.json` and `predictions.npz` with finite values where the target has sufficient variance/classes;
- mark binary metrics unavailable, with a reason, when validation has one class rather than fabricating a score.

**Step 2: Run tests to verify they fail**

Run: `/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python -m pytest tests/test_liftcube_frozen_probe.py -q`

Expected: missing sidecar/join/evaluation functions or schema assertions fail.

**Step 3: Implement probe matrix and persistence**

Add:

- continuous targets: `object_height`, `tcp_position`, `gripper_gap`;
- binary targets: `gripper_closed`, `cube_robot_contact`;
- representations: `true_spatial_next`, same-probe `predicted_spatial_next`, `temporal_current`, and `scene_plus_current_proprio` for robot-state targets;
- direct existing-head metrics for object position/height, TCP position, and gripper gap/closed;
- atomic JSON/NPZ writes, hashes, command/config metadata, and deterministic seed handling.

Keep the fitted real-spatial probe parameters in the report so same-probe reuse is auditable.

**Step 4: Run focused and relevant tests**

Run: `/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python -m pytest tests/test_frozen_probe.py tests/test_liftcube_frozen_probe.py tests/test_maniskill_dataset.py tests/test_maniskill_group1.py tests/test_maniskill_rollout_eval.py -q`

Expected: all pass.

**Step 5: Commit**

Commit only the evaluator/test changes with message `feat: evaluate frozen probes and contact labels`.

### Task 4: Generate exact cube-robot contact labels on A800

**Files:**
- Create: `scripts/generate_liftcube_contact_sidecar.py`
- Create: `tests/test_liftcube_contact_sidecar.py`
- Create: `scripts/run_liftcube_contact_sidecar_a800.sh`

**Step 1: Write failing unit tests**

Test pure helpers for:

- exact cube-versus-robot actor matching while excluding cube-table contact;
- frame-zero plus post-action indexing producing one label per observation frame;
- alignment statistics and hard failure when object/TCP position error exceeds tolerance;
- sidecar schema, deterministic episode/frame ordering, and duplicate rejection.

Use fake contact/actor objects; unit tests must not require ManiSkill.

**Step 2: Run tests to verify they fail**

Run: `/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python -m pytest tests/test_liftcube_contact_sidecar.py -q`

Expected: import failure because the generator does not yet exist.

**Step 3: Implement replay generator**

Read source HDF5/JSON trajectory actions and reset seeds. Create the matching `LiftCube-v0` control mode, restore/reset each episode, collect exact scene contacts at frame zero and after each atomic action, and compare replayed object/TCP positions with compact reference arrays. Write a compact NPZ sidecar plus JSON metadata atomically. Keep ManiSkill imports inside runtime functions so helper tests pass in the world-model environment.

**Step 4: Run unit tests and copy code to A800 staging**

Run focused tests on `lzh-24`, then copy the generator and runner to a dedicated staging checkout under `/workspace/users/lewm/` without modifying the older A800 repository.

**Step 5: Run one-episode A800 smoke replay**

Run with `/nix/lewm/4d-jepa-mpc/bin/python`, `--episode-ids 0`, and a strict alignment tolerance. Verify array lengths, both state errors, environment versions, and output hashes. If replay alignment fails, stop and report the exact mismatch instead of using the labels.

**Step 6: Run all 100 episodes and transfer the sidecar**

Persist under `/workspace/users/lewm/runs/liftcube_contact_labels_20260909_v1/`, then copy only the validated sidecar and metadata to the final `lzh-24` run directory.

**Step 7: Commit**

Commit the generator, tests, and runner with message `feat: generate aligned LiftCube contact sidecar`.

### Task 5: Run the frozen-probe diagnostic and verify results

**Files:**
- Create: `scripts/run_liftcube_frozen_probe_lzh24.sh`
- Create output outside Git: `/public/home/lzh_24/project/JEPA-4R/runs/liftcube_frozen_probe_20260909_v1/`

**Step 1: Add a reproducible runner**

Pin checkpoint, compact dataset, contact sidecar, seed, batch size, and output paths. Save the exact command and stream logs to the output directory.

**Step 2: Check GPU ownership and run extraction**

Run a fresh `nvidia-smi` check; select only a genuinely free GPU. Strict-load the checkpoint, extract every eligible train and validation window, fit train-only probes, and write `summary.json`, `predictions.npz`, `contact_sidecar.npz`, `command.txt`, `run.log`, and `REPORT.md`.

**Step 3: Determinism check**

Repeat the probe fitting/evaluation from cached extracted arrays with the same seed and compare metric JSON and fitted coefficients exactly or within documented floating-point tolerance.

**Step 4: Full verification**

Run: `/public/home/lzh_24/project/JEPA-4R/envs/worldmodel/bin/python -m pytest -q`

Also verify:

- `git diff --check`;
- strict checkpoint load succeeded;
- no model gradients/weight changes;
- summary and prediction arrays are finite where mathematically defined;
- all validation `(episode_id, frame_id)` contact keys matched exactly;
- output hashes and environment/version metadata are present.

**Step 5: Commit runner and report status**

Commit only the runner with message `chore: add reproducible LiftCube probe runner`. Summarize results as representation evidence, predictor-transfer evidence, and existing-head evidence. Do not reinterpret probe success as MPC/CEM physical success.

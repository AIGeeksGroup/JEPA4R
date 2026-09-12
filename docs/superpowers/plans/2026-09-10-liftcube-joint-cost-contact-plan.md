# LiftCube Joint Cost and Contact-Aware Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add contact-aware training and a goal-latent plus physical-head CEM objective, then train, audit, and transfer the essential result artifacts to A800.

**Architecture:** Contact labels remain an immutable sidecar joined by exact episode/frame keys. A configuration-gated contact head is trained on every predicted transition with balanced BCE and phase-critical window repetition. CEM receives an encoded goal cloud and combines latent, geometry, gripper, contact, hold, and stability penalties.

**Tech Stack:** Python 3.11, PyTorch/CUDA, NumPy, OmegaConf, ManiSkill2/SAPIEN, pytest, Git worktrees.

## Global Constraints

- Work only in `/public/home/lzh_24/project/JEPA-4R/worktrees/liftcube-joint-cost-contact` on `codex/liftcube-joint-cost-contact`.
- Preserve every existing data, checkpoint, run, and ranking-audit artifact.
- Preserve ordered 10-action chunks, `population=128`, `model_batch_size=32`, and global elite selection.
- Initialize from `best.pt` only after verifying SHA256 `681cdccebb2bdfbdeddbc9cdc72fa43d51aebb6b5770b9fc320d27edd43e0d28`.
- Use contact sidecar SHA256 `7c3868dcbeaf102e50461dc82f88c8cf605515c12692c0e47fc2199e5353af4f`.
- Physical success and ranking audit remain separate from offline loss.

---

### Task 1: Preserve and import the existing ranking audit

**Files:**
- Modify: `planning/latent_cem.py`
- Create: `scripts/eval_liftcube_model_sim_ranking.py`
- Create: `scripts/run_liftcube_model_sim_ranking_lzh24.sh`
- Create: `tests/test_liftcube_model_sim_ranking.py`

**Interfaces:**
- `LatentCEMPlanner.plan_with_goal_population(...) -> (plan, candidates, costs)`.
- `run_audit(...) -> dict` performs same-snapshot candidate execution.

- [ ] Copy only the four known audit files from the untouched source checkout.
- [ ] Run `python -m pytest tests/test_liftcube_model_sim_ranking.py tests/test_latent_cem.py -q`; expect all tests to pass before new behavior is added.
- [ ] Commit as `feat: add LiftCube model simulator ranking audit`.

### Task 2: Join contact labels and repeat phase-critical windows

**Files:**
- Create: `data/contact.py`
- Modify: `data/maniskill.py`
- Modify: `data/__init__.py`
- Modify: `scripts/train_maniskill_group1.py`
- Modify: `tests/test_maniskill_dataset.py`
- Modify: `tests/test_maniskill_group1.py`

**Interfaces:**
- `load_contact_sidecar(path, alignment_tolerance_m) -> ContactLabels` validates unique keys and alignment errors.
- `ManiSkillSequenceDataset(..., contact_sidecar=None, phase_repeat=1)` returns `cube_robot_contact[T+1]` and exposes deterministic per-index repeat counts.
- `CrossEpisodeBatchSampler(..., sample_repeats=None)` duplicates rare-phase indices while preserving distinct episodes per batch.

- [ ] Write failing tests for exact key joins, missing/duplicate labels, all-frame label alignment, deterministic 4x repeats, and unchanged legacy output.
- [ ] Run the focused tests and confirm the expected failures.
- [ ] Implement the loader, dataset fields, and repeat-aware sampler with no source-data writes.
- [ ] Run focused tests and commit as `feat: add contact-aware LiftCube sampling`.

### Task 3: Add the contact prediction loss and safe initialization

**Files:**
- Modify: `losses/lewm.py`
- Modify: `models/world_model.py`
- Modify: `scripts/train_maniskill_group1.py`
- Modify: `configs/maniskill_group1_sonata_proprio_goal_lzh24.yaml`
- Modify: `tests/test_shapes.py`
- Modify: `tests/test_maniskill_group1.py`

**Interfaces:**
- `LossConfig.contact_weight`, `contact_positive_weight`, and `contact_negative_weight` default to disabled-compatible values.
- `WorldModel.contact_head` exists only when contact loss is enabled.
- `TransitionPrediction.contact_probability` is optional for old checkpoints.
- `--init-checkpoint` accepts a checkpoint with only `contact_head.*` missing; all other missing/unexpected keys fail.

- [ ] Write failing tests for contact-logit shape, balanced BCE, finite gradients, disabled strict-load compatibility, and safe initialization.
- [ ] Run focused tests and confirm failures.
- [ ] Implement the minimal contact head/loss and initialization path.
- [ ] Run focused tests and commit as `feat: predict LiftCube contact transitions`.

### Task 4: Activate the complete joint CEM cost

**Files:**
- Modify: `planning/latent_cem.py`
- Modify: `planning/liftcube_goal.py`
- Modify: `scripts/eval_liftcube_latent_cem.py`
- Modify: `scripts/eval_liftcube_model_sim_ranking.py`
- Modify: `tests/test_latent_cem.py`
- Modify: `tests/test_liftcube_latent_cem_eval.py`
- Modify: `tests/test_liftcube_model_sim_ranking.py`

**Interfaces:**
- `AuxiliaryRollout.contact_probability[N,H]` carries model contact confidence.
- `LiftCubeCostWeights` adds independent `gripper` and `contact` weights.
- Proprio-enabled evaluation always builds `goal_latent` and supplies it to `GoalSpec`.

- [ ] Write failing tests proving non-zero goal-latent cost, `qpos[7:9]` gripper use, contact penalty gating, and unchanged CEM micro-batching.
- [ ] Run focused tests and confirm failures.
- [ ] Implement the joint non-negative cost and goal-cloud encoding.
- [ ] Run focused tests and commit as `feat: use joint latent physical LiftCube cost`.

### Task 5: Train and audit

**Files:**
- Create: `scripts/run_liftcube_contact_joint_2500_lzh24.sh`
- Create: `docs/experiments/liftcube-contact-joint-2026-09-10.md`

**Interfaces:**
- New run directory is `/public/home/lzh_24/project/JEPA-4R/runs/group1_skip10_sonata_contact_joint_2500`.
- New audit directory uses a new versioned name and never overwrites prior audits.

- [ ] Run the complete pytest suite.
- [ ] Verify checkpoint and sidecar hashes plus GPU ownership.
- [ ] Run a one-batch GPU preflight and a 20-step smoke; require finite contact loss and non-zero contact-head gradients.
- [ ] Run 2,500-step initialized training, preserving command, status, runtime, config, metrics, and logs.
- [ ] Evaluate the best checkpoint with corrected frozen probes and a two-state ranking smoke.
- [ ] If smoke artifacts are valid, run the 10-state x 128-candidate audit and record Spearman, Kendall, top-k overlap, regret, height, distance, contact, and penetration.
- [ ] Commit the runner and experiment report.

### Task 6: Transfer essential artifacts to A800

**Files:**
- Create remotely under `/workspace/users/lewm/runs/` only.

- [ ] Confirm `ssh A800` and exact destination capacity.
- [ ] Copy the best checkpoint, config, manifest, metrics, command, runtime, train log, audit summary/states, experiment report, and a git bundle or patch series.
- [ ] Hash source and destination artifacts and compare them.
- [ ] Do not copy environments, caches, full datasets, redundant checkpoints, or all videos.

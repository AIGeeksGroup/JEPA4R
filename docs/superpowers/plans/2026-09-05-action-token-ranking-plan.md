# Ordered Action Tokens and Motion Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and short-train a skip-frame=10 LiftCube model whose predictions measurably depend on the complete ordered action chunk.

**Architecture:** Preserve legacy action fusion paths while adding a token-preserving action encoder and per-block latent-to-action cross-attention. Add a training-only wrong-action forward pass and motion-weighted relative ranking term alongside the existing latent, flow-weighted, and SIGReg losses.

**Tech Stack:** Python 3.11, PyTorch 2.5, OmegaConf/YAML, pytest, Bash, Git.

## Global Constraints

- Use only the existing LiftCube 90 train / 10 validation split.
- Keep `skip_frame=10`; each latent transition consumes exactly ten ordered atomic actions.
- Train one new run for 2,500 steps and checkpoint every 500 steps.
- Do not overwrite `group1_skip10_motion_3k` or any older checkpoint.
- Do not configure or run MPC in this version.
- Treat shuffled-vs-correct MSE degradation below 10% as failure of the primary action-control criterion.

---

### Task 1: Action token encoder

**Files:**
- Modify: `models/action.py`
- Test: `tests/test_temporal_action_conditioning.py`

**Interfaces:**
- Produces: `ActionEncoder.encode_tokens(actions, env) -> Tensor[B,T,S,D]`.
- Preserves: `ActionEncoder.encode_chunks(actions, env) -> Tensor[B,T,D]` for legacy configurations.

- [x] Add a failing test that checks final-MLP output varies at normal scale and `encode_tokens` has shape `[B,T,S,D]` and changes when token order is swapped.
- [x] Run `pytest tests/test_temporal_action_conditioning.py -q` and confirm failure because `encode_tokens` is missing or the final layer remains gain 0.01.
- [x] Initialize only `BasicMLP.model[3]` with Xavier gain 1.0 after legacy initialization; add `preserve_tokens` and a learned `[1,S,D]` within-chunk position embedding.
- [x] Run the focused test and confirm it passes without changing legacy chunk output shapes.

### Task 2: Multi-layer cross-attention fusion

**Files:**
- Modify: `models/predictor.py`
- Test: `tests/test_temporal_action_conditioning.py`
- Test: `tests/test_shapes.py`

**Interfaces:**
- Consumes: aligned action tokens `[B,T,S,D]` for `fusion='cross_attention'`.
- Produces: predictions `[B,T,K,D]` or `[B,K,D]` with a residual cross-attention in every `STBlock`.

- [x] Add failing tests for cross-attention input shape, nonzero action gradients, and order sensitivity when positional action tokens are reordered.
- [x] Run the two focused test modules and confirm `cross_attention` is rejected by current `FUSIONS`.
- [x] Add `cross_attention` to `FUSIONS`; create `LayerNorm` plus `nn.MultiheadAttention(batch_first=True)` per block; apply `latent_queries += cross_attention(latent_queries, action_tokens, action_tokens)` before temporal attention.
- [x] Update predictor shape validation for `[B,T,S,D]` and run focused tests to green.

### Task 3: Motion-weighted action ranking loss

**Files:**
- Modify: `losses/lewm.py`
- Modify: `models/world_model.py`
- Test: `tests/test_shapes.py`

**Interfaces:**
- Adds `LossConfig.ranking_weight` and `LossConfig.ranking_margin`.
- Extends `LeWMLoss.forward(..., negative_pred=None)` and returns scalar `rank_loss`.
- Extends `WorldModel.forward(..., negative_actions=None)`.

- [x] Add failing tests showing indistinguishable correct/negative predictions incur ranking loss, sufficiently worse negative predictions do not, and `WorldModel` propagates gradients through a negative action pass.
- [x] Run the focused tests and confirm failure because ranking configuration/output is absent.
- [x] Reuse the matched target and motion weights from flow matching; compute per-patch errors and `relu(ranking_margin * e_correct.detach() + e_correct - e_negative)` with the target detached in this term.
- [x] Define total loss as existing prediction loss plus weighted ranking loss plus SIGReg; keep exact legacy behavior when `ranking_weight=0`.
- [x] Select `encode_tokens` only for cross-attention models and compute a second prediction from `negative_actions` only when supplied.
- [x] Run focused tests to green.

### Task 4: Training negatives and experiment configuration

**Files:**
- Modify: `scripts/train_maniskill_group1.py`
- Create: `configs/maniskill_group1_skip10_action_tokens_rank.yaml`
- Create: `scripts/run_skip10_action_tokens_rank_experiment.sh`
- Test: `tests/test_maniskill_group1.py`

**Interfaces:**
- Produces: `_wrong_actions(actions, chunk_size) -> Tensor` with batch-roll negatives or transition-roll fallback.
- Produces: new config/run directory `group1_skip10_action_tokens_rank_2500`.

- [x] Add failing tests for batch-roll alignment, batch-size-one fallback, and new config values.
- [x] Run focused tests and confirm `_wrong_actions` and the new config do not exist.
- [x] Pass wrong actions during training, validation, and preflight only when `ranking_weight > 0`; log `rank_loss` alongside existing metrics.
- [x] Add the new YAML with `output_scale=1`, `fusion=cross_attention`, `ranking_weight=1`, `ranking_margin=0.1`, `flow_weight=0.25`, and 2,500 steps.
- [x] Add a non-overwriting runner that evaluates best/last checkpoints for H=1,2,4 and writes durable status/log artifacts.
- [x] Run focused tests, `bash -n`, YAML parsing, and one GPU preflight.

### Task 5: Full verification, commit, and short experiment

**Files:**
- Verify all changed files.
- Create runtime artifacts only under `/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_action_tokens_rank_2500`.

**Interfaces:**
- Consumes: committed implementation and existing 90/10 LiftCube data.
- Produces: best/last checkpoints, metrics JSONL, rollout JSON files, status file, PID, and log.

- [x] Run `/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -p no:cacheprovider -q`.
- [x] Run Python bytecode compilation for changed Python files, `bash -n` for the runner, and Git whitespace checks.
- [x] Commit the implementation only after all verification commands exit zero.
- [x] Launch the runner detached on the available GPU with stdout/stderr redirected to an independent log and write its PID.
- [x] Confirm the process has completed preflight and at least one training step without non-finite loss, then stop polling and leave it resumable.
- [ ] When complete, evaluate primary 10% shuffled gap plus correct MSE, persistence, prediction delta, and moving-window secondary metrics before judging success.

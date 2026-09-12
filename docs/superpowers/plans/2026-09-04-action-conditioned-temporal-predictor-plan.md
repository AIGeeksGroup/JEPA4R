# Action-Conditioned Temporal Predictor Implementation Plan

> **For agentic workers:** Implement this plan inline in the current isolated worktree. Follow TDD: each production change starts with a failing test and ends with a focused test run.

**Goal:** Replace the last-action-only latent transition with a causal per-time-step action-conditioned predictor while preserving the existing public model interfaces.

**Architecture:** Encode context frames independently, keep the \`T\` frame axis, and use causal temporal attention with spatial tokens retained. Each frame receives its aligned action through the existing predictor fusion and predicts the next frame latent. The existing \`pred\` field remains the last prediction; \`pred_all\` exposes all transition predictions.

**Tech Stack:** Python, PyTorch, einops, pytest, existing WorldModel and LiftCube evaluator.

## Global Constraints

- Keep input shapes \`points [B,T+1,K,3]\` and \`actions [B,T,7]\`.
- Use a causal mask: prediction at time \`t\` may not read observation frames after \`t\`.
- Keep \`forward\` returning existing loss keys and \`rollout\` returning \`[B,H,K,D]\`.
- Do not regenerate datasets or add ManiSkill/SAPIEN dependencies in this first version.
- Existing checkpoint compatibility is not required; document that a new checkpoint is needed.

## File Map

- Modify: \`models/temporal.py\` — add action-conditioned causal temporal prediction over \`[B,T,K,D]\`.
- Modify: \`models/world_model.py\` — encode action sequence, train all next-frame transitions, preserve last-step compatibility.
- Modify: \`models/predictor.py\` — accept both single-step and sequence latent/action inputs.
- Modify: \`losses/lewm.py\` — accept both single-step and all-transition prediction tensors.
- Modify: \`scripts/eval_maniskill_group1_rollout.py\` — supply real context-action history during rollout evaluation.
- Test: \`tests/test_shapes.py\` — public shapes and action sensitivity.
- Test: \`tests/test_temporal_action_conditioning.py\` — causal masking and per-time action behavior.
- Create: \`docs/superpowers/specs/2026-09-04-action-conditioned-temporal-predictor-design.md\` — approved design.

### Task 1: Specify causal temporal action behavior

**Files:**
- Create: \`tests/test_temporal_action_conditioning.py\`

**Steps:**

- [x] Write tests for an action-conditioned temporal module with input \`latent1 [B,T,K,D]\` and \`action [B,T,D]\`: output shape \`[B,T,K,D]\`; changing action at time \`t\` changes output at \`t\`; changing an observation after \`t\` does not change output at \`t\` when causal mode is enabled.
- [x] Run \`pytest tests/test_temporal_action_conditioning.py -q\`; confirm the original implementation does not satisfy the new sequence behavior.

### Task 2: Implement the causal action-conditioned temporal module

**Files:**
- Modify: \`models/temporal.py\`
- Test: \`tests/test_temporal_action_conditioning.py\`

**Steps:**

- [x] Add sequence readout and frame-causal masking while preserving the old last-frame readout API.
- [x] Reuse the existing per-layer \`add/modulate\` predictor fusion. No new AdaLN-zero parameters are added in this version.
- [x] Run the focused temporal tests and confirm they pass.

### Task 3: Connect WorldModel training and rollout

**Files:**
- Modify: \`models/world_model.py\`
- Modify: \`models/predictor.py\` only if the temporal module needs a shared spatial projection.
- Test: \`tests/test_shapes.py\`

**Steps:**

- [x] Add the action-conditioned all-step path: \`latent1[:, :T]\` plus \`action_emb [B,T,D]\` predicts \`pred_all [B,T,K,D]\` against \`latent1_all[:,1:]\`.
- [x] Keep \`pred = pred_all[:, -1]\` and existing loss keys; make the loss accept all-step prediction while keeping single-step callers valid.
- [x] Keep rollout output \`[B,H,K,D]\` and call the same transition path per rollout step.
- [x] Add tests for \`forward\` and \`rollout\` shapes, causal action behavior, and explicit context-action history.
- [x] Run focused model tests and confirm they pass.

### Task 4: Regression and short empirical check

**Files:**
- Modify only if tests expose a compatibility issue.

**Steps:**

- [x] Run the complete test suite with the verified A800 Python interpreter.
- [x] Run syntax and import checks for the changed modules.
- [x] Run a 300-step continuation from step 8500 and evaluate all 804 LiftCube validation windows with correct, zero, and shuffled actions.
- [x] Record the empirical result and limitations. With real context actions, shuffled MSE is only 0.6%-2.4% above correct MSE, so the result demonstrates weak rather than strong action-state pairing.

## Completion evidence

- Continuation checkpoint: \`/workspace/users/lewm/runs/4d-jepa-architecture/group1_action_conditioned_smoke_20260904/last.pt\` (step 8800).
- Corrected evaluator output: \`/workspace/users/lewm/runs/4d-jepa-architecture/group1_action_conditioned_smoke_20260904/rollout_metrics_context_actions.json\` (804 windows).
- A rollout alignment regression was found during the real evaluation: \`T-1\` history actions were shifted before appending the current action. The fix forms \`T\` actions first, predicts, and shifts afterward; the \`T>1\` regression test covers both explicit and zero-filled history.

# LiftCube Batch-8 Ordered Action Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a batch-8 LiftCube model whose ordered 10-action conditioning is independent of motion-patch weighting.

**Architecture:** A deterministic cross-episode batch sampler supplies eight distinct episodes to each physical batch. A configurable action Transformer contextualizes ten sinusoidally positioned atomic actions before the predictor's existing per-layer cross-attention. Ranking uses unweighted same-index latent errors when motion weighting is disabled.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu121, OmegaConf, pytest, spconv, A800 CUDA.

## Global Constraints

- Work only in `/workspace/users/lewm/projects/4d-jepa-architecture/.worktrees/liftcube-multistep-evaluation`.
- Preserve backup branch `codex/liftcube-before-batch8-action-sequence-20260905` at `971dea6`.
- Use `/nix/lewm/4d-jepa-worldmodel`; do not install MPC dependencies.
- Use the existing LiftCube split of 90 train and 10 validation episodes.
- Use physical batch size 8, `skip_frame=10`, and 2500 training steps.
- Do not overwrite or resume an existing run directory.
- Do not launch training until focused tests, full pytest, syntax checks, and `git diff --check` pass.

---

### Task 1: Cross-Episode Physical Batches

**Files:**
- Modify: `data/maniskill.py`
- Modify: `scripts/train_maniskill_group1.py`
- Test: `tests/test_maniskill_dataset.py`

**Interfaces:**
- Consumes: `ManiSkillSequenceDataset.index: list[tuple[int, int]]`
- Produces: `CrossEpisodeBatchSampler(dataset, batch_size, shuffle, drop_last, seed)` yielding `list[int]`

- [x] **Step 1: Write failing sampler tests**

Add tests that build several episodes with multiple windows, consume two sampler epochs, and assert every full batch has unique episode ids. Assert `shuffle=False` is deterministic and invalid batch sizes are rejected.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -p no:cacheprovider tests/test_maniskill_dataset.py -q
```

Expected: import failure for `CrossEpisodeBatchSampler`.

- [x] **Step 3: Implement the sampler**

Group dataset indices by episode, shuffle with `torch.Generator().manual_seed(seed + epoch)`, then round-robin one index per active episode into each batch. Increment the epoch after each training iteration and drop only an incomplete final batch when requested.

- [x] **Step 4: Wire both loaders**

Replace `batch_size`, `shuffle`, and `drop_last` DataLoader arguments with explicit training and validation batch samplers. Retain `num_workers` and physical tensors of shape `[8, ...]`.

- [x] **Step 5: Verify GREEN and commit**

Run the focused test command and commit `data/maniskill.py`, `scripts/train_maniskill_group1.py`, and `tests/test_maniskill_dataset.py`.

### Task 2: Motion-Free Ranking

**Files:**
- Modify: `losses/lewm.py`
- Test: `tests/test_shapes.py`

**Interfaces:**
- Adds: `LossConfig.ranking_motion_weighted: bool = True`
- Preserves: `LeWMLoss.forward(pred, target, latent1_all, positions_all=None, negative_pred=None)`

- [x] **Step 1: Write failing loss tests**

Monkeypatch `match_motion_weights` to raise, configure `flow_weight=0`, `ranking_weight=1`, and `ranking_motion_weighted=False`, then assert ranking returns a finite positive scalar without positions. Add a compatibility test showing motion-weighted ranking still requires positions.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -p no:cacheprovider tests/test_shapes.py -q
```

Expected: `LossConfig` rejects the missing option or ranking still calls motion matching.

- [x] **Step 3: Implement separate ranking paths**

When `ranking_motion_weighted=False`, use:

```python
correct_error = (pred_seq - target_seq.detach()).pow(2).mean(dim=-1)
negative_error = (negative_seq - target_seq.detach()).pow(2).mean(dim=-1)
ranking_error = torch.relu(ranking_margin * correct_error.detach() + correct_error - negative_error)
rank_loss = ranking_error.mean()
```

Only compute motion matches when flow loss or explicitly motion-weighted ranking needs them.

- [x] **Step 4: Verify GREEN and commit**

Run the focused tests and commit `losses/lewm.py` plus its tests.

### Task 3: Sequence-Sensitive Action Tokens

**Files:**
- Modify: `models/action.py`
- Test: `tests/test_temporal_action_conditioning.py`

**Interfaces:**
- Adds ActionConfig fields: `token_encoder_layers`, `token_encoder_heads`, `token_mlp_ratio`, `token_position_encoding`, and `token_position_scale`
- Preserves: `ActionEncoder.encode_tokens(actions, env) -> [B, chunks, chunk_size, D]`

- [x] **Step 1: Write failing order-sensitivity tests**

Enable one token encoder layer with sinusoidal positions. Reverse actions inside each chunk, undo the output token ordering, and assert both the contextual token set mean and predictor output change. Verify defaults still create the legacy learned-position state dict.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -p no:cacheprovider tests/test_temporal_action_conditioning.py -q
```

Expected: new ActionConfig arguments are not accepted.

- [x] **Step 3: Implement sinusoidal positions and token encoder**

Construct a fixed `[1, chunk_size, D]` sinusoidal table when requested. For enabled layers, flatten chunks to `[B * chunks, chunk_size, D]`, run a pre-norm `nn.TransformerEncoder` with zero dropout and final LayerNorm, then restore the four-dimensional shape.

- [x] **Step 4: Verify GREEN and old checkpoint loading**

Run focused tests and strictly load the preceding `group1_skip10_action_tokens_rank_2500/best.pt` checkpoint.

- [x] **Step 5: Commit**

Commit `models/action.py` and `tests/test_temporal_action_conditioning.py`.

### Task 4: Experiment Configuration and Action Ablations

**Files:**
- Create: `configs/maniskill_group1_skip10_action_sequence_batch8.yaml`
- Modify: `scripts/eval_maniskill_group1_rollout.py`
- Test: `tests/test_maniskill_group1.py`
- Test: `tests/test_maniskill_rollout_eval.py`

**Interfaces:**
- Configuration enables batch 8, sinusoidal positions, one token Transformer layer, no flow, and unweighted ranking.
- Evaluator adds `model_reversed_action` and `model_repeat_last_action` with error and prediction-delta metrics.

- [x] **Step 1: Write failing configuration and evaluator tests**

Assert the new config values exactly. Extend the tiny action-driven evaluator test to require reverse and repeat-last result rows.

- [x] **Step 2: Verify RED**

Run both focused test files; expect missing config and metrics failures.

- [x] **Step 3: Add config and action variants**

Reverse only the ten atomic actions inside each chunk. Repeat the final atomic action ten times inside each chunk. Evaluate both through the existing rollout path and report degradation and prediction delta relative to correct action.

- [x] **Step 4: Verify GREEN and commit**

Run focused tests and commit the config, evaluator, and tests.

### Task 5: Verification, CUDA Preflight, and Experiment

**Files:**
- Create: `/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_action_sequence_batch8_2500/*`
- Modify: `docs/superpowers/plans/2026-09-05-liftcube-batch8-action-sequence-plan.md` only to record execution evidence

- [x] **Step 1: Run complete verification**

Run all focused tests, full pytest, `py_compile` on changed Python files, and `git diff --check`.

- [x] **Step 2: Run batch-8 CUDA preflight**

Use GPU 2 or 3 after checking `nvidia-smi`. Require finite losses/gradients, actual action shape `[8, 30, 7]`, and peak reserved memory below the selected 80 GB device capacity.

- [x] **Step 3: Commit implementation**

Commit all verified code and record the commit hash in the plan.

Execution evidence (2026-09-05):

- Backup branch: `codex/liftcube-before-batch8-action-sequence-20260905` at `971dea6`.
- Implementation commits: `30ad8ab`, `5528242`, `2aae67e`, `37b0bf3`, and `2aa98bd`.
- Focused configuration/evaluator RED: 3 expected failures; GREEN: 18 passed.
- Full suite after all code changes: 62 passed, 9 upstream deprecation warnings.
- `py_compile`, runner `bash -n`, and `git diff --check` passed.
- CUDA preflight on physical GPU 2: actions `[8,30,7]`, points `[8,4,512,3]`, finite loss/gradients, flow loss `0.0`, peak reserved memory `1,216,348,160` bytes.

- [x] **Step 4: Launch the 2500-step run**

Use a unique run directory, detached process, `.status` and `.pid` files, line-buffered log, and checkpoints every 500 steps. Do not restart automatically after failure.

- [x] **Step 5: Confirm stable progress and exit**

Confirm at least two advancing metric records, finite values, GPU utilization, and no duplicate process. If training is still active and healthy, leave it detached without polling.

Launch evidence (2026-09-05):

- Run directory: `/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_action_sequence_batch8_2500`.
- Detached runner PID `45777`, training PID `45782`, physical GPU 2, code commit `437b014`.
- Metrics advanced through steps 50, 100, and 150 with finite loss; latent loss decreased `0.03482 -> 0.02133 -> 0.01273`, and flow loss remained exactly `0.0`.
- Exactly one matching training Python process was running when the session exited.

- [ ] **Step 6: Evaluate terminal checkpoints**

On completion, evaluate best and last at horizons 1, 2, and 4 for correct, zero, shuffled, reversed, repeat-last, and persistence. Report the 10 percent shuffled criterion and reject action amplification that still loses to persistence.

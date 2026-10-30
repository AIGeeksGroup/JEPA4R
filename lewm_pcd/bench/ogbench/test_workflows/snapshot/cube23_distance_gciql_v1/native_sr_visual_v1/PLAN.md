# Native G05/G06 Visual SR Implementation Plan

**Goal:** Evaluate the accepted Cube2 RGB (G05) and PCD (G06) GCIQL checkpoints on 250 real online episodes each.

**Architecture:** A new isolated directory owns immutable source/config bindings, contract tests, diagnostic outputs, per-episode evidence, and serial execution state. The G05 adapter renders RGB online; the G06 adapter wraps the PCD Cube environment so its goal point cloud is captured while the simulator is at the true goal pose. Both adapters encode current and goal observations with the frozen LeWM checkpoint before constructing 211/198 dimensional policy inputs.

**Tech Stack:** Python 3.12, Gymnasium, MuJoCo, PyTorch frozen LeWM, JAX CPU GCIQL, NumPy.

## Global Constraints

- G05 and G06 run on one GPU serially; G06 starts only after G05 drains and passes its report.
- Each group uses tasks 1–5, 50 episodes per task, max 500 steps, seed `970000000 + task_id*100000 + zero_based_episode_index`, temperature 0, no Gaussian noise, and no warmup.
- Current input is online visual latent 192 plus proprioception 19; goal input is online goal visual latent 192 plus official two-cube XYZ6.
- No offline embedding cache is used for either current or goal policy observation.
- Only this new directory may be created or modified.

### Task 1: Contract tests and isolated files

Create `test_visual_sr_contract.py` and `visual_sr.py`. The first test run must fail because the production module is absent. The tests cover dimensions, seed formula, goal XYZ projection, finite float32 latent validation, and camera metadata.

### Task 2: Minimal online adapters

Implement frozen checkpoint loading, exact camera binding, RGB goal rendering, PCD goal capture, online encoder calls, GCIQL restore, and the shared episode evaluator. Preserve source SHA bindings and write status/exit evidence.

### Task 3: Diagnostic gates

Run one diagnostic episode per group with independent seeds. Verify real online encoder calls, current/goal shapes, finite outputs, physics/RNG invariants around goal PCD capture, and action forward. Stop a group on any failure.

### Task 4: Serial formal evaluation

Run G05 full 250 episodes, verify exit and complete episode evidence, release GPU resources, then run G06 full 250 episodes. Write reports and final CPU/resource receipts.

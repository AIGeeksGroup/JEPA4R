# PickCube Data-Scale Implementation Plan

**Goal:** Prepare 1,000 official PickCube trajectories in the current compact contract, run a guarded two-stage training experiment, and retain enough diagnostics to decide whether added manipulation coverage improves the JEPA world model.

**Architecture:** Add a small project-owned replay entry point for the known Gymnasium wrapper compatibility issue, generalize compact preparation and task selection without changing model internals, then run PickCube pretraining followed by LiftCube contact fine-tuning.

### Task 1: Replay compatibility entry point

- Add a failing test proving the robot-link compatibility patch is applied before ManiSkill replay starts.
- Add the minimal replay wrapper and rerun the two-episode pilot.
- Inspect the generated H5/JSON contract.

### Task 2: Generic compact converter

- Add a synthetic H5/JSON fixture and failing tests for alignment, deterministic point sampling, metadata, and split behavior.
- Implement conversion from replay observations to per-episode compact NPZ files.
- Run it on the two-episode pilot and validate every stored field.

### Task 3: Task-configurable training loader

- Add failing tests showing training accepts a non-LiftCube task and non-100-episode manifest while preserving strict split/disjoint/schema validation.
- Replace only the hard-coded task and episode-count assumptions with configuration values.
- Run affected tests and the full CPU test suite.

### Task 4: Full data publication

- Replay all 1,000 PickCube trajectories to a new directory with durable log, PID, status, source hash, and exact command metadata.
- Convert to compact NPZ with a 900/100 split.
- Verify episode counts, frame counts, finite values, action dimensions, proprioception layout, and source hashes.

### Task 5: Training preflight and formal stages

- Recheck GPU ownership.
- Run a short PickCube GPU forward/loss/backward job and inspect action-conditioning diagnostics.
- If the gate passes, run the formal PickCube stage from the non-contact base checkpoint.
- Fine-tune the resulting checkpoint on LiftCube with contact supervision.

### Task 6: Evaluation and handoff

- Report correct/zero/shuffled/persistence errors, shuffled degradation, validation losses, and auxiliary-head metrics.
- Run frozen probes on true and predicted latents.
- Only then run the same-state model-versus-simulator ranking audit.
- Preserve hashes and copy the most important final artifacts to A800 only when that server is reachable.

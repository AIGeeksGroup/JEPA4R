# LiftCube Latent CEM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure an isolated MPC runtime and measure whether the frozen step-1500 LiftCube world model can drive pure latent-cost receding-horizon CEM.

**Architecture:** A separate `/nix/lewm/4d-jepa-mpc` environment hosts the frozen world model runtime and ManiSkill2/SAPIEN without modifying the training environment. Online observations follow the compact-data point-cloud preprocessing, a torch-only CEM planner rolls `[N,2,10,7]` candidates from cached latent context, and the evaluator executes only the first 10-action chunk before replanning.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu121, spconv 2.3.6, ManiSkill2 0.5.3, SAPIEN 2.2.2, Gymnasium 1.3.0, pytest, A800 CUDA/Vulkan.

## Global Constraints

- Work only in `/workspace/users/lewm/projects/4d-jepa-architecture/.worktrees/liftcube-multistep-evaluation` on `codex/liftcube-multistep-eval`.
- Create backup branch `codex/liftcube-before-latent-cem-20260905` at design commit `fa23600` before implementation.
- Freeze `/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_action_sequence_batch8_2500/best.pt`, step 1500, SHA-256 `346ffd87ad74d9b55030258eb3c4cc66ab9b38a113d70e394b5679ab2976905d`.
- Use the checkpoint's compact data root `/workspace/datasets/maniskill2/v0.5.3/compact/LiftCube-v0` for the fixed validation-window cost diagnostic only; do not train or modify data.
- Do not modify `/nix/lewm/4d-jepa-worldmodel` or the dirty `/workspace/users/lewm/projects/4r-jepa-pt/.worktrees/maniskill2-group1` reference checkout.
- Keep the root `pyproject.toml`/`requirements.txt` as the upstream training contract; define the downstream runtime in `environments/mpc/pyproject.toml` and its matching `requirements-mpc.txt` lock, with `numpy==1.23.5` required by ManiSkill2 0.5.3.
- Install only under `/nix/lewm/4d-jepa-mpc`; keep install logs under `/workspace/users/lewm/runs/4d-jepa-architecture/mpc_environment/`.
- Stop before planning if SAPIEN/Vulkan reset-and-step smoke fails.
- Use pure latent cost only; do not add a state probe, privileged-state cost, geometric feedback, or hybrid phase controller.
- Smoke with seed 4100, 32 candidates, 2 iterations; formal evaluation uses seeds 4100-4104, 128 candidates, 5 iterations, 16 elites, horizon 2, chunk size 10, and at most 200 environment steps.
- Treat only `info["success"] == True` as task success.

---

### Task 1: Separate MPC Dependency Contract And Environment

**Files:**
- Modify: `pyproject.toml`
- Create: `environments/mpc/pyproject.toml`
- Create: `environments/mpc/apt-packages.txt`
- Create: `requirements-mpc.txt`
- Create: `scripts/setup_liftcube_mpc_env.sh`
- Create: `scripts/check_liftcube_mpc_env.py`
- Test: `tests/test_mpc_environment.py`

**Interfaces:**
- Consumes: repository `requirements.txt`, checkpoint path, and a Python 3.11 bootstrap.
- Produces: `/nix/lewm/4d-jepa-mpc/bin/python` and `check_environment(checkpoint: Path, require_renderer: bool) -> dict[str, object]`.

- [ ] **Step 1: Write failing dependency and setup tests**

Assert that `environments/mpc/pyproject.toml` exactly matches non-option entries in `requirements-mpc.txt`, repeats every upstream direct dependency except the incompatible NumPy pin, and uses `numpy==1.23.5`. Assert `environments/mpc/apt-packages.txt` locks `libgl1=1.4.0-1`, `libglib2.0-0=2.72.4-0ubuntu2.9`, and `libvulkan1=1.3.204.1-2`. Assert the setup targets `/nix/lewm/4d-jepa-mpc`, installs only `requirements-mpc.txt`, uses pip rather than uv, writes a persistent log, and never uses the world-model environment as its target. Import the checker and verify a fake checkpoint hash mismatch is an error.

```python
def test_mpc_requirements_are_standalone_and_model_compatible():
    upstream_project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    mpc_project = tomllib.loads(
        (ROOT / "environments/mpc/pyproject.toml").read_text()
    )
    locked = {
        line for line in (ROOT / "requirements-mpc.txt").read_text().splitlines()
        if line and not line.startswith(("#", "--"))
    }
    upstream = set(upstream_project["project"]["dependencies"])
    assert set(mpc_project["project"]["dependencies"]) == locked
    assert upstream - {"numpy==1.26.4"} <= locked
    assert "numpy==1.23.5" in locked

def test_setup_uses_separate_nix_environment():
    script = (ROOT / "scripts/setup_liftcube_mpc_env.sh").read_text()
    assert "env_dir=/nix/lewm/4d-jepa-mpc" in script
    assert "-r requirements.txt" not in script
    assert "-r requirements-mpc.txt" in script
    assert "/workspace/users/lewm/runs/4d-jepa-architecture/mpc_environment" in script
    assert "uv " not in script
```

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -p no:cacheprovider tests/test_mpc_environment.py -q
```

Expected: file-not-found or import failures because the MPC contract and checker do not exist.

- [ ] **Step 3: Declare pinned MPC dependencies**

Create a normal PEP 621 subproject at `environments/mpc/pyproject.toml` and mirror its direct dependency entries in `requirements-mpc.txt`. Include the upstream Torch/spconv/model pins, replace only `numpy==1.26.4` with `numpy==1.23.5`, and add these downstream packages:

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
spconv-cu120==2.3.6
timm==1.0.29
addict==2.4.0
einops==0.8.2
omegaconf==2.3.1
numpy==1.23.5
pytest==8.3.5
gymnasium==1.3.0
h5py==3.11.0
imageio==2.37.4
imageio-ffmpeg==0.6.0
mani-skill2==0.5.3
opencv-python==4.11.0.86
packaging==24.1
pyyaml==6.0.3
rtree==1.4.1
sapien==2.2.2
scipy==1.12.0
tqdm==4.66.5
transforms3d==0.4.2
trimesh==5.0.0
```

Add `planning` to the explicit setuptools package list.

- [ ] **Step 4: Implement the setup and checker**

The setup script must keep partial installs on failure and tee output to a timestamped persistent log:

```bash
env_dir=/nix/lewm/4d-jepa-mpc
log_root=/workspace/users/lewm/runs/4d-jepa-architecture/mpc_environment
python_bin="${MPC_BOOTSTRAP_PYTHON:-/nix/lewm/4d-jepa-worldmodel/bin/python}"
mkdir -p "$log_root"
"$python_bin" -m venv "$env_dir"  # no --system-site-packages; packages are not inherited
"$env_dir/bin/python" -m pip install --upgrade "pip==25.2" "setuptools==80.9.0" wheel
"$env_dir/bin/python" -m pip install -r requirements-mpc.txt
"$env_dir/bin/python" -m pip install -e . --no-deps
```

The checker validates Python 3.11, exact direct-package versions, CUDA 12.1 tensor arithmetic, checkpoint SHA and strict loading through `load_checkpoint_model`, ManiSkill2/SAPIEN imports, and optionally one headless `LiftCube-v0` reset/action-step/point-cloud extraction/close. It emits one JSON object and exits nonzero on any error.

- [ ] **Step 5: Verify GREEN and shell syntax**

```bash
PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -p no:cacheprovider tests/test_mpc_environment.py -q
bash -n scripts/setup_liftcube_mpc_env.sh
```

Expected: focused tests pass and shell parsing exits 0.

- [ ] **Step 6: Commit the environment contract**

```bash
git add pyproject.toml environments/mpc/pyproject.toml environments/mpc/apt-packages.txt requirements-mpc.txt scripts/setup_liftcube_mpc_env.sh scripts/check_liftcube_mpc_env.py tests/test_mpc_environment.py
git commit -m "build: add isolated LiftCube MPC environment"
```

- [ ] **Step 7: Install and verify the real environment**

```bash
bash scripts/setup_liftcube_mpc_env.sh
PYTHONNOUSERSITE=1 /nix/lewm/4d-jepa-mpc/bin/python scripts/check_liftcube_mpc_env.py --checkpoint /workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_action_sequence_batch8_2500/best.pt
PYTHONNOUSERSITE=1 /nix/lewm/4d-jepa-mpc/bin/python scripts/check_liftcube_mpc_env.py --checkpoint /workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_action_sequence_batch8_2500/best.pt --require-renderer
```

Expected: both JSON results contain `"ok": true`; the renderer result contains a finite point cloud and one completed step. Stop if it does not.

### Task 2: Online Point-Cloud Compatibility

**Files:**
- Create: `data/maniskill_online.py`
- Test: `tests/test_maniskill_online.py`

**Interfaces:**
- Consumes: `xyzw [N,4]`, `robot_seg [N]` or `[N,1]`, integer `sample_key`, and `point_count`.
- Produces: `preprocess_online_cloud(xyzw: np.ndarray, robot_seg: np.ndarray, *, sample_key: int, point_count: int) -> np.ndarray`, finite float32 `[point_count,3]`, robot-free, and within training bounds.

- [ ] **Step 1: Write failing preprocessing tests**

Build synthetic valid, invalid, out-of-bounds, and robot points. Assert deterministic output, sample-key sensitivity, robot exclusion, insufficient-point rejection, and shape `[512,3]`.

```python
def test_online_cloud_removes_robot_points_and_is_deterministic():
    first = preprocess_online_cloud(xyzw, robot_seg, sample_key=4100, point_count=512)
    second = preprocess_online_cloud(xyzw, robot_seg, sample_key=4100, point_count=512)
    assert first.shape == (512, 3)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    assert not np.any(np.all(first[:, None] == robot_xyz[None], axis=-1))
```

- [ ] **Step 2: Verify RED**

Run the new test file and expect import failure for `data.maniskill_online`.

- [ ] **Step 3: Implement the two-stage sampler**

Filter finite points with positive `w`, enforce bounds `[-0.75,-0.60,-0.05]` to `[0.75,0.60,0.80]`, remove `robot_seg`, deterministically sample 1024 scene points using `3072 + sample_key`, then call the current `_sample_points` to reduce to checkpoint `K`.

- [ ] **Step 4: Verify GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -p no:cacheprovider tests/test_maniskill_online.py -q
git add data/maniskill_online.py tests/test_maniskill_online.py
git commit -m "feat: preprocess online LiftCube point clouds"
```

### Task 3: Torch Latent CEM Core

**Files:**
- Create: `planning/__init__.py`
- Create: `planning/latent_cem.py`
- Test: `tests/test_latent_cem.py`

**Interfaces:**
- Produces: `CEMConfig`, `normalized_latent_mse(pred: torch.Tensor, goal: torch.Tensor) -> torch.Tensor`, `rollout_from_latent(model, latent_history: torch.Tensor, context_actions: torch.Tensor, action_candidates: torch.Tensor, env: str) -> torch.Tensor`, and `LatentCEMPlanner.plan(model, latent_history: torch.Tensor, context_actions: torch.Tensor, goal_latent: torch.Tensor, *, seed: int) -> torch.Tensor` with shape `[H,chunk,7]`.
- Consumes: latent history `[1,T,K,D]`, action history `[1,(T-1)*chunk,7]`, goal latent `[1,K,D]`, action mean/std, and integer seed.

- [ ] **Step 1: Write failing shape, rollout, and CEM tests**

Use a tiny deterministic fake model whose next latent equals the previous latent plus a weighted action sum. Assert exact H2 rollout values, action-history shifts by one chunk, seeded determinism, bounded arm actions, one repeated binary gripper value per chunk, and lower fake goal cost than zero actions.

```python
def test_cem_plan_is_binary_gripper_and_reduces_cost():
    plan = planner.plan(model, history, action_history, goal, seed=17)
    assert plan.shape == (2, 10, 7)
    assert torch.all((plan[..., -1] == -1) | (plan[..., -1] == 1))
    assert torch.all(plan[:, :, -1] == plan[:, :1, -1])
    assert cost(plan) < cost(torch.zeros_like(plan))
```

- [ ] **Step 2: Verify RED**

Run the focused test and expect import failure for `planning.latent_cem`.

- [ ] **Step 3: Implement cached rollout and pure latent cost**

For each candidate chunk, concatenate context action history with the candidate chunk, call `model._predict_sequence`, retain its final prediction, shift latent history one frame, and shift action history ten atomic actions. Normalize vectors along `D`; cost candidates as terminal MSE plus `0.25 * mean horizon MSE`.

- [ ] **Step 4: Implement constrained CEM**

Initialize arm means/stds from checkpoint action statistics with minimum std `0.03`. Sample arm dimensions with a seeded device generator and clamp to `[-1,1]`. Sample one Bernoulli gripper state per chunk, expand it over ten actions, select finite elites, update arm moments and gripper probabilities, and return final arm mean plus thresholded binary gripper.

- [ ] **Step 5: Verify GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -p no:cacheprovider tests/test_latent_cem.py -q
git add planning/__init__.py planning/latent_cem.py tests/test_latent_cem.py
git commit -m "feat: add skip10 latent CEM planner"
```

### Task 4: Closed-Loop LiftCube Evaluator

**Files:**
- Create: `scripts/eval_liftcube_latent_cem.py`
- Test: `tests/test_liftcube_latent_cem_eval.py`

**Interfaces:**
- Produces: `make_env(task: str = "LiftCube-v0")`, `build_goal_cloud(env, sample_key: int) -> np.ndarray`, `score_validation_reference(model, state: dict, device: torch.device) -> dict[str, float]`, `run_episode(env, model, planner, *, env_seed: int, cem_seed: int, max_steps: int) -> dict[str, object]`, and a CLI.
- Consumes: online preprocessor, latent CEM planner, checkpoint loader, and ManiSkill2 Gym API.

- [ ] **Step 1: Write failing fake-environment tests**

Assert goal capture restores complete state, warmup produces frames at 0/10/20 and 20 context actions, replanning occurs after ten executed actions, context/action histories shift together, termination records current success, and replan records include predicted cost, cube height, reward, and action statistics.

- [ ] **Step 2: Verify RED**

Run the focused evaluator tests and expect import failure for the new script.

- [ ] **Step 3: Implement environment compatibility and goal capture**

Register ManiSkill2, patch observation-wrapper `robot_link_ids` forwarding before `gym.make`, create `LiftCube-v0` in point-cloud robot-seg mode, preprocess every observation, and restore `base.get_state()` in `finally` after teleporting the cube to `goal_pos`.

Implement `score_validation_reference` with the first deterministic H2 window from `/workspace/datasets/maniskill2/v0.5.3/compact/LiftCube-v0`: aggregate its real target clouds with `model._aggregate`, roll out its recorded 20 atomic actions from the real T=3 context and 20 context actions, and return the same normalized terminal-plus-mean latent cost used by CEM. This is reporting-only and must never enter the planner objective.

- [ ] **Step 4: Implement skip-10 receding-horizon execution**

Warm up with `[0,0,0,0,0,0,1]` for 20 steps. Aggregate frames 0/10/20 per replan, call CEM with exact context actions, execute `plan[0]` for at most ten actions, append the real boundary observation/actions, and stop on success, termination, truncation, or 200 total steps.

- [ ] **Step 5: Implement CLI and durable output**

Require checkpoint, accept seeds/CEM settings/max steps/output, write configuration and provenance before execution, append and flush one JSON object per episode, and atomically write a summary with success count/rate, initial/zero/CEM costs, and the fixed validation demonstrated-action cost.

- [ ] **Step 6: Verify GREEN and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -p no:cacheprovider tests/test_liftcube_latent_cem_eval.py -q
git add scripts/eval_liftcube_latent_cem.py tests/test_liftcube_latent_cem_eval.py
git commit -m "feat: evaluate LiftCube with latent CEM"
```

### Task 5: Recoverable Runner And Full Verification

**Files:**
- Create: `scripts/run_liftcube_latent_cem_experiment.sh`
- Modify: `tests/test_mpc_environment.py`
- Modify: `docs/superpowers/plans/2026-09-05-liftcube-latent-cem.md` only for evidence.

**Interfaces:**
- Produces: `/workspace/users/lewm/runs/4d-jepa-architecture/liftcube_latent_cem_best1500_5seeds` plus adjacent `.status`, `.pid`, and `.log` files.

- [ ] **Step 1: Write a failing runner contract test**

Assert the runner uses `/nix/lewm/4d-jepa-mpc/bin/python`, exact checkpoint, seeds 4100-4104, formal CEM settings, atomic RUNNING/FAILED/COMPLETE status, PID, and non-overwrite guard.

- [ ] **Step 2: Verify RED**

Run the runner contract test and expect file-not-found failure.

- [ ] **Step 3: Implement the runner**

Use `set -euo pipefail`, refuse existing output or live PID, line-buffer logs, record Git/checkpoint provenance, trap failures to atomically write FAILED, run the evaluator once, and write COMPLETE only after summary creation.

- [ ] **Step 4: Run full verification**

```bash
PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -p no:cacheprovider -q
PYTHONPYCACHEPREFIX=/dev/shm/liftcube-mpc-pycache /nix/lewm/4d-jepa-worldmodel/bin/python -m py_compile data/maniskill_online.py planning/latent_cem.py scripts/check_liftcube_mpc_env.py scripts/eval_liftcube_latent_cem.py
bash -n scripts/setup_liftcube_mpc_env.sh scripts/run_liftcube_latent_cem_experiment.sh
git diff --check
```

Expected: all tests pass, shell/Python parsing exits 0, and diff check is empty. Remove `/dev/shm/liftcube-mpc-pycache` afterward.

- [ ] **Step 5: Commit runner and evidence**

```bash
git add scripts/run_liftcube_latent_cem_experiment.sh tests/test_mpc_environment.py docs/superpowers/plans/2026-09-05-liftcube-latent-cem.md
git commit -m "chore: add recoverable latent CEM runner"
```

### Task 6: Renderer Smoke, One-Episode Smoke, And Five Seeds

**Files:**
- Create: `/workspace/users/lewm/runs/4d-jepa-architecture/liftcube_latent_cem_smoke_4100/*`
- Create: `/workspace/users/lewm/runs/4d-jepa-architecture/liftcube_latent_cem_best1500_5seeds/*`
- Modify: `docs/superpowers/plans/2026-09-05-liftcube-latent-cem.md` only for terminal evidence.

- [ ] **Step 1: Re-run the real environment gate**

Run `check_liftcube_mpc_env.py --require-renderer` in the isolated MPC environment immediately before evaluation. Record package versions, CUDA device, checkpoint SHA, observation shape, and one-step result.

- [ ] **Step 2: Run one bounded smoke episode**

Use seed 4100, 32 candidates, 2 iterations, 4 elites, horizon 2, chunk 10, and max 60 total steps. Require finite costs, binary gripper, at least two replans, advancing environment steps, summary JSON, and no duplicate process.

- [ ] **Step 3: Evaluate the smoke result**

Stop on renderer/runtime/tensor failures or non-finite costs and report exact logs. A valid but unsuccessful episode may proceed because success is not the smoke criterion.

- [ ] **Step 4: Launch the five-seed run**

After checking live A800 GPU availability and absence of the named PID/run directory, launch exactly one detached runner on an idle GPU without automatic restart.

- [ ] **Step 5: Confirm stability and exit**

Confirm RUNNING status, live PID, GPU allocation, and at least two replans or one completed episode durably recorded. Leave the process detached rather than polling.

- [ ] **Step 6: Report terminal results**

On COMPLETE, report success count, per-seed success/steps/final cube height, initial/zero/CEM costs, and whether falling latent cost corresponded to real lifting. On FAILED, report exact exception/log tail. Clean only agent-created cache/upload files, verify Git, and stop terminal monitoring.

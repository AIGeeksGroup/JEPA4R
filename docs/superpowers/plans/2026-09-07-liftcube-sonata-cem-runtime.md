# LiftCube Sonata CEM Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Run a recoverable ten-seed LiftCube CEM evaluation on A800 with the frozen Sonata checkpoint and population-128 search.

**Architecture:** Preserve all 128 candidates and global elite selection, but split only world-model forwards into batches of at most 32. Keep the training environment untouched and make the separate MPC environment capable of loading Sonata before launching a GPU-2-only runner with durable status, log, PID, and per-episode output.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu121, Sonata/PTV3, torch-scatter, ManiSkill2 0.5.3, SAPIEN 2.2.2, pytest, Bash.

## Global Constraints

- Work only in '/workspace/users/lewm/projects/4d-jepa-architecture/.worktrees/liftcube-sonata-ptv3'.
- Keep '/nix/lewm/4d-jepa-worldmodel' unchanged; modify/install only '/nix/lewm/4d-jepa-mpc'.
- Use the migrated 'group1_skip10_sonata_frozen_batch8_2500/best.pt' with SHA-256 '38ca936f63af3fa552d0d8201876f5290f4a627f5e3e81e05d96963af4660b32'.
- Preserve candidates 128, iterations 5, elites 16, horizon 2, chunk 10, and maximum 200 steps.
- Limit only model forwards to 32; select elites globally over all 128 costs.
- Recheck GPU ownership immediately before launch and use physical GPU 2 only if idle.
- Never overwrite an existing run or start a duplicate process.
- Physical simulator success, not finite latent cost, is the primary outcome.

---

### Task 1: Micro-batch CEM model forwards

**Files:**
- Modify: 'tests/test_latent_cem.py'
- Modify: 'planning/latent_cem.py'

**Interfaces:**
- Consumes: 'rollout_from_latent(...) -> Tensor[N,H,K,D]' and 'latent_trajectory_cost(...) -> Tensor[N]'.
- Produces: 'CEMConfig.model_batch_size: int = 32'; unchanged 'LatentCEMPlanner.plan(...) -> Tensor[H,S,A]'.

- [ ] **Step 1: Write failing tests**

~~~python
def test_cem_microbatches_population_before_model_forward() -> None:
    config = CEMConfig(
        horizon=2,
        chunk_size=2,
        action_dim=2,
        candidates=128,
        iterations=1,
        elites=16,
        model_batch_size=32,
    )
    model = FakeModel(chunk_size=2)
    planner = LatentCEMPlanner(config, torch.zeros(2), torch.ones(2))
    planner.plan(
        model,
        _history(),
        torch.zeros(1, 4, 2),
        torch.tensor([[[0.0, 1.0]]]),
        env="maniskill2",
        seed=17,
    )
    batches = [actions.shape[0] for actions in model.seen_actions]
    assert max(batches) <= 32
    assert 32 in batches


def test_cem_rejects_nonpositive_model_batch_size() -> None:
    with pytest.raises(ValueError, match="model_batch_size"):
        CEMConfig(model_batch_size=0)
~~~

- [ ] **Step 2: Run RED**

~~~bash
PYTHONNOUSERSITE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest \
  tests/test_latent_cem.py::test_cem_microbatches_population_before_model_forward \
  tests/test_latent_cem.py::test_cem_rejects_nonpositive_model_batch_size -q
~~~

Expected: failure because 'CEMConfig' does not accept 'model_batch_size'.

- [ ] **Step 3: Implement minimal batching**

Add:

~~~python
model_batch_size: int = 32

if self.model_batch_size < 1:
    raise ValueError("model_batch_size must be positive")
~~~

Replace the single population rollout inside each CEM iteration with:

~~~python
costs = torch.cat(
    [
        latent_trajectory_cost(
            rollout_from_latent(
                model, latent_history, context_actions, action_batch, env
            ),
            goal_latent,
            trajectory_weight=cfg.trajectory_weight,
        )
        for action_batch in actions.split(cfg.model_batch_size)
    ]
)
~~~

- [ ] **Step 4: Run GREEN**

~~~bash
PYTHONNOUSERSITE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest tests/test_latent_cem.py -q
~~~

Expected: all planner tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add planning/latent_cem.py tests/test_latent_cem.py
git commit -m "fix: microbatch latent CEM rollouts"
~~~

---

### Task 2: Expose batching and adapt the isolated MPC contract

**Files:**
- Modify: 'scripts/eval_liftcube_latent_cem.py'
- Modify: 'scripts/check_liftcube_mpc_env.py'
- Modify: 'scripts/setup_liftcube_mpc_env.sh'
- Modify: 'requirements-mpc.txt'
- Modify: 'environments/mpc/pyproject.toml'
- Modify: 'tests/test_liftcube_latent_cem_eval.py'
- Modify: 'tests/test_mpc_environment.py'

**Interfaces:**
- Consumes: 'CEMConfig.model_batch_size'.
- Produces: '--model-batch-size'; '--expected-checkpoint-sha256'; an MPC lock that imports Sonata without installing the root training distribution.

- [ ] **Step 1: Write failing contract tests**

Require both MPC dependency files to contain:

~~~text
huggingface-hub==1.29.0
torch-scatter @ https://data.pyg.org/whl/torch-2.5.0+cu121/torch_scatter-2.1.2+pt25cu121-cp311-cp311-linux_x86_64.whl
sonata @ git+https://github.com/facebookresearch/sonata.git@18c09ff8d713494f78a8213792262b910977a65d
~~~

Also require the setup script not to contain 'pip install -e .'; test 'check_environment(..., expected_checkpoint_sha256=...)'; and test that '--model-batch-size 7' reaches 'CEMConfig' and provenance.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONNOUSERSITE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest \
  tests/test_mpc_environment.py tests/test_liftcube_latent_cem_eval.py -q
~~~

Expected: failures for missing dependencies, checker argument, and evaluator option.

- [ ] **Step 3: Update dependency/setup contracts**

Add the three exact entries to 'requirements-mpc.txt' and 'environments/mpc/pyproject.toml', retaining 'numpy==1.23.5'. Resolve 'repo_dir' from 'BASH_SOURCE[0]' and remove the editable root-package install. Continue installing only 'requirements-mpc.txt' into '/nix/lewm/4d-jepa-mpc'.

- [ ] **Step 4: Update checker and evaluator**

Use:

~~~python
def check_environment(
    checkpoint: Path,
    require_renderer: bool,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
) -> dict[str, Any]:
~~~

Add the checker CLI option with the legacy hash as default, compare the actual hash to the argument, and verify Sonata, torch-scatter, and huggingface-hub.

Add:

~~~python
parser.add_argument("--model-batch-size", type=int, default=32)
~~~

and pass it to 'CEMConfig'; 'vars(config)' records it.

- [ ] **Step 5: Run GREEN**

~~~bash
PYTHONNOUSERSITE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest \
  tests/test_mpc_environment.py tests/test_liftcube_latent_cem_eval.py \
  tests/test_latent_cem.py -q
~~~

- [ ] **Step 6: Commit**

~~~bash
git add requirements-mpc.txt environments/mpc/pyproject.toml \
  scripts/setup_liftcube_mpc_env.sh scripts/check_liftcube_mpc_env.py \
  scripts/eval_liftcube_latent_cem.py tests/test_mpc_environment.py \
  tests/test_liftcube_latent_cem_eval.py
git commit -m "feat: support Sonata in isolated MPC runtime"
~~~

---

### Task 3: Add the recoverable ten-seed Sonata runner

**Files:**
- Create: 'scripts/run_liftcube_sonata_latent_cem_10seeds.sh'
- Modify: 'tests/test_mpc_environment.py'

**Interfaces:**
- Consumes: Task 2 options and migrated Sonata 'best.pt'.
- Produces: 'liftcube_sonata_latent_cem_best_10seeds' plus sibling PID, log, status, and launcher log.

- [ ] **Step 1: Write the failing runner test**

Assert the new script contains the exact checkpoint/hash, seeds 4100-4109, candidates 128, model batch size 32, iterations 5, elites 16, horizon 2, chunk 10, max steps 200, 'CUDA_VISIBLE_DEVICES=2', renderer preflight, duplicate-PID detection, non-overwrite guards, and atomic terminal status.

- [ ] **Step 2: Run RED**

~~~bash
PYTHONNOUSERSITE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest \
  tests/test_mpc_environment.py::test_sonata_latent_cem_runner_is_recoverable_and_uses_ten_seeds -q
~~~

Expected: failure because the runner is absent.

- [ ] **Step 3: Create the runner**

Follow the existing recoverable runner and invoke:

~~~bash
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 stdbuf -oL -eL "$python_bin" \
  -m scripts.eval_liftcube_latent_cem \
  --checkpoint "$checkpoint" --output "$run_dir" \
  --seeds 4100 4101 4102 4103 4104 4105 4106 4107 4108 4109 \
  --cem-seed 31000 --candidates 128 --model-batch-size 32 \
  --iterations 5 --elites 16 --horizon 2 --chunk-size 10 \
  --max-steps 200 --device cuda
~~~

Run the checker with '--require-renderer' and the exact Sonata hash first.

- [ ] **Step 4: Run GREEN and shell syntax**

~~~bash
PYTHONNOUSERSITE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest \
  tests/test_mpc_environment.py -q
bash -n scripts/run_liftcube_sonata_latent_cem_10seeds.sh
~~~

- [ ] **Step 5: Commit**

~~~bash
git add scripts/run_liftcube_sonata_latent_cem_10seeds.sh tests/test_mpc_environment.py
git commit -m "feat: add Sonata CEM evaluation runner"
~~~

---

### Task 4: Verify code and update only the MPC environment

**Files:**
- Runtime target: '/nix/lewm/4d-jepa-mpc'
- Persistent log: '/workspace/users/lewm/runs/4d-jepa-architecture/mpc_environment/'

- [ ] **Step 1: Verify repository**

~~~bash
PYTHONNOUSERSITE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q
PYTHONPYCACHEPREFIX=/dev/shm/liftcube-sonata-cem-pycache \
  /nix/lewm/4d-jepa-worldmodel/bin/python -m py_compile \
  planning/latent_cem.py scripts/eval_liftcube_latent_cem.py \
  scripts/check_liftcube_mpc_env.py
git diff --check
~~~

- [ ] **Step 2: Update only MPC environment**

Log these commands persistently, uninstalling only the stale editable root distribution:

~~~bash
/nix/lewm/4d-jepa-mpc/bin/python -m pip uninstall -y pointcloud-world-model
/nix/lewm/4d-jepa-mpc/bin/python -m pip install -r requirements-mpc.txt
~~~

- [ ] **Step 3: Verify environment and renderer**

~~~bash
/nix/lewm/4d-jepa-mpc/bin/python -m pip check
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
  /nix/lewm/4d-jepa-mpc/bin/python scripts/check_liftcube_mpc_env.py \
  --checkpoint /workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_sonata_frozen_batch8_2500/best.pt \
  --expected-checkpoint-sha256 38ca936f63af3fa552d0d8201876f5290f4a627f5e3e81e05d96963af4660b32 \
  --require-renderer
~~~

Expected: 'pip check' is clean and checker JSON has '"ok": true', strict load, CUDA tensor result 6.0, and finite point cloud.

---

### Task 5: Smoke and launch the formal evaluation

**Runtime outputs:**
- Smoke: '/workspace/users/lewm/runs/4d-jepa-architecture/liftcube_sonata_latent_cem_smoke'
- Formal: '/workspace/users/lewm/runs/4d-jepa-architecture/liftcube_sonata_latent_cem_best_10seeds'

- [ ] **Step 1: Recheck GPU 2**

~~~bash
nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
~~~

Require no external compute process and at least 90% free memory; otherwise stop without preemption.

- [ ] **Step 2: Run non-overwriting smoke**

Run seed 4100 with candidates 32, model batch 32, one iteration, four elites, horizon 2, and 40 steps. Require config, one fsynced episode row, finite costs, and no CUDA/renderer exception. Do not interpret this as physical success evidence.

- [ ] **Step 3: Launch formal runner detached**

~~~bash
nohup env CUDA_VISIBLE_DEVICES=2 \
  bash scripts/run_liftcube_sonata_latent_cem_10seeds.sh \
  > /workspace/users/lewm/runs/4d-jepa-architecture/liftcube_sonata_latent_cem_best_10seeds.launcher.log \
  2>&1 < /dev/null &
~~~

Verify exactly one PID, RUNNING status, live log, and GPU-2 allocation. Once stable, exit instead of continuously polling.

- [ ] **Step 4: Interpret terminal results**

On COMPLETE, report success count/rate, per-seed success and final cube height, CEM versus zero-action costs, validation demonstrated/zero/persistence costs, and action saturation. On FAILED, report the exact traceback. Finite costs alone do not establish control.

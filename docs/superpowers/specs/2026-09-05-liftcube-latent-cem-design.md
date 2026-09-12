# LiftCube Latent CEM Evaluation Design

## Objective

Evaluate whether the frozen batch-8 action-conditioned world model can support
closed-loop CEM control in the real ManiSkill2 `LiftCube-v0` environment. This
is a diagnostic of the step-1500 checkpoint itself, not a hybrid-controller or
learned-state-probe experiment.

## Fixed Inputs

- Worktree: `/workspace/users/lewm/projects/4d-jepa-architecture/.worktrees/liftcube-multistep-evaluation`
- Checkpoint: `/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_action_sequence_batch8_2500/best.pt`
- Checkpoint step: 1500
- Checkpoint SHA-256: `346ffd87ad74d9b55030258eb3c4cc66ab9b38a113d70e394b5679ab2976905d`
- Task and control mode: `LiftCube-v0`, `pd_ee_delta_pose`
- Observation mode: `pointcloud_robot_seg`
- Evaluation seeds: 4100 through 4104

The old `/workspace/users/lewm/projects/4r-jepa-pt` MPC code is read-only
reference material. It targets a different FourRJEPA interface with five-action
blocks and must not be imported into this experiment.

## Separate MPC Environment

Create `/nix/lewm/4d-jepa-mpc` as a new Python 3.11 environment. Do not modify
or inherit packages from `/nix/lewm/4d-jepa-worldmodel`. Keep the root
`pyproject.toml` and `requirements.txt` as the upstream world-model contract.
Define the downstream environment separately in
`environments/mpc/pyproject.toml`, with `requirements-mpc.txt` as its matching
pip-installable lock. The downstream lock repeats the required model stack but
uses `numpy==1.23.5`, because `mani-skill2==0.5.3` declares `numpy<1.24`; all
other shared direct dependencies retain the upstream versions. Install only
the downstream lock, then install the current project editable with
`--no-deps`.

Keep non-Python renderer dependencies in `environments/mpc/apt-packages.txt`:
`libgl1=1.4.0-1`, `libglib2.0-0=2.72.4-0ubuntu2.9`, and
`libvulkan1=1.3.204.1-2`. These are system runtime libraries, not packages in
either Python environment.

Keep the installation log under
`/workspace/users/lewm/runs/4d-jepa-architecture/mpc_environment/`. If a large
CUDA wheel download fails, resume from the exact failed locked package instead
of recreating the environment.

The environment is usable only after all of these pass:

1. exact Python and package version checks;
2. a CUDA tensor operation on an A800;
3. current-project imports and strict loading of the named checkpoint;
4. ManiSkill2 and SAPIEN imports;
5. creation, reset, one action step, point-cloud extraction, and close of a
   headless `LiftCube-v0` environment.

SAPIEN/Vulkan failure is an environment failure, not an MPC result. Stop before
planning if the renderer smoke test fails.

## Online Observation Pipeline

Use the same world-space filtering and robot-point removal as the compact
dataset producer, then deterministic sampling to 1024 scene points and the
current world model's deterministic sampling to `K=512`. The model receives
only XYZ, matching checkpoint `C=3`.

After reset, execute 20 neutral open-gripper actions. Record observations at
steps 0, 10, and 20 and retain all 20 executed atomic actions. This supplies the
model's exact `T=3`, `skip_frame=10` context instead of inventing missing action
history.

Construct the goal point cloud by temporarily moving the cube to the
environment goal, capturing one observation, and restoring the complete
simulator state. Aggregate this goal cloud with the frozen spatial aggregator.
The CEM objective compares predicted spatial latents to this goal spatial
latent; it must not compare against the temporal `model.encode()` output.

## Latent CEM

Encode the three context clouds once per replan. A candidate tensor has shape
`[N, 2, 10, 7]`: two model transitions and ten ordered atomic actions per
transition. Roll out candidates directly from cached latent context using the
model's `_predict_sequence`, shifting both latent and action histories after
each transition. Do not re-encode point clouds per candidate.

Use a pure latent objective:

`terminal normalized latent MSE + 0.25 * mean normalized latent MSE`

Report initial-to-goal, zero-action, and best CEM costs online. Separately score
one fixed validation window with its recorded demonstrated action sequence so
the planner cost has an in-distribution reference. These diagnostics keep cost
insensitivity or model exploitation visible.

Initialize arm-action Gaussian means and standard deviations from checkpoint
training statistics, apply a minimum standard deviation of `0.03`, and clamp
actions to the environment bounds. Sample one categorical gripper value in
`{-1, +1}` per 10-action chunk, repeat it through that chunk, and update its
close probability from elites. Never produce continuous intermediate gripper
commands that are absent from the demonstrations. Use horizon 2 because the
checkpoint passed the shuffled action criterion at H1/H2 but not H4.

For the smoke run use 32 candidates, 2 iterations, and one seed. For the fixed
evaluation use 128 candidates, 5 iterations, 16 elites, and seeds 4100 through
4104. Execute only the first 10-action block, observe the real environment,
shift the context, and replan until success, termination, or 200 control steps.

## Outputs And Interpretation

Write a unique run directory with configuration, checkpoint provenance,
per-episode JSONL, status, PID, and log files. Per replan record predicted cost,
actual reward/success, cube height, and executed action statistics. Preserve
partial results after every episode.

The only success criterion is ManiSkill2 `info["success"] == True`. Lower CEM
cost, predicted lift, or small latent distance is not task success. Report the
fixed five-seed success count even if it is 0/5. A successful geometric or
hybrid fallback must not be included in this experiment because it would hide
whether the frozen latent world model is useful for planning.

## Verification And Failure Handling

Use test-first development for checkpoint loading, action chunk shapes, latent
rollout, categorical gripper output, context/action-history shifting, atomic
result writes, and fake-environment receding-horizon execution. Run focused
tests, the complete pytest suite, `py_compile`, `bash -n`, and `git diff --check`
before the renderer smoke test.

Do not start a second process if a PID or non-empty run directory exists. For a
long five-seed evaluation, launch one detached runner, verify two advancing log
or result records, and exit the session. Remove only agent-created upload files,
temporary staging, and Python caches after terminal completion.

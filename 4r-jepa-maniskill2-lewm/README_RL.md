# Isolated ManiSkill2 pointcloud PPO

The baseline is the official `haosulab/ManiSkill2-Learn` repository, commit
`e9860aff595da3ad34829d62c923be0b50001427`, using
`configs/mfrl/ppo/maniskill2_pn.py`. The vendor repository is never edited.
PyTorch3D's pure PyTorch transforms are imported from its unchanged official
v0.7.7 source (`89653419d0973396f3eff1a381ba09a07fffc2ed`); compiled 3D ops are
not used by the selected PointNet baseline.

The chosen task is `PickCube-v0`, `pointcloud`, `pd_ee_delta_pose`, with
1200 xyz+RGB points in the world coordinate frame. `world` is the effective
upstream `make_gym_env` default; this is explicitly recorded in the launcher.
The current shared encoder uses `configs/ppo_pointnet_identity.py`, which inherits
the official configuration and selects external `PointNetLastIdentity`. Channels
stay `[6,64,128,512]` with LN1d, activations `[ReLU,ReLU,Identity]`, global max
pooling, and no feature transform. Its pooled output is one 512-vector.
Only the final activation is replaced; the official forward is inherited and
the vendor files are unchanged. PPO and WM instantiate this same config.
RL also concatenates proprioceptive state in its Visuomotor policy/value heads.
The actual environment supplies 30 proprioceptive coordinates, so the critic
head is `[542,256,256,1]` with two ReLU hidden layers.

## Reproduction

All commands run inside this new task directory. Nothing is installed into the
global Python environment. The setup uses Python 3.10 and requires an NVIDIA
Vulkan driver for the SAPIEN renderer. Production training defaults to CUDA;
CPU is used only for smoke checks or the explicitly selected train-cpu mode.

```bash
bash scripts/setup_rl.sh
bash scripts/setup_rl_gpu.sh
bash scripts/run_rl.sh train

# Explicit CPU compatibility checks (rendering still requires free VRAM):
bash scripts/run_rl.sh smoke
bash scripts/run_rl.sh eval-smoke
```

`run_rl.sh` without a mode also selects production GPU training. It uses the
isolated `.venv-rl-gpu` (PyTorch2.8/CUDA12.8), while `.venv-rl` remains the
validated CPU environment. A missing CUDA environment or insufficient VRAM
causes a clear failure; no silent CPU fallback occurs. Preview the selected
command without starting anything using `bash scripts/run_rl.sh --print-command`.

For process/memory records as well, use `.venv-rl/bin/python
scripts/run_rl_checked.py smoke` and `eval-smoke` after sourcing
`scripts/rl_env.sh`.

The current smoke command targets `artifacts/rl_identity_smoke`; full training
targets `artifacts/rl_identity_train`. Existing `artifacts/rl_smoke` remains the
historical original-ReLU baseline. Historical tanh configs still select tanh;
their checkpoints and evidence are preserved.

The smoke command collects 32 actual simulator transitions, performs the
official PPO update, and saves official checkpoints. It keeps the 1200-point
encoder and official PPO epochs/losses, with a 16-step smoke episode horizon,
16-sample minibatch, and one environment. Evaluation runs in a separate process
so only one renderer is alive at a time. Smoke results do not establish task
success or convergence. See `artifacts/rl_smoke` and the saved verification
report for observed results.

Historical ReLU run verified on 2026-09-06: the original encoder received xyz/RGB tensors
`[1,1200,3]` and emitted `[1,512]`; all encoder/actor/critic parameters remained
finite and all three groups changed after PPO optimization. The 32-step
training run took 8.70 seconds inside the training function and approximately
15 seconds including setup. Separate checkpoint reload/evaluation completed
two 16-step episodes (returns 0.94 and 1.09, success 0/2). The renderer used a
sampled peak of 154,066,944 GPU bytes (~147 MiB); the network stayed on CPU.
This verifies execution and reload, not convergence or an expert policy.

The Identity version also has a CPU-only optimizer compatibility check in
`scripts/check_identity_ppo_replay.py`, using explicitly labeled test inputs,
the official PPO update, finite changes to encoder/actor/critic, and exact
model/optimizer reload. The latest GPU timing runs described below additionally
collected real simulator transitions and trained the Identity policy on CUDA;
they verify execution and throughput without establishing convergence.

Saved evidence: `artifacts/rl_smoke/verification.json`,
`artifacts/rl_smoke/models/model_final.ckpt`,
`artifacts/rl_smoke/test/statistics.csv`, and
`artifacts/rl_{smoke,eval-smoke}_resources.json`.
`real_transitions.npz` contains the actual 32 xyz/RGB/action transitions solely
for model integration checks; it is not the LeWM training dataset.

For sustained training, `bash scripts/run_rl.sh train` retains upstream's
5,000,000-step budget and PPO optimization defaults, with **16 parallel
environments** and a CUDA model on this host, as explicitly requested. This is
a local runtime override of upstream's five environments. The host provides a
16-core CPU quota; the environment workers share those cores with the PPO
driver and IPC. Sixteen environments is the selected configuration, not a
claim of a measured optimum.
CPU production is available only through `bash scripts/run_rl.sh train-cpu`,
with separate output.
The underlying upstream CLI accepts configuration
overrides; use the saved resolved config and a matching checkpoint when
resuming/evaluating. The default smoke and training output directories differ.

On this host's dedicated RTX 5090, two bounded 6,600-step GPU runs on
2026-09-07 measured about 280 control steps/s with five environments and
131 steps/s with one environment, including PPO updates after initial critic
warmup. Both retained 1,200 points, minibatches of 330 and two PPO epochs;
only the rollout window was shortened from 20,000 to 3,300 for timing.
The historical five-environment result projects to 4.96 hours for 5,000,000
steps (5–7 hours including operational overhead). Fitting
`time_per_step(n) = serial + parallel / n` to these two measurements gives
about **335 steps/s and 4.14 hours at 12 environments**, with a historical
4–6-hour scheduling estimate. That projection and the subsequent 12-environment
measurements below do not establish the throughput of the newly selected
16-environment configuration. GPU batching, IPC and the full 20,000-step
rollout window can change scaling. It estimates completion of the step budget, not
time to a guaranteed success rate. No additional environment-count sweep is
required or performed. See `validation/runtime-estimates.json`.

A **12-environment production run was started on 2026-09-07 at
02:19:18 UTC**, in `artifacts/rl_identity_train_p12_20260907`, with a
5,000,000-step budget and 20,000-step rollout window. It was subsequently
**stopped at the user's request**; the requested scope was timing measurements,
and full-budget training has not completed. The stop record is
`artifacts/rl_identity_train_p12_20260907/stopped_by_user.json`. That
12-environment run remains stopped. The user has since explicitly requested
a new 16-environment production run, **started on 2026-09-07 at
03:43:27.732030 UTC** (launcher PID 152283), in the separate directory
`artifacts/rl_identity_train_p16_20260907`. Its [launch record](artifacts/rl_identity_train_p16_20260907/launch.json)
records CUDA, 5,000,000 total steps, 20,000 rollout steps, minibatch 330 and
two PPO epochs. It is running with the pointcloud recording hook; this
startup does not establish 16-environment throughput or final success.

The first 20,000 transitions and PPO optimization took 78.575s (57.204s
collecting), including the one-time four-epoch critic warmup. At the recorded
220,000-step update, the post-warmup median was 64.9195s per 20,000 steps,
projecting to about **4.51 hours** for the full budget. These retained timing
observations are not evidence of a completed reproduction or convergence.
The GPU utilization log snapshot was 70%, not a whole-run average; no
environment-count benchmarking sequence was launched.

A separate bounded 40,000-step profile recorded resources every 100 ms and then exited. With 12 environments, rollout averaged 17.0% GPU utilization (observed peak 19%), while PPO updates averaged 72.3% (observed peak 92%). CPU usage against the 16-core quota averaged 61.9% in rollout and 81.5% in updates. These phases must not be confused with one whole-run utilization figure; setup and checkpoint/trace finalization are excluded. See `validation/resource-profiles.json` and `validation/resource-profiles.png`. This bounded profiling run has exited.

## Pointcloud input videos

Set `MS2_PPO_RECORD_VIDEO` to an output MP4 path when starting PPO to record
one real rollout episode from one environment. For example:

```bash
MS2_PPO_RECORD_VIDEO=artifacts/pointcloud_videos/ppo_rollout_pointcloud.mp4 \
  bash scripts/run_rl.sh train
```

The recorder copies the actual 1,200 XYZ+RGB points passed to the encoder,
then encodes a fixed-view pointcloud visualization in a background CPU
process. It does not render an RGB camera or change the sampled actions.
The running 16-environment job records its first real PPO input episode to
`artifacts/user_videos/ppo_training_pointcloud_episode0.mp4`. The companion
[heuristic pointcloud video](artifacts/user_videos/expert_pointcloud_episode0.mp4)
has been exported from one real Lance episode: 200 frames at 20 fps, using
the stored fp16 points converted to float32 as at the model input. The fixed
3D view adds no goal sphere or scene mesh. These videos show observations
used during sampling; they are not new evaluation episodes or evidence that
the policy has converged. The first PPO episode is from the start of training.

## CPU compatibility

Upstream `train_rl.py` and evaluation hardware logging unconditionally call
`torch.cuda.current_device()` even with `--num-gpus 0`. The external
`scripts/launch_official_rl.py` supplies a CPU logging shim and then executes
the original CLI via `runpy`. It changes no encoder, PPO loss, optimizer,
rollout, replay, or checkpoint implementation. Neither official source files
nor globally installed libraries are patched.

The launcher also adds observation/parameter-change instrumentation around the
upstream training call. It does not alter any loss or update. This host needs
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` to select the NVIDIA Vulkan
driver; `rl_env.sh` applies it only to child processes. The initial automatic
driver-selection attempt failed before allocating renderer GPU memory.

`scripts/rl_policy.py:load_policy(checkpoint, env, config_path=None)` rebuilds
the agent using the checkpoint's adjacent saved train configuration, or an
explicit config path, and strictly reloads the checkpoint on CPU. This retains
the saved ReLU/tanh/Identity semantics: activation functions have no weight entries.
A 32-step smoke checkpoint is not an expert policy and must
not be described as one when collecting world-model data.

## Encoder input boundary

The visual encoder computes features from colored point clouds only: XYZ+RGB,
1200 points and 6 channels. Proprioception, action and goal fields do not enter
PointNet. The original PPO architecture concatenates its 30-dimensional state
to the 512 visual vector AFTER encoding, before actor/critic heads. In contrast,
the frozen-WM PPO stage replaces latent1 with latent2 and RETAINS the original
current-state concatenation, so both heads still receive 542 dimensions.
The independently configured temporal encoder consumes only three latent1
vectors; state and WM actions do not enter either encoder. The WM action is
used only by its separate prediction head.

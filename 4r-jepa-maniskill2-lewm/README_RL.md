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
Vulkan driver for the SAPIEN renderer. Model computation uses CPU PyTorch.

```bash
bash scripts/setup_rl.sh
bash scripts/run_rl.sh smoke
bash scripts/run_rl.sh eval-smoke
```

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

The current Identity version has a CPU-only optimizer compatibility check in
`scripts/check_identity_ppo_replay.py`: real saved clouds/actions plus explicitly
synthetic robot-state/reward/done inputs, official PPO update, finite changes
to encoder/actor/critic, and exact model/optimizer reload. It is not a fresh
simulator rollout and does not establish unchanged PPO performance. Current
GPU headroom was insufficient, so no new renderer was launched for this edit.

Saved evidence: `artifacts/rl_smoke/verification.json`,
`artifacts/rl_smoke/models/model_final.ckpt`,
`artifacts/rl_smoke/test/statistics.csv`, and
`artifacts/rl_{smoke,eval-smoke}_resources.json`.
`real_transitions.npz` contains the actual 32 xyz/RGB/action transitions solely
for model integration checks; it is not the LeWM training dataset.

For sustained training, `bash scripts/run_rl.sh train` retains upstream's
5,000,000-step budget and PPO optimization defaults, with a single renderer and
CPU model. It is intentionally not started automatically. On the current busy
host this will be slow. The underlying upstream CLI accepts configuration
overrides; use the saved resolved config and a matching checkpoint when
resuming/evaluating. The default smoke and training output directories differ.

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

# JEPA4R: Embodied JEPA World Models with 4D Point Planning

Code implementation for the paper:

> **JEPA4R: Embodied JEPA World Models with 4D Point Planning**
>
> Jiarui Dong\*, Zhiheng Liang\*, Yanjun Li\*, Zeyu Zhang\*†, Heejeong Nam, Haoyu Chen, and Hao Tang‡
>
> \*Equal contribution. †Project lead. ‡Corresponding author.

<!-- Paper and project-page links, and the citation, are added once the paper is public. -->

---

## 🏃 Introduction

JEPA4R keeps JEPA's architecture and objective but reads single-depth-camera point clouds through a parameter-matched encoder and acts through a policy head on its frozen latents. Because point clouds already expose metric geometry, a world model only has to learn how this geometry changes under actions. Prediction, planning and control all happen in this 4D point latent space, and we call this 4D point planning. A decoder attached to the frozen world model, built on a pretrained PAGE-4D, shows what the model imagines as 4D point clouds.

<!-- Method overview (Fig. 2 of the paper): export to assets/method.png and show it here. -->

The framework has four parts:

- **Point clouds from one depth camera:** the eye-to-hand camera that renders each benchmark's pixels is treated as a time-of-flight sensor. Each depth image is lifted to world-frame points and cropped to a workspace box, the renderer's segmentation removes the table, floor, static fixtures and invisible goal markers, points within 5 mm voxels are merged, and 1,200 points with xyz and rgb are sampled.
- **JEPA world model:** a PointNet encoder widened to 5.51M parameters matches the 5.50M ViT-tiny of LeWM. With the projectors, the six-block predictor conditioned on actions through AdaLN-zero and the action encoder, JEPA4R has about 18.0M parameters and trains with a prediction loss and SIGReg.
- **Policy head on frozen latents:** a policy head is trained on the frozen latents under the policy-learning protocols of OGBench and CortexBench; JEPA4R and every baseline share the same heads.
- **Latent-space reconstruction:** a set generator modulated by the latent through FiLM feeds a 16×16 token grid into PAGE-4D; a learnable affine map and an rgb branch turn the point map into 1,200 points, for single frames and for imagined rollouts.

Images and point clouds record the same simulator rollouts row by row. JEPA4R is trained on the point clouds, LeWM, PLDM and LeJEPA on the paired images, and VISReg and V-JEPA 2.1 use the authors' released weights.

## 📦 Data Preparation

Each task is configured in `configs/env/<task>.yaml`; see the [data documentation](docs/data.md). Task names:

- OGBench: `cube_single`, `cube_double`, `cube_triple`, `cube_quadruple`, `scene`, `puzzle_3x3`, `puzzle_4x4`, `puzzle_4x5`
- CortexBench: `mw_assembly`, `mw_bin_pick`, `mw_button`, `mw_drawer`, `mw_hammer`, `adroit_relocate`
- Latent-space reconstruction: `cube_single_recon` (UR5e Cube-Single, plan-oracle data); RM75-B digital twin: `rm75b_cube`,
  `rm75b_mw_bin_pick`, `rm75b_mw_button`, `rm75b_mw_drawer`, `rm75b_mw_assembly` (EG2-4C2 gripper), `rm75b_rh56_relocate`
  (RH56 hand)

1. **World-model data.** Collect paired image / point-cloud trajectories:

```bash
jepa4r-run collect_wm_data env=cube_single
```

Cube uses 10,000 episodes of 200, 500, 1,000 and 600 steps for one to four cubes, mixing low-noise expert, high-noise expert, expert-then-random and random control in proportions `0.2`, `0.4`, `0.2` and `0.2`. Puzzle follows the noisy recipe and official scale of OGBench, with 1,000 episodes on 3×3 and 4×4 and 3,000 on 4×5, of 1,001 steps each. Scene, MetaWorld (per task) and Adroit Relocate use 10,000 episodes of 200 steps, mixing noisy expert control, expert control interrupted by random actions, and random actions in proportions `0.5`, `0.3` and `0.2`.

2. **Policy data.** Prepare the data of the policy learners: OGBench's official play data for Cube and Puzzle, and demonstrations for Scene and MetaWorld (25 per task) and Adroit Relocate (100):

```bash
jepa4r-run prepare_policy_data env=cube_single
```

Datasets are written to `data/wm/<task>/` and `data/policy/<task>/`; set `JEPA4R_DATA` to store them elsewhere.

## ⚙️ Environment Setup

The paper's experiments use PyTorch 2.8 and CUDA 12.8 on one NVIDIA RTX 5090 (32 GB) with a 16-vCPU Intel Xeon Gold 6459C and 90 GB of RAM. Install PyTorch for your CUDA environment, then install the project:

```bash
python -m pip install -e .
```

Dependencies are specified in `pyproject.toml`. Without installation, `python -m jepa4r <command>` replaces the `jepa4r-*` entry points.

The integrated simulators and policy learners keep the software stacks they were run with, so some stages start their
work in another interpreter, named by an environment variable (the default is the current Python):

| Variable | Stack | Used for |
|---|---|---|
| `JEPA4R_PY_COLLECT` | MuJoCo 3, OGBench, MetaWorld 3, gymnasium 1.2 | data collection, Scene demonstrations |
| `JEPA4R_PY_CORTEXBENCH` | eai-vc (CortexBench), MuJoCo 2.0 / mujoco-py, gym 0.23 | MetaWorld demonstrations, behavior cloning |
| `JEPA4R_PY_DQC` | JAX 0.6.2, Flax 0.10.7, OGBench | Puzzle latents and DQC |
| `JEPA4R_PY_TRAIN` | stable-pretraining 0.1.8, Lightning | PLDM training |
| `JEPA4R_PY_CUBE_WM` | Python 3.10, NumPy 2.2.6, MuJoCo 3.3.7, stable-pretraining 0.1.8 | Cube 1-4: collection, world models |
| `JEPA4R_PY_CUBE_RL` | Python 3.12, JAX 0.7.2, Flax 0.11.2, Torch | Cube 1-4: encoding, GCIQL, native evaluation |
| `JEPA4R_PY_RELOCATE_WM` | NumPy 1.26.4, stable-pretraining 0.1.8, Lightning 2.6.6 | Relocate / RH56: world models |
| `JEPA4R_PY_ADROIT_SIM` | eai-vc, MuJoCo 2.0 / mujoco-py, gym 0.23, NumPy 1.23.5 | Relocate: collection, behavior cloning |
| `JEPA4R_PY_RH56_SIM` | MuJoCo 3.12, gym 0.23, NumPy 1.26.4 | RM75-B + RH56 twin: collection, behavior cloning |

The Cube 1-4, Relocate and RH56 workflows (the OGBench-cube branch's; controllers in `jepa4r/policy/gciql` and
`jepa4r/policy/bc/relocate`) keep their temporary files under `JEPA4R_TMPDIR` / `TMPDIR` when set. Their five
interpreters are the `pyproject.toml` extras `cube-worldmodel`, `cube-policy`, `relocate-wm`, `relocate-adroit` and
`relocate-rh56` (`python -m pip install -e ".[cube-policy]" --extra-index-url https://download.pytorch.org/whl/cu128`);
their configurations and pinned manifests are in `configs/policy/gciql` and `configs/policy/bc/relocate`.

Upstream checkouts at pinned commits (`third_party/<name>` by default, or the variable) are listed in
[docs/architecture.md](docs/architecture.md); stable-worldmodel is vendored in `third_party/stable-worldmodel`
(`pip install -e third_party/stable-worldmodel`).

### Pretrained Checkpoints

Place third-party weights under `checkpoints/` (or set `JEPA4R_CHECKPOINTS`):

```text
checkpoints/
  visreg/visreg-vit-b-inet1k.pth                          # VISReg, ViT-B/16 (from the authors on request)
  cortexbench_baselines/vjepa2_1_vitb_dist_vitG_384.pt    # V-JEPA 2.1, ViT-B/16
  page4d/checkpoint_nomask.pt                             # PAGE-4D, decoder backbone
  vggt/model.pt                                           # VGGT-1B, decoder ablation
  da3/model.safetensors, da3/config.json                  # Depth Anything 3 Giant, decoder ablation
  vjepa/vjepa2_1_vitl_dist_vitG_384.pt                    # V-JEPA 2.1, ViT-L/16 (Cube; fetched by the Cube workflow)
  adroit/relocate-v0.pickle                               # DAPG's Relocate state expert (collection)
  rh56/adroit_rm75_rh56/                                  # the native RM75-B + RH56 bundle (JEPA4R_RH56_BUNDLE)
```

```bash
python scripts/download_checkpoints.py --download    # fetch the public ones, check sizes and sha256
```

These paths are the `weights` entries of `configs/method/{visreg,vjepa21}.yaml` and `configs/decoder/*.yaml`. Values
not given in the paper are taken from the source code of the paper's runs; each carries its source in a comment.

## 🚀 Training

Each command runs one task. `key=value` overrides any configuration value, and `tag=<name>` stores the run as a separate variant.

### Stage 1: World Models

```bash
jepa4r-run train_wm env=cube_single method=jepa4r
```

JEPA4R and LeWM (`method=lewm`) are identical except for the encoder. Following LeWM, clips have four frames spaced five environment steps apart, an action block concatenates the five actions between consecutive frames, latents have 192 dimensions, and the SIGReg weight is `0.09` with 1,024 random projections. Both train with AdamW for 10 epochs, with a learning rate of `5e-5`, a batch size of `128` and bf16 precision, on stable-worldmodel. PLDM (`method=pldm`) and LeJEPA (`method=lejepa`) are trained on the paired images for 10 epochs with their own recipes. VISReg and V-JEPA 2.1 use the authors' released weights and skip this stage.

### Stage 2: Policy Learning on Frozen Latents

```bash
jepa4r-run train_policy env=cube_single method=jepa4r
```

Every encoder is frozen. Cube uses GCIQL and Puzzle uses decoupled Q-chunking (DQC), both offline goal-conditioned reinforcement learning on OGBench's official play data for 10^6 steps; evaluation runs 50 episodes on each of the five official tasks, and Puzzle reports the best of its ten evaluations during training. MetaWorld and Adroit Relocate use the official imitation protocol of CortexBench, and Scene reuses the same code on its five evaluation tasks: behavior cloning for 100 epochs, scoring the best of 20 evaluations.

### Stage 3: Latent-Space Reconstruction

```bash
jepa4r-run train_decoder env=cube_single_recon decoder=page4d
```

The decoder trains on the OGBench Cube data with latents from the frozen epoch-10 JEPA4R model, in two stages: stage A trains the 5.75M new parameters with PAGE-4D (about 0.94B) frozen, and stage B adds rank-4 LoRA adapters (0.88M) to the query and value projections of all 72 attention blocks. The loss is the symmetric Chamfer distance plus the squared sliced Wasserstein-2 distance on xyz, plus `0.1` times the L1 color error of nearest-neighbor matched points.

## 📊 Evaluation

Policy success rates are computed in Stage 2. Decode frozen and imagined latents and measure the point error in centimeters:

```bash
jepa4r-run eval_recon env=cube_single_recon decoder=page4d
```

### Reproducing the Paper

Each table of the paper has an experiment specification that expands into all of its runs; finished runs are skipped.

| Paper | Specification |
|---|---|
| Tab. 1: OGBench | `experiments/tab1_ogbench.yaml` |
| Tab. 2: CortexBench | `experiments/tab2_cortexbench.yaml` |
| Tab. 3: latent-space reconstruction | `experiments/tab3_recon.yaml` |
| Tab. 4: ablations | `experiments/tab4_ablations.yaml` |
| Parameter counts (Sec. 4.2, 4.4) | `experiments/params.yaml` |
| RM75-B digital twin (Sec. 5.6) | `experiments/twin.yaml` |

```bash
jepa4r-reproduce experiments/tab1_ogbench.yaml          # list the runs and their status
jepa4r-reproduce experiments/tab1_ogbench.yaml --run    # run the missing ones
jepa4r-report                                           # outputs/report.md: paper vs. reproduced numbers
bash scripts/reproduce_all.sh                           # every table, then the report
```

The paper's numbers are stored in `paper/expected/`, and `paper/figures/` is where the figures would be regenerated into `outputs/figures/` (not integrated: this repository reproduces the paper's numbers). See [docs/reproduce.md](docs/reproduce.md) for the scope and settings, and [docs/architecture.md](docs/architecture.md) for the code layout. The RM75-B digital twin and everything trained in it are included; runs on the real arm are not.

This repository integrates the code of the paper's runs; its equivalence with that code was checked against the released
checkpoints and the paper's run records (data, model, training and evaluation gates) when it was integrated. What is
not integrated (the twin's look-ahead preview, a policy head on the twin's Cube, the figures) is listed with the reason
in [docs/reproduce.md](docs/reproduce.md); no experiment specification runs them.

## 😘 Acknowledgement

We thank the authors of LeWM, stable-worldmodel, PAGE-4D, DexArt, OGBench and CortexBench, whose work this implementation builds on.

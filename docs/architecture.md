# Code Architecture

## Layout

```text
configs/        one YAML per task (env/), table row (method/), encoder, policy learner, decoder
experiments/    one specification per paper table: which stage runs on which configurations
paper/
  expected/     the paper's numbers, compared with finished runs by jepa4r-report
  figures/      scripts that regenerate the paper's figures into outputs/figures/
jepa4r/
  cli.py        jepa4r-run and python -m jepa4r
  config.py     configuration composition and storage layout
  orchestrate.py  jepa4r-reproduce
  report.py     jepa4r-report
  methods.py    dispatch over the encoder rows of Tab. 1 / Tab. 2
  stages/       the seven pipeline stages
  envs/         wrappers with paired rgb / depth / segmentation: ogbench/ (Cube, Scene, Puzzle: envs, point clouds,
                experts), metaworld/ (the five tasks, point clouds, three-part collection policies), adroit/
                (Relocate: environment, point clouds, collection policy); twins/ RM75-B digital twin (cube_rm75_env.py,
                verify_rm75.py, descriptions/), its hand (hand.py), four MetaWorld tasks on it (metaworld_rm75/) and the
                RM75-B + RH56 Relocate twin (rh56/)
  data/         point-cloud pipeline, collection behaviors, storage, clips, official play data, demos/ (CortexBench
                and Scene demonstrations)
  models/       encoders, projector and action encoder, predictor, SIGReg, JEPA world model
  baselines/    PLDM, LeJEPA, VISReg, V-JEPA 2.1, random ViT-B
  policy/       frozen-encoder interface, latent caching, gciql/ (Cube 1-4: collection, world models, GCIQL, native
                evaluation), dqc/ (DQC harness), bc/ (CortexBench harness; relocate/ Relocate and RH56 workflow)
  eval/         OGBench, CortexBench and reconstruction evaluation
  decoder/      set generator, backbones/, head, LoRA, losses, training, imagination
  metrics/      point-cloud metrics in centimeters
scripts/        reproduce_all.sh, download_checkpoints.py
tests/          configuration, specification and report checks (unittest)
third_party/    vendored upstream code with UPSTREAM.md (stable-worldmodel)
```

Where a source branch kept several scripts together (the CortexBench and DQC harnesses, the environment packages, the
demonstration converters), they moved together into a package at the destination of the map below; the package's
`__init__.py` is the adapter to the stage interface, and every moved file starts with a `# Moved from <branch>:<path>`
line that names its source and what was adapted.

Storage (git-ignored): `data/` datasets, `outputs/` runs, report and figures, `checkpoints/` third-party weights;
override with `JEPA4R_DATA`, `JEPA4R_OUTPUTS` and `JEPA4R_CHECKPOINTS`.

## Pipeline

```text
collect_wm_data ------> train_wm ------> train_policy ---------------> Tab. 1, Tab. 2, Tab. 4b-d
prepare_policy_data ---------------------^
train_wm (cube_single_recon, jepa4r, epoch 10) --> train_decoder --> eval_recon --> Tab. 3, Tab. 4a
verify_twin + twin-trained train_wm / train_policy / train_decoder --> eval_preview --> Sec. 5.6 (twin)
count_params ---------------------------------------------------------> Sec. 4.2 / 4.4 parameter counts
                                     jepa4r-report <-- outputs/**/metrics.json + paper/expected/*.yaml
```

| Stage | Selects | Reads | Writes |
|---|---|---|---|
| `collect_wm_data` | env | simulator | `data/wm/<env>/` |
| `prepare_policy_data` | env | simulator, OGBench play data | `data/policy/<env>/` |
| `train_wm` | env, method | `data/wm/<env>/` | `outputs/train_wm/<env>/<method>/<tag>/`, a checkpoint per epoch |
| `train_policy` | env, method | frozen encoder, `data/policy/<env>/` | `outputs/train_policy/<env>/<method>/<tag>/metrics.json` |
| `train_decoder` | env, decoder | `data/wm/<env>/`, JEPA4R epoch 10 | `outputs/train_decoder/<env>/<decoder>/<tag>/` |
| `eval_recon` | env, decoder | decoder, world model, validation data | `outputs/eval_recon/<env>/<decoder>/<tag>/` |
| `verify_twin` | env | twin, vendor URDF | `outputs/verify_twin/<env>/<tag>/metrics.json` |
| `eval_preview` | env, decoder | twin-trained world model, policy head, decoder | `outputs/eval_preview/<env>/<decoder>/<tag>/metrics.json` |
| `count_params` | method or decoder | configurations only | `outputs/count_params/<method or decoder>/<tag>/` |

Each stage is `jepa4r/stages/<stage>.py: run(cfg, out_dir) -> dict | None`. `jepa4r-run` composes the configuration,
creates `out_dir`, saves `config.yaml`, calls `run`, writes the returned dict to `metrics.json` together with the run
key `{stage, env, method, decoder, tag}`, and finally writes `DONE`. `jepa4r-reproduce` skips jobs that are done.

| Stage | `metrics.json` keys |
|---|---|
| `train_policy` | `success` (%, the table entry), `success_per_task`, `evals` |
| `train_decoder` | `val_nn_stage_a`, `val_nn_stage_b` (cm) |
| `eval_recon` | `nn_mean`, `nn_median`, `nn_p90`, `outlier_pct`, `cube_err`, `shuffled_nn`; `imag_nn@S`, `true_nn@S`, `other_actions_nn@S`, `imag_cube@S` for S in 15 / 100 / 200; `imag_nn_max` (cm; `outlier_pct` in %) |
| `eval_preview` | `preview_nn`, `preview_cube`, `cube_to_target_success`, `cube_to_target_failure` (cm) |
| `count_params` | `encoder_params_M`, `total_params_M` (method); `new_params_M`, `lora_params_M`, `backbone_params_B` (decoder) |

## Configurations

`jepa4r-run <stage> env=E method=M decoder=D policy=P tag=T a.b.c=value`

- Groups live in `configs/{env,method,policy,decoder}/<name>.yaml`; `policy` defaults to `env.policy`.
- `method.encoder` names a file in `configs/encoder/`; `method.encoder=dgcnn` on the command line swaps it.
- `_base_: name` inherits from a file of the same group; files starting with `_` are bases only.
- `per_env: {"cube_*": {...}}` in a method patches it for matching tasks (PLDM: Impala on Cube; V-JEPA 2.1: ViT-L/16
  on Cube).
- `impl:` names the module that implements a configuration (`make(env_cfg)`, `build(enc_cfg)`,
  `train(cfg, out_dir)`, ...), so dispatch needs no registry.
- `???` marks a value that is not given in the paper; reading it raises an error.
- `tag` names a variant (default `default`); the ablation variants are declared in the experiment specifications.

`experiments/<table>.yaml` lists stage jobs as grids over configuration names, with optional variants (`tag` plus
overrides) and optional blocks (see `jepa4r/orchestrate.py`). `paper/expected/<table>.yaml` stores the paper's
numbers and where each is found among the runs (see `jepa4r/report.py`). The tests check that every paper number is
produced by some job.

## Integration Map

| Existing code | Destination | Upstream |
|---|---|---|
| depth → point cloud | `jepa4r/data/pointcloud.py`, `jepa4r/envs/camera.py` | — |
| simulator wrappers, cameras, experts | `jepa4r/envs/{ogbench,metaworld,adroit}/` (packages) | OGBench, MetaWorld (CortexBench) |
| RM75-B twin, hand twin, URDF check, look-ahead preview | `jepa4r/envs/twins/*`, `configs/env/rm75b_cube.yaml`, `jepa4r/stages/{verify_twin,eval_preview}.py` | vendor URDF |
| data collection with mixtures | `jepa4r/stages/collect_wm_data.py`, `jepa4r/data/{behaviors,storage}.py` | stable-worldmodel |
| official play data, demonstrations | `jepa4r/data/official_play.py`, `jepa4r/data/demos/` | OGBench, CortexBench |
| LeWM recipe and PointNet | `jepa4r/models/*` | stable-worldmodel, LeWM, DexArt |
| Tab. 4b encoders | `jepa4r/models/encoders/{pointnet2,dgcnn,point_patch_transformer}.py` | PointNet++, DGCNN, PPRL |
| PLDM / LeJEPA training, VISReg / V-JEPA 2.1 loading | `jepa4r/baselines/*`, `third_party/stable-worldmodel` (PLDM training) | PLDM, LeJEPA, VISReg, V-JEPA 2.1 |
| GCIQL, DQC, behavior cloning and their evaluation | `jepa4r/policy/{gciql/,dqc/,bc/}`, `jepa4r/eval/{ogbench,cortexbench}.py` | OGBench, DQC, CortexBench |
| decoder | `jepa4r/decoder/*` (`backbones/{page4d,vggt,depth_anything3}.py`) | PAGE-4D, VGGT, Depth Anything 3 |
| reconstruction metrics, open-loop evaluation | `jepa4r/metrics/pointcloud.py`, `jepa4r/eval/recon.py`, `jepa4r/decoder/imagine.py` | — |
| plotting and rendering | `paper/figures/*` | — |

Prefer installed packages for upstream code. When upstream code has to be modified, vendor it under `third_party/<name>/`
with an `UPSTREAM.md` that records the repository, the commit and the local changes.

Upstream code at pinned commits: stable-worldmodel is vendored with its patches; the others are used unchanged from a
checkout (the path in an environment variable, default `third_party/<name>`):

| Code | Commit | Used by | Variable |
|---|---|---|---|
| stable-worldmodel (vendored, patched) | `6f1e499` | data loading, collection, PLDM | `third_party/stable-worldmodel` |
| OGBench (`pip install` the checkout) | `a977253` | Scene, Puzzle, official play data; Cube 1-4 workflow | `JEPA4R_OGBENCH` (Cube) |
| DQC (ColinQiyangLi/dqc) | `df89825` | Puzzle policy | `JEPA4R_DQC` |
| eai-vc (CortexBench) | `76fe35e` | behavior cloning (MetaWorld, Scene, Relocate) | `JEPA4R_EAI_VC` |
| V-JEPA 2 (facebookresearch/vjepa2) | `204698b` | V-JEPA 2.1 encoder | `JEPA4R_VJEPA2` |
| PAGE-4D | `c4a39ac` | decoder backbone | `JEPA4R_PAGE4D` |
| Depth Anything 3 | `3d835ec` | decoder backbone (Tab. 4a) | `JEPA4R_DA3` |

The stages that run in other interpreters start them from `JEPA4R_PY_COLLECT` (MuJoCo 3 + OGBench + MetaWorld 3),
`JEPA4R_PY_CORTEXBENCH` (eai-vc, MuJoCo 2.0 / mujoco-py), `JEPA4R_PY_DQC` (JAX) and `JEPA4R_PY_TRAIN` (PLDM training).

The OGBench-cube branch's workflows run whole with their own stage receipts and sha256 pins; this repository's stages
run their shares of them:
- Cube 1-4: `jepa4r/policy/gciql` (`cube_stage`): collect_wm_data = collect + views, prepare_policy_data = prepare-data,
  train_wm = train, train_policy = prepare-encoder ... native-audit. Interpreters `JEPA4R_PY_CUBE_WM` (collection, world
  models; `jepa4r/policy/gciql/requirements/worldmodel.txt`) and `JEPA4R_PY_CUBE_RL` (encoding, GCIQL, native
  evaluation; `requirements/policy.txt` there); checkouts `JEPA4R_OGBENCH`, `JEPA4R_VJEPA2`; the PLDM authors' source (`1bd7e56`) is
  vendored in `jepa4r/policy/gciql/pldm/vendor`. A pinned file that changed stops a run, as in the branch.
- Adroit Relocate and the RM75-B + RH56 twin: `jepa4r/policy/bc/relocate` (`relocate_stage`), interpreters
  `JEPA4R_PY_RELOCATE_WM` (`jepa4r/policy/bc/relocate/requirements/wm.txt`), `JEPA4R_PY_ADROIT_SIM` (`adroit.txt`
  there, over the eai-vc stack), `JEPA4R_PY_RH56_SIM` (`rh56.txt` there); checkouts `JEPA4R_SWM` (stable-worldmodel `6f1e499`),
  `JEPA4R_EAI_VC_RELOCATE` (eai-vc `76fe35e` with `jepa4r/policy/bc/relocate/patches/eai-vc-train-loop.patch`),
  `JEPA4R_DMC_IMPORT_OVERLAY`; inputs `JEPA4R_ADROIT_DEMOS` (the official 100 demonstrations), `JEPA4R_RH56_BUNDLE`,
  `MUJOCO_PY_MUJOCO_PATH` / `MUJOCO_PY_MJKEY_PATH`.
- Temporary files of their children go to `JEPA4R_TMPDIR` (else `TMPDIR`, else `/tmp`).
- Their own test suites stay with them (pytest, in the workflows' interpreters): `jepa4r/policy/gciql/tests`,
  `jepa4r/policy/gciql/gates/tests`, `jepa4r/policy/bc/relocate/tests`. The branch's CEM planning (appendix Tab. S1)
  and its tests are not part of this repository.

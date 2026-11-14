# Code Architecture

## Layout

```text
configs/        one YAML per task (env/), table row (method/), encoder, policy learner, decoder; the Cube and Relocate
                workflows' own configurations sit in the group of the part that reads them: env/cube/, env/relocate/
                (collection configs and source lists, official data, point-cloud sampling), method/cube/,
                method/relocate/ (training recipes and Hydra configs, released encoder weights), policy/gciql/,
                policy/bc/ (GCIQL agent settings and source list, the Relocate dependencies and RH56 bundle file list,
                CortexBench embedding configs)
experiments/    one specification per paper table: which stage runs on which configurations
paper/
  expected/     the paper's numbers, compared with finished runs by jepa4r-report
assets/         figures of the README
jepa4r/
  cli.py        jepa4r-run and python -m jepa4r
  config.py     configuration composition and storage layout
  orchestrate.py  jepa4r-reproduce
  report.py     jepa4r-report
  methods.py    dispatch over the encoder rows of Tab. 1 / Tab. 2
  stages/       the pipeline stages
  envs/         wrappers with paired rgb / depth / segmentation: ogbench/ (Cube, Scene, Puzzle: envs, point clouds,
                experts), metaworld/ (the five tasks, point clouds, three-part collection policies), adroit/
                (Relocate: environment, point clouds, collection policy); twins/ RM75-B digital twin (cube_rm75_env.py,
                verify_rm75.py, descriptions/), its hand (hand.py), four MetaWorld tasks on it (metaworld_rm75/) and the
                RM75-B + RH56 Relocate twin (rh56/)
  data/         point-cloud pipeline, collection behaviors, storage, clips, official play data, demos/ (CortexBench
                and Scene demonstrations); cube/, relocate/ (collection and data of the Cube 1-4 and Relocate workflows)
  models/       encoders, projector and action encoder, predictor, SIGReg, JEPA world model; cube/ (world-model
                training of the Cube workflow)
  baselines/    PLDM, LeJEPA, VISReg, V-JEPA 2.1, random ViT-B; cube/, relocate/ (their training and loaders in the two
                workflows)
  policy/       frozen-encoder interface, latent caching, gciql/ (GCIQL on Cube 1-4, its online evaluation and the Cube
                workflow's controller), dqc/ (DQC), bc/ (CortexBench behavior cloning; relocate/ the Relocate workflow
                on Adroit and the RH56 twin)
  eval/         OGBench, CortexBench and reconstruction evaluation
  decoder/      set generator, backbones/, head, LoRA, losses, training, imagination
  metrics/      point-cloud metrics in centimeters
scripts/        reproduce_all.sh, download_checkpoints.py
tests/          configuration, specification, workflow, report and code-style checks (unittest)
third_party/    vendored upstream code with UPSTREAM.md (stable-worldmodel, PLDM)
```

Code that works as a unit (the CortexBench and DQC code, the environment packages, the demonstration converters)
is a package at the location given in the table below; its `__init__.py` adapts it to the stage interface.

Storage (git-ignored): `data/` datasets, `outputs/` runs, report and figures, `checkpoints/` third-party weights;
override with `JEPA4R_DATA`, `JEPA4R_OUTPUTS` and `JEPA4R_CHECKPOINTS`.

## Pipeline

```text
collect_wm_data ------> train_wm ------> train_policy ---------------> Tab. 1, Tab. 2, Tab. 4b-d
prepare_policy_data ---------------------^
train_wm (cube_single_recon, jepa4r, epoch 10) --> train_decoder --> eval_recon --> Tab. 3, Tab. 4a
verify_twin + twin-trained train_wm / train_policy / train_decoder --> Sec. 5.6 (twin)
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
| `count_params` | method or decoder | configurations only | `outputs/count_params/<method or decoder>/<tag>/` |

Each stage is `jepa4r/stages/<stage>.py: run(cfg, out_dir) -> dict | None`. `jepa4r-run` composes the configuration,
creates `out_dir`, saves `config.yaml`, calls `run`, writes the returned dict to `metrics.json` together with the run
key `{stage, env, method, decoder, tag}`, and finally writes `DONE`. `jepa4r-reproduce` skips jobs that are done.

| Stage | `metrics.json` keys |
|---|---|
| `train_policy` | `success` (%, the table entry), `success_per_task`, `evals` |
| `train_decoder` | `val_nn_stage_a`, `val_nn_stage_b` (cm) |
| `eval_recon` | `nn_mean`, `nn_median`, `nn_p90`, `outlier_pct`, `cube_err`, `shuffled_nn`; `imag_nn@S`, `true_nn@S`, `other_actions_nn@S`, `imag_cube@S` for S in 15 / 100 / 200; `imag_nn_max` (cm; `outlier_pct` in %) |
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

## Where the Code Lives

| Component | Location | Upstream |
|---|---|---|
| depth → point cloud | `jepa4r/data/pointcloud.py`, `jepa4r/envs/camera.py` | — |
| simulator wrappers, cameras, experts | `jepa4r/envs/{ogbench,metaworld,adroit}/` | OGBench, CortexBench |
| RM75-B twin, hand twin, URDF check | `jepa4r/envs/twins/*`, `configs/env/rm75b_cube.yaml`, `jepa4r/stages/verify_twin.py` | vendor URDF |
| data collection with mixtures | `jepa4r/stages/collect_wm_data.py`, `jepa4r/data/{behaviors,storage}.py` | — |
| official play data, demonstrations | `jepa4r/data/official_play.py`, `jepa4r/data/demos/` | OGBench, CortexBench |
| LeWM recipe and PointNet | `jepa4r/models/*` | stable-worldmodel, LeWM, DexArt |
| Tab. 4b encoders | `jepa4r/models/encoders/{pointnet2,dgcnn,point_patch_transformer}.py` | PointNet++, DGCNN, PPRL |
| PLDM / LeJEPA training, VISReg / V-JEPA 2.1 loading | `jepa4r/baselines/*` | PLDM, LeJEPA, VISReg, V-JEPA 2.1 |
| GCIQL, DQC, behavior cloning and their evaluation | `jepa4r/policy/{gciql,dqc,bc}/`, `jepa4r/eval/{ogbench,cortexbench}.py` | OGBench, DQC, CortexBench |
| decoder | `jepa4r/decoder/*` | PAGE-4D, VGGT, Depth Anything 3 |
| reconstruction metrics, open-loop evaluation | `jepa4r/metrics/pointcloud.py`, `jepa4r/eval/recon.py`, `jepa4r/decoder/imagine.py` | — |

Prefer installed packages for upstream code. When upstream code has to be modified, vendor it under `third_party/<name>/`
with an `UPSTREAM.md` that records the repository, the commit and the local changes.

Upstream code at the commits the paper used: stable-worldmodel and PLDM are vendored (`third_party/<name>/UPSTREAM.md`); the
others are used unchanged from a checkout (the path in an environment variable, default `third_party/<name>`):

| Code | Commit | Used by | Variable |
|---|---|---|---|
| stable-worldmodel (vendored, patched) | `6f1e499` | data loading, collection, PLDM | `third_party/stable-worldmodel` |
| PLDM (vladisai/PLDM, vendored) | `1bd7e56` | PLDM on Cube 1-4 | `third_party/pldm` |
| OGBench (`pip install` the checkout) | `a977253` | Scene, Puzzle, official play data; Cube 1-4 workflow | `JEPA4R_OGBENCH` (Cube) |
| DQC (ColinQiyangLi/dqc) | `df89825` | Puzzle policy | `JEPA4R_DQC` |
| eai-vc (CortexBench) | `76fe35e` | behavior cloning (MetaWorld, Scene, Relocate) | `JEPA4R_EAI_VC` |
| V-JEPA 2 (facebookresearch/vjepa2) | `204698b` | V-JEPA 2.1 encoder | `JEPA4R_VJEPA2` |
| PAGE-4D | `c4a39ac` | decoder backbone | `JEPA4R_PAGE4D` |
| Depth Anything 3 | `3d835ec` | decoder backbone (Tab. 4a) | `JEPA4R_DA3` |

The stages that run in other interpreters start them from `JEPA4R_PY_COLLECT` (MuJoCo 3 + OGBench + MetaWorld 3),
`JEPA4R_PY_CORTEXBENCH` (eai-vc, MuJoCo 2.0 / mujoco-py), `JEPA4R_PY_DQC` (JAX) and `JEPA4R_PY_TRAIN` (PLDM training).

Cube 1-4 and Adroit Relocate (with the RH56 hand twin) run as two workflows with their own stage controllers; their
code sits where the table above puts it, and the stages of this repository run their shares of them:
- Cube 1-4 (`cube_stage` in `jepa4r/policy/gciql`): the controller `workflow.py` (a stage is done when its output
  exists) with the GCIQL stages (`workflow_downstream.py`, `downstream/`); collection and official play data in
  `jepa4r/data/cube`, world-model training in `jepa4r/models/cube`, PLDM, LeJEPA and the released VISReg / V-JEPA 2.1
  weights in `jepa4r/baselines/cube`, the PLDM authors' source (`1bd7e56`) in `third_party/pldm`. collect_wm_data =
  collect + views, prepare_policy_data = prepare-data, train_wm = train, train_policy = prepare-encoder, camera, bind,
  encode, gciql, evaluate.
  Its configurations: `configs/env/cube` (collection Hydra configs and the list of collection sources, official play
  data), `configs/method/cube` (the JEPA4R / LeWM recipes, released VISReg / V-JEPA 2.1 weights), `configs/policy/gciql` (GCIQL
  agent settings and the list of its vendored sources). Interpreters `JEPA4R_PY_CUBE_WM` (collection,
  world models; pyproject extra `cube-worldmodel`) and `JEPA4R_PY_CUBE_RL` (encoding, GCIQL, online evaluation; extra
  `cube-policy`); checkouts `JEPA4R_OGBENCH`, `JEPA4R_VJEPA2`.
- Adroit Relocate and the RM75-B + RH56 twin (`relocate_stage` in `jepa4r/policy/bc`): the runner and the stage plans
  `jepa4r/policy/bc/relocate/{workflow,adroit_workflow,rh56_workflow}.py` (a stage is done when it exited with 0) with
  world-model training and behavior cloning (`adroit_learning.py` and the CortexBench runner of `jepa4r/policy/bc`)
  and their inputs (`configs/policy/bc/relocate/DEPENDENCIES.json`; the world-model training configs
  `configs/method/relocate/train`, the embedding configs `configs/policy/bc/conf/model`); collection and the official
  demonstrations in `jepa4r/data/relocate`, the PLDM and LeJEPA runners in `jepa4r/baselines/relocate`, the RH56
  twin's collection, world-model and BC code in `jepa4r/envs/twins/rh56`. Interpreters
  `JEPA4R_PY_RELOCATE_WM` (pyproject extra `relocate-wm`), `JEPA4R_PY_ADROIT_SIM` (extra `relocate-adroit`, over the
  eai-vc stack), `JEPA4R_PY_RH56_SIM` (extra `relocate-rh56`); checkouts `JEPA4R_SWM` (stable-worldmodel
  `6f1e499`), `JEPA4R_EAI_VC_RELOCATE` (eai-vc `76fe35e` with `third_party/eai-vc-relocate/eai-vc-train-loop.patch`),
  `JEPA4R_DMC_IMPORT_OVERLAY`; inputs `JEPA4R_ADROIT_DEMOS` (the official 100 demonstrations), `JEPA4R_RH56_BUNDLE`,
  `MUJOCO_PY_MUJOCO_PATH` / `MUJOCO_PY_MJKEY_PATH`.
- Temporary files of their children go to `JEPA4R_TMPDIR` (else `TMPDIR`, else `/tmp`).

# Reproducing the Paper

## Scope

This repository reproduces every number in the main text of the paper, except:

- runs on the real RM75-B arm (Sec. 5.6 real trials, Fig. 1c, the real-arm key frames); the RM75-B digital twin
  and its hand, and everything trained in them, are included (`experiments/twin.yaml`);
- results that appear only in the supplementary material (Tab. S1, planning with CEM under LeWM's setting);
- numbers quoted from other papers (the published rows of Tab. 2; LeWM vs. DINO-WM in Sec. 1).

Implementation details that the main text defers to the supplementary material (data mixtures, policy learners,
baseline recipes) are implemented, because the main-text numbers depend on them.

## Paper Items

| Paper item | Specification | Stages | Paper numbers |
|---|---|---|---|
| Tab. 1: 6 encoders on Cube-Single / Double / Triple, Scene, Puzzle 3×3 / 4×4 / 4×5; Cube-Quadruple 0.0 (Sec. 5.3) | `tab1_ogbench` | collect_wm_data, prepare_policy_data, train_wm, train_policy | `paper/expected/tab1_ogbench.yaml` |
| Tab. 2: 6 encoders (+ Random, optional) on 5 MetaWorld tasks and the mean, and Adroit Relocate | `tab2_cortexbench` | same | `paper/expected/tab2_cortexbench.yaml` |
| Tab. 3a/b; 12.31 cm shuffled control; decoder stage A → B (0.87 → 0.79 cm); within 1.1 cm over 200 steps (UR5e Cube-Single reconstruction data, `cube_single_recon`) | `tab3_recon` | collect_wm_data, train_wm, train_decoder, eval_recon | `paper/expected/tab3_recon.yaml` |
| Tab. 4a decoder backbone (Cube); 4b encoder, 4c temporal window, 4d hyper-parameters (Scene) | `tab4_ablations` | all of the above | `paper/expected/tab4_ablations.yaml` |
| 5.51M PointNet vs. 5.50M ViT-tiny, ~18.0M JEPA4R, 5.75M decoder, 0.88M LoRA, ~0.94B PAGE-4D | `params` | count_params | `paper/expected/params.yaml` |
| Sec. 5.6 in the RM75-B twin: twin vs. vendor URDF (2,000 joint configurations, limits, masses, gripper); twin-trained world models, policy heads (EG2-4C2 gripper: four CortexBench MetaWorld tasks; RH56 hand: Relocate) and decoder; the paper reports no twin numbers | `twin` | verify_twin, collect_wm_data, prepare_policy_data, train_wm, train_policy, train_decoder | `paper/expected/twin.yaml` (no table: the paper reports no twin numbers) |

```bash
jepa4r-reproduce experiments/tab1_ogbench.yaml                                  # list jobs and status
jepa4r-reproduce experiments/tab1_ogbench.yaml --run                            # run missing jobs
jepa4r-reproduce experiments/tab4_ablations.yaml --stage train_wm --where env=scene --run
jepa4r-report                                                                   # outputs/report.{md,csv}
```

## Settings Taken from the Paper

- **Observations:** one fixed eye-to-hand camera (the one rendering the benchmark's pixels); point clouds lifted from
  its depth, cropped, segmentation-filtered, merged in 5 mm voxels, 1,200 xyz+rgb points, fixed affine
  normalization; images 224 × 224.
- **World models (JEPA4R, LeWM):** clips of 4 frames 5 environment steps apart, action blocks of 5 actions, latent
  dimension 192, six-block AdaLN-zero causal predictor over at most 3 latents, MSE + 0.09 SIGReg (1,024
  projections), no stop-gradient or EMA target, AdamW with learning rate 5e-5, batch size 128, 10 epochs, bf16,
  stable-worldmodel. PointNet: shared MLP 6 → 1,080 → 4,320 (GELU), max-pool, linear head to 192.
- **Baselines:** PLDM and LeJEPA trained on our images for 10 epochs with their own recipes (ViT-tiny/14; PLDM uses an
  Impala encoder on Cube); VISReg and V-JEPA 2.1 released weights (ViT-B/16; V-JEPA 2.1 ViT-L/16 on Cube).
- **Policies:** Cube GCIQL and Puzzle DQC, offline goal-conditioned RL on OGBench's official play data for 10^6 steps,
  5 tasks × 50 episodes, Puzzle best of 10 evaluations; MetaWorld, Adroit Relocate and Scene behavior cloning for 100
  epochs on 25 demonstrations per task (100 on Relocate), best of 20 evaluations.
- **Decoder:** 256 queries and 3 FiLM residual MLP blocks → 16 × 16 × 1,024 tokens → frozen PAGE-4D → learnable affine
  map + rgb branch → 1,200 points; loss CD + SW₂² + 0.1 · L1 color; stage A trains the new parameters (5.75M), stage B
  adds rank-4 LoRA on q / v of all 72 attention blocks (0.88M); OGBench Cube with epoch-10 JEPA4R latents.
  Evaluation: 4,096 validation frames, outliers above 3 cm; open loop over 64 episodes with context frames at
  environment steps 0, 5 and 10, a horizon of 200 steps, and reports at steps 15, 100 and 200.

## Values Not Stated in the Paper

The configurations carry the values of the code the paper's runs used:

1. Task ids, play datasets, cameras, crop boxes, affine maps, point sampler: `configs/env/*.yaml`.
2. Collection policies, noise and switch schedules: `jepa4r/envs/{metaworld,ogbench}/` (`jepa4r/data/behaviors.py`
   lists them); shard plans and seeds: `jepa4r/stages/collect_wm_data.py`.
3. How the policy learners observe their data: CortexBench embeds every frame on the fly (`jepa4r/policy/bc`);
   DQC re-renders every row of OGBench's play data and encodes it (`jepa4r/policy/dqc`).
4. Projector, action encoder, predictor widths, weight decay, gradient clipping, split: `configs/method/_jepa_wm.yaml`.
   The training seed fixes the split and the shuffling, as in the paper runs; the initial weights come from torch's
   default generator, which those runs did not seed, so two trainings with the same seed differ as two seeds do.
5. DQC (`configs/policy/dqc.yaml`): the paper's "best of ten evaluations" are the evaluations at 100k ... 1M steps;
   the runner also evaluates the untrained agent at step 0 (kept: it shares the RNG of the batch sampling; 0.0 in all
   18 paper runs, so the best of ten equals the best of all eleven). BC: `configs/policy/bc.yaml` (eai-vc 76fe35e's
   configs, seeds 100 / 200 / 300, episode i at seed + i).
6. Decoder: `configs/decoder/_base.yaml` (pixel selection, 100 sliced-W2 projections, 14,140 batches of 32 per stage,
   learning rates 3e-4 / 1e-5, split 0.9 of seed 3072); Depth Anything 3 Giant; the cube position is read as the
   centroid of the points carrying the cube's colour.
7. Tab. 4b encoders: `configs/encoder/{pointnet2,dgcnn,point_patch_transformer}.yaml`.
8. Reconstruction data: the UR5e Cube-Single plan-oracle collection (`configs/env/cube_single_recon.yaml`), not Tab. 1's
   Cube-Single data.

## Notes

- **External inputs.** OGBench Cube 1-4 (Tab. 1 Cube columns, Sec. 5.3 Cube-Quadruple), Adroit Relocate (Tab. 2) and
  the RH56 hand twin run the Cube and Relocate workflows (controllers in `jepa4r/policy/gciql` and
  `jepa4r/policy/bc/relocate`; docs/architecture.md). Inputs they need from outside this repository: the V-JEPA 2.1
  ViT-L/16 weights (Cube; the workflow fetches them), DAPG's Relocate state expert, the official Relocate
  demonstrations, the native RM75-B + RH56 bundle (its files are listed in
  `configs/policy/bc/relocate/rh56_native_bundle_v1.json`) and MuJoCo 2.0 for mujoco-py.
- **Look-ahead preview** (Sec. 4.3): not part of this repository; the paper reports no twin numbers (Sec. 5.6
  "Results" is qualitative).
- **Policy head on the twin's Cube** (`configs/env/rm75b_cube.yaml` `policy`, `policy_data`: `???`): on the twin's
  Cube the paper's code only plans with CEM (the appendix-only setting of Tab. S1); the gripper's policy heads trained
  in the twin are the CortexBench ones of the four `rm75b_mw_*` tasks, so `experiments/twin.yaml` has no policy job on
  `rm75b_cube`.
- **Figures:** this repository reproduces the paper's numbers, not its figures; the stages still write the data some
  figures plot (`eval_recon`: `curves.npz`, `rollout_example.npz`).
- **Released weights:** VISReg's are shared by the authors on request (`scripts/download_checkpoints.py`).
- **Random (ViT-B) row of Tab. 2:** the paper lists it with the published numbers (caption: "Published numbers: the
  VC-1 study"), and its values equal the VC-1 study's Table 8 (`paper/expected/tab2_cortexbench.yaml` `published`); the
  paper's runs ran it only on bin-picking (`configs/method/random_vitb.yaml`); the job stays optional, on the five
  MetaWorld tasks (the Relocate workflow's behavior cloning has no random ViT-B).

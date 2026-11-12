# Reproducing the Paper

## Scope

This repository reproduces every number in the main text of the paper, except:

- the real-robot experiments (Sec. 5.6, Fig. 1c and the real-arm key frames); the RM75-B digital twin is used only
  there and in the supplementary material, so it is not included;
- results that appear only in the supplementary material (Tab. S1, planning with CEM under LeWM's setting);
- numbers quoted from other papers (the published rows of Tab. 2; LeWM vs. DINO-WM in Sec. 1).

Implementation details that the main text defers to the supplementary material (data mixtures, policy learners,
baseline recipes) are implemented, because the main-text numbers depend on them.

## Paper Items

| Paper item | Specification | Stages | Paper numbers |
|---|---|---|---|
| Tab. 1: 6 encoders on Cube-Single / Double / Triple, Scene, Puzzle 3×3 / 4×4 / 4×5; Cube-Quadruple 0.0 (Sec. 5.3) | `tab1_ogbench` | collect_wm_data, prepare_policy_data, train_wm, train_policy | `paper/expected/tab1_ogbench.yaml` |
| Tab. 2: 6 encoders (+ Random, optional) on 5 MetaWorld tasks and the mean, and Adroit Relocate | `tab2_cortexbench` | same | `paper/expected/tab2_cortexbench.yaml` |
| Tab. 3a/b; 12.31 cm shuffled control; decoder stage A → B (0.87 → 0.79 cm); within 1.1 cm over 200 steps | `tab3_recon` | collect_wm_data, train_wm, train_decoder, eval_recon | `paper/expected/tab3_recon.yaml` |
| Tab. 4a decoder backbone (Cube); 4b encoder, 4c temporal window, 4d hyper-parameters (Scene) | `tab4_ablations` | all of the above | `paper/expected/tab4_ablations.yaml` |
| 5.51M PointNet vs. 5.50M ViT-tiny, ~18.0M JEPA4R, 5.75M decoder, 0.88M LoRA, ~0.94B PAGE-4D | `params` | count_params | `paper/expected/params.yaml` |
| Fig. 1a, Fig. 1b, observations, success plots, CortexBench scatter, simulation key frames, open-loop errors | — | outputs of the stages above | `paper/figures/*.py` |

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

## Open Items

Values marked `???` in `configs/` are not given in the paper and are taken from the integrated code:

1. Benchmark task ids and the names of OGBench's official play datasets; camera names; crop boxes, affine maps,
   point sampler and rgb scaling.
2. Expert controllers, noise scales and interruption schedules of the data mixtures; how the Puzzle data follow
   OGBench's noisy recipe.
3. How the frozen encoders observe the official play data (images / point clouds for its recorded states).
4. Projector and action encoder inherited from LeWM; weight decay.
5. GCIQL, DQC and BC hyper-parameters; the frames the policy head reads; evaluation schedule; whether Cube reports
   the final evaluation.
6. CortexBench evaluation seeds and episodes per evaluation (every CortexBench value of the paper is a multiple of
   1/75, e.g. 97.3 = 73/75).
7. Whether the Random (ViT-B) row of Tab. 2 is our run (the specification marks it optional).
8. Tab. 4c: whether the policy head's input window changes together with the world-model context.
9. Decoder: pixel selection for the 1,200 points, sliced-W2 projections, epochs, learning rates and batch sizes per
   stage, train / validation split, which Cube task ("OGBench Cube"; the configurations use Cube-Single), Depth
   Anything 3 model size.
10. Metric definitions: how the bidirectional nearest-neighbor distances are aggregated; how the cube position is
    read from a point cloud.
11. Hyper-parameters of the Tab. 4b encoders at ~5.51M parameters; checkpoint files.

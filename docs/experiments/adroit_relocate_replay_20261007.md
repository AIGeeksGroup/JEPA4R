# Relocate visible-instance replay / WM / BC / fragment CEM

Final code and parameter entry points for the 2026-10-07 results. See [final results](../results/adroit_relocate_20261007.json). Dataset shards, RGB caches, checkpoints, private native runtime assets, logs and run controllers are external; this release adds no server-specific launcher or full parallel training implementation.

## Contracts and dependencies

- Original Adroit Relocate: action30, native observation39, qpos/qvel36, 200 raw steps, `vil_camera`, 256×256, MSAA0. Replay uses original executed controls and seeded native initial states; official BC uses the original raw expert labels, including values outside [-1,1] that native `step` clips internally.
- Point input: world XYZ + RGB0..255, 1200×6; blue object cap300, green marker ordinary-object cap300, remaining robot. Global5mm voxel and crop `[[-.6,-.8,-.25],[.6,.6,1.1]]`. Only raw visible robot pixels0 enables real same-camera background fill. No occluded or synthetic target geometry is inserted.
- WM: fixed affine `[0,-.2,.25]`, scale4, RGB/255; history3+prediction1, frameskip5, normalized raw30D controls grouped150D, latent192, DexArt[1080,4320], chunk300. Batch128/BF16/10epochs/16workers/prefetch2; original AdamW/loss/clip1/sampling/0.9 internal split remain. Only train fits normalization.
- BC: frozen encoder+projector, FLARE3×192 + proprio30 =606→30. Original100 demos/25 evaluation trajectories,100 epochs, evaluation5,10,…,95,99; seeds100/200/300; `num_cpu=1`. Best-of-20 SR and highest-mean-reward saved policy are distinct metrics.
- CEM: original Pen fragment protocol, seed42/50 starts/goal`t+25`/budget50, horizon5/receding5/action_block5/history1,300 candidates/top30/10 iterations. Relocate success is future-object-position distance<0.1m. This single-frame fragment metric differs from native full-task success requiring >25 successful frames. 31/50 starts already qualify; results include the19 initially-outside starts separately. No random/no-action baseline was run.

Use separate existing environments through `$WM_PY`, `$BC_PY`, `$CEM_PY`. Tested numerical stack: Python3.10.21/Torch2.8.0+cu128/torchvision0.23.0+cu128; WM/CEM NumPy1.26.4, BC NumPy1.23.4; stable-pretraining0.1.8/Lightning2.6.6, native gym0.23.1/mujoco_py2.0.2.13. The modern `mujoco` package used by eager dependency imports does not replace native MuJoCo200.

Before any native process, provide `MUJOCO_PY_MUJOCO_PATH`, `MUJOCO_PY_MJKEY_PATH`, `LD_LIBRARY_PATH` (MuJoCo200 `bin` and driver libraries), `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and native `mj_envs`/`mjrl` sources on `PYTHONPATH`. WM requires the recorded stable-worldmodel implementation: actual `world/world.py` SHA `3392bd31159daab859f2d73e5a52e59ac898259f829c9d1363e64eef151b91b3`; CEM solver SHA `c66625d42cf02f912fc4c99d5874c57aba686d899e510d91630df210ece2ee89`.

BC additionally needs the eai-vc76fe35e runtime, `mujoco_vc`, `vc_models` and its already-verified repaired `train_loop.py` SHA `c0819470fa0ca160544978faf2d4227b0bd2fdb49dd3c2d02ea59ffdf567eb45`; this is a repaired dependency, not pristine upstream. `CORTEXBENCH_EAI_VC_ROOT` selects that installation; it does not install missing packages or provide a simulator/license. These external dependencies must match before claiming reproduction of the recorded GPU BC result.

## Replay without a new expert

From `lewm_pcd/`, use the same numerical sampling values for replay and BC:

```sh
export ADROIT_PCD_CONFIG="$PWD/config/adroit_relocate_sampling.json"
export ADROIT_PCD_CONFIG_SHA256=$(sha256sum "$ADROIT_PCD_CONFIG" | awk '{print $1}')
$CEM_PY scripts/data/replay_adroit_relocate_pcd.py \
  --source /ABS/nonvisual_train_replay.h5 --initials /ABS/initial_states_train.npz \
  --sampling-config "$ADROIT_PCD_CONFIG" --start-episode 0 --num-episodes 625 \
  --out /ABS/fresh_shards/shard_000.h5
```

The HDF5 source must contain the original21 nonvisual columns, contiguous positive `ep_len`/`ep_offset`, unique per-episode seed, and matching initial-state SHA. The NPZ must retain full ordered `qpos/qvel/hand_qpos/obj_pos/palm_pos/target_pos`, `seed`, `reset_observation` and initial-state SHA. Every replayed state/reward/success is checked; raw field values/dtypes are preserved. New XYZ/RGB is recorded before action. Per-shard state and actual exit files are created in the output directory, outside Git.

For10000 train episodes, sixteen disjoint ranges `[625*i,625*(i+1))` give16 shards; test150 uses a separate source/NPZ and disjoint ranges. Concurrent processes each have independent output and sampler RNG. Limit active shards to actual simulator/GPU/CPU/memory capacity. No full recollection is launched by preparing this release.

The existing physical merger can retain original global replay IDs with the explicit new flag. It otherwise keeps its old behavior. To preserve input shards, copy the first shard to a fresh final path, then append the remaining ordered shards without `--delete`:

```sh
cp /ABS/fresh_shards/shard_000.h5 /ABS/fresh_train.h5
$CEM_PY scripts/data/merge_h5_shards.py --preserve-episode-indices \
  --out /ABS/fresh_train.h5 /ABS/fresh_shards/shard_001.h5 /ABS/fresh_shards/shard_002.h5
# Include all remaining shards in order. Reject overlapping episode IDs.
```

The completed dataset uses VDS and must retain all payload dependencies. Its published train/test are already usable external inputs; they are not rebuilt or relocated by this code release. The final replay-entry qualification covers normal test0/test1 and train4404 with actual BG frame65: all600 frames reproduce XYZ/RGB byte-for-byte and preserve all21 original fields. The real test0/test1 shards also append successfully with exact400-frame data and offsets. This verifies the thin entry and merger, not another full collection.

## World model

```sh
export RELOCATE_PCD_TRAIN=/ABS/published/train.h5
export STABLEWM_HOME=/ABS/fresh_wm_output
export SPT_CACHE_DIR=/ABS/fresh_spt_cache
$WM_PY train.py --config-name adroit_relocate_pcd
```

The configuration inherits the original `train.py`/LeWM/DexArt settings and changes only data path, output name and16/2 reader settings. If the runtime does not read `SPT_CACHE_DIR`, set `stable_pretraining.set(cache_dir=..., cleanup={'checkpoints':False,'logs':False,'callback_artifacts':False})` before invoking the unchanged `train.run`; the original recorded run configured this explicitly. Do not write cache/checkpoints to a small overlay disk.

For RGB, `data=adroit_relocate` requires `RELOCATE_RGB_TRAIN` pointing to a genuine `pixels` HDF5. The nonvisual replay and PCD-only datasets cannot train RGB. This release reuses the migrated RGB epoch10; it did not retrain RGB. The new configuration has been composed and checked, but no full training was repeated after release integration. Global model initialization was not fixed by the old data-split seed, so identical training outcomes are not promised.

## Official-demo BC

```sh
export RELOCATE_PCD_CHECKPOINT=/ABS/pcd_checkpoint_dir
export CORTEXBENCH_EAI_VC_ROOT=/ABS/verified_eai_vc
export ADROIT_FRESH_EVAL=1
# ADROIT_PCD_CONFIG remains the shared sampling JSON above.
$BC_PY bench/cortexbench/convert_adroit_demos_3d.py relocate-v0 \
  --input /ABS/official_light/demos --source-sha256 OFFICIAL_LIGHT_MANIFEST_SHA \
  --start-demo 0 --max-demos 100 --output /ABS/fresh_official_pcd/relocate-v0.pickle
$BC_PY bench/cortexbench/run_adroit_bc3d.py --config-name Adroit_BC_config \
  env=relocate-v0 embedding=lewm_adroit_relocate_pcd seed=100 \
  num_demos=100 eval_num_traj=25 epochs=100 eval_frequency=5 save_frequency=10 \
  num_cpu=1 data_dir=/ABS/fresh_official_pcd \
  hydra.run.dir=/ABS/fresh_BC_seed100 hydra.job.chdir=true \
  job_name=relocate_pcd_bc wandb.project=local wandb.entity=local
```

The converter also accepts the original official large pickle with its own SHA. The light source is the verified official100×200 derived copy with old images/features removed; it is not the mixed WM dataset. Repeat seeds200/300 in independent fresh output directories; no extra seed enters the three-seed mean. `ADROIT_FRESH_EVAL=1` disables the stale Adroit seed54321 evaluation shortcut only when explicitly requested. Default Pen/MetaWorld behavior is preserved; `train/*` logging still repeats evaluator paths and is not independent training-initial-state performance.

The RGB embedding configuration is `lewm_adroit_relocate_2d` with `RELOCATE_RGB_CHECKPOINT`; use original `run_bc.py` and genuine official RGB demos. It retains the existing legacy ViT key remapping. No new RGB BC result is reported here.

## Epoch10 fragment CEM

From the repository root, provide the complete150-test replay state package and a fresh output for every invocation:

```sh
$CEM_PY lewm_pcd/scripts/checks/prepare_relocate_fragment_rgb.py \
  --test /ABS/nonvisual_test_replay.h5 --initials /ABS/initial_states_test.npz \
  --out /ABS/fresh_rgb_windows
$CEM_PY lewm_pcd/scripts/checks/run_adroit_fragment_cem.py \
  --code-root "$PWD" --task relocate-v0 --route pix \
  --train /ABS/published/train.h5 --test /ABS/published/test.h5 \
  --initials /ABS/initial_states_test.npz \
  --rgb-cache /ABS/fresh_rgb_windows/rgb_windows.h5 \
  --checkpoint /ABS/rgb_checkpoint_dir/weights_epoch_10.pt --out /ABS/fresh_CEM_2D
$CEM_PY lewm_pcd/scripts/checks/run_adroit_fragment_cem.py \
  --code-root "$PWD" --task relocate-v0 --route pcd \
  --train /ABS/published/train.h5 --test /ABS/published/test.h5 \
  --initials /ABS/initial_states_test.npz \
  --checkpoint /ABS/pcd_checkpoint_dir/weights_epoch_10.pt --out /ABS/fresh_CEM_3D
```

Both checkpoint directories require their actual `config.json` and `config.yaml`. The RGB helper caches only the original sampler's selected26-state windows (1274 unique frames), keeps MSAA0, and fails on a missing real frame. No large old RGB dataset is necessary for this evaluation. CEM/BC share the sampler and values; CEM's YAML is explicitly checked against the JSON, not automatically loaded from that file. Train-only StandardScaler preserves the cloud Pen evaluation behavior; its population variance is not bitwise the training Torch `std(correction=1)` statistic.

The compatibility checks cover explicit switches, checkpoint/config contracts, original Pen CPU regression and exact native replay. They do not establish full GPU regressions for every other task, sim2real performance, or causal modality/pretraining advantage.

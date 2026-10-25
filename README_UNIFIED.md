# OGBench Cube unified switches — v3 with original Single LeWM

This checkout keeps the original JEPA4R CortexBench entry points at commit `6cec281879e8b866e567dc2e936072bb652b7326`. New adapters live in `lewm_pcd/bench/ogbench`; the canonical Tiny trainer is unchanged. Stable Worldmodel is bundled once at `6f1e499e9cc0c898d326112f485c1062c3d20f24`.

The accepted v1/v2 directories and code ZIPs are unchanged. Their original manifests/reports are retained under `evidence/unified_v1` and `evidence/unified_v2`; small byte-pinned command references support regression tests when v3 is unpacked alone.

## Scope and preserved contracts

| Collected profile | Cubes | Raw rows / original action rows | Rows used by this training view | Episodes train / validation |
|---|---:|---:|---:|---:|
| `single200` | 1 | 201 / 200 | 200 | 10000 / 1000 |
| `double500` | 2 | 501 / 500 | 500 | 10000 / 1000 |
| `triple_front600` | 3 | 1001 / 1000 | original 600-row prefix | 10000 / 1000 |
| `quadruple600` | 4 | 601 / 600 | 600 | 10000 / 1000 |

These are paired views of the same trajectories, never double-counted as RGB plus PCD episodes. Raw terminal rows stay in the original datasets and remain excluded from these training views. The registered profiles pin the accepted VDS and manifest hashes, original producer attribute validators, and collection provenance. The Single profile also retains its original backing-store stat guard. VDS paths and producer provenance are not rewritten by this release.

Single uses its historical `front_pixels` view; its sealed training manifest does not re-attest an exact camera pose, and the profile states that limit. Double/Triple retain front plus local-X 5 degrees without pullback. Quad retains that pitch plus its approved 0.1 m pullback. Camera, crop, task/source and reward bindings remain explicit.

`official1`..`official4` describe the separate official Play sources. They cannot be passed to the collected-data training reader. Downstream recipes must retain their real source NPZ, feature-cache/checkpoint hashes, camera, preprocessing, reward and evaluator protocol. The existing official state entry remains available unchanged.

## One command interface

From the repository root:

```bash
python lewm_pcd/scripts/run_ogbench.py --list
python lewm_pcd/scripts/run_ogbench.py --profile quadruple600 --method lejepa --stage train --data-root /absolute/accepted/view/root --output-root /absolute/new/run --workers 24 --prefetch 2
```

Both commands only print information. The second command produces a complete JSON command plan; it does not import a model, open a dataset, create an output directory, connect to a server, or start a child. `--cube-count` optionally cross-checks the selected profile. `--report` saves a new plan file and refuses overwrite. Machine JSON only supplies local data/output paths, Python and worker/prefetch settings; it does not connect over SSH or change dataset contracts.

An explicit `--execute` runs the described local child with a fresh output and a sibling `.dispatch` directory. The child and dispatcher keep logs, status, configuration and actual exits. Do not add `--execute` to the verification commands for this release. No formal training, collection or SR is part of code acceptance.

Supported switch combinations:

| Stage | Methods | Binding |
|---|---|---|
| `train` | `pldm`, `lejepa` | Sealed collected profiles only |
| `train` | `lewm2d` / `lewm_rgb`, `lewm3d` / `lewm_pcd` | Bound Double500, Triple-front600, Quad600 sources and explicit native recipes |
| `train` | Single `lewm2d` / `lewm3d`, including canonical names | Original raw201 training file, seed3072, native random window split |
| `gciql` | `state`, `pldm`, `lejepa`, `lewm_rgb`, `lewm_pcd` | Explicit `cube_all_distance_v1` recipe and existing audited inputs |
| `native-sr` | Visual methods above | Explicit accepted policy, frozen encoder, camera/preprocessing/evaluator bindings |

The visual Native loop rejects `state`; use the preserved original state Native entry. Single visual recipes require the profile's explicit original `front_pixels` identity, zero extra pitch/pullback and a cache-bound actual position/FOV. Metadata-only `official1` lacks that camera identity and fails closed; its original entry remains available. Double/Triple retain front5 and Quad retains its explicitly bound pullback. This release does not generate a feature cache or silently substitute a missing checkpoint. Legacy LeWM training, official-state protocol, collection scripts and prior experiment entry points remain in their original locations; they are not relabeled as newly accepted combinations.

## Method boundaries

- PLDM uses one byte-preserved Impala512/GRU method and one author-source closure, RGB16/action15, uint8/255 then antialiased Resize65, train-only sample-standard-deviation action normalization once, B128/BF16 and the original optimizer/schedule. The profile determines the epoch budget; it is never inferred from a directory name. The original author dataclasses require the existing Python 3.10 training environment; Python 3.12 is not compatible with that unchanged factory.
- `lejepa` selects only the canonical CortexBench Tiny/14 entry: head512, projector16, four views, B256, AdamW lr0.002, lambda0.02 and its native warmup/cosine. Worker/prefetch overrides are explicit. Its native export has backbone/head/proj weights and does not provide heldout validation or full optimizer/RNG resume. The native SWM loader, encoder, augmentations and training loop remain unchanged.
- Historical ViT-B code remains backed up and untouched in the original workspace. It is not registered in this switch and its checkpoints are not accepted as Tiny weights.
- Downstream actual dimensions are `current=L+19`, `goal=L+3N`; state-only uses `19+9N` and `3N`. PLDM and Tiny use `L=512`, LeWM uses `L=192`. No padding or projector features are introduced. Original GCIQL sampler/model/update/checkpoint code and Native loop are preserved; only task/path/dimension/encoder bindings are supplied.

World-model and downstream environments remain separate. The bound GCIQL train/gate source requires JAX CUDA; its reload and visual Native policy require JAX CPU, while the Native frozen visual encoder uses CUDA. The runtime enforces those phase-specific settings. No shared environment is installed or upgraded by this release.

## LeWM 2D and 3D switches

`lewm2d` normalizes to the existing `lewm_rgb` method; `lewm3d` normalizes to `lewm_pcd`. Both aliases work for existing downstream stages. The command plan records both `requested_method` and canonical `method`; downstream encoder identity remains controlled by its frozen checkpoint recipe.

Examples below only print plans:

```bash
python lewm_pcd/scripts/run_ogbench.py --profile double500 --method lewm2d --stage train --data-root /root/autodl-tmp/workspace/users/lewm/coordination/double_cube_lewm_full10ep_109_20261015_v1 --output-root /autodl-fs/data/lewm/cube_unified/new_double_rgb
python lewm_pcd/scripts/run_ogbench.py --profile quadruple600 --method lewm3d --stage train --machine-profile lewm_pcd/bench/ogbench/machines/4cube10225090_lewm.json
```

The LeWM recipe is explicit: RGB uses ViT-tiny/14 with latent192; PCD uses the accepted Cube DexArtPointNet with input XYZRGB6 and latent192. It does not fall through to the canonical `ms2` default or change to Sonata. The original `jepa.py`, `module.py`, `utils.py` and encoder bytes stay unchanged. The sealed2/3/4 adapter calls the preserved accepted Double trainer's original Hydra function; model, forward, SIGReg, AdamW, epoch scheduler and native SPT/Lightning loop are reused. Sealed2/3/4 defaults are fresh seed0, B128, BF16 mixed, 10 epochs, gradient clip1, AdamW lr5e-5/weight-decay0.001, SIGReg0.09/17knots/1024projections. Single's original seed/split/defaults are separate below. Resume is not inferred from an old output name.

This data seam uses four observations at offsets `[0,5,10,15]`, `frameskip=5`, and the original 20-row span. Normalize all 20 action rows before reshaping them into `4×25`; do not use PLDM's RGB16/action15 path. Independent sealed validation reuses the train-view action mean/std, with the original NaN-row exclusion and no new epsilon. H500 has 481 windows per episode; H600 has 581. RGB uses the original CPU image preprocessing; PCD uses `(XYZ-[.425,0,.2])*4` and RGB/255. No geometry or encoder-feature cache is introduced.

The three closed profiles retain their own exact manifest/producer guards; 3D view identity comes from its independently hashed PCD entry in the SHA-locked manifest. There is no substitution of RGB hashes for point-cloud hashes. The 4Cube example path is corrected in v2 to the currently verified `quadruple600_serial_v1` directory; the old v1 archive is preserved.

Single WM now uses the separate original protocol described below. `single200` remains the Cube1 task selector and retains its later sealed RGB/PLDM meaning for PLDM/Tiny. For LeWM training, the authoritative input is the explicitly printed `wm_training_data_binding`; the plan labels `profile` as task-selection metadata. Official Play profiles remain separate.

The LeWM child stores its resolved config, source/data binding, logs, status and actual exit code. A trainer returning0 is recorded as `lewm_native_returned` with `formal_acceptance_verified=false`; it is not a claim of final model acceptance. `--config`, when used for LeWM training, must be an exact copy of the packaged mode/protocol-specific recipe JSON; only explicit resource flags override workers/prefetch.

## Original Single protocol

The original code, both configs, raw training HDF5 files, PCD VDS backing and epoch1..10 checkpoint files were found on the user-specified `root@connect.westd.seetacloud.com:41134`. Code/configs were backed up with source-before/after hashes before this integration. The host currently exposes only0.5CPU/2GB for this session and its `nvidia-smi` file is empty; it was used only for light source/metadata/vector reads. No model or training ran there.

The same public switch selects Single automatically:

```bash
python lewm_pcd/scripts/run_ogbench.py --profile single200 --method lewm2d --stage train --machine-profile lewm_pcd/bench/ogbench/machines/10225090_single_original_lewm.json
python lewm_pcd/scripts/run_ogbench.py --profile single200 --method lewm3d --stage train --machine-profile lewm_pcd/bench/ogbench/machines/10225090_single_original_lewm.json
```

These commands only describe a plan. The generated child argv contains `--single-original-protocol`; a direct Single child without that selector is rejected. The selector is rejected for other Cube profiles. Existing `lewm_rgb/lewm_pcd` names also work. The data root here is the original `ogbench-train-20261003/datasets/ogbench` directory, not the later Single PLDM VDS folder.

Preserved Single behavior:

- RGB reads `fixed5_mixed_pix_train_chunk5.h5`; PCD reads `fixed5_mixed_pcd_train.h5` and its original `fixed5_mixed_all_pcd.h5` backing. It does not load the separate validation HDF5.
- One original training HDF5 has10000 episodes×201 rows. The stock SWM loader retains182 windows per episode, including start181 and its terminal NaN action. It keeps the original Python-list indices and does not substitute the sealed finite-action/compact reader.
- `seed3072` drives the original local generator, first for random window split and then for shuffled train sampling. Both action and observation are cached and normalized over the complete original source before that split, including the original NaN-row exclusion and zero-std behavior. Observation remains unused by the JEPA encoder.
- The original fractions are `[0.9, 1-0.9]`. Native remainder allocation on1820000 windows produces1638001 train and181999 validation windows; this is preserved rather than manually rounded. This is a window split, not an episode-heldout protocol.
- The original entry does not set a global seed before constructing the model and omits the Manager `seed` keyword. No `fresh_seed`, `pl.seed_everything` or Double seed0 hook is added. This preserves initialization semantics; it does not claim an identical retraining trajectory to historical weights.
- Original batch128, BF16-mixed,10epochs, optimizer/loss/model/scheduler are retained. RGB defaults to6workers/prefetch3/persistentTrue; PCD defaults to2workers/prefetch2/persistentTrue. Resource flags override only when explicitly supplied.
- Single calls byte-identical canonical `train.py` (`733564...`), while2/3/4 continue through their byte-identical accepted sealed native trainer. A fresh output directory prevents accidental reuse of the original checkpoint path without adding any seed call.

Runtime guards bind file size/mtime, full schema and episode-index hashes to the original accepted evidence. The16.5MB PCD VDS metadata file is rehashed and all25 source/target mappings plus the backing file are checked. The41.98GB RGB and43.67GB backing full-file hashes remain historical seal evidence with current stat/schema/index checks; they are not represented as freshly rehashed. No HDF5 is rewritten.

## Verification and provenance

From `lewm_pcd`, run the bounded suite:

```bash
PYTHONPATH=. python -m pytest bench/ogbench/tests -q
```

Tensor-reader/spawn tests need the existing Torch/HDF5/SWM runtime. The standalone `bench/ogbench/tests/verify_training_cpu.py` compares the original and merged PLDM preprocessing, losses, gradients and strict reload, and Tiny head512 export/reload. It runs on CPU with zero optimizer updates and no CUDA SIGReg call. It requires an explicit unchanged historical reference source and a fresh report directory.

`bench/ogbench/tests/verify_lewm_cpu.py` adds full LeWM RGB224/PCD model checks using fixed inputs, parameters and RNG: preprocessing, encode, predict, prediction/SIGReg/total loss, gradients and the original factory's strict checkpoint reload. The recorded v2 verification uses all1200 point-cloud points, CPU FP32, one thread and zero optimizer updates. CUDA/BF16 training, full epochs and SR are outside this acceptance. Existing sealed Quad views also receive a separate metadata/source-stat-only check; no dataset is rewritten or raw frame scan performed.

`verify_single_original_cpu.py` checks the new setup against the untouched original entry with actual first/last-episode action/state vectors and explicitly dummy visuals. It compares both normalizers, all diagnostic split IDs, shared split/shuffle generator and post-sampling RNG, controlled diagnostic model-initialization RNG, Manager seed omission, raw201 scalar windows and the terminal NaN action. It also hashes the complete1.82M-window native split and checks the original41134 splitter AST. Trainer/Manager/model spies prevent any fit or optimizer update. This is setup/contract proof; unchanged model/loss/factory evidence is reused from v2. Real production binding guards were separately run on41134 without importing Torch.

`UNIFIED_SOURCE_MANIFEST.json` pins shipped v3. The original workspace, prior releases and server production files remain unchanged. Current evidence is under `coordination/cube_unified_single_lewm_20261025_v3`; prior v1/v2 evidence is retained. Formal CUDA training, complete epochs, historical checkpoint re-acceptance and SR are outside this code-only acceptance.

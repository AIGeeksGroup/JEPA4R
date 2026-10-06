# Pen paired collection, world models and downstream evaluation (2026-10-06)

This release appends the completed Pen route to `cortexbench-adroit`; it does not replace its historical Pen/Relocate implementations or results. The tested downstream reference was `cortexbench@f1cc56dade6937de8ceec6834d0e796ff8bb407f`. Shared BC loops, head, optimizer, preprocessing, evaluation and save conditions remain unchanged. The new release loader reuses the historical encoder classes and has a checked tensor/output/RNG equivalence to that reference.

## Verified components and scope

| Component | Code | Verified scope |
|---|---|---|
| Native collection bridge | `lewm_pcd/envs/adroit/env.py`, `policy.py` | Pen24 actions/45 state/100 horizon; Relocate30/39/200 interface kept separately |
| Visible PCD | `envs/adroit/pcd.py` | Pen single-camera native depth/segmentation,1200 points,5mm; object≤300,target≤100,robot fill; no occluded geometry fabrication |
| Paired collection | `scripts/data/pen_paired_20261006/` | Real RGB rollout and exact executed-action replay in fresh PCD processes;16 shards/25episode batches,FD fix,per-stage exits,resume,hashes,VDS publication |
| WM training | `scripts/train/adroit_20261006/`, `config/train/data/adroit_pen*.yaml` | Original train.py/network/loss/optimizer; both10epochs/50910steps,finite,save/strictreload zero difference |
| Official BC | `bench/cortexbench/run_bc.py`, `run_adroit_bc3d.py`,versioned loaders/YAML | Official100demos/25eval/100epochs,three seeds,20evaluation rounds;both routes completed |
| Fragment CEM | `envs/adroit/fragment*.py`, `run_adroit_fragment_cem.py`,fragment YAML | Paired50starts from independenttest150,real prefix replay,original CEM algorithm;separate Adroit extension |

The code is not a bundle of all external runtime assets. MuJoCo2/license, the pinned native/VC libraries, published expert policy, official100demos, HDF5/VDS payloads and trained checkpoints are external prerequisites. The release includes the task code and versioned runner sources; hashes in historical sealed configs must be regenerated for a new checkout rather than bypassed. Historical WM/BC runner snapshots reject changed source/plan hashes deliberately.

## Fixed inputs and versions

- Camera `vil_camera`;RGB256/MSAA8 in its own process;PCD depth/seg256/MSAA0 before the first context. Both onlinePCD and official-demo conversion use the same backend. The published Pen expert pickle is an MJRL policy, not the100demo source.
- Mixed WM train:10000episodes,independent test150;episode probabilities50%expert+noise,30%expert+noisy/random segment,20%random. Actual executed controls and success/failure cases are preserved;100-demo BC and10000mixed WM data must not be confused.
- WM:Python3.10.21,Torch2.8.0+cu128,SPT0.1.8,Lightning2.6.6,HDF5reader3.16.0;see `scripts/train/adroit_20261006/requirements-wm.txt`. Original batch128/BF16/history3/frameskip5/24D→120D preserved. Read overrides only workers16,prefetch2.
- Stable-worldmodel Git:`6f1e499e9cc0c898d326112f485c1062c3d20f24` from `https://github.com/galilai-group/stable-worldmodel.git`.
- eai-vc:`76fe35e87b1937168f1ec4b236e863451883eaf3`;mj_envs:`c1557f5572977085f053df63f4e81f4b4e1fb17c`;mjrl:`83d35df95eb64274c5e93bb32a0a4e2f6576638a`;MuJoCoPy:`5f530c83a970a844bdafc3087a9f9fe7188b45d4` (2.0.2.13);dmc2gym:`0d7581f3a4b9e065e810ce978d422921e9ee110a`.
- BC has a separate runtime withNumPy1.23.5,timm0.6.11,gym0.23.1;native environment paths and library search paths must point to those pinned assets. CEM uses a separate modernWM/native overlay, not theBC timm0.6 environment. Existing WM prefixes were left unchanged.
- Official source:`https://dl.fbaipublicfiles.com/eai-vc/adroit-expert-v1.0.zip`. Pen pickleSHA `cb6e122487f3abaed13e722eec9b4ba59441ea232ccb8a423c7e3a2c5fcfb557`. All100×100 frames verified;original action labels outside[-1,1] remain raw and native.step clips execution.100-demo PCD conversion retained every non-image field and passed real state/reward replay tolerance1e-6/1e-5;PCD SHA `1724288fb2964f49da490bf0bbe553ccada9b09d40ea81d448a21367cb3b15dc`.

## Direct original BC commands

Install the pinned VC tree at the original `lewm_pcd/third_party/eai-vc` path;keep its official code unchanged. Set the two directories to actual external checkpoint directories containing config.json/config.yaml/weights_epoch_10.pt:

```sh
export PEN_20261006_RGB_CHECKPOINT=/ABS/rgb_checkpoint_dir
export PEN_20261006_PCD_CHECKPOINT=/ABS/pcd_checkpoint_dir
export STABLEWM_HOME=/ABS/fresh_output_root
# Configure MuJoCo2 path,license path,LD_LIBRARY_PATH and EGL before Python.
cd lewm_pcd/bench/cortexbench
$BC_PY run_bc.py --config-name Adroit_BC_config env=pen-v0 \
  embedding=lewm_adroit_pen_2d_20261006 seed=100 data_dir=/ABS/officialRGB/ \
  hydra.run.dir=/ABS/freshRGBseed100 hydra.job.chdir=true wandb.project=local wandb.entity=local
$BC_PY run_adroit_bc3d.py env=pen-v0 \
  embedding=lewm_adroit_pen_pcd_20261006 seed=100 data_dir=/ABS/officialPCD/ \
  hydra.run.dir=/ABS/freshPCDseed100 hydra.job.chdir=true wandb.project=local wandb.entity=local
```

Repeat seeds200/300 in freshdirectories. The official configs supply100demos/25eval/100epochs/evalfrequency5/savefrequency10/head256³/MSE/Adam/batch256. Actual zero-based evaluation indices are5,10,…,95,99;save indices10,20,…,90,99. The original best_policy saves by highestreward at save times, not highestsuccess. RGBseed300 did not produce a savedpolicy although its20score records are complete. The included historicalserialrunner needs a new source/head/config seal for this release;direct original CLI above is the portable entry,not a claim of replaying the historical absolute paths unchanged.

## Collection and WM entry points

`make_plan.py --help` accepts explicit worker/config/output root/code root/Python paths and disjointseed ranges. Its sibling `paired_scheduler.py` is the tested immediate-log-FD-close implementation;`publish_vds.py` retains sourcepayloads and checks everyVDS mapping. `legacy_worker.py` pins the released backend and publishedexpert SHA,asserts freshcontexts and uses real action replay. Provide a new workerconfig with actual pinnednative roots,expertpath,sourcefile SHAmanifest and image_size256. Do not reuse oldsealed plan SHA after changing sourcepaths or code.

For WM use the existing `train.py` with `data=adroit_pen`/`adroit_pen_pcd`,`model=lewm`/`lewm_pcd_dexart` and absolute `data.dataset.name` overrides. Match the completed protocol and verify the two epoch10checkpoints. The newdata YAML affine forPCD is center[0.075,-0.02,0.225],scale4,RGB/255. The historicalWM runner/gates are included for auditing and resealing,not as a no-configuration turnkey launcher.

## Fragment CEM commands and limitation

```sh
$CEM_PY lewm_pcd/scripts/checks/run_adroit_fragment_cem.py \
  --code-root /ABS/this_checkout --route pix \
  --train /ABS/train_rgb.h5 --test /ABS/test_rgb.h5 \
  --checkpoint /ABS/rgb/weights_epoch_10.pt --out /ABS/freshCEMpix
# route pcd uses the correspondingPCD train/test/checkpoint paths.
```

Original25step futuregoal,5macrosteps×5controls,receding5,budget50,300samples/top30/10iterations,seed42,historydefault1 remain. Only the exact newPenFragment environment binds normalization to train;existing targetbranch MetaWorld/DMC/default behavior is retained. Task readonly RGBviews resize256to224 using the originalWorld BILINEAR method to match itsinfo shape. SourceHDF bytes are unchanged. Mid-state cache mismatch is resolved by replaying stored actual controls from stored realinitialstate;6checkpoints had observationerror0.No qpos−dt*qvel approximation or tolerancerelaxation.

Goal predicate is the user-approvedAdroit extension:target=t+25objectposition/axis,distance<0.075m,dot>0.95,drop=failure.50starts came from40episodes,23already met thispredicate initially. RGB34/50 andPCD39/50 are rawfragment success,not officialBC scores or evidence of causalplanning advantage;initially-outside27 gave11/27 and16/27,no random/no-actioncontrol was added.

## Relocate and RH56

OldRelocate code,results andhistory remain at `bench/cortexbench/adroit/` and existingdocs. Native30D/39D/200h bridge is preserved without rerunning it in thisPenrelease. The oldtarget-aware PCD result used64 synthetic targetsite points;it must not be relabeled as thisrelease's visiblePen target quota. HistoricalRelocate has scopedexpert,replay,WM-reload andBC evidence;its matchedrandomcontrol and the latestnewinterface are not all verified.

The branch has an olderRM75-B/EG2 precedent. The separate local RM75/RH56 candidate is not included as task-ready code in this release. Its compile/render/motion interfaces have evidence,but nativeRelocate goal andPenholding gates failed;hardwareversion,mounting/calibration andcompleteactionmapping remain unconfirmed. This does not establish anRH56 sim2real system. Candidatehistory andstagedfiles were backedup separately rather than substituted for validatedPen/Relocate code.

Structured current results: `docs/results/adroit_pen_20261006_{paired_bc,fragment_cem,world_models,official_pcd_demos,bc_audit,cem_audit}.json`. All older datedresults remain separately recorded.

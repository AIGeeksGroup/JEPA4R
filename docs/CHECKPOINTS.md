# JEPA4R checkpoint download and loading

`codex/cube-adroit-release` combines `OGBench-cube` and `cortexbench-adroit` while preserving their validated C4 scientific code and guide gates. Only checkpoint publication, downloading and loading are added. The `reproduce` branch is the reference for the shared model repository and directory conventions; its replacement workflows are not imported here.

## Published scope

- `cube_single`, `cube_double`, `cube_triple`, `cube_quadruple`: original JEPA4R epoch-10 world model and final 1,000,000-update GCIQL policy, including the original configuration, RNG/run state and checkpoint manifest.
- `adroit_relocate`: the Oct6 JEPA4R world model and the three reward-selected best BC policies from seeds 100, 200 and 300. This is the lineage of the paper's 88.0% three-seed result. The Oct4 PCD checkpoint is excluded.
- `rh56_relocate`: the existing RH56 PCD world model and three reward-selected best BC policies, reused from their existing HF publication. This is the RM75-B/RH56 system, separate from Native Adroit.

The repository retains all six methods and the existing RGB/RH56 workflows. This checkpoint publication covers JEPA4R; it does not add releases for the other methods. Historical scores identify the original assets, and are not new measurements or a claim that loading reproduces the full statistical result.

## Download one complete bundle

The scripts need only Python's standard library to list, download and check files. Set `JEPA4R_CHECKPOINTS` or pass `--root` to select storage outside the source checkout.

```bash
python scripts/download_checkpoints.py --list
python scripts/download_checkpoints.py --task cube_single --download \
  --root /absolute/checkpoints --output /absolute/new/cube_single.bundle.json
python scripts/download_checkpoints.py --task cube_single --verify \
  --root /absolute/checkpoints --output /absolute/new/cube_single.verified.json
```

Create the parent directory for `--output` first. The output file must not already exist. Replace the task with another name from `--list`; Native/RH56 bundles include all three published policies. The destination is `<root>/<task>/jepa4r/`, followed by the original HF-relative paths.

`released_checkpoints.json` pins immutable HF commits and every required file's bytes/SHA256. Existing corrupt final files are rejected without replacement. Interrupted downloads retain owned partial files and resume with validated HTTP ranges. Do not run two downloads of the same file concurrently. If a process is killed and leaves a `.lock`, confirm that no download owns it before removing that lock; keep the associated `.part` and identity JSON to resume.

Original weights, configuration, RNG sidecars, manifests and training metadata retain their original bytes. Historical configuration paths describe the producer's machine; the download resolver uses the new local bundle paths without rewriting those originals or changing historical acceptance fields.

## Load-only smoke with the original environments

Use the separate environments already specified by `requirements/` and the baseline instructions. Cube WM and GCIQL use different processes; Native Adroit/RH56 BC also need their original `mjrl`/policy classes on `PYTHONPATH`. No GPU is required with `--device cpu`.

Some original `mjrl` builds check MuJoCo during package import. Set `MUJOCO_PY_MUJOCO_PATH`, `MUJOCO_PY_MJKEY_PATH` and the runtime's `LD_LIBRARY_PATH` as in the existing simulator/BC environment before loading those policies. The load-only smoke still performs zero simulator steps; it does not install or compile a simulator.

```bash
WM_PY=/absolute/worldmodel/env/bin/python
POLICY_PY=/absolute/policy/env/bin/python

"$WM_PY" scripts/checkpoint_smoke.py --bundle /absolute/new/cube_single.bundle.json \
  --component wm --device cpu --output /absolute/new/cube_single.wm_smoke.json
"$POLICY_PY" scripts/checkpoint_smoke.py --bundle /absolute/new/cube_single.bundle.json \
  --component policy --seed 0 --device cpu --output /absolute/new/cube_single.policy_smoke.json
```

For `adroit_relocate` or `rh56_relocate`, select `--seed 100`, `200` or `300` for the policy component. Use the original simulator/BC interpreter for unpickling its original policy class; the smoke does not instantiate a simulator. The world-model command uses the original `config.json`, strict weight loading, the original point-cloud preprocessor, and two fixed forwards. The policy command uses the original GCIQL or BC implementation and two deterministic action forwards. Receipts bind the registry revision, actual input files, script and package versions, outputs, and real exit status.

This interface checks downloaded checkpoint integrity and fixed-input loading. Existing `run.py` commands remain the entry points for collection, training, recovery, evaluation and their guide checks. Loading a published bundle does not manufacture a fresh-training acceptance receipt or bypass those checks. A new training run still writes to a new output directory.

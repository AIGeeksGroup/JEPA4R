# JEPA4R Relocate

Final-paper workflows for the native Adroit hand and the simulated RM75-B/RH56 hand. One entry selects the environment and model and connects collection, data checks, world-model training, strict reload, official-source demonstrations, BC and online evaluation.

| `--env` | `--model` |
|---|---|
| `adroit-relocate` | `jepa4r`, `lewm`, `pldm`, `lejepa`, `visreg`, `vjepa` |
| `rh56-relocate` | `jepa4r`, `lewm` |

RH56 uses the paper's MuJoCo simulation.

## Run

Prepare the Python environments and external assets below, then copy `assets.example.json` to a file outside this checkout and replace its paths.

```bash
python run.py --list

# Print the complete plan without starting collection or learning.
python run.py --env adroit-relocate --model jepa4r --mode smoke \
  --assets /absolute/assets.runtime.json --output /absolute/new/adroit-smoke

# Execute the bounded workflow.
python run.py --env adroit-relocate --model jepa4r --mode smoke \
  --assets /absolute/assets.runtime.json --output /absolute/new/adroit-smoke --execute

# Same entry, RH56 environment.
python run.py --env rh56-relocate --model lewm --mode smoke \
  --assets /absolute/assets.runtime.json --output /absolute/new/rh56-rgb-smoke --execute
```

Every output must be a fresh absolute directory outside the checkout. `--stage NAME` selects one stage from the printed plan. `--resume` rechecks accepted artifacts and continues stages that have not started; a failed partial stage requires a fresh directory so its evidence is retained. A changed source, dependency/input content or stage plan cannot resume an older run.

## Environments and assets

Use Linux with a CUDA-capable PyTorch installation. Keep training and simulator/BC dependencies in separate environments. Exact tested versions and outcomes are recorded in `SMOKE_VERIFICATION.json` when a run has been accepted.

- **WM:** Python3.10, Torch2.8/cu128, torchvision0.23, NumPy1.26.4 and `requirements/wm.txt`. Install CUDA wheels with `pip install -r requirements/wm.txt`. The bundled stable-worldmodel library and PLDM entry are pinned to `6f1e499`; their per-file hashes are in its `SOURCE.json`.
- **Adroit simulator/BC:** `requirements/adroit.txt`, native MuJoCo2.0 and the pinned `mujoco-py` fork. A lightweight environment can share the training environment's Torch installation while overriding its simulator packages. The unlocked legacy activation key is available from [Roboti's license page](https://www.roboti.us/license.html).
- **MPC:** use a separate lightweight environment with the WM Torch/SPT stack and `requirements/mpc.txt`, plus the pinned legacy MuJoCo sources. It retains modern timm for the original world-model loader.
- **RH56 simulator/BC:** `requirements/rh56.txt`, MuJoCo3.12.0 and the external native RM75-B/RH56 bundle (`env.py`, assets, policy code and `experts/relocate_expert.pt`). The original BC import chain also needs the pinned legacy MJRL dependencies.

`DEPENDENCIES.json` pins EAI-VC, MJRL, mj_envs, the Adroit mesh submodule and dmc2gym. Apply `patches/eai-vc-train-loop.patch` to the pinned EAI source. This is the final training loop: it records finite loss/gradient/device evidence and saves a new best-reward policy at its evaluation. Its optimizer, loss and evaluation schedule are unchanged. Preserve upstream licenses when installing dependencies.

Prepare official sources without altering their raw labels:

```bash
"$WM_PY" scripts/prepare_official_assets.py \
  --source /absolute/official/relocate-v0.pickle \
  --source-sha256 c71c7af5dda7b9a3d133a7e2baaa821bfe1ab7c4445dfde5624b947d13865694 \
  --output /absolute/new/official-assets
```

This creates the light100 directory and RH56 initialization JSON while preserving all nonvisual fields.

The preparer uses the recorded WM environment (NumPy1.26.4). The native BC source gate verifies the ordered100 original initializations and policy-critical arrays before selecting the smoke demo.

Native mixed collection requires the trusted [official DAPG state-feedback policy](https://github.com/aravindr93/hand_dapg/tree/master/dapg/policies). Native BC requires the original official Relocate100 demonstrations or a verified light copy with all nonvisual values preserved. Their raw action labels remain unchanged, including values outside [-1,1]; the simulator performs its native execution clipping. The RH56 initialization JSON retains the ordered100 official object/target positions and their source hashes. Configure these assets explicitly in the JSON. Pretrained VISReg and V-JEPA use their released weights; their WM training stage is omitted.

## Protocol and verification

Adroit uses30D actions,30D proprio and200 pre-action rows per episode. Its mixed WM train dataset and official BC demonstrations are distinct sources. Point clouds are1200 XYZRGB visible surface points from the final camera/depth/segmentation path, with5mm voxel deduplication and the final object/marker quotas. No synthetic target points are inserted.

RH56 uses12D executed actions and13D measured proprio, T actions plus T+1 observations, the final opposite camera, whole-episode WM splits and train-only normalization. Its BC smoke delegates to the existing `profile` backend, which has the minimum budget and the final camera.

`smoke` reduces dataset/update/evaluation budgets and verifies actual execution and artifact compatibility. It does not reproduce a formal success rate. `full` retains the paper budgets: Adroit10000 train/150 test mixed episodes, RH569000 train/1000 validation episodes, ten WM epochs,100 official-source BC demos and BC seeds100/200/300 with100 epochs and25 trajectories at each of20 evaluations. Use storage sized for the full paired datasets.

`PLAN.json`, `STATUS.json`, stage logs and actual exit files are written outside the source tree. Acceptance requires data/seed/alignment checks, real positive-LR parameter updates, finite gradients, strict reload, frozen-encoder checks and actual online rollouts. BC Best-of-20 and the policy saved by best mean reward are different outputs.

The separate final Relocate CEM fragment entry is `lewm_pcd/scripts/checks/run_adroit_fragment_cem.py`; its full task uses50 paired fragments and future object-position goals. Its explicit smoke mode is a small diagnostic. Fragment success is reported separately from full-task BC success.

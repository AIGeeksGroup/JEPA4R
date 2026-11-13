# stable-worldmodel (vendored)

- Repository: https://github.com/galilai-group/stable-worldmodel
- Commit: `6f1e499e9cc0c898d326112f485c1062c3d20f24` ("Freeze TD-MPC2 running scale during validation (#322)")
- License: MIT (`LICENSE`, unchanged)

Vendored here (bytes of the upstream commit unless listed under local changes): the library package
`stable_worldmodel/` (all 120 files), `LICENSE`, `README.md`, `pyproject.toml`, and the PLDM training script with its
configs, `scripts/train/pldm.py`, `scripts/train/config/{pldm.yaml, data/ogb.yaml, launcher/local.yaml}` (the PLDM
baseline of Tab. 1 / Tab. 2 is the framework's own training, `jepa4r/baselines/pldm.py`). The same file set (unmodified)
is bundled by the OGBench-cube branch (`OGBench-cube@4fbb0c6:lewm_pcd/third_party/stable-worldmodel`, `SOURCE.json`).
Install with `pip install -e third_party/stable-worldmodel` (or put this directory on `PYTHONPATH`).

## Local changes

The four patches of `cortexbench@9331494:lewm_pcd/scripts/baselines/*.patch`, kept in `patches/`. None changes a number:

| Patch | File | Applied here | Change |
|---|---|---|---|
| `pldm_logger_fix.patch` | `scripts/train/pldm.py` | yes | `logger.info(...)` -> `logging.getLogger(__name__).info(...)` (the name `logger` is rebound to the wandb logger later in the function: UnboundLocalError); checkpoint every epoch (`epoch_interval=1`, upstream 5) so that epoch 10 of a 10-epoch run is saved as `weights_epoch_10.pt` |
| `gcrl_interpolate_fix.patch` | `stable_worldmodel/wm/gcrl/gcrl.py` | yes | `interpolate_pos_encoding=False` default (transformers 4.50's Dinov2Model has no such argument and always interpolates) |
| `eval_ff_api_fix.patch` | `scripts/plan/eval_ff.py` | no (file not vendored) | API fix of the planning evaluation script; no paper number uses it |
| `gcbc_ckpt_path_fix.patch` | `scripts/train/gcbc.py` | no (file not vendored) | checkpoint path of the GCBC training script; no paper number uses it |

GATES.md G1 pins this dependency as "stable-worldmodel @ 6f1e499 + 4 patched files" (the four above).

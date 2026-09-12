# LiftCube Sonata PTv3 Latent Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an authentic frozen Sonata Small PTv3 spatial encoder with point-aligned RGB, train one comparable 2500-step LiftCube skip-10 candidate on the existing 90/10 episode split, and compare latent health and action use with the current scratch checkpoint before any CEM evaluation.

**Architecture:** Preserve the temporal encoder, ordered 10-action tokens, cross-attention predictor, ranking loss, SIGReg, and rollout semantics. Add a selectable `sonata_ptv3` spatial backend that strictly loads the pinned official checkpoint, consumes `[centered_xyz, rgb/255, zero_normal]`, lifts encoder-only features to input resolution, and projects them to `D=64`. Generate a new RGB compact dataset with the exact mask and deterministic indices that produced the existing XYZ. Freeze the PTv3 backbone; train the projection and downstream world model only.

**Tech Stack:** Python 3.11, PyTorch 2.5.1 CUDA 12.1, official `facebookresearch/sonata` commit `18c09ff8d713494f78a8213792262b910977a65d`, h5py, NumPy, OmegaConf, pytest, A800 GPU 3.

## Global Constraints

- Work only in `/workspace/users/lewm/projects/4d-jepa-architecture/.worktrees/liftcube-sonata-ptv3` on `codex/liftcube-sonata-ptv3`.
- Preserve backup `codex/liftcube-before-sonata-ptv3-20260906` at `567d0d7`; do not alter the current scratch checkpoint.
- Use `/nix/lewm/4d-jepa-worldmodel`; do not install ManiSkill/SAPIEN/Vulkan or touch the MPC environment.
- Replay HDF5: `/workspace/datasets/maniskill2/v0.5.3/replayed/LiftCube-v0/count_0100/trajectory.pointcloud_robot_seg.pd_ee_delta_pose.h5`.
- XYZ reference: `/workspace/datasets/maniskill2/v0.5.3/compact/LiftCube-v0`.
- RGB output: `/workspace/datasets/maniskill2/v0.5.3/compact_rgb/LiftCube-v0`; never overwrite a non-empty target.
- Weight: `/workspace/users/lewm/checkpoints/sonata/sonata_small.pth`, SHA-256 `ee5dae277dde1d164499579a6b17d0bd38df4e4286b0f1ba21a82f8cc14270d1`.
- First candidate: frozen backbone, `T=3`, `skip_frame=10`, cross-episode batch 8, 2500 steps, checkpoint every 500, `flow_weight=0`.
- GPU 3 only. Never stop or displace external GPU processes; wait if the live capacity gate fails.
- No CEM, simulator rollout, privileged state cost, or downstream environment work in this phase.
- Follow red-green-refactor for every source change. Publish data/JSON/checkpoints atomically and persist logs/status for long work.

---

### Task 1: Pin the upstream Sonata dependency contract

**Files:**
- Create: `tests/test_sonata_dependency_contract.py`
- Modify: `pyproject.toml:12-25`
- Modify: `requirements.txt:1-12`

- [ ] **Step 1: Write the failing contract test**

```python
from pathlib import Path
import tomllib

REVISION = "18c09ff8d713494f78a8213792262b910977a65d"


def test_upstream_contract_pins_sonata_rgb_reader_and_scatter_wheel():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = "\n".join(project["project"]["dependencies"])
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "h5py==3.11.0" in dependencies and "h5py==3.11.0" in requirements
    assert REVISION in dependencies and REVISION in requirements
    wheel = "torch_scatter-2.1.2+pt25cu121-cp311-cp311-linux_x86_64.whl"
    assert wheel in dependencies and wheel in requirements
```

- [ ] **Step 2: Confirm RED**

Run:

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q tests/test_sonata_dependency_contract.py
```

Expected: FAIL because the dependency files do not contain these pins.

- [ ] **Step 3: Add exact upstream-only dependencies**

Add to both dependency files without changing existing Torch/CUDA pins:

```text
h5py==3.11.0
huggingface-hub==1.29.0
packaging==25.0
torch-scatter @ https://data.pyg.org/whl/torch-2.5.0+cu121/torch_scatter-2.1.2+pt25cu121-cp311-cp311-linux_x86_64.whl
sonata @ git+https://github.com/facebookresearch/sonata.git@18c09ff8d713494f78a8213792262b910977a65d
```

- [ ] **Step 4: Confirm GREEN and valid TOML**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q tests/test_sonata_dependency_contract.py
/nix/lewm/4d-jepa-worldmodel/bin/python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('pyproject ok')"
```

Expected: `1 passed` and `pyproject ok`.

- [ ] **Step 5: Install only the missing upstream packages and log the result**

Write install output under `/workspace/users/lewm/runs/4d-jepa-architecture/sonata_environment/`:

```bash
/nix/lewm/4d-jepa-worldmodel/bin/pip install \
  'https://data.pyg.org/whl/torch-2.5.0+cu121/torch_scatter-2.1.2+pt25cu121-cp311-cp311-linux_x86_64.whl' \
  'h5py==3.11.0' \
  'git+https://github.com/facebookresearch/sonata.git@18c09ff8d713494f78a8213792262b910977a65d'
```

Verify `h5py==3.11.0`, `numpy==1.26.4`, `torch==2.5.1+cu121`, CUDA 12.1, `torch-scatter==2.1.2`, and that Sonata resolves inside `/nix/lewm/4d-jepa-worldmodel`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt tests/test_sonata_dependency_contract.py
git commit -m "build: pin Sonata world-model dependencies"
```

---

### Task 2: Reconstruct aligned RGB without changing XYZ

**Files:**
- Create: `scripts/prepare_liftcube_rgb_compact.py`
- Create: `tests/test_prepare_liftcube_rgb_compact.py`

- [ ] **Step 1: Write RED tests with a tiny replay HDF5**

Cover tests named `test_extract_frame_uses_one_mask_and_one_sample_for_xyz_and_rgb`,
`test_prepare_episode_requires_exact_reference_xyz`,
`test_prepare_episode_keeps_source_fields_and_adds_uint8_rgb`,
`test_prepare_root_refuses_nonempty_output`, and
`test_atomic_npz_leaves_no_temporary_file`.

The correspondence assertion must compare RGB to the exact raw indices selected for XYZ, not merely compare shapes.

- [ ] **Step 2: Confirm RED**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q tests/test_prepare_liftcube_rgb_compact.py
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the historical mask and RNG contract once**

```python
XYZ_LOW = np.array([-0.75, -0.60, -0.05], dtype=np.float32)
XYZ_HIGH = np.array([0.75, 0.60, 0.80], dtype=np.float32)
SAMPLING_SEED = 3072


def aligned_sample_indices(xyzw, robot_seg, episode_id, frame_id, point_count=1024):
    xyzw = np.asarray(xyzw, dtype=np.float32)
    robot = np.asarray(robot_seg, dtype=bool).reshape(-1)
    xyz = xyzw[:, :3]
    valid = (
        np.isfinite(xyzw).all(axis=1)
        & (xyzw[:, 3] > 0)
        & (xyz >= XYZ_LOW).all(axis=1)
        & (xyz <= XYZ_HIGH).all(axis=1)
        & ~robot
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < point_count:
        raise ValueError(f"only {len(valid_indices)} valid scene points")
    rng = np.random.default_rng(
        SAMPLING_SEED + 1_000_003 * int(episode_id) + int(frame_id)
    )
    local = np.sort(rng.choice(len(valid_indices), point_count, replace=False))
    return valid_indices[local]


def extract_aligned_frame(xyzw, rgb, robot_seg, episode_id, frame_id, point_count=1024):
    index = aligned_sample_indices(
        xyzw, robot_seg, episode_id, frame_id, point_count
    )
    return (
        np.ascontiguousarray(np.asarray(xyzw, np.float32)[index, :3]),
        np.ascontiguousarray(np.asarray(rgb, np.uint8)[index]),
    )
```

The CLI accepts `--replay-h5`, `--source-compact`, `--output-root`, and `--verify-only`. It reads `xyzw`, `rgb`, and `robot_seg`, regenerates every frame, and requires `np.array_equal(regenerated_xyz, source["xyz"])` before publishing. Copy `action`, `object_pos`, `goal_pos`, and `success`; add `rgb [F,1024,3] uint8`. Write `.npz.tmp`, reopen/validate, then `os.replace`; write the manifest last. Fail on an existing non-empty output root.

- [ ] **Step 4: Confirm GREEN**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q tests/test_prepare_liftcube_rgb_compact.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_liftcube_rgb_compact.py tests/test_prepare_liftcube_rgb_compact.py
git commit -m "feat: reconstruct aligned LiftCube RGB compact data"
```

---

### Task 3: Load RGB with the same second-stage point indices

**Files:**
- Modify: `data/maniskill.py:31-48,137-219`
- Modify: `scripts/eval_maniskill_group1_rollout.py:25,33-125`
- Modify: `tests/test_maniskill_dataset.py`
- Modify: `tests/test_maniskill_rollout_eval.py`

- [ ] **Step 1: Add RED tests for paired sampling**

```python
def test_rgb_sequence_uses_same_indices_as_xyz(tmp_path):
    dataset = ManiSkillSequenceDataset(
        root, split="train", T=3, point_count=4,
        skip_frame=2, include_rgb=True,
    )
    points, actions = dataset[0]
    assert points.shape == (4, 4, 6)
    assert torch.allclose(points[..., 3:], encoded_rgb(points[..., :3]))
    assert 0 <= points[..., 3:].min() <= points[..., 3:].max() <= 1


```

Also add `test_rgb_mode_rejects_missing_or_misaligned_rgb`,
`test_xyz_only_mode_remains_bitwise_compatible`, and
`test_rollout_dataset_can_emit_six_channel_points`.

- [ ] **Step 2: Confirm RED**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q \
  tests/test_maniskill_dataset.py tests/test_maniskill_rollout_eval.py
```

- [ ] **Step 3: Separate index generation from feature gathering**

```python
def _sample_indices(count, point_count, episode_id, frame_id):
    if count < point_count:
        raise ValueError(f"only {count} points; {point_count} required")
    if count == point_count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(_SEED + 1_000_003 * episode_id + frame_id)
    return np.sort(rng.choice(count, size=point_count, replace=False))


def _sample_features(episode, frame_id, point_count, episode_id, include_rgb):
    index = _sample_indices(
        episode["xyz"].shape[1], point_count, episode_id, frame_id
    )
    xyz = np.asarray(episode["xyz"][frame_id, index], np.float32)
    if not include_rgb:
        return np.ascontiguousarray(xyz)
    rgb = np.asarray(episode["rgb"][frame_id, index], np.float32) / 255.0
    return np.ascontiguousarray(np.concatenate([xyz, rgb], axis=-1))
```

Add `include_rgb: bool = False` to both datasets. In RGB mode validate exact `[frames,points,3] uint8` alignment. Keep the default C=3 path unchanged.

- [ ] **Step 4: Confirm GREEN and backward compatibility**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q \
  tests/test_maniskill_dataset.py tests/test_maniskill_rollout_eval.py
```

- [ ] **Step 5: Commit**

```bash
git add data/maniskill.py scripts/eval_maniskill_group1_rollout.py \
  tests/test_maniskill_dataset.py tests/test_maniskill_rollout_eval.py
git commit -m "feat: load aligned RGB point features"
```

---

### Task 4: Implement the strictly loaded frozen Sonata aggregator

**Files:**
- Create: `models/sonata_aggregator.py`
- Create: `tests/test_sonata_aggregator.py`

- [ ] **Step 1: Write RED tests using a fake hierarchical backbone**

Cover CenterShift, voxel restore, two-level feature lifting, hash mismatch, strict-load failure, frozen/eval/no-grad behavior, output shape, deterministic order, and untouched world-frame positions. Unit tests must not require the 155 MB weight or CUDA.

- [ ] **Step 2: Confirm RED**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q tests/test_sonata_aggregator.py
```

- [ ] **Step 3: Add explicit config and pure helpers**

```python
@dataclass
class SonataAggregatorConfig:
    checkpoint_path: str = ""
    expected_sha256: str = ""
    freeze_backbone: bool = True
    enable_flash: bool = False
    grid_size: float = 0.02
    lift_concat_levels: int = 2
```

Implement typed helpers named `file_sha256`, `center_shift_xyz`, and
`lift_encoder_features`; their behavior is exercised independently by the tests above.

Fail unless `in_channels==6`, the file/hash match, `freeze_backbone is True`, the checkpoint config says `in_channels==9`, and strict state loading succeeds.

- [ ] **Step 4: Construct the exact official backbone locally**

```python
state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
backbone_cfg = dict(state["config"])
backbone_cfg.update(enable_flash=False, shuffle_orders=False, drop_path=0.0)
self.backbone = PointTransformerV3(**backbone_cfg)
self.backbone.load_state_dict(state["state_dict"], strict=True)
self.backbone.requires_grad_(False)
self.backbone.eval()

lifted_channels = sum(
    backbone_cfg["enc_channels"][-(cfg.lift_concat_levels + 1):]
)
self.projection = nn.Sequential(
    nn.LayerNorm(lifted_channels),
    nn.Linear(lifted_channels, out_channels),
    nn.LayerNorm(out_channels),
)
```

Do not call `from_pretrained()` and do not download during training.

- [ ] **Step 5: Build the 9D input and restore K-point resolution**

For `points=[world_xyz,rgb]`: CenterShift XYZ per sample, 2 cm voxelize, average features per voxel, run the official encoder under `torch.no_grad()`, perform Sonata's published two concatenating lifts and remaining broadcast lifts, restore K points with the initial inverse, project to D=64, then gather tokens and original world XYZ with the existing `Ordering` permutation.

```python
normal = torch.zeros_like(rgb)
feat9 = torch.cat([centered_xyz, rgb, normal], dim=-1)
assert feat9.shape[-1] == checkpoint_config["in_channels"] == 9
```

Override `train()` so the backbone remains in eval while the projection follows model mode.

- [ ] **Step 6: Confirm GREEN**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q tests/test_sonata_aggregator.py
```

- [ ] **Step 7: Run the real strict-load and GPU-3 forward gate**

After a live ownership check:

```bash
CUDA_VISIBLE_DEVICES=3 /nix/lewm/4d-jepa-worldmodel/bin/python - <<'PY'
import torch
from models.sonata_aggregator import SonataAggregatorConfig, SonataSpatialAggregator

cfg = SonataAggregatorConfig(
    checkpoint_path="/workspace/users/lewm/checkpoints/sonata/sonata_small.pth",
    expected_sha256="ee5dae277dde1d164499579a6b17d0bd38df4e4286b0f1ba21a82f8cc14270d1",
)
model = SonataSpatialAggregator(cfg, in_channels=6, out_channels=64).cuda().train()
points = torch.zeros(2, 512, 6, device="cuda")
points[..., :3].uniform_(-0.5, 0.5)
points[..., 3:].uniform_(0, 1)
tokens, positions = model(points)
assert tokens.shape == (2, 512, 64) and positions.shape == (2, 512, 3)
assert torch.isfinite(tokens).all() and not model.backbone.training
assert all(not p.requires_grad for p in model.backbone.parameters())
print("strict Sonata GPU forward ok")
PY
```

- [ ] **Step 8: Commit**

```bash
git add models/sonata_aggregator.py tests/test_sonata_aggregator.py
git commit -m "feat: add frozen Sonata PTv3 spatial encoder"
```

---

### Task 5: Select the backend without breaking scratch checkpoints

**Files:**
- Modify: `models/aggregator.py:21-31,383-454`
- Modify: `models/world_model.py:17-57`
- Modify: `scripts/train_maniskill_group1.py:32-50,198-289`
- Create: `configs/maniskill_group1_sonata_frozen.yaml`
- Modify: `tests/test_shapes.py`
- Modify: `tests/test_maniskill_group1.py`

- [ ] **Step 1: Add RED tests**

Add tests named `test_default_aggregator_backend_remains_scratch`,
`test_world_model_builds_sonata_backend_when_selected`,
`test_sonata_config_requires_six_channel_rgb_data`,
`test_optimizer_excludes_frozen_parameters`, and
`test_existing_scratch_config_round_trips_unchanged`.

- [ ] **Step 2: Confirm RED**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q \
  tests/test_shapes.py tests/test_maniskill_group1.py
```

- [ ] **Step 3: Add backend config and a lazy factory**

```python
@dataclass
class AggregatorConfig:
    backend: str = "scratch_ptv3"
    sonata: SonataAggregatorConfig = field(default_factory=SonataAggregatorConfig)
    # retain every existing scratch field


def build_spatial_aggregator(cfg, in_channels, out_channels):
    if cfg.backend == "scratch_ptv3":
        return SpatialAggregator(cfg, in_channels, out_channels)
    if cfg.backend == "sonata_ptv3":
        from models.sonata_aggregator import SonataSpatialAggregator
        return SonataSpatialAggregator(cfg.sonata, in_channels, out_channels)
    raise ValueError(f"unknown aggregator backend: {cfg.backend!r}")
```

`WorldModel` calls the factory. Existing configs omit `backend`, preserving scratch module names and checkpoint compatibility.

- [ ] **Step 4: Make RGB and trainable parameters explicit**

Add `include_rgb: bool=False` to `Group1TrainConfig`; require model C=6 when true and C=3 when false. Pass it to train/val datasets. Construct AdamW only from `requires_grad=True` parameters. Store spatial provenance and frozen/trainable counts in `config.json` and checkpoints.

Preserve every scheduled checkpoint instead of overwriting only `last.pt`:

```python
if step % train_cfg.checkpoint_every == 0 or step == train_cfg.steps:
    _save_checkpoint(
        run_dir / f"checkpoint_{step:06d}.pt",
        model, optimizer, step, best_val, model_cfg, train_cfg, mean, std,
    )
    _save_checkpoint(
        run_dir / "last.pt",
        model, optimizer, step, best_val, model_cfg, train_cfg, mean, std,
    )
```

Add a test that steps 500 and 1000 remain independently readable after the later save.

- [ ] **Step 5: Add the comparable Sonata config**

Copy `configs/maniskill_group1_skip10_action_sequence_batch8.yaml` and change only:

```yaml
model:
  C: 6
  aggregator:
    backend: sonata_ptv3
    sonata:
      checkpoint_path: /workspace/users/lewm/checkpoints/sonata/sonata_small.pth
      expected_sha256: ee5dae277dde1d164499579a6b17d0bd38df4e4286b0f1ba21a82f8cc14270d1
      freeze_backbone: true
      enable_flash: false
      grid_size: 0.02
      lift_concat_levels: 2
  loss:
    flow_weight: 0.0
    ranking_weight: 1.0
train:
  data_root: /workspace/datasets/maniskill2/v0.5.3/compact_rgb
  run_name: group1_skip10_sonata_frozen_batch8_2500
  include_rgb: true
  steps: 2500
  batch_size: 8
  checkpoint_every: 500
  device: cuda
```

Retain T=3, skip-frame 10, action tokens, cross-attention, loss coefficients, learning rate, and 90/10 split.

- [ ] **Step 6: Confirm GREEN and scratch compatibility**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q \
  tests/test_shapes.py tests/test_maniskill_group1.py \
  tests/test_maniskill_dataset.py tests/test_temporal_action_conditioning.py
```

- [ ] **Step 7: Commit**

```bash
git add models/aggregator.py models/world_model.py scripts/train_maniskill_group1.py \
  configs/maniskill_group1_sonata_frozen.yaml tests/test_shapes.py \
  tests/test_maniskill_group1.py
git commit -m "feat: select Sonata encoder for LiftCube training"
```

---

### Task 6: Add latent-health and geometric-match metrics

**Files:**
- Create: `scripts/latent_diagnostics.py`
- Create: `tests/test_latent_diagnostics.py`

- [ ] **Step 1: Write RED tests with collapsed, isotropic, and permuted tensors**

Add tests named `test_representation_health_detects_constant_channels`,
`test_representation_health_reports_two_rank_measures`,
`test_off_diagonal_cosine_excludes_self_pairs_and_is_seeded`,
`test_mutual_nearest_matches_ignore_array_index`,
`test_action_gate_rejects_amplification_that_loses_to_persistence`, and
`test_action_gradient_norm_is_nonzero_for_action_driven_predictor`.

- [ ] **Step 2: Confirm RED**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q tests/test_latent_diagnostics.py
```

- [ ] **Step 3: Implement deterministic metrics**

Expose:

Expose typed functions `representation_health(tokens, pair_samples=65536,
seed=3072)`, `mutual_nearest_matches(left_xyz, right_xyz,
max_distance=0.03)`, `matched_temporal_signal`, `action_gradient_norm`, and
`judge_action_control(correct, shuffled, persistence, threshold=0.10)`.

For covariance eigenvalues, report entropy effective rank and participation ratio. Define near-constant channels as std `<1e-6`. Seed cosine pair sampling and exclude self-pairs. Match independent point samples by mutual nearest world-coordinate neighbors within 3 cm, then report all matches and the bottom/top displacement quartiles as static/moving diagnostics; never use them in optimization.

Pass only when `(shuffled_mse-correct_mse)/correct_mse >= 0.10` and `correct_mse < persistence_mse`.

- [ ] **Step 4: Confirm GREEN**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q tests/test_latent_diagnostics.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/latent_diagnostics.py tests/test_latent_diagnostics.py
git commit -m "feat: measure spatial latent health and action use"
```

---

### Task 7: Build the checkpoint-level latent evaluator

**Files:**
- Create: `scripts/eval_liftcube_latent_diagnostics.py`
- Create: `tests/test_liftcube_latent_diagnostics_eval.py`
- Modify: `scripts/eval_maniskill_group1_rollout.py`
- Modify: `models/sonata_aggregator.py`

- [ ] **Step 1: Write RED evaluator tests with a fake action-driven model**

Cover identical validation window IDs, separate raw/projected/temporal/predicted stages, correct/zero/shuffled/repeat-last H1/H2/H4, prediction delta/cosine, gradient, persistence, false amplification rejection, and atomic provenance JSON.

- [ ] **Step 2: Confirm RED**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q \
  tests/test_latent_diagnostics.py tests/test_liftcube_latent_diagnostics_eval.py
```

- [ ] **Step 3: Expose diagnostic-only raw spatial features**

```python
@torch.no_grad()
def diagnostic_features(self, points):
    raw, projected, positions = self._encode(points, return_raw=True)
    return {
        "raw_spatial": raw,
        "projected_spatial": projected,
        "positions": positions,
    }
```

For scratch, record raw spatial as unavailable and use the existing output as projected spatial; do not compare unlike raw widths as if they were the same representation.

- [ ] **Step 4: Implement deterministic evaluation**

CLI:

```bash
python scripts/eval_liftcube_latent_diagnostics.py \
  --checkpoint CHECKPOINT --data-root DATA_ROOT --output OUTPUT.json \
  --horizons 1 2 4 --batch-size 8 --seed 3072 --device cuda
```

Infer RGB from model C, reuse existing skip-10 action alignment/shuffle controls, use identical validation indices, and report health for raw spatial, projected spatial, temporal context, and predicted latents. Match geometry by returned world positions. Report correct/zero/shuffled/repeat-last errors, shuffled degradation, prediction delta/cosine, action gradient, persistence, and the two-part action gate. Include checkpoint hash/step/config, dataset manifest hash, window count, and timing. Write with temporary + `os.replace`.

- [ ] **Step 5: Confirm GREEN**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q \
  tests/test_maniskill_rollout_eval.py tests/test_latent_diagnostics.py \
  tests/test_liftcube_latent_diagnostics_eval.py
```

- [ ] **Step 6: Commit**

```bash
git add models/sonata_aggregator.py scripts/eval_maniskill_group1_rollout.py \
  scripts/latent_diagnostics.py scripts/eval_liftcube_latent_diagnostics.py \
  tests/test_liftcube_latent_diagnostics_eval.py
git commit -m "feat: evaluate LiftCube latent representations"
```

---

### Task 8: Add a recoverable GPU-3 runner

**Files:**
- Create: `scripts/run_liftcube_sonata_frozen_2500.sh`
- Create: `tests/test_sonata_runner.py`

- [ ] **Step 1: Write RED safety tests**

Require `CUDA_VISIBLE_DEVICES=3`, non-empty run refusal, live-PID refusal, atomic PREPARING/RUNNING/EVALUATING/COMPLETE-or-FAILED status, one PID/log/run directory, trapped failure code, checkpoint diagnostics, and no CEM invocation.

- [ ] **Step 2: Confirm RED**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q tests/test_sonata_runner.py
```

- [ ] **Step 3: Implement the runner**

```bash
RUN_DIR=/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_sonata_frozen_batch8_2500
STATUS="$RUN_DIR/group1_skip10_sonata_frozen_batch8_2500.status"
PID_FILE="$RUN_DIR/group1_skip10_sonata_frozen_batch8_2500.pid"
LOG="$RUN_DIR/group1_skip10_sonata_frozen_batch8_2500.log"
export CUDA_VISIBLE_DEVICES=3
export PYTHONDONTWRITEBYTECODE=1
```

Flush JSONL/log output. On terminal success, evaluate steps 500/1000/1500/2000/2500 and `best.pt`; preserve all artifacts.

- [ ] **Step 4: Confirm GREEN and shell syntax**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q tests/test_sonata_runner.py
bash -n scripts/run_liftcube_sonata_frozen_2500.sh
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_liftcube_sonata_frozen_2500.sh tests/test_sonata_runner.py
git commit -m "feat: run recoverable Sonata latent experiment"
```

---

### Task 9: Verify the branch before real data/GPU work

**Files:** None expected

- [ ] **Step 1: Run focused tests together**

```bash
PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest \
  -p no:cacheprovider -q \
  tests/test_sonata_dependency_contract.py \
  tests/test_prepare_liftcube_rgb_compact.py \
  tests/test_maniskill_dataset.py tests/test_sonata_aggregator.py \
  tests/test_maniskill_group1.py tests/test_shapes.py \
  tests/test_latent_diagnostics.py \
  tests/test_liftcube_latent_diagnostics_eval.py tests/test_sonata_runner.py
```

- [ ] **Step 2: Run full regression**

```bash
PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4d-jepa-worldmodel/bin/python -m pytest \
  -p no:cacheprovider -q
```

Expected: all tests pass; no skip conceals a required Sonata path.

- [ ] **Step 3: Compile and check the tree**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m py_compile \
  data/maniskill.py models/aggregator.py models/sonata_aggregator.py \
  models/world_model.py scripts/prepare_liftcube_rgb_compact.py \
  scripts/train_maniskill_group1.py scripts/latent_diagnostics.py \
  scripts/eval_liftcube_latent_diagnostics.py
bash -n scripts/run_liftcube_sonata_frozen_2500.sh
git diff --check
git status --short
```

Expected: every command exits zero and committed status is clean.

---

### Task 10: Materialize and audit the real RGB compact data

**Files:** Data only under `/workspace/datasets/maniskill2/v0.5.3/compact_rgb/LiftCube-v0`

- [ ] **Step 1: Verify all 100 episodes without publishing**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python scripts/prepare_liftcube_rgb_compact.py \
  --replay-h5 /workspace/datasets/maniskill2/v0.5.3/replayed/LiftCube-v0/count_0100/trajectory.pointcloud_robot_seg.pd_ee_delta_pose.h5 \
  --source-compact /workspace/datasets/maniskill2/v0.5.3/compact/LiftCube-v0 \
  --output-root /workspace/datasets/maniskill2/v0.5.3/compact_rgb/LiftCube-v0 \
  --verify-only
```

Expected: every regenerated frame's XYZ is exactly equal to the reference compact XYZ; RGB is `[F,1024,3] uint8` and aligned by the same selected raw indices.

- [ ] **Step 2: Publish once, then audit**

Run without `--verify-only`. Require 100 episodes, the unchanged 90/10 IDs, exact equality of every XYZ/action/label array to source, valid RGB range/diversity, one manifest, and no temporary files. Any XYZ mismatch blocks training.

---

### Task 11: Preflight, smoke, and launch one detached run on GPU 3

**Files:** Runtime outputs under `/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_sonata_frozen_batch8_2500`

- [ ] **Step 1: Recheck live GPU ownership/capacity**

```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv,noheader
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader
```

Do not rely on a stale GPU reading and do not displace another process.

- [ ] **Step 2: Run finite-gradient preflight**

```bash
CUDA_VISIBLE_DEVICES=3 /nix/lewm/4d-jepa-worldmodel/bin/python \
  scripts/train_maniskill_group1.py \
  --config configs/maniskill_group1_sonata_frozen.yaml --preflight
```

Require points `[8,4,512,6]`, actions `[8,30,7]`, finite loss/downstream gradients, zero backbone gradients, backbone eval mode, strict-load provenance, and recorded peak memory.

- [ ] **Step 3: Run a non-overwriting 20-step smoke**

```bash
CUDA_VISIBLE_DEVICES=3 /nix/lewm/4d-jepa-worldmodel/bin/python \
  scripts/train_maniskill_group1.py \
  --config configs/maniskill_group1_sonata_frozen.yaml \
  --run-dir /workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_sonata_frozen_smoke20 \
  train.steps=20 train.log_every=1 train.val_every=10 \
  train.checkpoint_every=10 train.val_batches=2
```

Require stable memory, finite validation, readable checkpoints, and identical frozen-backbone tensor hashes before/after.

- [ ] **Step 4: Launch exactly one formal process**

```bash
nohup bash scripts/run_liftcube_sonata_frozen_2500.sh \
  >/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_sonata_frozen_batch8_2500.launcher.log \
  2>&1 &
```

Check once that the PID/command/status/log/first JSONL record are coherent and GPU memory is stable. Then exit instead of polling.

---

### Task 12: Compare terminal Sonata checkpoints with the scratch baseline

**Files:** Runtime reports only

- [ ] **Step 1: Resume only after terminal status**

Do not duplicate a RUNNING process. For FAILED, report the exact exception and useful log tail before changing code. For COMPLETE, validate checkpoint steps and readability.

- [ ] **Step 2: Evaluate the fixed scratch baseline**

Use `/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_action_sequence_batch8_2500/best.pt`, expected step 1500 and SHA-256 `346ffd87ad74d9b55030258eb3c4cc66ab9b38a113d70e394b5679ab2976905d`, with the original C=3 compact data.

- [ ] **Step 3: Evaluate Sonata at 500/1000/1500/2000/2500 and best**

Use identical validation window IDs/seed and save one atomic JSON per checkpoint.

- [ ] **Step 4: Write `latent_comparison.json` and `.md`**

Report stage-wise latent health, matched temporal signal, correct/zero/shuffled/repeat-last H1/H2/H4 MSE, shuffled degradation, prediction delta/cosine, action gradient, persistence, memory/time, and trainable parameters. Label action-valid only when shuffled is at least 10% worse and correct beats persistence. Report the lowest-correct-MSE and largest-gap checkpoints separately if none passes.

Interpretation:

- healthy raw Sonata features plus weak action separation means the downstream dynamics/action path remains the bottleneck;
- collapsed raw Sonata features mean preprocessing/lifting is wrong, not that pretraining failed;
- healthy raw but collapsed projected features isolate the projection/loss boundary;
- weak matched temporal response in otherwise healthy raw features is the only reason to propose later last-stage unfreezing;
- a large prediction delta without better target alignment is not success;
- do not run CEM.

- [ ] **Step 5: Clean generated caches and audit Git**

Remove only agent-created `__pycache__`, `.pytest_cache`, and `pointcloud_world_model.egg-info` under the named worktree. Preserve data, logs, checkpoints, and reports. Require clean `git status --short`.

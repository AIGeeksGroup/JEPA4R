# LiftCube Proprioception and Goal-Aware Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add train-only-normalized Panda proprioception and object-state prediction to the LiftCube world model, then make CEM optimize predicted lift-and-hold behavior while preserving legacy checkpoints and evaluator truth boundaries.

**Architecture:** An opt-in 25-dimensional proprioception stream creates one robot token per sampled frame. Every predictor layer cross-attends scene queries to the current robot token plus the existing ten ordered action tokens, while auxiliary heads predict next proprioception and object position; planner rollouts recursively feed predicted proprioception and score predicted height, hold, stability, approach, and early-close behavior through a structured `GoalSpec`.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu121, NumPy, h5py, OmegaConf, ManiSkill2/SAPIEN, pytest, Git worktrees.

## Global Constraints

- Work only in `/workspace/users/lewm/projects/4d-jepa-architecture/.worktrees/liftcube-sonata-ptv3` on `codex/liftcube-proprio-goal`.
- Preserve backup branch `codex/liftcube-before-proprio-goal-2026-09-08` at base commit `51a1baba694e1f7007ad624b7e3b66be5bd1fbdf`.
- Never overwrite `/workspace/datasets/maniskill2/v0.5.3/compact_rgb/LiftCube-v0`, the baseline checkpoint, old runs, or old CEM results.
- Publish data only to `/workspace/datasets/maniskill2/v0.5.3/compact_rgb_proprio/LiftCube-v0` through verified atomic writes.
- Preserve the exact ordered 90 train / 10 validation episode IDs from the existing manifest.
- Compute proprioception mean/std from train episode IDs only; floor standard deviations at `1e-6`.
- Preserve every ordered atomic action for `skip_frame=10`; do not replace ten control actions with one aggregate action.
- Use `/nix/lewm/4d-jepa-worldmodel` for dataset/model/training and `/nix/lewm/4d-jepa-mpc` for ManiSkill/SAPIEN/renderer/CEM.
- Do not create virtual environments in `/workspace`.
- Do not use `strict=False` to hide missing keys. Legacy configs instantiate no proprioception modules and must load legacy checkpoints strictly.
- Run test-first red/green cycles for every behavior change.
- A finite loss, passing test, or lower predicted cost is not evidence of physical control success.
- After verification, start at most one new no-resume training run capped at 2,500 steps; do not launch a formal ten-seed CEM run.

## File Map

- Create `scripts/prepare_liftcube_proprio_compact.py`: copy compact RGB fields, extract replay proprioception, validate provenance/splits, compute train-only statistics, and publish atomically.
- Create `tests/test_prepare_liftcube_proprio_compact.py`: fixture-based data alignment, normalization leakage, manifest, and atomic-write tests.
- Modify `data/maniskill.py:31-49,176-275`: validate/load aligned proprioception and object-position windows while preserving legacy tuple output.
- Modify `tests/test_maniskill_dataset.py:10-194`: exercise opt-in mapping output and episode boundaries.
- Create `models/proprio.py`: configuration, normalization buffers, robot encoder, and grouped metric helpers.
- Modify `models/predictor.py:102-227`: concatenate one robot token with ordered action KV tokens in every decoder layer and expose transition features.
- Modify `models/world_model.py:25-170`: configure the branch, predict auxiliary outputs, compute loss metrics, and provide structured transition prediction.
- Modify `losses/lewm.py:22-167`: add configured proprioception/object losses without changing disabled-path behavior.
- Modify `tests/test_temporal_action_conditioning.py`: prove robot/action conditioning and gradients.
- Modify `tests/test_shapes.py`: prove enabled/disabled world-model output and strict legacy behavior.
- Modify `scripts/train_maniskill_group1.py:21-299`: consume mapping batches, inject manifest statistics, log/save auxiliary metrics.
- Create `configs/maniskill_group1_sonata_proprio_goal.yaml`: new 25-dimensional architecture and unique 2,500-step run.
- Create `scripts/run_liftcube_sonata_proprio_goal_2500.sh`: environment/version/GPU gates and recoverable launch.
- Create `planning/liftcube_goal.py`: `GoalSpec`, cost weights, cost components, validation, and JSON serialization.
- Modify `planning/latent_cem.py:12-215`: add structured autoregressive rollout and goal-aware CEM scoring while retaining latent-only entry points.
- Modify `tests/test_latent_cem.py`: prove no future-proprioception access and preserve old latent planner behavior.
- Modify `scripts/eval_liftcube_latent_cem.py:45-707`: online proprioception capture, new goal/cost plumbing, predicted traces, and unchanged physical evaluator fields.
- Modify `tests/test_liftcube_latent_cem_eval.py`: cost semantics, NaN/empty/terminated behavior, and evaluator/planner separation.
- Modify `planning/__init__.py` and `models/__init__.py`: export only the new public dataclasses/functions used by scripts and tests.

---

### Task 1: Prepare a Non-Destructive RGB-Proprioception Dataset

**Files:**
- Create: `scripts/prepare_liftcube_proprio_compact.py`
- Create: `tests/test_prepare_liftcube_proprio_compact.py`

**Interfaces:**
- Consumes: source compact manifest/NPZ files and replay HDF5/JSON groups named `traj_<episode_id>`.
- Produces: `prepare_episode(replay_group: h5py.Group, source: Mapping[str, np.ndarray], *, episode_id: int) -> dict[str, np.ndarray]`, `compute_proprio_stats(episodes: Mapping[int, Mapping[str, np.ndarray]], train_episode_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray, int]`, and `prepare_root(replay_h5: Path, replay_json: Path, source_compact: Path, output_root: Path, *, verified_robot_uid: str, verify_only: bool = False) -> dict[str, object]`.

- [ ] **Step 1: Write failing extraction and episode-alignment tests**

```python
def test_prepare_episode_preserves_source_and_adds_aligned_proprio():
    source = make_compact_episode(frames=4)
    with replay_fixture(source, qpos_value=2.0) as group:
        result = prepare_episode(group, source, episode_id=7)
    for key in ("xyz", "rgb", "action", "object_pos", "goal_pos", "success"):
        np.testing.assert_array_equal(result[key], source[key])
    assert result["qpos"].shape == (4, 9)
    assert result["qvel"].shape == (4, 9)
    assert result["tcp_pose"].shape == (4, 7)
    assert result["base_pose"].shape == (4, 7)

@pytest.mark.parametrize("field", ["qpos", "qvel", "tcp_pose", "base_pose"])
def test_prepare_episode_rejects_proprio_length_mismatch(field):
    source = make_compact_episode(frames=4)
    with replay_fixture(source, shorten=field) as group:
        with pytest.raises(ValueError, match=field):
            prepare_episode(group, source, episode_id=7)
```

- [ ] **Step 2: Run the tests and verify the RED failure**

Run:

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest tests/test_prepare_liftcube_proprio_compact.py -q
```

Expected: collection fails because `scripts.prepare_liftcube_proprio_compact` does not exist.

- [ ] **Step 3: Implement field extraction and strict validation**

```python
SOURCE_FIELDS = ("xyz", "rgb", "action", "object_pos", "goal_pos", "success")
REPLAY_FIELDS = {
    "qpos": "obs/agent/qpos",
    "qvel": "obs/agent/qvel",
    "tcp_pose": "obs/extra/tcp_pose",
    "base_pose": "obs/agent/base_pose",
}
PROPRIO_LAYOUT = (
    {"name": "qpos", "start": 0, "stop": 9},
    {"name": "qvel", "start": 9, "stop": 18},
    {"name": "tcp_pose", "start": 18, "stop": 25},
)

def prepare_episode(replay_group, source, *, episode_id):
    frames = int(np.asarray(source["xyz"]).shape[0])
    action = np.asarray(source["action"])
    if np.asarray(source["rgb"]).shape[:1] != (frames,) or np.asarray(source["object_pos"]).shape != (frames, 3):
        raise ValueError(f"episode {episode_id}: observation fields are misaligned")
    if np.asarray(source["goal_pos"]).shape != (3,) or np.asarray(source["success"]).shape != (frames - 1,):
        raise ValueError(f"episode {episode_id}: static/action fields are misaligned")
    if action.shape != (frames - 1, 7):
        raise ValueError(f"episode {episode_id}: action must be [{frames - 1},7]")
    replay_action = np.asarray(replay_group["actions"], dtype=np.float32)
    if not np.array_equal(action, replay_action):
        raise ValueError(f"episode {episode_id}: replay actions differ")
    result = {name: np.asarray(source[name]) for name in SOURCE_FIELDS}
    expected_dims = {"qpos": 9, "qvel": 9, "tcp_pose": 7, "base_pose": 7}
    for name, replay_name in REPLAY_FIELDS.items():
        value = np.asarray(replay_group[replay_name], dtype=np.float32)
        if value.shape != (frames, expected_dims[name]):
            raise ValueError(f"episode {episode_id}: {name} shape {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"episode {episode_id}: {name} is not finite")
        result[name] = value
    return result
```

- [ ] **Step 4: Add failing train-only statistics and manifest tests**

```python
def test_proprio_stats_never_open_validation_episode():
    episodes = RecordingEpisodes({0: proprio(1.0), 1: proprio(3.0), 9: proprio(999.0)})
    mean, std, count = compute_proprio_stats(episodes, [0, 1])
    assert episodes.opened == [0, 1]
    np.testing.assert_allclose(mean, np.full(25, 2.0))
    np.testing.assert_allclose(std, np.full(25, 1.0))
    assert count == 8

def test_prepare_root_preserves_exact_ordered_split(tmp_path):
    result = prepare_root(replay_h5, replay_json, source_root, tmp_path / "out")
    assert result["train_episode_ids"] == source_manifest["train_episode_ids"]
    assert result["val_episode_ids"] == source_manifest["val_episode_ids"]
    assert result["proprio_normalization"]["source_split"] == "train"
    assert result["robot_uid"] == "panda"
    assert result["control_mode"] == "pd_ee_delta_pose"
```

- [ ] **Step 5: Implement streaming train-only statistics and atomic publication**

```python
def proprio_array(episode):
    return np.concatenate(
        [episode["qpos"], episode["qvel"], episode["tcp_pose"]], axis=-1
    ).astype(np.float32, copy=False)

def compute_proprio_stats(episodes, train_episode_ids):
    total = np.zeros(25, dtype=np.float64)
    squares = np.zeros(25, dtype=np.float64)
    count = 0
    for episode_id in train_episode_ids:
        values = proprio_array(episodes[int(episode_id)]).astype(np.float64)
        total += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
        count += len(values)
    mean = total / count
    variance = np.maximum(squares / count - np.square(mean), 0.0)
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), 1e-6).astype(np.float32), count
```

Reuse the existing verified `atomic_save_npz`, `_atomic_write_json`, and SHA-256 patterns. Publish `manifest.json` after all 100 episodes reopen successfully. Store replay JSON control-mode verification and runtime Panda verification provenance.

- [ ] **Step 6: Run focused tests and commit the data code**

Run:

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest tests/test_prepare_liftcube_proprio_compact.py -q
git diff --check
git add scripts/prepare_liftcube_proprio_compact.py tests/test_prepare_liftcube_proprio_compact.py
git commit -m "feat: prepare LiftCube proprio compact data"
```

Expected: focused tests pass; commit contains no generated dataset files.

### Task 2: Emit Aligned Dataset Windows and Configure the New Run

**Files:**
- Modify: `data/maniskill.py:31-49,176-275`
- Modify: `tests/test_maniskill_dataset.py:10-194`
- Modify: `scripts/train_maniskill_group1.py:21-299`
- Create: `configs/maniskill_group1_sonata_proprio_goal.yaml`

**Interfaces:**
- Consumes: Task 1 NPZ and manifest schema.
- Produces: legacy `(points, actions)` or enabled mapping with `points`, `actions`, `proprio`, and `object_pos`; `load_proprio_stats(root) -> tuple[np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write failing aligned-window and validation-leakage tests**

```python
def test_dataset_emits_aligned_proprio_mapping(tmp_path):
    root = write_task(tmp_path, include_proprio=True)
    dataset = ManiSkillSequenceDataset(
        root, split="train", T=2, skip_frame=3, point_count=8,
        include_rgb=True, include_proprio=True,
    )
    batch = dataset[0]
    assert set(batch) == {"points", "actions", "proprio", "object_pos"}
    assert batch["points"].shape == (3, 8, 6)
    assert batch["actions"].shape == (6, 7)
    assert batch["proprio"].shape == (3, 25)
    torch.testing.assert_close(batch["proprio"][:, 0], torch.tensor([0.0, 3.0, 6.0]))

def test_legacy_dataset_output_remains_tuple(tmp_path):
    dataset = ManiSkillSequenceDataset(write_task(tmp_path), point_count=8)
    assert isinstance(dataset[0], tuple)
    assert len(dataset[0]) == 2
```

- [ ] **Step 2: Run the dataset tests and verify RED**

Run:

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest tests/test_maniskill_dataset.py -q
```

Expected: the new test fails because `include_proprio` is not accepted.

- [ ] **Step 3: Implement opt-in mapping output**

```python
def _proprio(episode):
    return np.ascontiguousarray(
        np.concatenate([episode["qpos"], episode["qvel"], episode["tcp_pose"]], axis=-1),
        dtype=np.float32,
    )

# In __getitem__ after frame_ids/actions are built:
if not self.include_proprio:
    return torch.from_numpy(points), torch.from_numpy(actions)
proprio = _proprio(episode)[frame_ids]
object_pos = np.ascontiguousarray(episode["object_pos"][frame_ids], dtype=np.float32)
return {
    "points": torch.from_numpy(points),
    "actions": torch.from_numpy(actions),
    "proprio": torch.from_numpy(proprio),
    "object_pos": torch.from_numpy(object_pos),
}
```

Extend `_validate_episode` to require exact `[F,9]`, `[F,9]`, `[F,7]`, and `[F,3]` finite arrays only when proprioception is enabled.

- [ ] **Step 4: Write failing config and batch-unpacking tests**

```python
def test_sonata_proprio_config_uses_new_data_and_aux_losses():
    model_cfg, train_cfg = load_group1_config(CONFIG, [])
    assert train_cfg.data_root.endswith("compact_rgb_proprio")
    assert train_cfg.include_proprio is True
    assert model_cfg.proprio.enabled is True
    assert model_cfg.proprio.dim == 25
    assert model_cfg.loss.proprio_weight > 0
    assert model_cfg.loss.object_weight > 0
```

- [ ] **Step 5: Implement training config plumbing without model behavior yet**

Add `include_proprio: bool = False` to `Group1TrainConfig`, read manifest statistics with exact length 25, copy them into `model_cfg.proprio.mean/std`, construct enabled datasets, and add `_unpack_batch`:

```python
def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["points"], batch["actions"], batch["proprio"], batch["object_pos"]
    points, actions = batch
    return points, actions, None, None
```

Create the new YAML with `T=3`, `skip_frame=10`, `K=512`, `D=64`, `C=6`, frozen Sonata, cross-attention, ranking weight `1.0`, proprioception enabled at 25 dimensions, non-zero configured auxiliary weights, batch size 8, and unique run name `group1_skip10_sonata_proprio_goal_batch8_2500`.

- [ ] **Step 6: Run focused tests and commit dataset/config plumbing**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest tests/test_maniskill_dataset.py tests/test_maniskill_group1.py -q
git diff --check
git add data/maniskill.py scripts/train_maniskill_group1.py configs/maniskill_group1_sonata_proprio_goal.yaml tests/test_maniskill_dataset.py tests/test_maniskill_group1.py
git commit -m "feat: emit aligned LiftCube proprio windows"
```

### Task 3: Inject Robot Tokens and Predict Auxiliary State

**Files:**
- Create: `models/proprio.py`
- Modify: `models/predictor.py:102-227`
- Modify: `models/world_model.py:25-170`
- Modify: `models/__init__.py`
- Modify: `losses/lewm.py:22-167`
- Modify: `tests/test_temporal_action_conditioning.py`
- Modify: `tests/test_shapes.py`

**Interfaces:**
- Consumes: raw proprioception `[B,T+1,25]`, object position `[B,T+1,3]`, and train-only normalization buffers.
- Produces: `TransitionPrediction(scene_latent, temporal_features, decoder_features, next_proprio_normalized, next_proprio, object_pos)` and auxiliary loss/metric fields. Only the normalized field is fed into later model transitions; only the raw field is used for metrics and geometric cost.

- [ ] **Step 1: Write failing conditioning and gradient tests**

```python
def test_same_scene_action_different_proprio_changes_prediction():
    model = enabled_world_model().eval()
    points, actions, proprio = inputs()
    first = model(points, actions, proprio=proprio, object_pos=torch.zeros(2, 4, 3))["pred"]
    changed = proprio.clone()
    changed[:, :, 0] += 2.0
    second = model(points, actions, proprio=changed, object_pos=torch.zeros(2, 4, 3))["pred"]
    assert not torch.allclose(first, second)

def test_action_and_proprio_conditioning_receive_gradients():
    output = enabled_world_model()(points, actions.requires_grad_(), proprio=proprio.requires_grad_(), object_pos=object_pos)
    output["loss"].backward()
    assert actions.grad is not None and actions.grad.abs().sum() > 0
    assert proprio.grad is not None and proprio.grad.abs().sum() > 0
```

- [ ] **Step 2: Run the two tests and verify RED**

Expected: `WorldModel.forward()` rejects `proprio`.

- [ ] **Step 3: Implement normalization and robot encoding**

```python
@dataclass
class ProprioConfig:
    enabled: bool = False
    dim: int = 25
    hidden_dim: int = 128
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

class ProprioEncoder(nn.Module):
    def __init__(self, cfg, D):
        super().__init__()
        self.register_buffer("mean", torch.tensor(cfg.mean).view(1, 1, cfg.dim))
        self.register_buffer("std", torch.tensor(cfg.std).view(1, 1, cfg.dim).clamp_min(1e-6))
        self.net = nn.Sequential(nn.Linear(cfg.dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, D))
        self.robot_type = nn.Parameter(torch.zeros(1, 1, D))

    def normalize(self, value):
        return (value - self.mean) / self.std

    def forward_normalized(self, normalized):
        return self.net(normalized) + self.robot_type
```

- [ ] **Step 4: Make every predictor layer attend to robot plus action tokens**

Change predictor signatures to accept `robot_tokens: torch.Tensor | None`. Inside `STBlock.forward`:

```python
condition = action_ids
if robot_tokens is not None:
    robot = rearrange(robot_tokens, "B T C -> (B T) 1 C")
    condition = torch.cat([robot, action_SC], dim=1)
key_value = self.action_norm(condition)
action_update, _ = self.action_cross_attn(query, key_value, key_value, need_weights=False)
```

Return decoder features when `return_features=True`; otherwise preserve the current tensor return type.

- [ ] **Step 5: Implement auxiliary heads and configured losses**

```python
features = prediction.decoder_features.mean(dim=2)
predicted_proprio_normalized = self.next_proprio_head(features)
predicted_proprio = self.proprio_encoder.denormalize(predicted_proprio_normalized)
predicted_object = self.object_position_head(features)
proprio_loss = F.mse_loss(predicted_proprio_normalized, normalized_proprio[:, 1:])
object_loss = F.mse_loss(predicted_object, object_pos[:, 1:])
loss = base["loss"] + cfg.loss.proprio_weight * proprio_loss + cfg.loss.object_weight * object_loss
```

Denormalize predicted proprioception for raw qpos/qvel/TCP MSE. Return `proprio_loss`, `object_loss`, `qpos_mse`, `qvel_mse`, `tcp_mse`, `object_xyz_mse`, and `object_z_mse` from `WorldModel.forward`.

- [ ] **Step 6: Prove disabled behavior and strict legacy loading**

```python
def test_disabled_proprio_model_has_no_proprio_parameters_and_loads_strictly():
    legacy = WorldModel(WorldModelConfig())
    clone = WorldModel(WorldModelConfig())
    clone.load_state_dict(legacy.state_dict(), strict=True)
    assert not any(name.startswith("proprio") for name, _ in clone.named_parameters())
```

- [ ] **Step 7: Run focused tests and commit the model slice**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest tests/test_shapes.py tests/test_temporal_action_conditioning.py -q
git diff --check
git add models/proprio.py models/predictor.py models/world_model.py models/__init__.py losses/lewm.py tests/test_shapes.py tests/test_temporal_action_conditioning.py
git commit -m "feat: condition LiftCube dynamics on proprio"
```

### Task 4: Add Autoregressive Proprioception Rollout

**Files:**
- Modify: `planning/latent_cem.py:62-101`
- Modify: `tests/test_latent_cem.py:17-215`

**Interfaces:**
- Consumes: model transition prediction, initial scene/proprio/action histories, candidate chunks.
- Produces: `AuxiliaryRollout(scene_latent, proprio_normalized, proprio, object_pos)` from `rollout_with_aux(model, latent_history: Tensor, proprio_history_normalized: Tensor, context_actions: Tensor, action_candidates: Tensor, env: str) -> AuxiliaryRollout` while keeping `rollout_from_latent(model, latent_history: Tensor, context_actions: Tensor, action_candidates: Tensor, env: str) -> Tensor` unchanged. The normalized field is recurrence-only; raw `proprio` is planner/metric-only.

- [ ] **Step 1: Write the failing no-future-proprioception test**

```python
def test_rollout_second_step_uses_first_predicted_proprio():
    model = RecordingAuxModel(next_values=[torch.full((1, 25), 7.0), torch.full((1, 25), 9.0)])
    initial = torch.zeros(1, 3, 25)
    result = rollout_with_aux(model, latent_history(), initial, context_actions(), candidates(), "maniskill2")
    torch.testing.assert_close(model.proprio_inputs[1][:, -1], torch.full((1, 25), 7.0))
    torch.testing.assert_close(result.proprio[:, 1], torch.full((1, 25), 9.0))
```

- [ ] **Step 2: Run the test and verify RED**

Expected: import fails because `rollout_with_aux` does not exist.

- [ ] **Step 3: Implement explicit parallel-history recurrence**

```python
@dataclass
class AuxiliaryRollout:
    scene_latent: torch.Tensor
    proprio_normalized: torch.Tensor
    proprio: torch.Tensor
    object_pos: torch.Tensor

def rollout_with_aux(model, latent_history, proprio_history_normalized, context_actions, action_candidates, env):
    scene_predictions, normalized_predictions, raw_predictions, object_predictions = [], [], [], []
    proprio_history = proprio_history_normalized
    for step in range(action_candidates.shape[1]):
        transition = model._predict_sequence_with_aux(history, proprio_history, model_actions, env)
        next_scene = transition.scene_latent[:, -1]
        next_proprio_normalized = transition.next_proprio_normalized[:, -1]
        next_proprio = transition.next_proprio[:, -1]
        next_object = transition.object_pos[:, -1]
        scene_predictions.append(next_scene)
        normalized_predictions.append(next_proprio_normalized)
        raw_predictions.append(next_proprio)
        object_predictions.append(next_object)
        history = torch.cat([history[:, 1:], next_scene[:, None]], dim=1)
        proprio_history = torch.cat([proprio_history[:, 1:], next_proprio_normalized[:, None]], dim=1)
        action_history = model_actions[:, chunk_size:]
    return AuxiliaryRollout(
        torch.stack(scene_predictions, 1),
        torch.stack(normalized_predictions, 1),
        torch.stack(raw_predictions, 1),
        torch.stack(object_predictions, 1),
    )
```

- [ ] **Step 4: Run new and legacy rollout tests and commit**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest tests/test_latent_cem.py -q
git diff --check
git add planning/latent_cem.py tests/test_latent_cem.py
git commit -m "feat: recurse predicted robot state in rollout"
```

### Task 5: Implement GoalSpec and Lift-Hold-Stability Cost

**Files:**
- Create: `planning/liftcube_goal.py`
- Modify: `planning/__init__.py`
- Modify: `planning/latent_cem.py:103-215`
- Modify: `tests/test_latent_cem.py`

**Interfaces:**
- Consumes: `AuxiliaryRollout`, candidate actions, model chunk size, and optional demonstrated latent.
- Produces: `GoalSpec`, `LiftCubeCostWeights`, `LiftCubeCost(total, components)`, and `liftcube_trajectory_cost(rollout: AuxiliaryRollout, action_candidates: Tensor, goal: GoalSpec, *, chunk_size: int) -> LiftCubeCost`.

- [ ] **Step 1: Write failing threshold, hold, stability, and invalid-input tests**

```python
def test_below_threshold_has_height_penalty():
    cost = score_heights([0.10, 0.15], threshold=0.22)
    assert cost.components["height"].item() > 0

def test_one_peak_then_fall_fails_hold_and_costs_more_than_stable_hold():
    transient = score_heights([0.24, 0.10], threshold=0.22)
    stable = score_heights([0.23, 0.23], threshold=0.22)
    assert transient.components["hold"].item() > 0
    assert transient.total.item() > stable.total.item()

def test_two_chunk_boundaries_encode_ten_control_step_hold():
    goal = GoalSpec(0.02, 0.22, hold_control_steps=10, height_tolerance=0.01)
    result = liftcube_trajectory_cost(aux_rollout([0.23, 0.23]), actions(), goal, chunk_size=10)
    assert result.components["hold"].item() == 0

def test_nan_and_empty_rollout_are_rejected():
    with pytest.raises(ValueError, match="horizon"):
        liftcube_trajectory_cost(aux_rollout([]), actions(horizon=0), goal(), chunk_size=10)
    assert torch.isinf(score_heights([0.23, float("nan")]).total).all()
```

- [ ] **Step 2: Run tests and verify RED**

Expected: `planning.liftcube_goal` import fails.

- [ ] **Step 3: Implement validated dataclasses and exact cost components**

```python
@dataclass(frozen=True)
class LiftCubeCostWeights:
    height: float = 1.0
    hold: float = 1.0
    stability: float = 0.5
    latent: float = 0.05
    approach: float = 0.1
    early_close: float = 0.1

@dataclass(frozen=True)
class GoalSpec:
    initial_cube_height: float
    lift_threshold: float
    hold_control_steps: int = 10
    height_tolerance: float = 0.01
    approach_radius: float = 0.04
    demonstrated_goal_latent: torch.Tensor | None = None
    weights: LiftCubeCostWeights = field(default_factory=LiftCubeCostWeights)
```

Implement:

```python
required_nodes = math.ceil(goal.hold_control_steps / chunk_size) + 1
tail = height[:, -required_nodes:]
height_cost = torch.relu(goal.lift_threshold - height[:, -1]).square()
hold_cost = torch.relu(goal.lift_threshold - tail).square().mean(dim=1)
delta = tail[:, 1:] - tail[:, :-1]
stability_cost = (
    torch.relu(delta.abs() - goal.height_tolerance).square().mean(dim=1)
    + torch.relu(-delta).square().mean(dim=1)
)
```

Use denormalized raw `proprio[...,18:21]` as TCP xyz, predicted raw `object_pos` for distance, `(1 - gripper)/2` as close amount, and mask approach shaping to predicted pre-lift nodes. Never use `proprio_normalized` for geometric cost. Any non-finite prediction receives `inf` total cost. Serialize every weight and component.

- [ ] **Step 4: Route new CEM planning through goal-aware scoring**

Add an explicit enabled planner call accepting `proprio_history` and `goal_spec`; keep the old `goal_latent` signature for disabled models. Split candidates at `model_batch_size=32`, concatenate all component tensors, and select elites from total cost without reducing population 128.

- [ ] **Step 5: Run focused planner tests and commit**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest tests/test_latent_cem.py -q
git diff --check
git add planning/liftcube_goal.py planning/__init__.py planning/latent_cem.py tests/test_latent_cem.py
git commit -m "feat: plan for LiftCube lift and hold"
```

### Task 6: Integrate Training Metrics and Online Evaluator State

**Files:**
- Modify: `scripts/train_maniskill_group1.py:115-299`
- Create: `scripts/run_liftcube_sonata_proprio_goal_2500.sh`
- Modify: `scripts/eval_liftcube_latent_cem.py:45-707`
- Modify: `tests/test_maniskill_group1.py`
- Modify: `tests/test_liftcube_latent_cem_eval.py`

**Interfaces:**
- Consumes: enabled mapping batches, `GoalSpec`, online observation dictionaries, and goal-aware planner.
- Produces: checkpointed normalization/config, predicted cost traces, and unchanged physical success records.

- [ ] **Step 1: Write failing online extraction and evaluator-boundary tests**

```python
def test_extract_online_proprio_uses_manifest_layout():
    obs = {"agent": {"qpos": np.arange(9), "qvel": np.arange(9) + 10},
           "extra": {"tcp_pose": np.arange(7) + 20}}
    value = extract_proprio(obs)
    np.testing.assert_array_equal(value[:9], np.arange(9))
    np.testing.assert_array_equal(value[9:18], np.arange(9) + 10)
    np.testing.assert_array_equal(value[18:], np.arange(7) + 20)

def test_physical_success_is_not_replaced_by_predicted_success():
    result = run_episode(fake_env(height_trace=[0.02] * 20), aux_model(predicted_height=0.30), planner(), max_steps=20, env_seed=1, cem_seed=2)
    assert result["planner_predicted_success"] is True
    assert result["success"] is False
    assert result["success_by_peak_and_hold"] is False
```

- [ ] **Step 2: Run evaluator/training tests and verify RED**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest tests/test_maniskill_group1.py tests/test_liftcube_latent_cem_eval.py -q
```

Expected: new extraction and auxiliary logging interfaces are absent.

- [ ] **Step 3: Complete training batch, metric, config, and checkpoint plumbing**

Pass `proprio` and `object_pos` through `_forward`, `run_preflight`, `_evaluate`, and `train`. Extend totals and records with every auxiliary metric. Store `proprio_mean`, `proprio_std`, copied data manifest, model config, and unique run config in each checkpoint; continue atomic checkpoint replacement.

- [ ] **Step 4: Replace enabled evaluator goal-cloud dependence with GoalSpec**

At reset, capture initial scene and proprioception together. Build:

```python
goal_spec = GoalSpec(
    initial_cube_height=cube_z0,
    lift_threshold=resolved_lift_threshold,
    hold_control_steps=hold_steps,
    height_tolerance=height_tolerance,
    weights=weights_from_cli(args),
)
```

Warm up aligned point/proprio histories at steps 0, 10, and 20. Replan with structured rollouts. Record total and component costs plus predicted height/object/TCP traces. Continue to compute real `height_trace`, peak, hold, stability, environment success, and physical `success` solely from simulator observations.

- [ ] **Step 5: Add runner safety gates**

The shell runner uses `/nix/lewm/4d-jepa-worldmodel/bin/python`, refuses a non-empty run directory, records `nvidia-smi`, exact Python/Torch/CUDA versions, branch/HEAD/status, config hash, command, PID, and log path. It never calls `--resume` and does not choose a GPU that has an external process.

- [ ] **Step 6: Run focused tests and commit integration**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest tests/test_maniskill_group1.py tests/test_liftcube_latent_cem_eval.py -q
git diff --check
git add scripts/train_maniskill_group1.py scripts/eval_liftcube_latent_cem.py scripts/run_liftcube_sonata_proprio_goal_2500.sh tests/test_maniskill_group1.py tests/test_liftcube_latent_cem_eval.py
git commit -m "feat: integrate proprio training and CEM evaluation"
```

### Task 7: Prepare Real Data and Run Complete Verification

**Files:**
- Modify only if a failing regression exposes a task-scoped bug.
- Write persistent outputs under the new dataset/run paths, never into Git.

**Interfaces:**
- Consumes: all committed implementation slices.
- Produces: verified 100-episode dataset, clean regression evidence, and two finite smokes.

- [ ] **Step 1: Verify exact runtime robot/control identity in the MPC environment**

```bash
/nix/lewm/4d-jepa-mpc/bin/python -c "from scripts.eval_liftcube_latent_cem import make_env; e=make_env(); print(e.unwrapped.agent.__class__.__name__, e.unwrapped.control_mode); e.close()"
```

Expected: Panda agent identity and `pd_ee_delta_pose`. Record the exact attribute/value in the dataset manifest provenance.

- [ ] **Step 2: Prepare and atomically publish the new 100-episode dataset**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python scripts/prepare_liftcube_proprio_compact.py \
  --replay-h5 /workspace/datasets/maniskill2/v0.5.3/replayed/LiftCube-v0/count_0100/trajectory.pointcloud_robot_seg.pd_ee_delta_pose.h5 \
  --replay-json /workspace/datasets/maniskill2/v0.5.3/replayed/LiftCube-v0/count_0100/trajectory.pointcloud_robot_seg.pd_ee_delta_pose.json \
  --source-compact /workspace/datasets/maniskill2/v0.5.3/compact_rgb/LiftCube-v0 \
  --output-root /workspace/datasets/maniskill2/v0.5.3/compact_rgb_proprio/LiftCube-v0 \
  --verified-robot-uid panda
```

Expected: 100 episodes published; exact ordered 90/10 split retained; all episode records verified; no `.tmp` files.

- [ ] **Step 3: Reopen the complete dataset in verification-only mode**

Run the same command with `--verify-only`. Expected: all 100 files, manifest, statistics, provenance, shapes, and finite checks pass without changing hashes.

- [ ] **Step 4: Run dedicated and complete tests**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest \
  tests/test_prepare_liftcube_proprio_compact.py \
  tests/test_maniskill_dataset.py \
  tests/test_shapes.py \
  tests/test_temporal_action_conditioning.py \
  tests/test_latent_cem.py \
  tests/test_liftcube_latent_cem_eval.py -q
/nix/lewm/4d-jepa-worldmodel/bin/python -m pytest -q
```

Expected: all focused and full tests pass. Existing dependency warnings are recorded, not misreported as new failures.

- [ ] **Step 5: Run syntax and Git integrity checks**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python -m compileall -q data losses models planning scripts tests
git diff --check
git status --short --branch
```

Remove generated `__pycache__` after compilation/testing and confirm tracked files remain clean.

- [ ] **Step 6: Run one real world-model forward/loss/backward smoke**

```bash
/nix/lewm/4d-jepa-worldmodel/bin/python scripts/train_maniskill_group1.py \
  --config configs/maniskill_group1_sonata_proprio_goal.yaml \
  --preflight
```

Expected: points `[8,4,512,6]`, actions `[8,30,7]`, proprio `[8,4,25]`, finite total/latent/rank/proprio/object losses, non-zero finite action and proprioception gradients, and bounded CUDA memory.

- [ ] **Step 7: Run one renderer/proprioception/CEM-cost smoke**

Use the MPC environment with one seed, `candidates<=8`, `iterations=1`, `elites<=2`, `horizon=2`, and `max_steps<=30`. Expected: renderer observation, proprioception `[25]`, two-node autoregressive prediction, finite component costs, and real evaluator fields. Do not interpret physical failure as a code failure if the smoke remains finite and correctly recorded.

- [ ] **Step 8: Commit any verification-only support and record final clean state**

If no task-scoped fixes were needed, create no empty commit. Otherwise rerun the relevant RED/GREEN test before committing. Record branch, HEAD, log, dataset manifest hash, test totals, smoke JSON, and unverified assumptions.

### Task 8: Launch and Monitor One Bounded Training Run

**Files:**
- Do not modify source files after the verified launch unless a failing test first reproduces a task-scoped defect.
- Persist run config, logs, metrics, checkpoints, and preflight output under `/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_sonata_proprio_goal_batch8_2500`.

**Interfaces:**
- Consumes: verified new dataset and config.
- Produces: at most 2,500 training steps plus validation/checkpoints; no CEM success claim.

- [ ] **Step 1: Re-query GPU ownership and repository state immediately before launch**

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
git status --short --branch
git rev-parse HEAD
```

Select only a genuinely idle GPU with no external compute process. If ownership is ambiguous, do not launch.

- [ ] **Step 2: Launch the unique no-resume run and persist process evidence**

```bash
CUDA_VISIBLE_DEVICES=$LIFTCUBE_GPU \
  /nix/lewm/4d-jepa-worldmodel/bin/python scripts/train_maniskill_group1.py \
  --config configs/maniskill_group1_sonata_proprio_goal.yaml \
  --run-dir /workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_sonata_proprio_goal_batch8_2500
```

Set `LIFTCUBE_GPU` in the same shell to the concrete idle index returned and verified in Step 1, then launch through the checked runner so PID, command, environment, and combined log are saved. The runner rejects an unset value or a GPU with an existing compute process.

- [ ] **Step 3: Confirm stable early progress**

Verify the process remains alive, GPU ownership matches the saved PID, step 1 and the first configured log interval are finite, auxiliary losses are present, and checkpoint/config provenance is correct. Stop the run on NaN, non-finite gradients, unexpected external GPU contention, or repeated runtime failure.

- [ ] **Step 4: Monitor terminal completion without token-heavy polling**

After stable progress, exit the interactive SSH session. Reconnect at bounded intervals to read the latest persisted metrics/log and process status. Stop monitoring when the process exits or step 2,500 is recorded.

- [ ] **Step 5: Report training evidence without overclaiming**

Report final/best step, train/validation total and component losses, qpos/qvel/TCP/object xyz/z metrics, checkpoint paths and hashes, GPU/time/memory evidence, failures/restarts, and remaining assumptions. State explicitly that no formal real CEM rollout has been run and control success remains unverified.

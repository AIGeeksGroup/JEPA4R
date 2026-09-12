# LiftCube Multi-step Offline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a deterministic LiftCube validation-only evaluator that measures H=1/2/4/8 latent rollout accuracy against persistence, zero-action, and shuffled-action baselines for the existing best.pt checkpoint.

**Architecture:** Add one evaluation-specific Dataset and CLI under scripts/eval_maniskill_group1_rollout.py, leaving the WorldModel, training path, and existing one-step evaluator unchanged. The evaluator uses the manifest validation split, constructs one common H_max-eligible cohort, batches correct/zero/shuffled action rollouts through the frozen model, and writes provenance-rich JSON atomically outside the source tree.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu121, NumPy 1.26.4, OmegaConf 2.3.1, pytest 8.3.5, existing WorldModel and compact ManiSkill2 NPZ data.

## Global Constraints

- Evaluate the existing LiftCube best.pt only; do not modify model parameters and do not train.
- Read only manifest.json val_episode_ids; no training episode may enter the evaluation cohort.
- Default horizons are exactly H=1/2/4/8 and all horizons use the same H_max-eligible window cohort.
- Use checkpoint T and K, currently T=3 and K=512, and require seven-dimensional actions.
- Do not install, import, or start ManiSkill2/SAPIEN; do not modify the verified WM Python environment or dependency files.
- Do not implement MPC, CEM, a reward model, a success head, simulator control, or cross-task evaluation.
- Preserve scripts/eval_maniskill_group1.py and offline_metrics.json unchanged.
- Write rollout_metrics.json atomically and never overwrite checkpoints, training logs, or offline_metrics.json.

---

## File Map

- Create: scripts/eval_maniskill_group1_rollout.py
  - Owns validation-only multi-step windows, action controls, metric accumulation, checkpoint loading, provenance, CLI, and atomic JSON output.
- Create: tests/test_maniskill_rollout_eval.py
  - Owns small synthetic-NPZ tests for alignment, common cohorts, deterministic action permutation, metrics, and CLI output.
- Read only: data/maniskill.py
  - Reuse _validate_episode and _sample_points so evaluation preprocessing matches training.
- Read only: scripts/train_maniskill_group1.py
  - Reuse validate_group1_manifest so the exact LiftCube 90/10 boundary is enforced.
- Read only: models/world_model.py
  - Use WorldModel.rollout and _aggregate without changing model behavior.
- Runtime output only: /workspace/users/lewm/runs/4d-jepa-architecture/group1_scratch_liftcube/rollout_metrics.json

---

### Task 1: Validation-only rollout windows and deterministic action controls

**Files:**
- Create: scripts/eval_maniskill_group1_rollout.py
- Create: tests/test_maniskill_rollout_eval.py

**Interfaces:**
- Consumes: validate_group1_manifest(root: Path) -> dict, _validate_episode(episode: dict[str, np.ndarray], path: Path) -> None, and _sample_points(xyz, point_count, episode_id, frame_id) -> np.ndarray.
- Produces: WindowRef, LiftCubeRolloutDataset, LiftCubeRolloutDataset.actions_for_index(index), and build_action_shuffle_indices(count, seed).

- [ ] **Step 1: Write failing dataset-alignment and action-permutation tests**

Create tests/test_maniskill_rollout_eval.py with:

    import json
    from pathlib import Path

    import numpy as np
    import pytest
    import torch

    from scripts.eval_maniskill_group1_rollout import (
        LiftCubeRolloutDataset,
        build_action_shuffle_indices,
    )


    def _write_eval_root(root: Path, *, frames: int = 8, raw_points: int = 4) -> Path:
        root.mkdir(parents=True)
        train_ids = list(range(90))
        val_ids = list(range(90, 100))
        manifest = {
            "task": "LiftCube-v0",
            "episode_ids": train_ids + val_ids,
            "train_episode_ids": train_ids,
            "val_episode_ids": val_ids,
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for episode_id in val_ids:
            frame_values = np.arange(frames, dtype=np.float32) + episode_id * 100
            xyz = np.broadcast_to(
                frame_values[:, None, None],
                (frames, raw_points, 3),
            ).copy()
            action = np.stack(
                [
                    np.arange(frames - 1, dtype=np.float32) + episode_id,
                    np.zeros(frames - 1, dtype=np.float32),
                    np.ones(frames - 1, dtype=np.float32),
                    np.full(frames - 1, 2.0, dtype=np.float32),
                    np.full(frames - 1, 3.0, dtype=np.float32),
                    np.full(frames - 1, 4.0, dtype=np.float32),
                    np.full(frames - 1, 5.0, dtype=np.float32),
                ],
                axis=1,
            )
            np.savez(root / f"episode_{episode_id:04d}.npz", xyz=xyz, action=action)
        return root


    def test_rollout_dataset_aligns_context_actions_and_targets(tmp_path):
        root = _write_eval_root(tmp_path / "LiftCube-v0")
        dataset = LiftCubeRolloutDataset(root, T=2, point_count=4, horizon=3)

        index, context, future, actions = dataset[0]

        assert index == 0
        assert len(dataset) == 10 * (8 - 2 - 3 + 1)
        assert dataset.episode_ids == list(range(90, 100))
        assert context.shape == (2, 4, 3)
        assert future.shape == (3, 4, 3)
        assert actions.shape == (3, 7)
        assert torch.all(context[0] == 9000)
        assert torch.all(context[1] == 9001)
        assert torch.all(future[0] == 9002)
        assert actions[0, 0].item() == pytest.approx(91.0)
        assert torch.equal(actions, dataset.actions_for_index(0))


    def test_action_shuffle_is_seeded_permutation_without_fixed_points():
        first = build_action_shuffle_indices(11, seed=3072)
        second = build_action_shuffle_indices(11, seed=3072)

        assert np.array_equal(first, second)
        assert sorted(first.tolist()) == list(range(11))
        assert np.all(first != np.arange(11))


    def test_action_shuffle_rejects_single_window():
        with pytest.raises(ValueError, match="at least two"):
            build_action_shuffle_indices(1, seed=3072)

- [ ] **Step 2: Run the focused tests and verify the missing-module failure**

Run:

    cd /workspace/users/lewm/projects/4d-jepa-architecture
    PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4r-jepa-ms2/env/bin/python -m pytest +      -p no:cacheprovider tests/test_maniskill_rollout_eval.py -q

Expected: collection fails with ModuleNotFoundError for scripts.eval_maniskill_group1_rollout.

- [ ] **Step 3: Implement the rollout-window dataset and deterministic permutation**

Create scripts/eval_maniskill_group1_rollout.py with:

    """Validation-only multi-step latent rollout evaluation for LiftCube Group 1."""

    from __future__ import annotations

    from collections import OrderedDict
    from dataclasses import dataclass
    from pathlib import Path

    import numpy as np
    import torch
    from torch.utils.data import Dataset

    from data.maniskill import _sample_points, _validate_episode
    from scripts.train_maniskill_group1 import validate_group1_manifest


    ACTION_DIM = 7
    DEFAULT_HORIZONS = (1, 2, 4, 8)


    @dataclass(frozen=True)
    class WindowRef:
        episode_id: int
        start: int


    class LiftCubeRolloutDataset(Dataset):
        """Common H_max-eligible windows from Group 1 validation episodes."""

        def __init__(
            self,
            root: str | Path,
            *,
            T: int,
            point_count: int,
            horizon: int,
            cache_size: int = 2,
        ):
            if T < 1 or point_count < 1 or horizon < 1:
                raise ValueError("T, point_count, and horizon must be positive")
            self.root = Path(root)
            self.T = int(T)
            self.point_count = int(point_count)
            self.horizon = int(horizon)
            self.cache_size = max(1, int(cache_size))
            manifest = validate_group1_manifest(self.root)
            self.episode_ids = [int(x) for x in manifest["val_episode_ids"]]
            self.index: list[WindowRef] = []
            self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

            for episode_id in self.episode_ids:
                path = self.root / f"episode_{episode_id:04d}.npz"
                episode = self._read_episode(path)
                frames = int(episode["xyz"].shape[0])
                if episode["xyz"].shape[1] < self.point_count:
                    raise ValueError(
                        f"{path}: has {episode['xyz'].shape[1]} points; "
                        f"{self.point_count} required"
                    )
                window_count = frames - self.T - self.horizon + 1
                self.index.extend(
                    WindowRef(episode_id, start)
                    for start in range(max(0, window_count))
                )
            if not self.index:
                raise ValueError(
                    f"no complete T={self.T}, H={self.horizon} validation windows "
                    f"under {self.root}"
                )

        @staticmethod
        def _read_episode(path: Path) -> dict[str, np.ndarray]:
            if not path.is_file():
                raise FileNotFoundError(f"missing validation episode: {path}")
            with np.load(path, allow_pickle=False) as archive:
                episode = {key: np.asarray(archive[key]) for key in archive.files}
            _validate_episode(episode, path)
            return episode

        def _load_episode(self, episode_id: int) -> dict[str, np.ndarray]:
            if episode_id in self._cache:
                self._cache.move_to_end(episode_id)
                return self._cache[episode_id]
            episode = self._read_episode(self.root / f"episode_{episode_id:04d}.npz")
            self._cache[episode_id] = episode
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return episode

        def __len__(self) -> int:
            return len(self.index)

        def actions_for_index(self, index: int) -> torch.Tensor:
            ref = self.index[index]
            episode = self._load_episode(ref.episode_id)
            first_action = ref.start + self.T - 1
            actions = np.ascontiguousarray(
                episode["action"][first_action : first_action + self.horizon],
                dtype=np.float32,
            )
            if actions.shape != (self.horizon, ACTION_DIM):
                raise ValueError(
                    f"episode {ref.episode_id} start {ref.start}: expected actions "
                    f"[{self.horizon},{ACTION_DIM}], got {actions.shape}"
                )
            return torch.from_numpy(actions)

        def __getitem__(
            self, index: int
        ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
            ref = self.index[index]
            episode = self._load_episode(ref.episode_id)
            frame_ids = range(ref.start, ref.start + self.T + self.horizon)
            sampled = np.stack(
                [
                    _sample_points(
                        episode["xyz"][frame_id],
                        self.point_count,
                        ref.episode_id,
                        frame_id,
                    )
                    for frame_id in frame_ids
                ]
            )
            context = torch.from_numpy(sampled[: self.T])
            future = torch.from_numpy(sampled[self.T :])
            return index, context, future, self.actions_for_index(index)


    def build_action_shuffle_indices(count: int, seed: int) -> np.ndarray:
        """Return a deterministic cyclic derangement of range(count)."""
        if count < 2:
            raise ValueError("action shuffling requires at least two windows")
        rng = np.random.default_rng(seed)
        offset = int(rng.integers(1, count))
        return (np.arange(count, dtype=np.int64) + offset) % count

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

    cd /workspace/users/lewm/projects/4d-jepa-architecture
    PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4r-jepa-ms2/env/bin/python -m pytest +      -p no:cacheprovider tests/test_maniskill_rollout_eval.py -q

Expected: 3 passed.

- [ ] **Step 5: Commit the dataset slice**

Run:

    git add scripts/eval_maniskill_group1_rollout.py tests/test_maniskill_rollout_eval.py
    git commit -m "feat: add LiftCube rollout evaluation windows"

Expected: one commit containing only the new evaluator and its focused tests.

---

### Task 2: Per-horizon metrics and action-sensitivity evaluation

**Files:**
- Modify: scripts/eval_maniskill_group1_rollout.py
- Modify: tests/test_maniskill_rollout_eval.py

**Interfaces:**
- Consumes: LiftCubeRolloutDataset and WorldModel-compatible _aggregate(points) and rollout(points, actions, env) methods.
- Produces: ErrorTotals.update, ErrorTotals.result, SquaredErrorTotals.update, and evaluate_rollouts(model, dataset, batch_size, device, horizons, seed).

- [ ] **Step 1: Add failing metric and fake-model evaluation tests**

Append to tests/test_maniskill_rollout_eval.py:

    from types import SimpleNamespace

    from scripts.eval_maniskill_group1_rollout import (
        ErrorTotals,
        evaluate_rollouts,
    )


    def test_error_totals_weights_elements_and_feature_vectors():
        totals = ErrorTotals()
        prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

        totals.update(prediction, target)
        result = totals.result()

        assert result["mse"] == pytest.approx(0.5)
        assert result["cosine_distance"] == pytest.approx(0.5)


    class _TinyRolloutDataset(torch.utils.data.Dataset):
        horizon = 2
        episode_ids = [90, 91]

        def __init__(self):
            self._actions = torch.tensor(
                [
                    [[1.0, 0, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0, 0]],
                    [[2.0, 0, 0, 0, 0, 0, 0], [4.0, 0, 0, 0, 0, 0, 0]],
                ]
            )

        def __len__(self):
            return 2

        def actions_for_index(self, index):
            return self._actions[index]

        def __getitem__(self, index):
            actions = self._actions[index]
            context = torch.zeros(1, 1, 3)
            values = actions[:, 0].cumsum(0)
            future = values[:, None, None].expand(2, 1, 3).clone()
            return index, context, future, actions


    class _ActionDrivenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = SimpleNamespace(T=1, K=1, D=1)

        def _aggregate(self, points):
            return points[..., :1]

        def rollout(self, points, actions, env=None):
            start = points[:, -1:, :, :1]
            increments = actions[..., 0].cumsum(1)[:, :, None, None]
            return start + increments


    def test_evaluate_rollouts_reports_horizons_and_action_controls():
        result = evaluate_rollouts(
            _ActionDrivenModel(),
            _TinyRolloutDataset(),
            batch_size=2,
            device=torch.device("cpu"),
            horizons=(1, 2),
            seed=3072,
        )

        h2 = result["per_horizon"]["2"]
        assert result["window_count"] == 2
        assert h2["model_correct_action"]["mse"] == pytest.approx(0.0)
        assert h2["model_zero_action"]["mse"] > 0
        assert h2["model_shuffled_action"]["mse"] > 0
        assert h2["gain_vs_persistence"] > 1
        assert h2["gain_vs_shuffled_action"] > 1
        assert h2["prediction_delta_shuffled_mse"] > 0

- [ ] **Step 2: Run the new tests and verify the missing-interface failures**

Run:

    cd /workspace/users/lewm/projects/4d-jepa-architecture
    PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4r-jepa-ms2/env/bin/python -m pytest +      -p no:cacheprovider +      tests/test_maniskill_rollout_eval.py::test_error_totals_weights_elements_and_feature_vectors +      tests/test_maniskill_rollout_eval.py::test_evaluate_rollouts_reports_horizons_and_action_controls +      -q

Expected: collection fails because ErrorTotals and evaluate_rollouts are not defined.

- [ ] **Step 3: Implement weighted metrics and batched evaluation**

Add these imports to scripts/eval_maniskill_group1_rollout.py:

    from dataclasses import dataclass

    import torch.nn.functional as F
    from torch.utils.data import DataLoader

Keep the existing dataclass import as a single line; do not duplicate it. Then append:

    @dataclass
    class ErrorTotals:
        squared_error: float = 0.0
        element_count: int = 0
        cosine_distance: float = 0.0
        vector_count: int = 0

        def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
            if prediction.shape != target.shape:
                raise ValueError(
                    f"prediction and target shapes differ: "
                    f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
                )
            if prediction.shape[-1] < 1:
                raise ValueError("latent feature dimension must be positive")
            delta = prediction - target
            self.squared_error += float(delta.square().sum().detach().cpu())
            self.element_count += int(delta.numel())
            cosine = 1.0 - F.cosine_similarity(
                prediction.float(), target.float(), dim=-1, eps=1e-8
            )
            self.cosine_distance += float(cosine.sum().detach().cpu())
            self.vector_count += int(cosine.numel())

        def result(self) -> dict[str, float]:
            if self.element_count == 0 or self.vector_count == 0:
                raise RuntimeError("cannot finalize empty error totals")
            return {
                "mse": self.squared_error / self.element_count,
                "cosine_distance": self.cosine_distance / self.vector_count,
            }


    @dataclass
    class SquaredErrorTotals:
        squared_error: float = 0.0
        element_count: int = 0

        def update(self, left: torch.Tensor, right: torch.Tensor) -> None:
            if left.shape != right.shape:
                raise ValueError(
                    f"delta shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}"
                )
            delta = left - right
            self.squared_error += float(delta.square().sum().detach().cpu())
            self.element_count += int(delta.numel())

        def result(self) -> float:
            if self.element_count == 0:
                raise RuntimeError("cannot finalize empty squared-error totals")
            return self.squared_error / self.element_count


    def evaluate_rollouts(
        model: torch.nn.Module,
        dataset: LiftCubeRolloutDataset,
        *,
        batch_size: int,
        device: torch.device,
        horizons: tuple[int, ...],
        seed: int,
    ) -> dict[str, object]:
        ordered_horizons = tuple(sorted(set(int(x) for x in horizons)))
        if not ordered_horizons or ordered_horizons[0] < 1:
            raise ValueError("horizons must contain positive integers")
        if ordered_horizons[-1] != dataset.horizon:
            raise ValueError(
                f"dataset horizon {dataset.horizon} must equal max horizon "
                f"{ordered_horizons[-1]}"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        shuffle_indices = build_action_shuffle_indices(len(dataset), seed)
        action_bank = torch.stack(
            [dataset.actions_for_index(index) for index in range(len(dataset))]
        )
        methods = (
            "model_correct_action",
            "persistence",
            "model_zero_action",
            "model_shuffled_action",
        )
        totals = {
            horizon: {method: ErrorTotals() for method in methods}
            for horizon in ordered_horizons
        }
        deltas = {
            horizon: {
                "prediction_delta_zero_mse": SquaredErrorTotals(),
                "prediction_delta_shuffled_mse": SquaredErrorTotals(),
            }
            for horizon in ordered_horizons
        }

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        model.eval()
        with torch.no_grad():
            for indices, context, future, correct_actions in loader:
                context = context.to(device)
                future = future.to(device)
                correct_actions = correct_actions.to(device)
                shuffled_actions = action_bank[
                    torch.from_numpy(shuffle_indices[indices.numpy()])
                ].to(device)
                zero_actions = torch.zeros_like(correct_actions)

                truth = model._aggregate(torch.cat([context, future], dim=1))
                target = truth[:, model.cfg.T :]
                persistence = truth[:, model.cfg.T - 1 : model.cfg.T].expand_as(target)
                correct = model.rollout(context, correct_actions, env="maniskill2")
                zero = model.rollout(context, zero_actions, env="maniskill2")
                shuffled = model.rollout(context, shuffled_actions, env="maniskill2")

                expected = (
                    context.shape[0],
                    dataset.horizon,
                    model.cfg.K,
                    model.cfg.D,
                )
                for name, value in (
                    ("target", target),
                    ("correct", correct),
                    ("zero", zero),
                    ("shuffled", shuffled),
                ):
                    if tuple(value.shape) != expected:
                        raise ValueError(
                            f"{name} has shape {tuple(value.shape)}, expected {expected}"
                        )
                    if not torch.isfinite(value).all().item():
                        raise RuntimeError(f"{name} contains NaN or Inf")

                predictions = {
                    "model_correct_action": correct,
                    "persistence": persistence,
                    "model_zero_action": zero,
                    "model_shuffled_action": shuffled,
                }
                for horizon in ordered_horizons:
                    step = horizon - 1
                    for method, prediction in predictions.items():
                        totals[horizon][method].update(
                            prediction[:, step], target[:, step]
                        )
                    deltas[horizon]["prediction_delta_zero_mse"].update(
                        correct[:, step], zero[:, step]
                    )
                    deltas[horizon]["prediction_delta_shuffled_mse"].update(
                        correct[:, step], shuffled[:, step]
                    )

        per_horizon: dict[str, object] = {}
        for horizon in ordered_horizons:
            row = {
                method: accumulator.result()
                for method, accumulator in totals[horizon].items()
            }
            correct_mse = row["model_correct_action"]["mse"]
            denominator = max(correct_mse, 1e-12)
            row.update(
                {
                    "gain_vs_persistence": row["persistence"]["mse"] / denominator,
                    "gain_vs_zero_action": row["model_zero_action"]["mse"] / denominator,
                    "gain_vs_shuffled_action": (
                        row["model_shuffled_action"]["mse"] / denominator
                    ),
                    "prediction_delta_zero_mse": deltas[horizon][
                        "prediction_delta_zero_mse"
                    ].result(),
                    "prediction_delta_shuffled_mse": deltas[horizon][
                        "prediction_delta_shuffled_mse"
                    ].result(),
                }
            )
            per_horizon[str(horizon)] = row
        return {
            "window_count": len(dataset),
            "per_horizon": per_horizon,
        }

- [ ] **Step 4: Run the focused tests and verify all five tests pass**

Run:

    cd /workspace/users/lewm/projects/4d-jepa-architecture
    PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4r-jepa-ms2/env/bin/python -m pytest +      -p no:cacheprovider tests/test_maniskill_rollout_eval.py -q

Expected: 5 passed.

- [ ] **Step 5: Commit the metric slice**

Run:

    git add scripts/eval_maniskill_group1_rollout.py tests/test_maniskill_rollout_eval.py
    git commit -m "feat: evaluate LiftCube latent rollout baselines"

Expected: one commit containing the metric accumulator, batched evaluator, and tests.

---

### Task 3: Checkpoint provenance, atomic report, and CLI

**Files:**
- Modify: scripts/eval_maniskill_group1_rollout.py
- Modify: tests/test_maniskill_rollout_eval.py

**Interfaces:**
- Consumes: evaluate_rollouts, WorldModel, WorldModelConfig, OmegaConf, and a Group 1 checkpoint dictionary.
- Produces: sha256_file, load_checkpoint_model, atomic_write_json, build_report, print_summary, and main(argv=None).

- [ ] **Step 1: Add failing provenance, atomic-output, and CLI smoke tests**

Append to tests/test_maniskill_rollout_eval.py:

    import hashlib

    import scripts.eval_maniskill_group1_rollout as rollout_eval


    def test_atomic_write_json_replaces_complete_document(tmp_path):
        output = tmp_path / "rollout_metrics.json"
        rollout_eval.atomic_write_json(output, {"value": 1})
        rollout_eval.atomic_write_json(output, {"value": 2})

        assert json.loads(output.read_text(encoding="utf-8")) == {"value": 2}
        assert not output.with_suffix(".json.tmp").exists()


    def test_main_writes_provenance_without_loading_simulator(tmp_path, monkeypatch):
        checkpoint = tmp_path / "best.pt"
        checkpoint.write_bytes(b"checkpoint")
        data_root = _write_eval_root(tmp_path / "LiftCube-v0")
        output = tmp_path / "rollout_metrics.json"
        fake_model = _ActionDrivenModel()
        state = {
            "step": 8500,
            "model_config": {"T": 1, "K": 1, "D": 1, "C": 3},
            "train_config": {"task": "LiftCube-v0"},
        }

        class _FakeDataset:
            episode_ids = list(range(90, 100))

            def __init__(self, root, *, T, point_count, horizon):
                self.root = Path(root)
                self.T = T
                self.point_count = point_count
                self.horizon = horizon

            def __len__(self):
                return 7

        monkeypatch.setattr(
            rollout_eval,
            "load_checkpoint_model",
            lambda path, device: (fake_model, state),
        )
        monkeypatch.setattr(rollout_eval, "LiftCubeRolloutDataset", _FakeDataset)
        monkeypatch.setattr(
            rollout_eval,
            "evaluate_rollouts",
            lambda *args, **kwargs: {
                "window_count": 7,
                "per_horizon": {
                    str(h): {
                        "model_correct_action": {"mse": 1.0, "cosine_distance": 0.1},
                        "persistence": {"mse": 2.0, "cosine_distance": 0.2},
                        "model_zero_action": {"mse": 3.0, "cosine_distance": 0.3},
                        "model_shuffled_action": {"mse": 4.0, "cosine_distance": 0.4},
                        "gain_vs_persistence": 2.0,
                        "gain_vs_zero_action": 3.0,
                        "gain_vs_shuffled_action": 4.0,
                        "prediction_delta_zero_mse": 0.5,
                        "prediction_delta_shuffled_mse": 0.6,
                    }
                    for h in (1, 2, 4, 8)
                },
            },
        )

        result = rollout_eval.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-root",
                str(data_root),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )

        saved = json.loads(output.read_text(encoding="utf-8"))
        expected_hash = hashlib.sha256(b"checkpoint").hexdigest()
        assert result == saved
        assert saved["checkpoint"]["sha256"] == expected_hash
        assert saved["checkpoint"]["step"] == 8500
        assert saved["data"]["episode_count"] == 10
        assert saved["evaluation"]["horizons"] == [1, 2, 4, 8]
        assert saved["evaluation"]["window_count"] == 7

- [ ] **Step 2: Run the CLI tests and verify the missing-function failures**

Run:

    cd /workspace/users/lewm/projects/4d-jepa-architecture
    PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4r-jepa-ms2/env/bin/python -m pytest +      -p no:cacheprovider +      tests/test_maniskill_rollout_eval.py::test_atomic_write_json_replaces_complete_document +      tests/test_maniskill_rollout_eval.py::test_main_writes_provenance_without_loading_simulator +      -q

Expected: failures because atomic_write_json and main are not yet defined.

- [ ] **Step 3: Implement checkpoint loading, provenance, summary, and CLI**

Add these imports to scripts/eval_maniskill_group1_rollout.py:

    import argparse
    import hashlib
    import json
    import os
    import time

    from omegaconf import OmegaConf

    from models import WorldModel, WorldModelConfig

Append:

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def load_checkpoint_model(
        checkpoint: Path, device: torch.device
    ) -> tuple[WorldModel, dict]:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        required = {"step", "model", "model_config", "train_config"}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"checkpoint is missing keys: {missing}")
        if state["train_config"].get("task") != "LiftCube-v0":
            raise ValueError(
                f"checkpoint task must be LiftCube-v0, got "
                f"{state['train_config'].get('task')!r}"
            )
        cfg = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(WorldModelConfig()),
                OmegaConf.create(state["model_config"]),
            )
        )
        if cfg.C != 3:
            raise ValueError(f"LiftCube evaluator requires C=3, got {cfg.C}")
        if cfg.action.envs["maniskill2"].dim != ACTION_DIM:
            raise ValueError(
                f"LiftCube evaluator requires action dim {ACTION_DIM}, got "
                f"{cfg.action.envs['maniskill2'].dim}"
            )
        model = WorldModel(cfg).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        return model, state


    def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


    def build_report(
        *,
        checkpoint: Path,
        data_root: Path,
        state: dict,
        dataset: LiftCubeRolloutDataset,
        metrics: dict[str, object],
        horizons: tuple[int, ...],
        batch_size: int,
        seed: int,
        elapsed_sec: float,
    ) -> dict[str, object]:
        manifest = data_root / "manifest.json"
        return {
            "schema_version": 1,
            "task": "LiftCube-v0",
            "split": "val",
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": sha256_file(checkpoint),
                "step": int(state["step"]),
            },
            "data": {
                "root": str(data_root.resolve()),
                "manifest_sha256": sha256_file(manifest),
                "episode_count": len(dataset.episode_ids),
            },
            "model_config": state["model_config"],
            "evaluation": {
                "T": int(dataset.T),
                "K": int(dataset.point_count),
                "horizons": list(horizons),
                "window_count": int(metrics["window_count"]),
                "batch_size": int(batch_size),
                "seed": int(seed),
                "elapsed_sec": float(elapsed_sec),
            },
            "per_horizon": metrics["per_horizon"],
        }


    def print_summary(report: dict[str, object]) -> None:
        print("H  model_mse  persist_mse  zero_mse  shuffled_mse  persist_gain")
        for horizon in report["evaluation"]["horizons"]:
            row = report["per_horizon"][str(horizon)]
            print(
                f"{horizon:<2} "
                f"{row['model_correct_action']['mse']:.6e} "
                f"{row['persistence']['mse']:.6e} "
                f"{row['model_zero_action']['mse']:.6e} "
                f"{row['model_shuffled_action']['mse']:.6e} "
                f"{row['gain_vs_persistence']:.4f}"
            )


    def main(argv: list[str] | None = None) -> dict[str, object]:
        parser = argparse.ArgumentParser(
            description="Evaluate LiftCube multi-step latent rollouts."
        )
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--data-root", type=Path, required=True)
        parser.add_argument("--output", type=Path, default=None)
        parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--seed", type=int, default=3072)
        parser.add_argument(
            "--device",
            default="cuda" if torch.cuda.is_available() else "cpu",
        )
        args = parser.parse_args(argv)

        horizons = tuple(sorted(set(args.horizons)))
        if horizons != DEFAULT_HORIZONS:
            raise ValueError(
                f"Group 1 rollout evaluation requires horizons "
                f"{DEFAULT_HORIZONS}, got {horizons}"
            )
        device = torch.device(args.device)
        model, state = load_checkpoint_model(args.checkpoint, device)
        dataset = LiftCubeRolloutDataset(
            args.data_root,
            T=model.cfg.T,
            point_count=model.cfg.K,
            horizon=max(horizons),
        )
        started = time.perf_counter()
        metrics = evaluate_rollouts(
            model,
            dataset,
            batch_size=args.batch_size,
            device=device,
            horizons=horizons,
            seed=args.seed,
        )
        elapsed_sec = time.perf_counter() - started
        report = build_report(
            checkpoint=args.checkpoint,
            data_root=args.data_root,
            state=state,
            dataset=dataset,
            metrics=metrics,
            horizons=horizons,
            batch_size=args.batch_size,
            seed=args.seed,
            elapsed_sec=elapsed_sec,
        )
        output = args.output or args.checkpoint.parent / "rollout_metrics.json"
        atomic_write_json(output, report)
        print_summary(report)
        print(f"wrote {output}")
        return report


    if __name__ == "__main__":
        main()

- [ ] **Step 4: Run the evaluator tests and then the full suite**

Run:

    cd /workspace/users/lewm/projects/4d-jepa-architecture
    PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4r-jepa-ms2/env/bin/python -m pytest +      -p no:cacheprovider tests/test_maniskill_rollout_eval.py -q

Expected: 7 passed.

Then run:

    PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4r-jepa-ms2/env/bin/python -m pytest +      -p no:cacheprovider -q

Expected: all existing 24 tests plus the 7 new tests pass; warnings may remain but no failures or errors.

- [ ] **Step 5: Check formatting and import-time behavior**

Run:

    /nix/lewm/4r-jepa-ms2/env/bin/python -m py_compile +      scripts/eval_maniskill_group1_rollout.py +      tests/test_maniskill_rollout_eval.py
    git diff --check

Expected: both commands exit 0 and git diff --check prints no output.

- [ ] **Step 6: Commit the CLI slice**

Run:

    git add scripts/eval_maniskill_group1_rollout.py tests/test_maniskill_rollout_eval.py
    git commit -m "feat: add LiftCube rollout evaluation CLI"

Expected: one commit containing the checkpoint loader, report writer, CLI, and tests.

---

### Task 4: Run the real best.pt evaluation and verify preservation guarantees

**Files:**
- Create at runtime only: /workspace/users/lewm/runs/4d-jepa-architecture/group1_scratch_liftcube/rollout_metrics.json
- Do not modify tracked source files.

**Interfaces:**
- Consumes: the committed CLI, best.pt, and compact/LiftCube-v0.
- Produces: a complete rollout_metrics.json diagnostic artifact for checkpoint step 8500.

- [ ] **Step 1: Record immutable input hashes before evaluation**

Run:

    sha256sum +      /workspace/users/lewm/runs/4d-jepa-architecture/group1_scratch_liftcube/best.pt +      /workspace/datasets/maniskill2/v0.5.3/compact/LiftCube-v0/manifest.json

Expected best.pt SHA-256:

    a9622210d351a48a6c9853e4945c2764173012c708b8367e796e8e2c9e1cce2

Save both printed hashes in the execution notes for comparison after the run.

- [ ] **Step 2: Run the full validation-cohort multi-step evaluation**

Run:

    cd /workspace/users/lewm/projects/4d-jepa-architecture
    PYTHONDONTWRITEBYTECODE=1 /nix/lewm/4r-jepa-ms2/env/bin/python +      scripts/eval_maniskill_group1_rollout.py +      --checkpoint /workspace/users/lewm/runs/4d-jepa-architecture/group1_scratch_liftcube/best.pt +      --data-root /workspace/datasets/maniskill2/v0.5.3/compact/LiftCube-v0 +      --output /workspace/users/lewm/runs/4d-jepa-architecture/group1_scratch_liftcube/rollout_metrics.json +      --device cuda:0 +      --batch-size 4

Expected: exit 0, one summary row for each of H=1, H=2, H=4, and H=8, followed by the output path. Do not abort merely because a gain is below 1; this task measures the checkpoint rather than tuning it.

- [ ] **Step 3: Validate report schema and finiteness**

Run:

    /nix/lewm/4r-jepa-ms2/env/bin/python - <<'PY'
    import json
    import math
    from pathlib import Path

    path = Path("/workspace/users/lewm/runs/4d-jepa-architecture/group1_scratch_liftcube/rollout_metrics.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["task"] == "LiftCube-v0"
    assert report["split"] == "val"
    assert report["checkpoint"]["step"] == 8500
    assert report["checkpoint"]["sha256"] == "a9622210d351a48a6c9853e4945c2764173012c708b8367e796e8e2c9e1cce2"
    assert report["data"]["episode_count"] == 10
    assert report["evaluation"]["horizons"] == [1, 2, 4, 8]
    assert report["evaluation"]["window_count"] > 0
    for horizon in ("1", "2", "4", "8"):
        row = report["per_horizon"][horizon]
        for method in (
            "model_correct_action",
            "persistence",
            "model_zero_action",
            "model_shuffled_action",
        ):
            assert math.isfinite(row[method]["mse"])
            assert math.isfinite(row[method]["cosine_distance"])
        for name in (
            "gain_vs_persistence",
            "gain_vs_zero_action",
            "gain_vs_shuffled_action",
            "prediction_delta_zero_mse",
            "prediction_delta_shuffled_mse",
        ):
            assert math.isfinite(row[name])
    print("rollout report: PASS")
    PY

Expected: rollout report: PASS.

- [ ] **Step 4: Re-run and check deterministic metric content**

Run the Step 2 command a second time with the same seed and arguments, saving it temporarily as:

    --output /workspace/users/lewm/runs/4d-jepa-architecture/group1_scratch_liftcube/rollout_metrics.repeat.json

Then run:

    /nix/lewm/4r-jepa-ms2/env/bin/python - <<'PY'
    import json
    import math
    from pathlib import Path

    root = Path("/workspace/users/lewm/runs/4d-jepa-architecture/group1_scratch_liftcube")
    first = json.loads((root / "rollout_metrics.json").read_text(encoding="utf-8"))
    repeat = json.loads((root / "rollout_metrics.repeat.json").read_text(encoding="utf-8"))
    for horizon in ("1", "2", "4", "8"):
        left = first["per_horizon"][horizon]
        right = repeat["per_horizon"][horizon]
        for method in (
            "model_correct_action",
            "persistence",
            "model_zero_action",
            "model_shuffled_action",
        ):
            for metric in ("mse", "cosine_distance"):
                assert math.isclose(
                    left[method][metric],
                    right[method][metric],
                    rel_tol=1e-6,
                    abs_tol=1e-12,
                )
        for metric in (
            "gain_vs_persistence",
            "gain_vs_zero_action",
            "gain_vs_shuffled_action",
            "prediction_delta_zero_mse",
            "prediction_delta_shuffled_mse",
        ):
            assert math.isclose(
                left[metric], right[metric], rel_tol=1e-6, abs_tol=1e-12
            )
    assert first["evaluation"]["window_count"] == repeat["evaluation"]["window_count"]
    assert first["checkpoint"] == repeat["checkpoint"]
    assert first["data"] == repeat["data"]
    print("deterministic metrics: PASS")
    PY

Expected: deterministic metrics: PASS. Remove only rollout_metrics.repeat.json after this check; retain rollout_metrics.json.

- [ ] **Step 5: Verify source inputs and repository state remain unchanged**

Run:

    sha256sum +      /workspace/users/lewm/runs/4d-jepa-architecture/group1_scratch_liftcube/best.pt +      /workspace/datasets/maniskill2/v0.5.3/compact/LiftCube-v0/manifest.json
    git status --short --branch

Expected: both hashes match Step 1, including the exact best.pt hash above; git status shows the current branch with no tracked or untracked source changes.

- [ ] **Step 6: Report the diagnostic honestly**

Summarize for each horizon:

- model MSE and cosine distance;
- persistence gain, where values greater than 1 mean the model wins;
- zero-action and shuffled-action gains;
- prediction deltas showing whether actions change the rollout;
- window count, elapsed time, checkpoint hash, and output path.

Explicitly state any horizon where the model fails to beat persistence or where correct actions fail to beat zero/shuffled actions. Do not infer MPC readiness from this offline evaluation alone.

---

## Final Verification Checklist

- [ ] Focused tests pass: tests/test_maniskill_rollout_eval.py.
- [ ] Full existing test suite passes with no failures.
- [ ] git diff --check and Python compilation pass.
- [ ] Real best.pt evaluation covers the same validation cohort at H=1/2/4/8.
- [ ] rollout_metrics.json is finite, complete, deterministic, and provenance-rich.
- [ ] best.pt and manifest hashes are unchanged.
- [ ] No ManiSkill/SAPIEN process was started and no dependency/environment file changed.
- [ ] Repository worktree is clean after the implementation commits and runtime evaluation.

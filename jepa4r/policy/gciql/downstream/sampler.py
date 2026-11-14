"""Thin multi-Cube input projection and per-object ALL-distance reward.

The original GCDataset owns all sampling and RNG. Features are one readonly
FP32 RAM array per split, shared by same-batch lookup threads, never workers
that sample future batches. State-only constructs no visual dimensions.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ._source import install_impls

install_impls()
from utils.datasets import GCDataset
from .vendor.cube_inputs import CubeInputs


def load_cached_features(cache_dir, split, episodes, feature_dim):
    """The cached latents of one split, episode by episode (``<cache_dir>/<split>/episode_<i>/latent.npy``, 1001 rows
    each), as one readonly FP32 array in RAM."""
    root = Path(cache_dir)
    result = np.empty((episodes * 1001, feature_dim), dtype=np.float32)
    for episode in range(episodes):
        values = np.load(root / split / f"episode_{episode:06d}" / "latent.npy", allow_pickle=False)
        if values.shape != (1001, feature_dim) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"the cached latents of {split} episode {episode} have the wrong shape, dtype or values")
        result[episode * 1001 : (episode + 1) * 1001] = values
    result.setflags(write=False)
    return result


def object_xyz(states, cubes):
    return np.stack([states[:, 19 + 9 * i : 22 + 9 * i] for i in range(cubes)], axis=1)


@dataclass
class DistanceInputs(GCDataset):
    cubes: int = 2
    mode: str = "state"
    features: object = None
    workers: int = 0
    latent_dim: int = 192

    def __post_init__(self):
        super().__post_init__()
        if self.cubes not in (1, 2, 3, 4) or self.mode not in ("state", "rgb", "pcd"):
            raise ValueError("Explicit Cube1..4 and state/RGB/PCD mode required")
        if self.config["frame_stack"] is not None or self.config["p_aug"] != 0:
            raise ValueError("Keep the original no-stack/no-augmentation state profile")
        states = self.dataset["observations"]
        if states.shape != (self.size, 19 + 9 * self.cubes) or states.dtype != np.float32:
            raise ValueError("Original matching official 19+9N state required")
        if states.flags.writeable:
            raise ValueError("Original state must remain readonly")
        if self.mode == "state":
            if self.features is not None or self.workers != 0:
                raise ValueError("State-only uses no latent array or synthetic lookup workers")
            self.current_dim, self.goal_dim = 19 + 9 * self.cubes, 3 * self.cubes
            self.contract = None
        else:
            # source: (192,512); paper Tab. 1 VISReg 768 / V-JEPA 2.1 1024
            if self.latent_dim not in (192, 512, 768, 1024):
                raise ValueError("Selected encoder must produce actual192, actual512, actual768 or actual1024")
            if not isinstance(self.features, np.ndarray) or self.features.shape != (self.size, self.latent_dim):
                raise ValueError("One aligned readonly FP32 feature array at the actual encoder width required")
            if self.features.dtype != np.float32 or self.features.flags.writeable:
                raise ValueError("Features must be readonly FP32 RAM")
            if self.workers not in (0, 4):
                raise ValueError("Use selected four role threads or zero-thread reference")
            self.current_dim, self.goal_dim = self.latent_dim + 19, self.latent_dim + 3 * self.cubes
            self.contract = CubeInputs(self.cubes, self.latent_dim)
        self.executor = (
            ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="cube-role-lookup")
            if self.workers
            else None
        )

    def sample_goals(self, *args, **kwargs):
        ids = super().sample_goals(*args, **kwargs)
        self._goal_ids.append(ids.copy())
        return ids

    def sample(self, batch_size, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)
        idxs = np.asarray(idxs)
        if idxs.shape != (batch_size,) or not np.issubdtype(idxs.dtype, np.integer):
            raise ValueError("Original integer current IDs required")
        if np.any(idxs < 0) or np.any(idxs >= self.size - 1):
            raise ValueError("Current IDs cannot be the final raw terminal")
        if "valids" in self.dataset and not np.all(self.dataset["valids"][idxs] > 0):
            raise ValueError("An invalid raw terminal cannot be sampled as current")
        self._goal_ids = []
        batch = super().sample(batch_size, idxs=idxs, evaluation=evaluation)
        value_ids, actor_ids = self._goal_ids
        current = batch["observations"]
        value_goal = batch["value_goals"]
        success = np.all(
            np.linalg.norm(object_xyz(current, self.cubes) - object_xyz(value_goal, self.cubes), axis=-1) <= 0.4,
            axis=-1,
        )
        # Preserve the original sampler's reward/mask dtype; only labels change.
        batch["rewards"] = success.astype(batch["rewards"].dtype) - 1.0
        batch["masks"] = 1.0 - success.astype(batch["masks"].dtype)
        if self.mode == "state":
            # state inputs: the goals are the cubes' xyz (3 per cube), not CubeInputs' oracle path.
            batch["value_goals"] = object_xyz(value_goal, self.cubes).reshape(batch_size, 3 * self.cubes)
            batch["actor_goals"] = object_xyz(batch["actor_goals"], self.cubes).reshape(batch_size, 3 * self.cubes)
        else:
            roles = ("observations", "next_observations", "value_goals", "actor_goals")
            ids = (idxs, idxs + 1, value_ids, actor_ids)
            if self.executor:
                futures = [self.executor.submit(self.features.__getitem__, item) for item in ids]
                visual = [future.result() for future in futures]
            else:
                visual = [self.features[item] for item in ids]
            for role, latent in zip(roles, visual):
                operation = (
                    self.contract.current if role in ("observations", "next_observations") else self.contract.goal
                )
                batch[role] = operation(latent, batch[role], oracle=False)
        return batch

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None


def create_agent(agent_class, seed, batch, config):
    """Reuse the already-adapted original GCIQL factory at actual input widths."""
    return agent_class.create(
        seed, batch["observations"][:1], batch["actions"][:1], config, ex_goals=batch["actor_goals"][:1]
    )

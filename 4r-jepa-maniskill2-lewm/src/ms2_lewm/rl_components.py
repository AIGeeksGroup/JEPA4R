"""Reuse the shared ManiSkill2-Learn point-cloud PPO model components.

Both modules are instantiated by the upstream registry. The shared local
``configs/ppo_pointnet_identity.py`` inherits the official PPO configuration and
changes only the final PointNet activation, as requested by the user. The frame
adapter merely folds/unfolds the time dimension. The critic adapter changes
the input interface from ``latent + robot_state`` to ``latent + action`` and
the scalar output to the frame latent dimension.  That input-size adaptation
is indispensable for an action-conditioned world model: the official PPO
critic is a state-value V head, not an action-conditioned Q head.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from maniskill2_learn.networks import build_model
from maniskill2_learn.networks.utils import (
    get_kwargs_from_shape,
    replace_placeholder_with_args,
)
from maniskill2_learn.utils.meta import Config


ORIGINAL_RL_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "vendor/ManiSkill2-Learn/configs/mfrl/ppo/maniskill2_pn.py"
)
DEFAULT_RL_CONFIG = Path(__file__).resolve().parents[2] / "configs/ppo_pointnet_identity.py"
DEFAULT_OBS_SHAPE = {"xyz": (1200, 3), "rgb": (1200, 3)}
POINT_KEYS = ("xyz", "rgb", "seg")


def load_ppo_config(config_path: str | Path | None = None) -> Config:
    """Read the shared configuration with the official loader/custom imports."""
    path = Path(config_path) if config_path is not None else DEFAULT_RL_CONFIG
    if not path.is_absolute():
        # Saved WM configs use project-relative paths so another checkout and
        # an arbitrary caller working directory select the same configuration.
        path = Path(__file__).resolve().parents[2] / path
    return Config.fromfile(str(path.resolve()))


class OfficialPointNetFrameEncoder(nn.Module):
    """The shared PPO PointNet, applied independently to every frame.

    Tensor inputs are ``[B,T,N,C]`` (or a single frame ``[B,N,C]``), with
    channels in the same order used upstream: xyz, optional rgb, optional seg.
    Dictionaries hold those same per-point tensors.  Robot state is not part
    of the visual encoder's output.  The result is ``[B,T,D]`` (or ``[B,D]``).

    PointNet itself detaches point-cloud preprocessing, as it does in PPO;
    this adapter never detaches its output or any encoder parameters.  In
    particular, the target frame must also pass through this shared instance.
    Its final activation is Identity in the current shared configuration; previous
    layers and the inherited global max-pooling forward remain unchanged.
    """

    def __init__(
        self,
        obs_shape: Mapping[str, Any] | None = None,
        action_dim: int = 7,
        config_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        cfg = load_ppo_config(config_path)
        self.config_path = str(Path(cfg.filename).resolve())
        self.obs_shape = deepcopy(dict(obs_shape or DEFAULT_OBS_SHAPE))
        self.source_visual_nn_cfg = deepcopy(
            cfg.agent_cfg.actor_cfg.nn_cfg.visual_nn_cfg
        )
        self.visual_nn_cfg = replace_placeholder_with_args(
            deepcopy(self.source_visual_nn_cfg),
            **get_kwargs_from_shape(self.obs_shape, action_dim),
        )
        self.pointnet = build_model(deepcopy(self.visual_nn_cfg))
        self.output_dim = int(self.visual_nn_cfg.mlp_spec[-1])
        self.latent_dim = self.output_dim
        self.input_channels = int(self.visual_nn_cfg.feat_dim)

    def forward(self, points: Tensor | Mapping[str, Tensor]) -> Tensor:
        if isinstance(points, Mapping):
            if "xyz" not in points:
                raise ValueError("Point-cloud dictionaries must contain 'xyz'.")
            reference = points["xyz"]
            selected = {key: points[key] for key in POINT_KEYS if key in points}
        elif isinstance(points, Tensor):
            reference = points
            selected = None
        else:
            raise TypeError("Expected a point-cloud tensor or xyz/rgb/seg dictionary.")

        if not isinstance(reference, Tensor) or reference.ndim not in (3, 4):
            raise ValueError("Expected point-cloud shape [B,N,C] or [B,T,N,C].")
        prefix = reference.shape[:-2]
        point_count = reference.shape[-2]
        if point_count == 0 or any(size == 0 for size in prefix):
            raise ValueError("Point clouds, batch, and time dimensions must be nonempty.")

        if selected is None:
            channel_count = reference.shape[-1]
            flat_input = reference.reshape(-1, point_count, channel_count)
        else:
            channel_count = 0
            flat_input = {}
            for key, value in selected.items():
                if not isinstance(value, Tensor) or value.shape[:-1] != reference.shape[:-1]:
                    raise ValueError(f"Point-cloud component {key!r} has incompatible shape.")
                channel_count += value.shape[-1]
                flat_input[key] = value.reshape(-1, point_count, value.shape[-1])

        if channel_count != self.input_channels:
            raise ValueError(
                f"PPO PointNet expects {self.input_channels} channels from obs_shape; "
                f"received {channel_count}."
            )
        latent1 = self.pointnet(flat_input)
        return latent1.reshape(*prefix, self.output_dim)


class OfficialCriticPredictor(nn.Module):
    """The PPO critic's official LinearMLP with the necessary boundary sizes.

    The source ``[512 + agent_shape, 256, 256, 1]`` head becomes
    ``[D + action_dim, 256, 256, D]``.  Hidden widths, activation, normalization,
    output activation, and upstream ``zero_init_output`` initialization are
    unchanged.  Upstream's latter option multiplies output weights by 0.01
    and zeros the bias; it does not zero the output weights.
    """

    def __init__(
        self,
        action_dim: int,
        latent_dim: int = 512,
        config_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or latent_dim <= 0:
            raise ValueError("action_dim and latent_dim must be positive.")
        cfg = load_ppo_config(config_path)
        self.config_path = str(Path(cfg.filename).resolve())
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        source_latent_dim = int(
            cfg.agent_cfg.actor_cfg.nn_cfg.visual_nn_cfg.mlp_spec[-1]
        )
        if self.latent_dim != source_latent_dim:
            raise ValueError(
                f"latent_dim must equal the official frame encoder output {source_latent_dim}."
            )
        self.source_mlp_cfg = deepcopy(cfg.agent_cfg.critic_cfg.nn_cfg.mlp_cfg)
        self.mlp_cfg = deepcopy(self.source_mlp_cfg)
        self.mlp_cfg.mlp_spec[0] = self.latent_dim + self.action_dim
        self.mlp_cfg.mlp_spec[-1] = self.latent_dim
        self.head = build_model(deepcopy(self.mlp_cfg))

    def forward(self, latent2: Tensor, action: Tensor) -> Tensor:
        if latent2.shape[:-1] != action.shape[:-1]:
            raise ValueError("latent2 and action must have the same batch dimensions.")
        if latent2.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected latent2 dimension {self.latent_dim}.")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dimension {self.action_dim}.")
        return self.head(torch.cat((latent2, action), dim=-1))

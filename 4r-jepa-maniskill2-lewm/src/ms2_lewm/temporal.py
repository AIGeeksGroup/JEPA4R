"""Independent, action-free temporal aggregation for the RL frame encoder.

Every frame contributes one token of width D. Native PyTorch Transformer
layers preserve that width throughout attention and residual paths; no
projection compresses D to the LeWM predictor's hidden width. Architecture
choices live in this project's config, independently of LeWM's predictor.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPORAL_CONFIG = PROJECT_ROOT / "configs" / "temporal_encoder.yaml"


def resolve_temporal_config(config=None) -> dict:
    """Return validated plain values, merging overrides over our own defaults.

    ``config`` may be a mapping or a YAML path. Relative paths are resolved
    from the project root so saved checkpoint configs are independent of cwd.
    Neither this function nor the encoder imports any upstream LeWM module.
    """
    with DEFAULT_TEMPORAL_CONFIG.open(encoding="utf-8") as source:
        defaults = yaml.safe_load(source)
    if config is None:
        overrides = {}
    elif isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        with config_path.open(encoding="utf-8") as source:
            overrides = yaml.safe_load(source)
    elif isinstance(config, Mapping):
        overrides = dict(config)
    else:
        raise TypeError("Temporal config must be a mapping or a YAML path.")
    if not isinstance(overrides, Mapping):
        raise ValueError("Temporal config YAML must contain a mapping.")
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown temporal config fields: {sorted(unknown)}")
    values = {**defaults, **overrides}
    if values["architecture"] != "temporal_encoder_v1":
        raise ValueError("Unsupported temporal architecture; expected temporal_encoder_v1.")
    for key in ("history_size", "num_layers", "num_heads", "mlp_ratio"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Temporal {key} must be a positive integer.")
        values[key] = int(value)
    dropout = values["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("Temporal dropout must be a number in [0, 1).")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("Temporal dropout must be a number in [0, 1).")
    values["dropout"] = float(dropout)
    if values["activation"] != "gelu":
        raise ValueError("temporal_encoder_v1 uses GELU activation.")
    if not isinstance(values["norm_first"], bool):
        raise ValueError("Temporal norm_first must be boolean.")
    values["norm_first"] = bool(values["norm_first"])
    values["architecture"] = str(values["architecture"])
    values["activation"] = str(values["activation"])
    return values


class TemporalEncoder(nn.Module):
    """Aggregate consecutive frame vectors [B,H,D] to [B,D], for 1 <= H <= T.

    K=1 means one vector per frame; [B,H,1,D] is accepted as an equivalent
    view. Positions are learned and attention is causal. Only the last
    contextualized token is returned. Actions never enter this module.
    """

    def __init__(self, latent_dim: int, K: int = 1, config=None):
        super().__init__()
        if isinstance(K, bool) or K != 1:
            raise ValueError("This model requires K=1: one vector per point-cloud frame.")
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer.")
        self.config = resolve_temporal_config(config)
        self.K = 1
        self.latent_dim = latent_dim
        self.num_frames = self.config["history_size"]
        if latent_dim % self.config["num_heads"]:
            raise ValueError("latent_dim must be divisible by temporal num_heads.")
        self.pos_embedding = nn.Parameter(torch.empty(1, self.num_frames, latent_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        # Construct separately: TransformerEncoder's cloning would otherwise
        # start all layers from identical values.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=self.config["num_heads"],
                dim_feedforward=latent_dim * self.config["mlp_ratio"],
                dropout=self.config["dropout"],
                activation=self.config["activation"],
                batch_first=True,
                norm_first=self.config["norm_first"],
            )
            for _ in range(self.config["num_layers"])
        ])
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, latent1: Tensor) -> Tensor:
        if latent1.ndim == 4:
            if latent1.shape[2] != 1:
                raise ValueError("Expected K=1 in [B,T,K,D].")
            latent1 = latent1.squeeze(2)
        if latent1.ndim != 3 or latent1.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected [B,T,{self.latent_dim}].")
        history = latent1.shape[1]
        if not 1 <= history <= self.num_frames:
            raise ValueError(f"Temporal context length must be in 1..{self.num_frames}.")
        x = latent1 + self.pos_embedding[:, :history]
        causal_mask = torch.ones(history, history, dtype=torch.bool, device=x.device).triu(1)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        return self.norm(x[:, -1])

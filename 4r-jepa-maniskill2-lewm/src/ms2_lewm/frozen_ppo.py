"""Stage 3: frozen checkpoint encoder through latent2, trainable PPO heads.

The WM prediction head and loss are never attached to PPO. Each observation
contains its own complete history; there is no hidden rolling state in this
encoder, so PPO's shuffled minibatches remain valid.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from maniskill2_learn.networks.backbones.visuomotor import Visuomotor
from maniskill2_learn.networks.builder import BACKBONES
from maniskill2_learn.utils.torch import ExtendedModule
from maniskill2_learn.utils.meta import Config

from .model import PointCloudEncoder
from .temporal import resolve_temporal_config


ROOT = Path(__file__).resolve().parents[2]


def _saved_config_path(value, checkpoint_dir):
    """Resolve a bundle-local snapshot or a project-relative training config."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    local = checkpoint_dir / candidate
    return local if local.is_file() else ROOT / candidate


def read_encoder_checkpoint(checkpoint, expected_sha256=None):
    """Read our exported WM, full Lightning checkpoint, or encoder-only bundle.

These are local, user-selected training checkpoints. Full Lightning files
contain optimizer/config objects and use the trusted-checkpoint loader.
    """
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Trained world-model checkpoint not found: {path}")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError("World-model checkpoint SHA256 differs from the recorded source.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a world-model state dictionary/checkpoint.")
    if payload.get("format") in ("ms2_lewm_frozen_encoder_v1", "ms2_lewm_frozen_encoder_v2",
                                "ms2_lewm_frozen_encoder_v3"):
        config = dict(payload["encoder_config"])
        state = payload["encoder_state_dict"]
        history_size = int(payload["history_size"])
        frameskip = int(payload.get("pretraining_frameskip", payload.get("frameskip", 0)))
        source = dict(payload["source"])
        if payload.get("format") in ("ms2_lewm_frozen_encoder_v2", "ms2_lewm_frozen_encoder_v3"):
            if payload.get("history_stride") != 1:
                raise ValueError("Frozen encoder bundle requires history_stride=1.")
            snapshot = _saved_config_path(config["rl_config_path"], path.parent)
            if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != payload["encoder_config_sha256"]:
                raise ValueError("Frozen encoder configuration SHA256 mismatch.")
            for filename, digest in payload["encoder_definition_sha256"].items():
                definition = Path(filename)
                if not definition.is_absolute():
                    definition = ROOT / definition
                if not definition.is_file() or hashlib.sha256(definition.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Frozen encoder implementation SHA256 mismatch: {filename}")
    else:
        config_path = path.parent / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text())
        else:
            config_path = path.parent / "config.yaml"
            if not config_path.is_file():
                raise ValueError("Checkpoint needs its saved config.json or config.yaml to identify the encoder.")
            saved = OmegaConf.load(config_path)
            config = OmegaConf.to_container(saved.get("model", saved), resolve=True)
        target = config.pop("_target_", "ms2_lewm.model.WorldModel")
        if target not in ("ms2_lewm.model.WorldModel", "ms2_lewm.model.PointCloudWorldModel"):
            raise ValueError(f"Expected our trained point-cloud WorldModel, got {target!r}.")
        if not config.get("rl_config_path"):
            raise ValueError("Saved model must explicitly record rl_config_path; activation cannot be inferred from weights.")
        recorded_stride = config.pop("history_stride", None)
        if recorded_stride is None:
            training_config = path.parent / "config.yaml"
            if training_config.is_file():
                recorded_stride = OmegaConf.select(OmegaConf.load(training_config), "data.dataset.frameskip")
        if recorded_stride is None:
            raise ValueError("Checkpoint does not record history_stride; old/unknown temporal spacing cannot be assumed to be 1.")
        weights = payload.get("state_dict", payload)
        prefix = "model.encoder." if any(k.startswith("model.encoder.") for k in weights) else "encoder."
        state = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        if not state:
            raise ValueError("Checkpoint contains no encoder-through-latent2 weights.")
        history_size = int(config.get("temporal_config", {}).get("history_size", 0))
        frameskip = int(recorded_stride)
        source = {"checkpoint": str(path), "sha256": sha256, "config": str(config_path)}
    allowed = {"action_dim", "obs_shape", "rl_config_path", "temporal_config"}
    if set(config) - allowed:
        raise ValueError(f"Unsupported saved encoder configuration keys: {sorted(set(config) - allowed)}")
    if "temporal_config" not in config:
        raise ValueError("Checkpoint must record the independent temporal_config; legacy LeWM-predictor-derived encoders are rejected.")
    if not isinstance(config["temporal_config"], Mapping) or set(resolve_temporal_config()) - set(config["temporal_config"]):
        raise ValueError("Saved temporal_config must record all independent architecture parameters explicitly.")
    config["temporal_config"] = resolve_temporal_config(config["temporal_config"])
    if history_size != config["temporal_config"]["history_size"] or frameskip != 1:
        raise ValueError("Frozen PPO requires a world model pretrained on consecutive frames (history_stride=1); legacy stride 5 is rejected.")
    config["rl_config_path"] = str(_saved_config_path(config["rl_config_path"], path.parent).resolve())
    source["loaded_checkpoint"] = str(path)
    source["loaded_sha256"] = sha256
    return config, state, history_size, frameskip, source


def export_encoder_bundle(checkpoint, output):
    """Snapshot only the learned encoder and its configuration for stage 3."""
    config, state, history_size, frameskip, source = read_encoder_checkpoint(checkpoint)
    # Strict reconstruction validates that both PointNet and temporal weights
    # exist and match before writing a reusable encoder bundle.
    encoder = PointCloudEncoder(**config)
    encoder.load_state_dict(state, strict=True)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Encoder bundle already exists: {output}")
    snapshot = output.with_suffix(".ppo_config.py")
    if snapshot.exists():
        raise FileExistsError(f"Encoder configuration snapshot already exists: {snapshot}")
    Config.fromfile(config["rl_config_path"]).dump(str(snapshot))
    config = dict(config, rl_config_path=snapshot.name)
    root = ROOT
    definitions = [root / "src/ms2_lewm/model.py", root / "src/ms2_lewm/pointnet_identity.py",
                   root / "src/ms2_lewm/rl_components.py", root / "src/ms2_lewm/temporal.py",
                   root / "vendor/ManiSkill2-Learn/maniskill2_learn/networks/backbones/pointnet.py",
                   root / "vendor/ManiSkill2-Learn/maniskill2_learn/networks/backbones/mlp.py"]
    payload = {"format": "ms2_lewm_frozen_encoder_v3", "encoder_config": config,
               "encoder_state_dict": encoder.state_dict(), "history_size": history_size,
               "history_stride": 1, "pretraining_frameskip": frameskip,
               "encoder_config_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
               "encoder_definition_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in definitions},
               "latent_dim": encoder.output_dim, "source": source}
    torch.save(payload, output)
    return output


@BACKBONES.register_module()
class FrozenWorldModelEncoder(ExtendedModule):
    """One checkpoint PointCloudEncoder, permanently frozen and deterministic."""

    def __init__(self, checkpoint, expected_sha256=None):
        super().__init__()
        if checkpoint is None:
            raise ValueError("Stage 3 requires --wm-checkpoint from completed world-model training.")
        config, state, self.history_size, self.pretraining_frameskip, self.provenance = read_encoder_checkpoint(
            checkpoint, expected_sha256
        )
        self.history_stride = 1
        self.encoder = PointCloudEncoder(**config)
        self.encoder.load_state_dict(state, strict=True)
        self.output_dim = self.latent_dim = self.encoder.output_dim
        self.checkpoint_path = str(Path(checkpoint).resolve())
        self.register_buffer("source_sha256_bytes", torch.tensor(
            list(bytes.fromhex(self.provenance["sha256"])), dtype=torch.uint8
        ))
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Parent PPO.train() recursively visits this module. Keep the learned
        # representation in eval mode (especially Transformer dropout).
        super().train(False)
        self.requires_grad_(False)
        return self

    def forward(self, inputs, **kwargs):
        if isinstance(inputs, Mapping):
            if "xyz" in inputs:
                points = {k: inputs[k] for k in ("xyz", "rgb", "seg") if k in inputs}
                reference = points["xyz"]
            elif "pointcloud" in inputs:
                points = inputs["pointcloud"]
                reference = points
            else:
                raise ValueError("Expected xyz/rgb history or pointcloud history.")
        else:
            points = reference = inputs
        if not torch.is_tensor(reference) or reference.ndim != 4 or reference.shape[1] != self.history_size:
            raise ValueError(f"Frozen PPO expects self-contained [B,{self.history_size},N,C] histories.")
        dtype = next(self.encoder.parameters()).dtype
        if isinstance(points, Mapping):
            points = {k: v.to(dtype=dtype) for k, v in points.items()}
        else:
            points = points.to(dtype=dtype)
        with torch.no_grad():
            return self.encoder(points)


@BACKBONES.register_module()
class FrozenWMVisuomotor(Visuomotor):
    """Original Visuomotor forward: frozen latent2 plus the current state.

    The inherited forward removes state before calling the encoder and then
    concatenates it with latent2 for the unchanged actor/critic heads.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.visual_nn, FrozenWorldModelEncoder):
            raise TypeError("FrozenWMVisuomotor requires a frozen world-model encoder.")

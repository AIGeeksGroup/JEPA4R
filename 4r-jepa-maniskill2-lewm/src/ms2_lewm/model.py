"""PointNet -> temporal encoder -> unchanged PPO critic hidden layers.

The public LeWM repository calls its model JEPA, its temporal predictor
ARPredictor, and its regularizer SIGReg. TemporalEncoder, LeWMLoss and
WorldModel are the requested local model classes, not invented upstream APIs.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .official import lewm_config, lewm_jepa, lewm_module
from .temporal import TemporalEncoder, resolve_temporal_config



class LeWMLoss(nn.Module):
    """Final-step MSE plus official SIGReg on every frame, across batch.

No stop-gradient, EMA target, latent2 regularization, or time/batch merging.
The final-step prediction target is the explicit user-specified model change;
public LeWM's training script instead predicts all context-aligned steps.
"""

    def __init__(self, history_size=None):
        super().__init__()
        cfg = lewm_config()
        self.history_size = int(resolve_temporal_config()["history_size"] if history_size is None else history_size)
        self.weight = float(cfg.loss.sigreg.weight)
        self.sigreg = lewm_module.SIGReg(**dict(cfg.loss.sigreg.kwargs))

    def forward(self, prediction: Tensor, latent1: Tensor) -> dict[str, Tensor]:
        if latent1.ndim != 3 or latent1.shape[1] != self.history_size + 1:
            raise ValueError(f"SIGReg requires all T+1={self.history_size + 1} frames [B,T+1,D].")
        target = latent1[:, -1]
        if prediction.shape != target.shape:
            raise ValueError("Prediction and next-frame latent1 must both be [B,D].")
        pred_loss = F.mse_loss(prediction, target)
        sigreg_loss = self.sigreg(latent1.transpose(0, 1))
        return {
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "loss": pred_loss + self.weight * sigreg_loss,
        }


class PointCloudEncoder(nn.Module):
    """The complete 4D encoder: independent 3D frames, then time aggregation."""

    def __init__(self, action_dim, obs_shape=None, rl_config_path=None, temporal_config=None):
        super().__init__()
        from .rl_components import OfficialPointNetFrameEncoder
        self.frame_encoder = OfficialPointNetFrameEncoder(
            obs_shape=obs_shape, action_dim=action_dim, config_path=rl_config_path
        )
        self.output_dim = self.frame_encoder.output_dim
        self.temporal_encoder = TemporalEncoder(self.output_dim, K=1, config=temporal_config)
        self.temporal_config = dict(self.temporal_encoder.config)

    def forward(self, pointcloud):
        latent1 = self.frame_encoder(pointcloud)
        if latent1.ndim == 2:
            latent1 = latent1.unsqueeze(1)
        return self.temporal_encoder(latent1)


class WorldModel(lewm_jepa.JEPA):
    """LeWM's dict interface with the requested point-cloud model replacement.

forward: {pointcloud:[B,T+1,N,C], action:[B,>=T,A]} -> prediction/loss dict.
encode: {pointcloud:[B,H,N,C]} -> emb/latent1:[B,H,D], latent2:[B,D].
rollout: info pointcloud:[B,S,H,N,C], actions:[B,S,L,A] ->
         predicted_emb:[B,S,L+1,D], with L>=H.

As in the pinned public le-wm JEPA, rollout actions INCLUDE context actions;
action k leaves frame k. This is different from newer stable-worldmodel
releases that accept strictly-future actions. The matching stable revision
is pinned alongside this project. ``pixels`` is accepted as the unchanged
official I/O key, containing point clouds rather than RGB images.
"""

    def __init__(self, action_dim: int = 7, obs_shape=None, rl_config_path=None, history_stride: int = 1, temporal_config=None):
        if history_stride != 1:
            raise ValueError("Current temporal encoder uses consecutive control-step frames (history_stride=1).")
        from .rl_components import OfficialCriticPredictor
        encoder = PointCloudEncoder(action_dim, obs_shape, rl_config_path, temporal_config)
        super().__init__(
            encoder=encoder,
            predictor=OfficialCriticPredictor(action_dim, encoder.output_dim, rl_config_path),
            action_encoder=nn.Identity(),
        )
        self.action_dim = int(action_dim)
        self.history_stride = 1
        self.latent_dim = encoder.output_dim
        self.history_size = encoder.temporal_encoder.num_frames
        self.temporal_config = dict(encoder.temporal_config)
        self.loss_fn = LeWMLoss(history_size=self.history_size)

    @property
    def frame_encoder(self):
        return self.encoder.frame_encoder

    @property
    def temporal_encoder(self):
        return self.encoder.temporal_encoder

    def _points(self, info):
        if isinstance(info, Tensor):
            points = info
        elif "pointcloud" in info:
            points = info["pointcloud"]
        elif "xyz" in info:
            points = {k: info[k] for k in ("xyz", "rgb", "seg") if k in info}
        elif "pixels" in info:
            points = info["pixels"]
        else:
            raise ValueError("Expected pointcloud, xyz/rgb, or official pixels alias.")
        dtype = next(self.frame_encoder.parameters()).dtype
        if isinstance(points, Mapping):
            return {k: v.to(dtype=dtype) for k, v in points.items()}
        # The official LanceWriter flattens non-image columns. Restore the
        # lossless shape inside the model; the collector/writer stay intact.
        channels = self.frame_encoder.input_channels
        n_points = int(self.frame_encoder.obs_shape["xyz"][0])
        if points.shape[-1] == n_points * channels:
            points = points.reshape(*points.shape[:-1], n_points, channels)
        return points.to(dtype=dtype)

    def encode(self, info):
        output = dict(info) if isinstance(info, Mapping) else {"pointcloud": info}
        latent1 = self.frame_encoder(self._points(info))
        if latent1.ndim == 2:
            latent1 = latent1.unsqueeze(1)
        output["emb"] = output["latent1"] = latent1
        output["latent2"] = self.temporal_encoder(latent1[:, -self.history_size:])
        if "action" in output:
            output["act_emb"] = torch.nan_to_num(output["action"], 0.0)
        return output

    def predict(self, emb, act_emb):
        if emb.ndim != 3 or act_emb.ndim != 3 or emb.shape[:2] != act_emb.shape[:2]:
            raise ValueError("Expected aligned emb [B,H,D] and actions [B,H,A].")
        latent2 = self.temporal_encoder(emb[:, -self.history_size:])
        return self.predictor(latent2, act_emb[:, -1]).unsqueeze(1)

    def forward(self, info, action=None):
        output = dict(info) if isinstance(info, Mapping) else {"pointcloud": info}
        if action is not None:
            output["action"] = action
        if "action" not in output:
            raise ValueError("Training requires the action sequence.")
        latent1 = self.frame_encoder(self._points(output))
        T = self.history_size
        if latent1.ndim != 3 or latent1.shape[1] != T + 1:
            raise ValueError(f"Training requires exactly T+1={T+1} frames.")
        actions = torch.nan_to_num(output["action"], 0.0)
        if actions.ndim != 3 or actions.shape[0] != latent1.shape[0] or actions.shape[1] < T:
            raise ValueError("Expected action [B,>=T,A], paired with its departure frame.")
        latent2 = self.temporal_encoder(latent1[:, :T])
        prediction = self.predictor(latent2, actions[:, T - 1])
        output.update(emb=latent1, latent1=latent1, latent2=latent2,
                      act_emb=actions, prediction=prediction, target=latent1[:, T])
        output.update(self.loss_fn(prediction, latent1))
        return output

    def rollout(self, info, action_sequence, history_size=None):
        # Keep the exact public JEPA rollout algorithm and action indexing.
        output = dict(info)
        points = self._points(output)
        if isinstance(points, Mapping):
            points = torch.cat([points[k] for k in ("xyz", "rgb", "seg") if k in points], -1)
        if points.ndim != 5 or action_sequence.ndim != 4:
            raise ValueError("Expected context [B,S,H,N,C] and actions [B,S,L,A].")
        if points.shape[:2] != action_sequence.shape[:2] or action_sequence.shape[-1] != self.action_dim:
            raise ValueError("Rollout batch/sample/action dimensions do not match.")
        if not 1 <= points.shape[2] <= action_sequence.shape[2]:
            raise ValueError("LeWM rollout actions must include H context actions (L>=H>=1).")
        if history_size is not None and history_size != self.history_size:
            raise ValueError("history_size must match the official training config.")
        output["pixels"] = points
        return super().rollout(output, action_sequence, history_size=self.history_size)

    def get_cost(self, info_dict, action_candidates):
        # Official LeWM final-state latent MSE, with point-cloud goal I/O.
        device = next(self.parameters()).device
        info = {k: v.to(device) if torch.is_tensor(v) else v for k, v in info_dict.items()}
        goal_points = info.get("goal_pointcloud", info.get("goal"))
        if goal_points is None:
            raise ValueError("Planning requires a goal point cloud under 'goal'.")
        goal = self.encode({"pointcloud": goal_points[:, 0]})
        info["goal_emb"] = goal["emb"].unsqueeze(1)
        return self.criterion(self.rollout(info, action_candidates))


PointCloudWorldModel = WorldModel

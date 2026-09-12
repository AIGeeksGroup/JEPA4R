"""WorldModel：整条流水线本身。

    点云序列 [B, T, K, C]
      → SpatialAggregator（逐帧，仓库 A）                  → latent1 [B, T, K, D]
      → TemporalEncoder（+ 位置编码 + 时间编码）            → latent2 [B, K, D]
      → Predictor（架构与动作输入来自仓库 B）               → 预测的下一帧 latent1 [B, K, D]
      损失来自仓库 C（losses/lewm.py）。
"""

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from losses import LeWMLoss, LossConfig
from models.action import ActionConfig, ActionEncoder
from models.aggregator import AggregatorConfig, build_spatial_aggregator
from models.predictor import Predictor, PredictorConfig
from models.proprio import ProprioConfig, ProprioEncoder
from models.temporal import TemporalConfig, TemporalEncoder


@dataclass
class WorldModelConfig:
    K: int = 512  # 每帧点数
    D: int = 64  # 全程统一的向量维度
    T: int = 4  # 上下文帧数
    skip_frame: int = 1  # 原子动作/观测帧间隔；每个 latent transition 消耗 skip_frame 个动作
    C: int = 6  # 点通道数（前三维 xyz）
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    proprio: ProprioConfig = field(default_factory=ProprioConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_yaml(cls, path: str = "configs/default.yaml", overrides: Sequence[str] = ()) -> "WorldModelConfig":
        """读 YAML 的 ``model`` 段 + 命令行 dotlist 覆盖（如 ``model.K=256``），经 dataclass 模式校验后返回。"""
        cfg = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.load(path).model)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([o[len("model.") :] for o in overrides if o.startswith("model.")]))
        return OmegaConf.to_object(cfg)


@dataclass
class TransitionPrediction:
    scene_latent: torch.Tensor
    temporal_features: torch.Tensor
    decoder_features: torch.Tensor
    next_proprio_normalized: torch.Tensor
    next_proprio: torch.Tensor
    object_pos: torch.Tensor
    contact_probability: torch.Tensor | None


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        assert cfg.C >= 3 and cfg.T >= 1 and cfg.K >= 1 and cfg.skip_frame >= 1
        self.cfg = cfg
        self.aggregator = build_spatial_aggregator(
            cfg.aggregator, in_channels=cfg.C, out_channels=cfg.D
        )
        self.temporal = TemporalEncoder(cfg.temporal, K=cfg.K, T=cfg.T, D=cfg.D)
        self.ordered_action_tokens = cfg.predictor.fusion == "cross_attention"
        self.action_encoder = ActionEncoder(
            cfg.action, D=cfg.D, chunk_size=cfg.skip_frame, preserve_tokens=self.ordered_action_tokens,
        )
        self.predictor = Predictor(cfg.predictor, D=cfg.D, envs=list(cfg.action.envs))
        self.criterion = LeWMLoss(cfg.loss)
        self.proprio_encoder: ProprioEncoder | None = None
        self.next_proprio_head: nn.Linear | None = None
        self.object_position_head: nn.Linear | None = None
        self.contact_head: nn.Linear | None = None
        if cfg.loss.contact_weight > 0 and not cfg.proprio.enabled:
            raise ValueError("contact prediction requires enabled proprioception")
        if cfg.proprio.enabled:
            if cfg.predictor.fusion != "cross_attention":
                raise ValueError(
                    "proprioception requires predictor fusion='cross_attention'"
                )
            self.proprio_encoder = ProprioEncoder(cfg.proprio, D=cfg.D)
            self.next_proprio_head = nn.Linear(cfg.D, cfg.proprio.dim)
            self.object_position_head = nn.Linear(cfg.D, 3)
            if cfg.loss.contact_weight > 0:
                self.contact_head = nn.Linear(cfg.D, 1)

    # ---------------------------------------------------------------- helpers
    def _env(self, env):
        env = next(iter(self.cfg.action.envs)) if env is None else env  # 一个 batch 只含一个环境；缺省取配置里的第一个
        assert env in self.cfg.action.envs, f"未登记的环境 {env!r}"
        return env

    def spatial_provenance(self) -> dict[str, object]:
        if hasattr(self.aggregator, "provenance"):
            return self.aggregator.provenance()
        return {"backend": "scratch_ptv3"}

    def _aggregate(self, points: torch.Tensor, return_positions: bool = False):
        """[B, N, K, C] -> latent1 [B, N, K, D]，逐帧独立（各帧展平到骨干的 batch 维，帧间不共享信息）。"""
        assert points.dim() == 4 and points.shape[2:] == (self.cfg.K, self.cfg.C), \
            f"points 应为 [B, N, {self.cfg.K}, {self.cfg.C}]，得到 {tuple(points.shape)}"
        B, N, K, C = points.shape
        tokens, positions = self.aggregator(points.reshape(B * N, K, C))
        tokens = tokens.view(B, N, K, self.cfg.D)
        if return_positions:
            return tokens, positions.view(B, N, K, 3)
        return tokens

    def _predict_sequence(
        self,
        latent1: torch.Tensor,
        actions: torch.Tensor,
        env: str,
        *,
        proprio_normalized: torch.Tensor | None = None,
        return_features: bool = False,
    ):
        """latent1 [B,T,K,D] + 原子动作 [B,T*skip_frame,A_env] -> predictions."""
        assert latent1.dim() == 4 and latent1.shape[1:] == (self.cfg.T, self.cfg.K, self.cfg.D)
        assert actions.dim() == 3 and actions.shape[:2] == (latent1.shape[0], self.cfg.T * self.cfg.skip_frame)
        temporal_features = self.temporal(latent1, return_sequence=True, causal=True)
        action_emb = self._encode_actions(actions, env)
        robot_tokens = None
        if self.proprio_encoder is not None:
            if proprio_normalized is None or proprio_normalized.shape != (
                latent1.shape[0],
                self.cfg.T,
                self.cfg.proprio.dim,
            ):
                raise ValueError(
                    "enabled proprio_normalized must be "
                    f"[B,{self.cfg.T},{self.cfg.proprio.dim}]"
                )
            robot_tokens = self.proprio_encoder.forward_normalized(
                proprio_normalized
            )
        prediction = self.predictor(
            temporal_features,
            action_emb,
            env,
            robot_tokens=robot_tokens,
            return_features=return_features,
        )
        if return_features:
            pred_all, decoder_features = prediction
        else:
            pred_all = prediction
        assert pred_all.shape == latent1.shape
        if return_features:
            return pred_all, temporal_features, decoder_features
        return pred_all, temporal_features

    def _predict(self, latent2: torch.Tensor, action: torch.Tensor, env: str) -> torch.Tensor:
        """latent2 [B, K, D] + 原子动作 [B, S, A_env] -> [B, K, D]。"""
        return self.predictor(latent2, self._encode_actions(action, env), env)

    def _encode_actions(self, actions: torch.Tensor, env: str) -> torch.Tensor:
        if self.ordered_action_tokens:
            return self.action_encoder.encode_tokens(actions, env)
        return self.action_encoder.encode_chunks(actions, env)

    # ---------------------------------------------------------------- 对外接口
    def encode(self, points: torch.Tensor):
        """points [B, T, K, C] -> (latent1 [B, T, K, D], latent2 [B, K, D])，只编码不预测。"""
        latent1 = self._aggregate(points)
        return latent1, self.temporal(latent1, causal=True)

    def forward(
        self,
        points: torch.Tensor,
        actions: torch.Tensor,
        env: str = None,
        negative_actions: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        object_pos: torch.Tensor | None = None,
        cube_robot_contact: torch.Tensor | None = None,
    ) -> dict:
        """points [B,T+1,K,C]，actions [B,T*skip_frame,A_env]；每段原子动作预测下一采样帧 latent。"""
        T = self.cfg.T
        assert points.dim() == 4 and points.shape[1] == T + 1, f"points 应为 [B, {T + 1}, K, C]，得到 {tuple(points.shape)}"
        assert actions.dim() == 3 and actions.shape[:2] == (points.shape[0], T * self.cfg.skip_frame), (
            f"actions 应为 [B, {T * self.cfg.skip_frame}, A]，得到 {tuple(actions.shape)}"
        )
        env = self._env(env)
        latent1_all, positions_all = self._aggregate(points, return_positions=True)
        latent1, target = latent1_all[:, :T], latent1_all[:, T]
        normalized_proprio = None
        if self.proprio_encoder is not None:
            expected_proprio = (points.shape[0], T + 1, self.cfg.proprio.dim)
            if proprio is None or proprio.shape != expected_proprio:
                raise ValueError(
                    f"enabled proprio must be {expected_proprio}, "
                    f"got {None if proprio is None else tuple(proprio.shape)}"
                )
            if object_pos is None or object_pos.shape != (points.shape[0], T + 1, 3):
                raise ValueError(
                    f"enabled object_pos must be [B,{T + 1},3], "
                    f"got {None if object_pos is None else tuple(object_pos.shape)}"
                )
            if not torch.isfinite(object_pos).all():
                raise ValueError("object_pos must be finite")
            if self.contact_head is not None:
                expected_contact = (points.shape[0], T + 1)
                if cube_robot_contact is None or cube_robot_contact.shape != expected_contact:
                    raise ValueError(
                        f"enabled cube_robot_contact must be {expected_contact}, "
                        f"got {None if cube_robot_contact is None else tuple(cube_robot_contact.shape)}"
                    )
                if not torch.isfinite(cube_robot_contact).all():
                    raise ValueError("cube_robot_contact must be finite")
                if not torch.all(
                    (cube_robot_contact == 0) | (cube_robot_contact == 1)
                ):
                    raise ValueError("cube_robot_contact must contain binary labels")
            elif cube_robot_contact is not None:
                raise ValueError("cube_robot_contact was provided to a disabled contact head")
            normalized_proprio = self.proprio_encoder.normalize(proprio)
            pred_all, temporal_features, decoder_features = self._predict_sequence(
                latent1,
                actions,
                env,
                proprio_normalized=normalized_proprio[:, :T],
                return_features=True,
            )
        else:
            if proprio is not None or object_pos is not None or cube_robot_contact is not None:
                raise ValueError("proprio/object_pos/contact were provided to a disabled model")
            pred_all, temporal_features = self._predict_sequence(latent1, actions, env)
        latent2 = temporal_features[:, -1]
        target_all = latent1_all[:, 1:]
        pred = pred_all[:, -1]
        negative_pred_all = None
        if negative_actions is not None:
            assert negative_actions.shape == actions.shape, (
                f"negative_actions 应与 actions 同形，得到 {tuple(negative_actions.shape)} 与 {tuple(actions.shape)}"
            )
            robot_tokens = None
            if self.proprio_encoder is not None:
                robot_tokens = self.proprio_encoder.forward_normalized(
                    normalized_proprio[:, :T]
                )
            negative_pred_all = self.predictor(
                temporal_features,
                self._encode_actions(negative_actions, env),
                env,
                robot_tokens=robot_tokens,
            )
        losses = self.criterion(
            pred_all, target_all, latent1_all, positions_all, negative_pred=negative_pred_all,
        )
        output = {"pred": pred, "pred_all": pred_all, "target": target, "target_all": target_all,
                "loss": losses["loss"], "latent1": latent1, "latent2": latent2, "latent2_all": temporal_features,
                "pred_loss": losses["pred_loss"], "latent_loss": losses["latent_loss"],
                "flow_loss": losses["flow_loss"], "rank_loss": losses["rank_loss"], "reg_loss": losses["reg_loss"]}
        if self.proprio_encoder is not None:
            assert self.next_proprio_head is not None
            assert self.object_position_head is not None
            pooled = decoder_features.mean(dim=2)
            predicted_proprio_normalized = self.next_proprio_head(pooled)
            predicted_proprio = self.proprio_encoder.denormalize(
                predicted_proprio_normalized
            )
            predicted_object = self.object_position_head(pooled)
            proprio_loss = F.mse_loss(
                predicted_proprio_normalized, normalized_proprio[:, 1:]
            )
            object_loss = F.mse_loss(predicted_object, object_pos[:, 1:])
            output.update(
                {
                    "loss": losses["loss"]
                    + self.cfg.loss.proprio_weight * proprio_loss
                    + self.cfg.loss.object_weight * object_loss,
                    "next_proprio_normalized": predicted_proprio_normalized,
                    "next_proprio": predicted_proprio,
                    "object_pos": predicted_object,
                    "proprio_loss": proprio_loss,
                    "object_loss": object_loss,
                    "qpos_mse": F.mse_loss(
                        predicted_proprio[..., :9], proprio[:, 1:, :9]
                    ),
                    "qvel_mse": F.mse_loss(
                        predicted_proprio[..., 9:18], proprio[:, 1:, 9:18]
                    ),
                    "tcp_mse": F.mse_loss(
                        predicted_proprio[..., 18:25], proprio[:, 1:, 18:25]
                    ),
                    "object_xyz_mse": object_loss,
                    "object_z_mse": F.mse_loss(
                        predicted_object[..., 2], object_pos[:, 1:, 2]
                    ),
                }
            )
            if self.contact_head is not None:
                contact_logits = self.contact_head(pooled).squeeze(-1)
                contact_target = cube_robot_contact[:, 1:].to(contact_logits.dtype)
                raw_contact_loss = F.binary_cross_entropy_with_logits(
                    contact_logits, contact_target, reduction="none"
                )
                contact_weights = torch.where(
                    contact_target > 0.5,
                    raw_contact_loss.new_tensor(
                        self.cfg.loss.contact_positive_weight
                    ),
                    raw_contact_loss.new_tensor(
                        self.cfg.loss.contact_negative_weight
                    ),
                )
                contact_loss = (
                    raw_contact_loss * contact_weights
                ).sum() / contact_weights.sum()
                contact_probability = contact_logits.sigmoid()
                output.update(
                    {
                        "loss": output["loss"]
                        + self.cfg.loss.contact_weight * contact_loss,
                        "contact_logits": contact_logits,
                        "contact_probability": contact_probability,
                        "contact_loss": contact_loss,
                        "contact_accuracy": (
                            (contact_probability >= 0.5) == (contact_target >= 0.5)
                        )
                        .to(contact_logits.dtype)
                        .mean(),
                    }
                )
        return output

    def predict_transition(
        self,
        latent_history: torch.Tensor,
        proprio_history_normalized: torch.Tensor,
        actions: torch.Tensor,
        env: str = None,
    ) -> TransitionPrediction:
        """Predict one transition from normalized recurrent robot history."""
        if self.proprio_encoder is None:
            raise RuntimeError("predict_transition requires enabled proprioception")
        if latent_history.shape[1:] != (self.cfg.T, self.cfg.K, self.cfg.D):
            raise ValueError(
                "latent_history must be "
                f"[B,{self.cfg.T},{self.cfg.K},{self.cfg.D}]"
            )
        env = self._env(env)
        pred_all, temporal, decoder = self._predict_sequence(
            latent_history,
            actions,
            env,
            proprio_normalized=proprio_history_normalized,
            return_features=True,
        )
        pooled = decoder[:, -1].mean(dim=1)
        assert self.next_proprio_head is not None
        assert self.object_position_head is not None
        next_normalized = self.next_proprio_head(pooled)
        return TransitionPrediction(
            scene_latent=pred_all[:, -1],
            temporal_features=temporal[:, -1],
            decoder_features=decoder[:, -1],
            next_proprio_normalized=next_normalized,
            next_proprio=self.proprio_encoder.denormalize(next_normalized),
            object_pos=self.object_position_head(pooled),
            contact_probability=None
            if self.contact_head is None
            else self.contact_head(pooled).squeeze(-1).sigmoid(),
        )

    def rollout(self, points: torch.Tensor, actions: torch.Tensor, env: str = None,
                context_actions: torch.Tensor | None = None) -> torch.Tensor:
        """points [B, T, K, C]，actions [B, H*skip_frame, A_env] -> [B, H, K, D]。
        每步把 T-1 个历史动作块与当前动作块对齐到 T 帧。"""
        assert points.dim() == 4 and points.shape[1] == self.cfg.T, f"points 应为 [B, {self.cfg.T}, K, C]，得到 {tuple(points.shape)}"
        assert actions.dim() == 3 and actions.shape[0] == points.shape[0]
        env = self._env(env)
        latent1 = self._aggregate(points)
        chunk = self.cfg.skip_frame
        if actions.shape[1] % chunk:
            raise AssertionError(f"rollout actions length must be divisible by skip_frame={chunk}")
        history_length = (self.cfg.T - 1) * chunk
        if context_actions is None:
            action_history = actions.new_zeros((actions.shape[0], history_length, actions.shape[-1]))
        else:
            assert context_actions.shape == (actions.shape[0], history_length, actions.shape[-1]), (
                f"context_actions 应为 [B, {history_length}, A]，得到 {tuple(context_actions.shape)}"
            )
            action_history = context_actions
        preds = []
        horizon = actions.shape[1] // chunk
        for h in range(horizon):
            current = actions[:, h * chunk : (h + 1) * chunk]
            step_actions = torch.cat([action_history, current], dim=1)
            pred_all, _ = self._predict_sequence(latent1, step_actions, env)
            pred = pred_all[:, -1]
            preds.append(pred)
            latent1 = torch.cat([latent1[:, 1:], pred.unsqueeze(1)], dim=1)
            action_history = step_actions[:, chunk:]
        out = torch.stack(preds, dim=1)
        assert out.shape == (points.shape[0], horizon, self.cfg.K, self.cfg.D)
        return out

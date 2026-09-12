"""损失（移植自 LeWM，仓库 C）：

    预测损失 = 预测值与目标逐向量的均方误差          # 目标不截断梯度，不用 EMA 教师，梯度双向流回编码器
    正则损失 = SIGReg( 全部 T+1 帧的 latent1，重排为 (位置 = 帧×槽位, 样本, D) )
    总损失   = 预测损失 + λ · 正则损失                  # λ 默认 0.09

- SIGReg 来自 ``le-wm/module.py``（Sketched Isotropic Gaussian Regularizer，默认 17 个节点、1024 个投影）；
  它的统计量沿样本维计算、沿位置维平均，所以沿位置分块累加（同一份随机投影复用）与原实现精确等价。
  位置维扩成帧数×K 后中间张量 (位置, 样本, 投影, 节点) 会占几十 GB，因此这里分块并用 checkpoint 逐块重算，
  峰值显存只与块大小有关。
- 预测/总损失来自 ``le-wm/train.py`` 的 ``lejepa_forward``。
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class LossConfig:
    reg_weight: float = 0.09  # λ
    flow_weight: float = 0.0  # λ_flow；0 保持原始 LeWM 损失
    motion_threshold: float = 0.01
    motion_temperature: float = 0.002
    motion_floor: float = 0.25
    motion_gamma: float = 1.0
    ranking_weight: float = 0.0
    ranking_margin: float = 0.1
    ranking_motion_weighted: bool = True
    proprio_weight: float = 0.0
    object_weight: float = 0.0
    contact_weight: float = 0.0
    contact_positive_weight: float = 1.0
    contact_negative_weight: float = 1.0
    knots: int = 17
    num_proj: int = 1024
    chunk: int = 64  # 沿位置维的分块大小


@torch.no_grad()
def match_motion_weights(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float,
    temperature: float,
    floor: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutual-NN patch matching and normalized displacement weights."""
    if source.shape != target.shape or source.dim() != 4 or source.shape[-1] != 3:
        raise ValueError("source and target positions must have shape [B,T,K,3]")
    if temperature <= 0 or not 0 <= floor <= 1 or gamma <= 0:
        raise ValueError("motion_temperature > 0, motion_floor in [0,1], and motion_gamma > 0 are required")
    B, T, K, _ = source.shape
    source_flat = source.reshape(B * T, K, 3).float()
    target_flat = target.reshape(B * T, K, 3).float()
    distances = torch.cdist(source_flat, target_flat)
    forward = distances.argmin(dim=-1)
    backward = distances.argmin(dim=-2)
    source_ids = torch.arange(K, device=source.device).expand(B * T, K)
    mutual = backward.gather(1, forward).eq(source_ids)
    displacement = distances.gather(2, forward.unsqueeze(-1)).squeeze(-1)
    motion = torch.sigmoid((displacement - threshold) / temperature).pow(gamma)
    raw = mutual.to(motion.dtype) * (floor + (1.0 - floor) * motion)
    weights = raw / raw.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    return forward.view(B, T, K), weights.view(B, T, K).to(source.dtype)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer：输入 (位置, 样本, D)，把每个位置上的样本分布拉向各向同性高斯。"""

    def __init__(self, knots: int = 17, num_proj: int = 1024, chunk: int = 64):
        super().__init__()
        self.num_proj, self.chunk = num_proj, chunk
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def statistic(self, proj: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Epps–Pulley 统计量：proj (P, B, D)，A (D, num_proj) -> (P, num_proj)。"""
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (P, B, num_proj, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return (err @ self.weights) * proj.size(-2)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        assert proj.dim() == 3, f"SIGReg 输入应为 (位置, 样本, D)，得到 {tuple(proj.shape)}"
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A / A.norm(p=2, dim=0)  # 同一份随机投影供所有分块复用
        total = proj.new_zeros(())
        for piece in proj.split(self.chunk, dim=0):
            total = total + checkpoint(self.statistic, piece, A, use_reentrant=False).sum()
        return total / (proj.size(0) * self.num_proj)  # = 原实现 statistic.mean()


class LeWMLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.reg_weight = cfg.reg_weight
        self.flow_weight = cfg.flow_weight
        self.motion_threshold = cfg.motion_threshold
        self.motion_temperature = cfg.motion_temperature
        self.motion_floor = cfg.motion_floor
        self.motion_gamma = cfg.motion_gamma
        self.ranking_weight = cfg.ranking_weight
        self.ranking_margin = cfg.ranking_margin
        self.ranking_motion_weighted = cfg.ranking_motion_weighted
        self.proprio_weight = cfg.proprio_weight
        self.object_weight = cfg.object_weight
        if self.ranking_weight < 0 or self.ranking_margin < 0:
            raise ValueError("ranking_weight and ranking_margin must be non-negative")
        if self.proprio_weight < 0 or self.object_weight < 0:
            raise ValueError("proprio_weight and object_weight must be non-negative")
        if cfg.contact_weight < 0:
            raise ValueError("contact_weight must be non-negative")
        if cfg.contact_positive_weight <= 0 or cfg.contact_negative_weight <= 0:
            raise ValueError("contact class weights must be positive")
        self.sigreg = SIGReg(cfg.knots, cfg.num_proj, cfg.chunk)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        latent1_all: torch.Tensor,
        positions_all: torch.Tensor | None = None,
        negative_pred: torch.Tensor | None = None,
    ) -> dict:
        """pred / target: [B,K,D] 或 [B,T,K,D]；latent1_all: [B,T+1,K,D]。"""
        assert pred.shape == target.shape and pred.dim() in (3, 4), (
            f"pred {tuple(pred.shape)} 与 target {tuple(target.shape)} 应同为 [B,K,D] 或 [B,T,K,D]"
        )
        B, N, K, D = latent1_all.shape
        if pred.dim() == 3:
            assert pred.shape == (B, K, D)
        else:
            assert pred.shape[0] == B and pred.shape[2:] == (K, D) and pred.shape[1] == N - 1
        latent_loss = (pred - target).pow(2).mean()  # 目标不 detach
        flow_loss = latent_loss.new_zeros(())
        rank_loss = latent_loss.new_zeros(())
        pred_seq = pred.unsqueeze(1) if pred.dim() == 3 else pred
        target_seq = target.unsqueeze(1) if target.dim() == 3 else target
        matched_target = None
        weights = None
        if self.flow_weight > 0 or (self.ranking_weight > 0 and self.ranking_motion_weighted):
            if positions_all is None or positions_all.shape != (B, N, K, 3):
                raise ValueError(f"motion-weighted loss requires positions_all [B,N,K,3], got {None if positions_all is None else tuple(positions_all.shape)}")
            pos_source = positions_all[:, -2:-1] if pred.dim() == 3 else positions_all[:, :-1]
            pos_target = positions_all[:, -1:] if pred.dim() == 3 else positions_all[:, 1:]
            match, weights = match_motion_weights(
                pos_source,
                pos_target,
                threshold=self.motion_threshold,
                temperature=self.motion_temperature,
                floor=self.motion_floor,
                gamma=self.motion_gamma,
            )
            matched_target = target_seq.gather(2, match.unsqueeze(-1).expand(-1, -1, -1, D))
        if self.flow_weight > 0:
            assert matched_target is not None and weights is not None
            patch_error = (pred_seq - matched_target).pow(2).mean(dim=-1)
            flow_loss = (patch_error * weights).sum() / weights.sum().clamp_min(1e-6)
        if self.ranking_weight > 0:
            if negative_pred is None or negative_pred.shape != pred.shape:
                raise ValueError("ranking loss requires negative_pred with the same shape as pred")
            negative_seq = negative_pred.unsqueeze(1) if negative_pred.dim() == 3 else negative_pred
            ranking_target = matched_target.detach() if self.ranking_motion_weighted else target_seq.detach()
            correct_error = (pred_seq - ranking_target).pow(2).mean(dim=-1)
            negative_error = (negative_seq - ranking_target).pow(2).mean(dim=-1)
            relative_margin = self.ranking_margin * correct_error.detach()
            ranking_error = torch.relu(relative_margin + correct_error - negative_error)
            if self.ranking_motion_weighted:
                assert weights is not None
                rank_loss = (ranking_error * weights).sum() / weights.sum().clamp_min(1e-6)
            else:
                rank_loss = ranking_error.mean()
        pred_loss = (latent_loss + self.flow_weight * flow_loss) / (1.0 + self.flow_weight)
        reg_loss = self.sigreg(latent1_all.permute(1, 2, 0, 3).reshape(N * K, B, D))  # (位置 = 帧×槽位, 样本, D)
        loss = pred_loss + self.ranking_weight * rank_loss + self.reg_weight * reg_loss
        return {"loss": loss, "pred_loss": pred_loss, "latent_loss": latent_loss, "flow_loss": flow_loss,
                "rank_loss": rank_loss, "reg_loss": reg_loss}

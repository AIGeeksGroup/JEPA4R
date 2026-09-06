"""冻结的世界模型编码器（latent2）：T 帧历史观测 -> 各帧 latent1（PPRL encoder）-> 时间 Transformer -> 当前帧 latent2。

- encoder 与 temporal 全部冻结、始终 eval、前向 no_grad；
- T 帧点云拼成一个 jagged batch 只过一次 encoder（tokenizer 逐点云独立，random_start=False，结果与逐帧调用完全相同）；
- 输入键与 HistoryObsWrapper 一致：points_k / state_k，k=0 最旧，k=T-1 当前帧；输出 [B, D]，D = encoder.embed_dim。"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrozenWorldModelEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, temporal: nn.Module, num_frames: int, points_key: str = "points", state_key: str = "state") -> None:
        super().__init__()
        self.encoder, self.temporal = encoder, temporal
        self.num_frames, self.points_key, self.state_key = num_frames, points_key, state_key
        for p in self.parameters():
            p.requires_grad_(False)
        self.encoder.eval(); self.temporal.eval()

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim

    def train(self, mode: bool = True) -> "FrozenWorldModelEncoder":
        super().train(mode)
        self.encoder.eval(); self.temporal.eval()
        return self

    def forward(self, observation) -> torch.Tensor:
        with torch.no_grad():
            pos, ptr, state, offset = [], [torch.zeros(1, dtype=torch.long)], [], 0
            for k in range(self.num_frames):
                pts, st = observation[f"{self.points_key}_{k}"], observation[f"{self.state_key}_{k}"]
                p, pt = pts["pos"], pts["ptr"]
                ptr[0] = ptr[0].to(pt.device)
                pos.append(p); ptr.append(pt[1:] + offset); offset += int(pt[-1]); state.append(st)
            pos, ptr, state = torch.cat(pos, 0), torch.cat(ptr, 0), torch.cat(state, 0)
            B = state.shape[0] // self.num_frames
            lat1 = self.encoder({self.points_key: {"pos": pos, "ptr": ptr}, self.state_key: state})  # [T*B, D]
            lat1 = lat1.reshape(self.num_frames, B, -1).transpose(0, 1)  # [B, T, D]，旧 -> 新
            return self.temporal(lat1)[:, -1]

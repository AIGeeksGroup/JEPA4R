"""Stage B (Sec. 4.4): rank-4 LoRA [hu2022lora] on the query and value projections of all attention blocks of
the backbone (72 on PAGE-4D, 0.88M parameters).

``add_lora(module, rank)`` wraps every fused qkv projection under ``module`` in ``LoRAQV`` (update restricted to the q
and v slices); the backbone decoders call it on their attention blocks (recon branch's lora.py, moved unchanged).
"""
import torch
from torch import nn


# Moved from lewm-pcd-recon-page4d@d3c2d07:lewm_pcd/recon/decoders/lora.py:LoRAQV, add_lora (lines 7-36; verbatim)
class LoRAQV(nn.Module):
    """LoRA on the fused qkv projection, update restricted to the q and v slices: W x + s * [B_q; 0; B_v] (A x).
    The pretrained qkv weight is untouched; A is random, B is zero, so training starts exactly at the pretrained model."""

    def __init__(self, qkv, rank, alpha=None):
        super().__init__()
        self.qkv, self.rank, self.scale = qkv, rank, (alpha or rank) / rank
        self.dim = qkv.in_features
        self.down = nn.Linear(self.dim, rank, bias=False)
        self.up_q = nn.Linear(rank, self.dim, bias=False)
        self.up_v = nn.Linear(rank, self.dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up_q.weight); nn.init.zeros_(self.up_v.weight)

    def forward(self, x):
        h = self.down(x)
        delta = torch.cat([self.up_q(h), torch.zeros_like(h[..., :1]).expand(*h.shape[:-1], self.dim), self.up_v(h)], dim=-1)
        return self.qkv(x) + self.scale * delta


def add_lora(module, rank):
    """Wrap every attention's fused qkv in the given blocks; returns the wrappers."""
    wrappers = []
    for name, child in list(module.named_children()):
        if name == 'qkv' and isinstance(child, nn.Linear):
            w = LoRAQV(child, rank)
            setattr(module, name, w); wrappers.append(w)
        else:
            wrappers += add_lora(child, rank)
    return wrappers

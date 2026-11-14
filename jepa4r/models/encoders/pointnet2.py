"""PointNet++ [qi2017pointnetpp] for the Tab. 4b ablation, sized to ~5.51M parameters.

Original module docstring:
PointNet++ (Qi et al., NeurIPS 2017): the single-scale-grouping classifier of the authors' code
(charlesq34/pointnet2: models/pointnet2_cls_ssg.py, utils/pointnet_util.py, tf_ops), in plain PyTorch.

Up to the global feature it is the original:
  SA1  farthest-point sampling of 512 centres (the authors' kernel: the first centre is point 0), ball query of
       radius 0.2: the first 32 points in index order with distance < 0.2, the slots left filled with the first of
       them; per neighbour [xyz - centre, its features] -> shared MLP [64, 64, 128] -> max over the neighbours
  SA2  128 centres, radius 0.4, 64 neighbours, MLP [128, 128, 256] on [xyz - centre, SA1 features]
  SA3  one group of all 128 points, MLP [256, 512, 1024] on [xyz, SA2 features] -> max -> the global feature
  MLP layer = 1x1 conv with bias -> BatchNorm -> ReLU (tf_util.conv2d: Xavier-uniform weights, zero biases);
  BatchNorm with PyTorch's defaults (eps 1e-5, momentum 0.1), as the PyTorch ports.
Input (B, N, C): xyz in the first 3 channels, the other C - 3 (rgb) as point features, which the authors' network
takes the same way (its classifier with normals). Its fully connected classification layers are not part of the
encoder.

Adaptations to LeWM: a linear head from the global feature to out_dim (the 192-d latent), and the MLP widths x 2.5
(ratios kept), so that the parameter count matches the pixel model's ViT-tiny encoder: 5.51 M (5.50 M). The radii
are the authors' (their objects are normalised to the unit sphere; the JEPA4R point clouds, (xyz - centre) x 4, have
about that extent).

Computation: the first layer of each set abstraction is a linear map of [xyz_j - centre, feat_j], evaluated as W_xyz
xyz_j + W_feat feat_j (per point) + b - W_xyz centre (per centre), the same map with fewer products, in float32. Memory:
in training every BatchNorm normalises over the whole batch (all clouds, centres and neighbours), and the grouped
activations of a 128 x 4-frame batch do not fit in GPU memory. The centres are therefore processed in chunks under
activation checkpointing; each BatchNorm's batch statistics are summed over all chunks first (one pass per layer) and
then applied, so the result is the unchunked network's up to floating-point summation order, and the running statistics
are updated once per step as nn.BatchNorm does. In eval mode the running statistics are used. With bn_group
(config/train/model/encoder/pointnet2.yaml: 32; the authors trained PointNet++ with batches of 16) the statistics are
per group of that many clouds instead: each group is computed in one piece under activation checkpointing, and the
running statistics are updated once per group.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class ChunkedSharedMLP(nn.Module):
    """[1x1 conv (= Linear on the last dim) -> BatchNorm -> activation] for each width, on positions (..., C).
    forward(make, chunks, reduce): make(chunk) -> that chunk's layer-0 input (..., C0), recomputed when needed (with
    premade=True: layer 0's linear output itself, which the caller computes with lins[0]'s weights); chunks:
    [(chunk, number of positions in it)]; reduce(y) -> the chunk's result from its last activation.
    In training the BatchNorm statistics are those of all chunks' positions together (see the module docstring):
    one chunk is computed once, in one piece; several are processed under activation checkpointing, each layer's
    per-chunk mean and variance combined exactly over the chunks before the layer is applied."""

    def __init__(
        self,
        in_dim: int,
        widths: Sequence[int],
        bias: bool = True,
        act: str = "relu",
        eps: float = 1e-5,
        momentum: float = 0.1,
        xavier: bool = False,
        premade: bool = False,
    ):
        super().__init__()
        self.lins, self.bns = nn.ModuleList(), nn.ModuleList()
        d = in_dim
        for h in widths:
            lin = nn.Linear(d, h, bias=bias)
            if xavier:
                nn.init.xavier_uniform_(lin.weight)
                if bias:
                    nn.init.zeros_(lin.bias)
            self.lins.append(lin)
            self.bns.append(nn.BatchNorm1d(h, eps=eps, momentum=momentum))
            d = h
        self.act, self.premade = act, premade

    def _pre(self, i, x):  # layer i's linear output
        return x if (i == 0 and self.premade) else self.lins[i](x)

    def _bn_act(self, i, y, stat):  # BatchNorm with the given (mean, var) as one multiply-add, then the activation
        bn = self.bns[i]
        mean, var = stat
        a = bn.weight * torch.rsqrt(var + bn.eps)
        y = torch.addcmul(bn.bias - mean * a, y, a)  # float32
        return F.relu(y, inplace=True) if self.act == "relu" else F.leaky_relu(y, 0.2, inplace=True)

    def _update(self, i, mean, var, m):  # nn.BatchNorm's running update (unbiased variance)
        bn = self.bns[i]
        with torch.no_grad():
            bn.num_batches_tracked += 1
            bn.running_mean.mul_(1 - bn.momentum).add_(mean.detach(), alpha=bn.momentum)
            bn.running_var.mul_(1 - bn.momentum).add_(var.detach() * (m / max(m - 1, 1)), alpha=bn.momentum)

    def _stats(self, make, chunk, stats, i):  # mean and (biased) variance of layer i's linear output on a chunk
        x = make(chunk)
        for j in range(i):
            x = self._bn_act(j, self._pre(j, x), stats[j])
        var, mean = torch.var_mean(self._pre(i, x).float().flatten(0, -2), dim=0, correction=0)
        return mean, var

    def _out(self, make, chunk, stats, reduce):
        x = make(chunk)
        for j in range(len(self.lins)):
            x = self._bn_act(j, self._pre(j, x), stats[j])
        return reduce(x)

    def piece(self, x, reduce):
        """Training mode in one piece: every BatchNorm normalises over all positions of x (layer-0 input, or layer 0's
        linear output with premade). Returns reduce(last activation) and the per-layer (mean, variance); the running
        statistics are not touched (update_running), so that a recomputation under checkpointing does not repeat it."""
        stats = []
        for i in range(len(self.lins)):
            y = self._pre(i, x)
            var, mean = torch.var_mean(y.float().flatten(0, -2), dim=0, correction=0)
            stats.append((mean, var))
            x = self._bn_act(i, y, (mean, var))
        return reduce(x), stats

    def update_running(self, stats, m):
        for i, (mean, var) in enumerate(stats):
            self._update(i, mean, var, m)

    def forward(self, make, chunks, reduce):
        if not self.training:
            stats = [(bn.running_mean, bn.running_var) for bn in self.bns]
            return [self._out(make, c, stats, reduce) for c, _ in chunks]
        if len(chunks) == 1:  # in one piece, as nn.BatchNorm in training
            ((c, m),) = chunks
            out, stats = self.piece(make(c), reduce)
            self.update_running(stats, m)
            return [out]
        run = (
            (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if torch.is_grad_enabled() else (lambda f, *a: f(*a))
        )
        ns = [float(n) for _, n in chunks]
        m = sum(ns)
        stats = []
        for i in range(len(self.lins)):
            parts = [run(self._stats, make, c, stats, i) for c, _ in chunks]
            mean = sum(n * mu for n, (mu, _) in zip(ns, parts)) / m
            var = sum(n * (v + (mu - mean) ** 2) for n, (mu, v) in zip(ns, parts)) / m  # exact pooled variance
            self._update(i, mean, var, m)
            stats.append((mean, var))
        return [run(self._out, make, c, stats, reduce) for c, _ in chunks]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """The authors' farthestpointsampling kernel: the first index is 0, then repeatedly the point with the largest
    squared distance to the chosen set. xyz (B, N, 3) -> (B, npoint)."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e38, device=xyz.device)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = far
        dist = torch.minimum(dist, ((xyz - xyz[ar, far][:, None]) ** 2).sum(-1))
        far = dist.argmax(-1)
    return idx


def ball_query(xyz: torch.Tensor, centres: torch.Tensor, radius: float, nsample: int, chunk: int = 64) -> torch.Tensor:
    """The authors' query_ball_point kernel: per centre the first nsample points in index order with distance
    < radius, the slots left filled with the first of them. xyz (B, N, 3), centres (B, S, 3) -> (B, S, nsample)."""
    N = xyz.shape[1]
    ar = torch.arange(N, device=xyz.device)
    out = []
    for s in range(0, centres.shape[1], chunk):
        d2 = ((centres[:, s : s + chunk, None] - xyz[:, None]) ** 2).sum(-1)  # (B, s, N)
        key = torch.where(d2 < radius * radius, ar, N)
        first = key.topk(nsample, dim=-1, largest=False, sorted=True).values
        out.append(torch.where(first == N, first[..., :1], first))
    return torch.cat(out, 1)


def _gather(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:  # t (B, N, C), idx (B, ...) -> (B, ..., C)
    b = torch.arange(t.shape[0], device=t.device).view(-1, *[1] * (idx.dim() - 1))
    return t[b, idx]


def _per_group(mlp, n_clouds, group, make, reduce, positions_per_cloud):
    """Training with BatchNorm statistics per group of `group` clouds: the clouds in order, group by group, each in one
    piece under activation checkpointing, its running statistics updated once per group (as nn.BatchNorm called on
    the groups one after the other). make(slice of clouds) -> the group's layer-0 input; returns the groups' results
    concatenated over the clouds."""
    outs = []
    for b in range(0, n_clouds, group):
        sl = slice(b, min(b + group, n_clouds))

        def fn(sl=sl):
            return mlp.piece(make(sl), reduce)

        out, stats = checkpoint(fn, use_reentrant=False) if torch.is_grad_enabled() else fn()
        mlp.update_running(stats, (sl.stop - sl.start) * positions_per_cloud)
        outs.append(out)
    return torch.cat(outs, 0)


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_feat, widths, chunk_elems, bn_group=None):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.chunk_elems = npoint, radius, nsample, chunk_elems
        self.bn_group = bn_group
        self.mlp = ChunkedSharedMLP(3 + in_feat, widths, bias=True, act="relu", xavier=True, premade=True)
        self.width = max(widths)

    def forward(self, xyz, feat):  # xyz (B, N, 3), feat (B, N, C) or None -> centres (B, S, 3), (B, S, C')
        with torch.no_grad():
            centres = _gather(xyz, farthest_point_sample(xyz, self.npoint))
            nidx = ball_query(xyz, centres, self.radius, self.nsample)  # (B, S, K)
        B, S, K = nidx.shape
        # the first layer's linear map on [xyz_j - centre, feat_j] = W_xyz xyz_j + W_feat feat_j - W_xyz centre + b:
        # per point and per centre (K times fewer products), in float32 (no cancellation in xyz_j - centre)
        lin = self.mlp.lins[0]
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            w = lin.weight.float()
            per_point = xyz @ w[:, :3].t() + (feat.float() @ w[:, 3:].t() if feat is not None else 0)
            per_centre = lin.bias.float() - centres @ w[:, :3].t()

        def make(sl):  # = lins[0]([xyz_j - centre, feat_j]) for the chunk's centres
            return _gather(per_point, nidx[:, sl]) + per_centre[:, sl, None]

        if self.training and self.bn_group and B > self.bn_group:
            out = _per_group(
                self.mlp,
                B,
                self.bn_group,
                lambda b: _gather(per_point[b], nidx[b]) + per_centre[b][:, :, None],
                lambda y: y.max(dim=2).values,
                S * K,
            )
            return centres, out
        step = max(1, self.chunk_elems // (B * K * self.width))
        chunks = [(slice(s, s + step), B * K * len(range(s, min(s + step, S)))) for s in range(0, S, step)]
        out = self.mlp(make, chunks, lambda y: y.max(dim=2).values)
        return centres, torch.cat(out, 1)


class SetAbstractionAll(nn.Module):
    def __init__(self, in_feat, widths, bn_group=None):
        super().__init__()
        self.mlp = ChunkedSharedMLP(3 + in_feat, widths, bias=True, act="relu", xavier=True)
        self.bn_group = bn_group

    def forward(self, xyz, feat):  # one group of all points: [xyz, feat] -> max over the points
        x = torch.cat([xyz, feat], -1)
        if self.training and self.bn_group and x.shape[0] > self.bn_group:
            return _per_group(
                self.mlp, x.shape[0], self.bn_group, lambda b: x[b], lambda y: y.max(dim=1).values, x.shape[1]
            )
        return self.mlp(lambda _: x, [(None, x.shape[0] * x.shape[1])], lambda y: y.max(dim=1).values)[0]


class PointNet2(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        npoints: Sequence[int] = (512, 128),
        radii: Sequence[float] = (0.2, 0.4),
        nsamples: Sequence[int] = (32, 64),
        mlps: Sequence[Sequence[int]] = ((64, 64, 128), (128, 128, 256), (256, 512, 1024)),
        out_dim: int = 192,
        chunk_elems: int = 2**27,
        bn_group: int | None = None,
    ):
        """chunk_elems: positions x the widest layer per chunk in training (memory only; the result does not
        depend on it). bn_group: in training, BatchNorm statistics per group of this many clouds (the batch's clouds
        in order) instead of over the whole batch; None = the whole batch, as nn.BatchNorm on the batch."""
        super().__init__()
        f = in_dim - 3
        self.sa1 = SetAbstraction(npoints[0], radii[0], nsamples[0], f, mlps[0], chunk_elems, bn_group)
        self.sa2 = SetAbstraction(npoints[1], radii[1], nsamples[1], mlps[0][-1], mlps[1], chunk_elems, bn_group)
        self.sa3 = SetAbstractionAll(mlps[1][-1], mlps[2], bn_group)
        self.head = nn.Linear(mlps[2][-1], out_dim)

    def forward(self, x):
        """x: (B, N, C) -> (B, out_dim)"""
        x = x.float()
        xyz = x[..., :3].contiguous()
        feat = x[..., 3:] if x.shape[-1] > 3 else None
        xyz, feat = self.sa1(xyz, feat)
        xyz, feat = self.sa2(xyz, feat)
        return self.head(self.sa3(xyz, feat))


def build(enc_cfg):
    """configs/encoder/pointnet2.yaml (hparams = LeWM's PointNet++ encoder config)."""
    h = enc_cfg.hparams
    return PointNet2(
        in_dim=int(enc_cfg.in_dim),
        npoints=list(h.npoints),
        radii=list(h.radii),
        nsamples=list(h.nsamples),
        mlps=[list(m) for m in h.mlps],
        out_dim=int(enc_cfg.out_dim),
        bn_group=int(h.bn_group),
    )

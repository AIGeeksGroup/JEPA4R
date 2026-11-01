"""DGCNN (Wang et al., ACM TOG 2019), the authors' PyTorch classifier (WangYueFt/dgcnn, pytorch/model.py, class
DGCNN), in plain PyTorch with the same layers.

Up to the global feature it is the original:
  EdgeConv x 4  per point its k = 20 nearest neighbours (squared Euclidean distance on the layer's input, the point
                itself included); edge feature [neighbour - point, point] -> 1x1 conv (no bias) -> BatchNorm ->
                LeakyReLU(0.2) -> max over the neighbours; widths 64, 64, 128, 256; each layer's neighbours from
                its own input (the dynamic graph)
  conv5         the four outputs concatenated (512) -> 1x1 conv (no bias) -> BatchNorm -> LeakyReLU(0.2), emb 1024
  pooling       [max over the points, mean over the points] -> the global feature (2 x emb)
  PyTorch default initialisation and BatchNorm (eps 1e-5, momentum 0.1), as the authors' code.
Input (B, N, C): xyz in the first 3 channels, then the other point features (rgb). The first layer's neighbours are
found on xyz only and its edge features use all C channels, as the authors' segmentation network does with its
extra input channels (tensorflow/sem_seg/model.py: the distances on the coordinates, the edge features on all nine
channels). The classifier's fully connected layers after the pooling are not part of the encoder.

Adaptations to LeWM: a linear head from the global feature to out_dim (the 192-d latent), and all widths x 43/16
(ratios kept: 172, 172, 344, 688, emb 2752), so that the parameter count matches the pixel model's ViT-tiny
encoder: 5.50 M (5.50 M).

Computation: each EdgeConv is evaluated without materialising its (point, neighbour) pairs (see _edge_conv:
the max over the neighbours commutes with BatchNorm + LeakyReLU per channel, and the batch statistics over
the pairs have a closed form); its 1x1 conv of [x_j - x_i, x_i] as W_a x_j + (W_b - W_a) x_i, in float32. conv5
is processed in chunks of points under activation checkpointing, its BatchNorm statistics combined exactly over
the chunks (ChunkedSharedMLP). In training the BatchNorms normalise over the whole batch, so the result is
the authors' network's up to floating-point rounding, and the running statistics are updated once per step as
nn.BatchNorm does; in eval mode the running statistics are used.
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
    (The same class as in encoders/pointnet2.py, repeated so that each encoder file loads on its own.)
    In training the BatchNorm statistics are those of all chunks' positions together (see the module docstring):
    one chunk is computed once, in one piece; several are processed under activation checkpointing, each layer's
    per-chunk mean and variance combined exactly over the chunks before the layer is applied."""

    def __init__(self, in_dim: int, widths: Sequence[int], bias: bool = True, act: str = 'relu',
                 eps: float = 1e-5, momentum: float = 0.1, xavier: bool = False, premade: bool = False):
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
        return F.relu(y, inplace=True) if self.act == 'relu' else F.leaky_relu(y, 0.2, inplace=True)

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

    def forward(self, make, chunks, reduce):
        if not self.training:
            stats = [(bn.running_mean, bn.running_var) for bn in self.bns]
            return [self._out(make, c, stats, reduce) for c, _ in chunks]
        if len(chunks) == 1:  # in one piece, as nn.BatchNorm in training
            (c, m), = chunks
            x = make(c)
            for i in range(len(self.lins)):
                y = self._pre(i, x)
                var, mean = torch.var_mean(y.float().flatten(0, -2), dim=0, correction=0)
                self._update(i, mean, var, m)
                x = self._bn_act(i, y, (mean, var))
            return [reduce(x)]
        run = (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if torch.is_grad_enabled() else (lambda f, *a: f(*a))
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


def knn(x: torch.Tensor, k: int, chunk: int = 256) -> torch.Tensor:
    """The authors' knn: the k points with the smallest squared Euclidean distance (the point itself included),
    computed in float32. x (B, N, C) -> (B, N, k)."""
    with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        xx = (x * x).sum(-1)
        out = []
        for s in range(0, x.shape[1], chunk):
            d = xx[:, s:s + chunk, None] - 2 * x[:, s:s + chunk] @ x.transpose(1, 2) + xx[:, None]
            out.append(d.topk(k, dim=-1, largest=False).indices)
        return torch.cat(out, 1)


def _gather(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:  # t (B, N, C), idx (B, ...) -> (B, ..., C)
    b = torch.arange(t.shape[0], device=t.device).view(-1, *[1] * (idx.dim() - 1))
    return t[b, idx]


def _neighbour_argmax_argmin(v: torch.Tensor, idx: torch.Tensor, chunk_elems: int):
    """Per point i and channel: the neighbour j in idx[i] with the largest and the smallest v_j. v (B, N, C),
    idx (B, N, k) -> two (B, N, C) index tensors into v's points (computed in chunks of points)."""
    B, N, C = v.shape
    k = idx.shape[-1]
    step = max(1, chunk_elems // (B * k * C))
    jmax, jmin = [], []
    for s in range(0, N, step):
        ix = idx[:, s:s + step]
        g = _gather(v, ix)  # (B, n, k, C)
        jmax.append(torch.take_along_dim(ix[..., None], g.argmax(dim=2)[:, :, None], dim=2)[:, :, 0])
        jmin.append(torch.take_along_dim(ix[..., None], g.argmin(dim=2)[:, :, None], dim=2)[:, :, 0])
    return torch.cat(jmax, 1), torch.cat(jmin, 1)


def _adjacency(idx: torch.Tensor):
    """The neighbour graph idx (B, N, k) as sparse (B N x B N) matrices (the graph and its transpose): row b N + i of
    the graph has ones at the columns b N + j of i's neighbours j."""
    B, N, k = idx.shape
    rows = torch.arange(B * N, device=idx.device).repeat_interleave(k)
    cols = (idx + torch.arange(B, device=idx.device)[:, None, None] * N).reshape(-1)
    ones = torch.ones(rows.numel(), device=idx.device)
    return (torch.sparse_coo_tensor(torch.stack([rows, cols]), ones, (B * N, B * N)).coalesce(),
            torch.sparse_coo_tensor(torch.stack([cols, rows]), ones, (B * N, B * N)).coalesce())


class _NeighbourSum(torch.autograd.Function):
    """S_i = sum of v_j over i's neighbours j as a sparse-dense product, its gradient the transposed product; both in
    float32 with autocast off (the trainer runs the backward pass inside autocast, and sparse products have no
    bfloat16 kernel)."""

    @staticmethod
    def forward(ctx, v, adj):
        ctx.adj_t = adj[1]
        B, N, C = v.shape
        with torch.autocast(device_type=v.device.type, enabled=False):
            return torch.sparse.mm(adj[0], v.float().reshape(B * N, C)).view(B, N, C)

    @staticmethod
    def backward(ctx, g):
        B, N, C = g.shape
        with torch.autocast(device_type=g.device.type, enabled=False):
            return torch.sparse.mm(ctx.adj_t, g.float().reshape(B * N, C)).view(B, N, C), None


def _neighbour_sum(v: torch.Tensor, adj) -> torch.Tensor:
    """S_i = sum of v_j over i's neighbours j. v (B, N, C), adj from _adjacency -> (B, N, C)."""
    return _NeighbourSum.apply(v, adj)


class DGCNN(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        k: int = 20,
        widths: Sequence[int] = (64, 64, 128, 256),
        emb_dim: int = 1024,
        out_dim: int = 192,
        chunk_elems: int = 2 ** 27,
    ):
        """chunk_elems: positions x the widest channel count per chunk in training (memory only; the result does
        not depend on it)."""
        super().__init__()
        self.k, self.chunk_elems = k, chunk_elems
        self.edge = nn.ModuleList()
        d = in_dim
        for w in widths:
            self.edge.append(ChunkedSharedMLP(2 * d, [w], bias=False, act='leaky'))  # its weights; computed in _edge_conv
            d = w
        self.conv5 = ChunkedSharedMLP(sum(widths), [emb_dim], bias=False, act='leaky')
        self.head = nn.Linear(2 * emb_dim, out_dim)

    def _chunks(self, B, N, per_point, width):
        step = max(1, self.chunk_elems // (B * per_point * width))
        return [(slice(s, s + step), B * per_point * len(range(s, min(s + step, N)))) for s in range(0, N, step)]

    def _edge_conv(self, mlp, x, idx):
        """EdgeConv on x (B, N, C) with neighbours idx (B, N, k) -> (B, N, C'). Per pair (i, j): y_ij = W [x_j - x_i,
        x_i] = nbr_j + ctr_i (nbr = W_a x, ctr = (W_b - W_a) x); BatchNorm over all B N k pairs; LeakyReLU; max over j.
        BatchNorm then LeakyReLU is monotone in each channel (increasing where the BatchNorm scale
        a = gamma / sqrt(var + eps) is >= 0, decreasing where it is < 0), so the max over j of the activations is the
        activation of the normalised max_j y_ij = ctr_i + max_j nbr_j (a >= 0) or min_j y_ij (a < 0); the batch mean
        and variance over all pairs follow exactly from nbr, ctr, the in-degrees and S_i = sum_{j in N(i)} nbr_j. The
        same values and gradients as with the (B, N, k, C') pairs materialised, without them. Channels are independent
        throughout, so they are processed in groups (activation checkpointing in training; the running statistics
        updated outside the recomputed part, once per step)."""
        B, N, C = x.shape
        k = idx.shape[-1]
        bn, weight = mlp.bns[0], mlp.lins[0].weight
        train = self.training
        if train:
            with torch.no_grad():
                deg = torch.zeros(B, N, device=x.device).scatter_add_(1, idx.flatten(1),
                                                                      torch.ones(B, N * k, device=x.device))
                adj = _adjacency(idx)
        step = max(1, self.chunk_elems // (B * N))
        outs = []
        for c0 in range(0, weight.shape[0], step):
            g = slice(c0, c0 + step)

            def group(x, g=g):
                with torch.autocast(device_type=x.device.type, enabled=False):  # float32 (no cancellation)
                    w = weight[g].float()
                    nbr, ctr = x @ w[:, :C].t(), x @ (w[:, C:] - w[:, :C]).t()
                    with torch.no_grad():
                        jmax, jmin = _neighbour_argmax_argmin(nbr, idx, self.chunk_elems)
                    ymax, ymin = nbr.gather(1, jmax) + ctr, nbr.gather(1, jmin) + ctr
                    if train:
                        m = B * N * k
                        alpha, cbar = (deg[..., None] * nbr).sum((0, 1)) / m, ctr.mean((0, 1))
                        u, v = nbr - alpha, ctr - cbar  # centred: their sums over the pairs are both 0
                        m2 = ((deg[..., None] * u * u).sum((0, 1)) + 2 * (v * _neighbour_sum(u, adj)).sum((0, 1))
                              + k * (v * v).sum((0, 1)))
                        mean, var = alpha + cbar, m2 / m  # batch mean and (biased) variance over all pairs
                    else:
                        mean, var = bn.running_mean[g], bn.running_var[g]
                    a = bn.weight[g] * torch.rsqrt(var + bn.eps)
                    out = F.leaky_relu(torch.where(a >= 0, ymax, ymin) * a + (bn.bias[g] - mean * a), 0.2)
                return (out, mean, var) if train else out

            if not train:
                outs.append(group(x))
                continue
            out, mean, var = (checkpoint(group, x, use_reentrant=False) if torch.is_grad_enabled() else group(x))
            m = B * N * k
            with torch.no_grad():  # nn.BatchNorm's running update (unbiased variance), this group's channels
                bn.running_mean[g] = (1 - bn.momentum) * bn.running_mean[g] + bn.momentum * mean.detach()
                bn.running_var[g] = (1 - bn.momentum) * bn.running_var[g] + bn.momentum * var.detach() * (m / (m - 1))
            outs.append(out)
        if train:
            with torch.no_grad():
                bn.num_batches_tracked += 1
        return torch.cat(outs, -1)

    def forward(self, x):
        """x: (B, N, C) -> (B, out_dim)"""
        x = x.float()
        feats, h = [], x
        for i, mlp in enumerate(self.edge):
            idx = knn(h[..., :3] if i == 0 else h, self.k)  # the first layer's neighbours on xyz
            h = self._edge_conv(mlp, h, idx)
            feats.append(h)
        h = torch.cat(feats, -1)  # (B, N, sum(widths))
        B, N, _ = h.shape
        chunks = self._chunks(B, N, 1, max(h.shape[-1], self.conv5.lins[0].out_features))
        parts = self.conv5(lambda sl: h[:, sl], chunks, lambda y: torch.stack([y.max(dim=1).values, y.sum(dim=1)]))
        mx = torch.stack([p[0] for p in parts]).max(dim=0).values
        mean = torch.stack([p[1] for p in parts]).sum(dim=0) / N
        return self.head(torch.cat([mx, mean], -1))

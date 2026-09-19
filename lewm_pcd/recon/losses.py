"""Point-set losses for the reconstruction decoders. Everything here works on xyz alone; colour (channels 3:)
is only ever compared between already-matched pairs, and only when both sides carry it.

All functions run in float32 regardless of the surrounding autocast (coordinates lose ~1 mm in bfloat16).
"""

import math

import torch
from torch.utils.checkpoint import checkpoint


def _fp32(*tensors):
    return [t.float() for t in tensors]


def _nn_index(a, b, chunk=256):
    """argmin_j ||a_i - b_j|| over xyz for every i, without autograd memory. a (B, N, C), b (B, M, C) -> (B, N)."""
    with torch.no_grad():
        return torch.cat([torch.cdist(a[:, s:s + chunk, :3], b[..., :3]).argmin(dim=-1)
                          for s in range(0, a.size(1), chunk)], dim=1)


def _gather(x, idx):
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def chamfer(pred, gt, color_weight=0.1, chunk=256):
    """Symmetric Chamfer on xyz (squared distances) + L1 colour on the matched pairs.

    The nearest-neighbour search runs under ``no_grad`` and only the matched pairs are differentiated: the
    argmin is locally constant, so this is the gradient of the min at O(B*N) instead of O(B*N*M) memory."""
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred, gt = _fp32(pred, gt)
        g_of_p = _gather(gt, _nn_index(pred, gt, chunk))                   # nearest real point of every decoded point
        p_of_g = _gather(pred, _nn_index(gt, pred, chunk))                 # nearest decoded point of every real point
        d_pg = (pred[..., :3] - g_of_p[..., :3]).pow(2).sum(-1).mean()
        d_gp = (gt[..., :3] - p_of_g[..., :3]).pow(2).sum(-1).mean()
        loss = d_pg + d_gp
        logs = {'cd': loss.detach()}
        if color_weight > 0 and pred.size(-1) > 3 and gt.size(-1) > 3:
            col = (pred[..., 3:] - g_of_p[..., 3:]).abs().mean() + (gt[..., 3:] - p_of_g[..., 3:]).abs().mean()
            loss = loss + color_weight * col
            logs['col'] = col.detach()
        return loss, logs


def sliced_wasserstein(pred, gt, n_proj=100):
    """Squared sliced Wasserstein-2 distance between the xyz of two point sets (Bonneel et al. 2015; as a
    point-cloud reconstruction loss: Nguyen et al., ICCV 2021).

    Both sets are projected on ``n_proj`` random directions and sorted; on a line the optimal one-to-one
    matching pairs equal ranks. Unlike a nearest-neighbour match no real point can share its partner, so a
    region holding a fifth of the real points must receive a fifth of the decoded ones. Sets of different
    sizes are compared at the quantiles of the larger one."""
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, g = _fp32(pred[..., :3], gt[..., :3])
        theta = torch.nn.functional.normalize(torch.randn(n_proj, 3, device=p.device), dim=-1)
        sp = (p @ theta.T).sort(dim=1).values.transpose(1, 2)              # (B, n_proj, N)
        sg = (g @ theta.T).sort(dim=1).values.transpose(1, 2)              # (B, n_proj, M)
        if sp.size(-1) != sg.size(-1):
            n = max(sp.size(-1), sg.size(-1))
            resample = lambda s: s if s.size(-1) == n else torch.nn.functional.interpolate(s, size=n, mode='linear', align_corners=True)
            sp, sg = resample(sp), resample(sg)
        return 3.0 * (sp - sg).pow(2).mean()                               # x3: E[(theta . d)^2] = |d|^2 / 3, same units as Chamfer


@torch.no_grad()
def nn_distance(pred, gt, chunk=256):
    """Mean of the two one-sided mean nearest-neighbour distances (not squared), per sample, in the input's units."""
    pred, gt = _fp32(pred, gt)
    a = (pred[..., :3] - _gather(gt[..., :3], _nn_index(pred, gt, chunk))).norm(dim=-1).mean(-1)
    b = (gt[..., :3] - _gather(pred[..., :3], _nn_index(gt, pred, chunk))).norm(dim=-1).mean(-1)
    return 0.5 * (a + b)


def quat_to_rot(q):
    """(..., 4) unit quaternion wxyz -> (..., 3, 3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def _mixture_terms(mu, log_s, quat, weight_logit, rgb, gt):
    """Per-sample sums of -log p(x) and of the colour error, for one chunk of the batch."""
    R = quat_to_rot(quat)                                                  # (B, K, 3, 3) gaussian frame -> world
    d = gt[:, :, None, :3] - mu[:, None]                                   # (B, N, K, 3)
    local = torch.einsum('bnkj,bkji->bnki', d, R)                          # R^T d
    maha = (local / log_s.exp()[:, None]).pow(2).sum(-1)                   # (B, N, K)
    logp = (torch.log_softmax(weight_logit, dim=-1) - log_s.sum(-1))[:, None] - 0.5 * maha - 1.5 * math.log(2 * math.pi)
    nll = -torch.logsumexp(logp, dim=-1).sum()
    if rgb is None:
        return nll, nll.new_zeros(())
    expected = torch.einsum('bnk,bkc->bnc', torch.softmax(logp, dim=-1), rgb)   # colour the mixture assigns to each real point
    return nll, (expected - gt[..., 3:]).abs().sum()


def gaussian_nll(g, gt, color_weight=0.1, spread_weight=1.0, chunk=16):
    """Negative log-likelihood of the real points under the decoded mixture of 3D Gaussians, + L1 between the
    mixture's expected colour and the real colour, + a one-sided Chamfer that keeps every mean on the geometry
    (components with a vanishing weight get no likelihood gradient and would drift).

    ``g``: mu (B, K, 3), log_s (B, K, 3), quat (B, K, 4), weight_logit (B, K), rgb (B, K, 3) or None.
    The (B, N, K) terms are evaluated ``chunk`` samples at a time and recomputed in the backward pass."""
    with torch.autocast(device_type=gt.device.type, enabled=False):
        gt = gt.float()
        mu, log_s, quat, wl = _fp32(g['mu'], g['log_s'], g['quat'], g['weight_logit'])
        rgb = g['rgb'].float() if g.get('rgb') is not None and gt.size(-1) > 3 else None
        nll, col = gt.new_zeros(()), gt.new_zeros(())
        for s in range(0, gt.size(0), chunk):
            sl = slice(s, s + chunk)
            args = (mu[sl], log_s[sl], quat[sl], wl[sl], None if rgb is None else rgb[sl], gt[sl])
            a, b = checkpoint(_mixture_terms, *args, use_reentrant=False) if torch.is_grad_enabled() else _mixture_terms(*args)
            nll, col = nll + a, col + b
        n = gt.size(0) * gt.size(1)
        nll, col = nll / n, col / (n * 3)
        spread = (mu - _gather(gt[..., :3], _nn_index(mu, gt))).pow(2).sum(-1).mean()
        loss = nll + spread_weight * spread
        logs = {'nll': nll.detach(), 'spread': spread.detach()}
        if rgb is not None and color_weight > 0:
            loss = loss + color_weight * col
            logs['col'] = col.detach()
        return loss, logs


@torch.no_grad()
def sample_gaussians(g, n):
    """``n`` points drawn from the mixture (components by weight, then a Gaussian offset): (B, n, 3 or 6)."""
    w = torch.softmax(g['weight_logit'].float(), dim=-1)
    idx = torch.multinomial(w, n, replacement=True)                        # (B, n)
    mu, log_s, quat = (_gather(g[k].float(), idx) for k in ('mu', 'log_s', 'quat'))
    pts = mu + torch.einsum('bnij,bnj->bni', quat_to_rot(quat), torch.randn_like(mu) * log_s.exp())
    return pts if g.get('rgb') is None else torch.cat([pts, _gather(g['rgb'].float(), idx)], dim=-1)

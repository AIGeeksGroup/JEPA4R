"""Decoder loss, Eq. 2:  L_dec = CD(P_hat, P) + SW_2^2(P_hat, P) + 0.1 * L_rgb(P_hat, P)

CD     symmetric Chamfer distance on xyz
SW_2^2 squared sliced Wasserstein-2 distance on xyz [bonneel2015sliced] (cfg.decoder.loss.sw_projections)
L_rgb  L1 colour error of nearest-neighbour matched points

The paper's loss code (chamfer = CD + colour term, sliced_wasserstein, nn_distance = the
validation readout); ``decoder_loss`` adds them up as the backbone decoders' ``loss`` does.
"""
import torch


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
            resample = lambda s: (
                s if s.size(-1) == n else torch.nn.functional.interpolate(s, size=n, mode='linear', align_corners=True)
            )
            sp, sg = resample(sp), resample(sg)
        # x3: E[(theta . d)^2] = |d|^2 / 3, same units as Chamfer
        return 3.0 * (sp - sg).pow(2).mean()


@torch.no_grad()
def nn_distance(pred, gt, chunk=256):
    """Mean of the two one-sided mean nearest-neighbour distances (not squared), per sample, in the input's units."""
    pred, gt = _fp32(pred, gt)
    a = (pred[..., :3] - _gather(gt[..., :3], _nn_index(pred, gt, chunk))).norm(dim=-1).mean(-1)
    b = (gt[..., :3] - _gather(pred[..., :3], _nn_index(gt, pred, chunk))).norm(dim=-1).mean(-1)
    return 0.5 * (a + b)


def decoder_loss(pred, target, loss_cfg):
    """Eq. 2 for decoded points pred and the observed cloud target (B, N, 6), as PAGE4DDecoder.loss adds the terms up:
    chamfer (CD on xyz + loss_cfg.rgb_l1 x L1 colour of NN-matched pairs) + loss_cfg.sliced_w2 x SW_2^2
    (loss_cfg.sw_projections random directions) -> (loss, {'cd', 'col', 'sw'})."""
    assert float(loss_cfg.chamfer) == 1.0, 'the Chamfer term has weight 1 (Eq. 2)'
    loss, logs = chamfer(pred, target, float(loss_cfg.rgb_l1))
    if float(loss_cfg.sliced_w2) > 0:
        sw = sliced_wasserstein(pred, target, int(loss_cfg.sw_projections))
        loss, logs['sw'] = loss + float(loss_cfg.sliced_w2) * sw, sw.detach()
    return loss, logs

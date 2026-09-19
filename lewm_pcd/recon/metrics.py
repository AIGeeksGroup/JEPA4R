"""Error measures between a decoded and a real point cloud. Everything is in METRES (``to_metres`` first) and works
on xyz; an object is located through its position column in the dataset (e.g. ``privileged_block_0_pos``), never
through a hard-coded colour. Colour is used only to READ OUT where the decoder put that object, with a colour
signature measured on the real clouds of the same data; without colour that one readout is NaN, the rest stands."""

import torch

from recon.losses import _gather, _nn_index, sliced_wasserstein


def to_metres(cloud, center, scale):
    """target frame -> metres (inverse of ``utils.PointCloudPreprocessor``); colour channels untouched."""
    c = torch.as_tensor(center, dtype=cloud.dtype, device=cloud.device)
    return torch.cat([cloud[..., :3] / scale + c, cloud[..., 3:]], dim=-1)


@torch.no_grad()
def nn_both_sides(pred, gt):
    """Per sample: mean distance decoded -> nearest real point (spurious geometry), real -> nearest decoded point
    (missing geometry), and the per-point decoded -> real distances."""
    d_pg = (pred[..., :3] - _gather(gt[..., :3], _nn_index(pred, gt))).norm(dim=-1)
    d_gp = (gt[..., :3] - _gather(pred[..., :3], _nn_index(gt, pred))).norm(dim=-1)
    return d_pg.mean(-1), d_gp.mean(-1), d_pg


@torch.no_grad()
def colour_error(pred, gt):
    """Per sample L1 between the rgb of every decoded point and of its nearest real point."""
    return (pred[..., 3:] - _gather(gt[..., 3:], _nn_index(pred, gt))).abs().mean((-1, -2))


@torch.no_grad()
def region_error(pred, gt, centre, radius):
    """The part of the scene within ``radius`` of ``centre`` (B, 3), e.g. the cube or the end effector.
    Returns per sample: mean distance from the real points of the region to the decoded cloud (NaN if the region is
    empty in the real cloud), and the decoded / real point-count ratio inside the region (1 = right amount)."""
    in_gt = (gt[..., :3] - centre[:, None]).norm(dim=-1) < radius
    in_pred = (pred[..., :3] - centre[:, None]).norm(dim=-1) < radius
    d = (gt[..., :3] - _gather(pred[..., :3], _nn_index(gt, pred))).norm(dim=-1)
    n = in_gt.sum(-1)
    err = (d * in_gt).sum(-1) / n.clamp_min(1)
    err[n == 0] = float('nan')
    return err, in_pred.sum(-1) / n.clamp_min(1)


@torch.no_grad()
def colour_signature(gt, centre, radius=0.025):
    """Median rgb of the real points within ``radius`` of the object's true position, over all given frames (the
    median ignores the few gripper points that enter the ball during a grasp), and how exclusive that colour is:
    the share of real points farther than 3 x radius that still match it. None without colour / without such points."""
    if gt.size(-1) <= 3:
        return None, None
    dist = (gt[..., :3] - centre[:, None]).norm(dim=-1)
    if (dist < radius).sum() < 50:
        return None, None
    sig = gt[..., 3:][dist < radius].median(dim=0).values
    return sig, colour_match(gt, sig)[dist > 3 * radius].float().mean()


def colour_match(cloud, signature, tol=0.2):
    return (cloud[..., 3:] - signature).norm(dim=-1) < tol


@torch.no_grad()
def object_centroid(cloud, signature, min_points=5):
    """(B, N, 6) -> (B, 3): centroid of the points carrying the object's colour; NaN where fewer than ``min_points``.
    A surface centroid (visible faces only): compare it with the same readout of the real cloud, not only with the
    simulator's object centre."""
    if signature is None:
        return cloud.new_full((cloud.size(0), 3), float('nan'))
    m = colour_match(cloud, signature).float()
    n = m.sum(-1, keepdim=True)
    c = (cloud[..., :3] * m.unsqueeze(-1)).sum(1) / n.clamp_min(1)
    c[n.squeeze(-1) < min_points] = float('nan')
    return c


@torch.no_grad()
def sliced_w(pred, gt, n_proj=200):
    """Per sample root sliced-Wasserstein distance: how far mass has to move for the decoded points to be
    distributed like the real ones (also sees a region with the right shape but the wrong number of points)."""
    return torch.stack([sliced_wasserstein(p[None], g[None], n_proj).sqrt() for p, g in zip(pred, gt)])

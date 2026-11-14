"""Point-cloud metrics of Tab. 3 / Tab. 4a, in the world frame (undo the affine normalisation), reported in cm.

nn_stats(pred, gt)       bidirectional nearest-neighbour distances between decoded and observed points
                         -> per frame: nn = the mean of the two one-sided mean NN distances (decoded -> real,
                         real -> decoded), outlier_pct (share of decoded points above 3 cm); Tab. 3 reports mean /
                         median / p90 over the frames
cube_position_error      distance between the cube positions read from the two clouds
                         (rule: cfg.decoder.eval.cube_position: the centroid of the points carrying the cube's colour)
The paper's metric code (everything in metres; eval_frames / eval_rollout use it directly).
"""
import torch

from jepa4r.decoder.losses import _gather, _nn_index, sliced_wasserstein


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


@torch.no_grad()
def nn_stats(pred_xyz, gt_xyz, outlier_cm=3.0) -> dict:
    """Per frame (B, N, 3+) clouds in metres -> cm, as eval_frames computes them: nn, nn_spurious (decoded -> real),
    nn_missing (real -> decoded), outlier_pct (decoded points farther than outlier_cm from every real point, %)."""
    spurious, missing, d_pg = nn_both_sides(pred_xyz, gt_xyz)
    return {'nn': 50 * (spurious + missing), 'nn_spurious': 100 * spurious, 'nn_missing': 100 * missing,
            'outlier_pct': 100 * (d_pg > outlier_cm / 100).float().mean(-1)}


@torch.no_grad()
def cube_position_error(pred, gt, rule, centre=None, signature=None):
    """Per frame, cm: |centroid of the cube's colour in the decoded cloud - in the real cloud| (eval_frames'
    object_pos). The colour signature: the median rgb of the real points within 0.85 x rule.radius_cm of the simulator's
    cube position ``centre`` (B, 3) over all given frames (colour_signature), or ``signature`` if given."""
    if signature is None:
        signature, _ = colour_signature(gt, centre, float(rule.radius_cm) / 100 * 0.85)
    return (object_centroid(pred, signature) - object_centroid(gt, signature)).norm(dim=-1) * 100

"""Small GPU rasteriser to LOOK at what the decoders produce; nothing here takes part in training.

* ``render_points``     z-buffered perspective disc splats.
* ``render_gaussians``  EWA splatting of 3D Gaussians (3D covariance -> screen-space 2x2, front-to-back alpha
  compositing). The decoder's weight is a mass, not an opacity; the display rule used here is
  opacity = min(1, weight x K): a component carrying at least its uniform share is opaque.

Everything is in metres. Cameras look at ``target``: azimuth from +x towards +y, elevation above the table."""

import numpy as np
import torch
from PIL import Image, ImageDraw

from recon.losses import quat_to_rot

BG = (0.09, 0.10, 0.12)


def make_cam(az_deg=45.0, el_deg=40.0, dist=1.2, target=(0.36, 0.0, 0.2), fovy_deg=45.0, size=320, device='cuda'):
    """World -> camera rotation / translation and intrinsics (OpenCV axes: x right, y down, z forward)."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    pos = np.asarray(target, float) + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    fwd = np.asarray(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    f = 0.5 * size / np.tan(np.deg2rad(fovy_deg) / 2)
    return dict(R=torch.tensor(R, dtype=torch.float32, device=device), t=torch.tensor(-R @ pos, dtype=torch.float32, device=device),
                f=f, c=size / 2, size=size, device=device)


def _shade(col, z):  # cheap depth cue so the geometry reads in a still image
    return col * (1.0 - 0.45 * (z - z.min()) / (z.max() - z.min() + 1e-6)).unsqueeze(-1)


@torch.no_grad()
def render_points(cloud, cam, point_radius=0.006):
    """cloud (N, 3 or 6) metres (+ rgb in [0, 1]; grey without colour) -> (size, size, 3) float image."""
    S, dev = cam['size'], cam['device']
    cloud = torch.as_tensor(cloud, dtype=torch.float32, device=dev)
    rgb = cloud[:, 3:6].clamp(0, 1) if cloud.size(-1) > 3 else torch.full_like(cloud[:, :3], 0.75)
    pc = cloud[:, :3] @ cam['R'].T + cam['t']
    keep = pc[:, 2] > 1e-3
    pc, rgb = pc[keep], rgb[keep]
    img = torch.tensor(BG, device=dev).expand(S, S, 3).clone()
    if pc.numel() == 0:
        return img.cpu().numpy()
    z = pc[:, 2]
    u, v = cam['f'] * pc[:, 0] / z + cam['c'], cam['f'] * pc[:, 1] / z + cam['c']
    rad = (cam['f'] * point_radius / z).clamp(0.7, 12.0)
    rgb = _shade(rgb, z)

    r = int(rad.max().ceil().item())                                       # every disc's pixels at once, then one z-buffer resolve
    off = torch.arange(-r, r + 1, device=dev)
    dy, dx = (o.reshape(-1) for o in torch.meshgrid(off, off, indexing='ij'))
    ui, vi = u.round().long()[:, None] + dx, v.round().long()[:, None] + dy
    ok = ((dx * dx + dy * dy)[None] <= rad.pow(2)[:, None]) & (ui >= 0) & (ui < S) & (vi >= 0) & (vi < S)
    flat = (vi * S + ui).clamp_(0, S * S - 1)
    zz = z[:, None].expand_as(flat)
    depth = torch.full((S * S,), 1e9, device=dev).scatter_reduce_(0, flat[ok], zz[ok], reduce='amin')
    win = ok & (zz <= depth[flat])
    img.view(-1, 3).scatter_(0, flat[win].unsqueeze(-1).expand(-1, 3), rgb[:, None, :].expand(-1, flat.size(1), -1)[win])
    return img.clamp(0, 1).cpu().numpy()


@torch.no_grad()
def render_gaussians(g, cam, min_alpha=1e-3):
    """``g``: one sample of ``GaussianDecoder`` output converted by ``gaussians_to_metres`` -> (size, size, 3)."""
    S, dev = cam['size'], cam['device']
    mu, s = g['mu'].to(dev).float(), g['log_s'].to(dev).float().exp()
    w = torch.softmax(g['weight_logit'].to(dev).float(), -1)
    alpha0 = (w * w.numel()).clamp(0, 1)
    col = g['rgb'].to(dev).float().clamp(0, 1) if g.get('rgb') is not None else torch.full_like(mu, 0.75)
    Rg = quat_to_rot(g['quat'].to(dev).float())

    mc = mu @ cam['R'].T + cam['t']
    keep = mc[:, 2] > 1e-2
    mc, Rg, s, alpha0, col = mc[keep], Rg[keep], s[keep], alpha0[keep], col[keep]
    img, T = torch.zeros(S, S, 3, device=dev), torch.ones(S, S, device=dev)
    if mc.numel():
        x, y, z = mc.unbind(-1)
        cov = cam['R'] @ (Rg @ torch.diag_embed(s.pow(2)) @ Rg.transpose(-1, -2)) @ cam['R'].T
        J = torch.zeros(mc.size(0), 2, 3, device=dev)
        J[:, 0, 0] = J[:, 1, 1] = cam['f'] / z
        J[:, 0, 2], J[:, 1, 2] = -cam['f'] * x / z.pow(2), -cam['f'] * y / z.pow(2)
        S2 = J @ cov @ J.transpose(-1, -2)
        S2[:, 0, 0] += 0.3; S2[:, 1, 1] += 0.3                             # EWA low-pass
        det = (S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] * S2[:, 1, 0]).clamp_min(1e-8)
        a_, b_, c_ = S2[:, 1, 1] / det, -S2[:, 0, 1] / det, S2[:, 0, 0] / det   # inverse screen covariance
        uu, vv = cam['f'] * x / z + cam['c'], cam['f'] * y / z + cam['c']
        order = torch.argsort(z)                                           # front to back
        uu, vv, a_, b_, c_, alpha0, col = (t[order] for t in (uu, vv, a_, b_, c_, alpha0, _shade(col, z)))
        ys, xs = torch.meshgrid(torch.arange(S, device=dev, dtype=torch.float32), torch.arange(S, device=dev, dtype=torch.float32), indexing='ij')
        for i in range(0, uu.numel(), 128):                                # chunk: (k, S, S) stays small
            sl = slice(i, i + 128)
            du, dv = xs[None] - uu[sl, None, None], ys[None] - vv[sl, None, None]
            q = a_[sl, None, None] * du * du + 2 * b_[sl, None, None] * du * dv + c_[sl, None, None] * dv * dv
            a = (alpha0[sl, None, None] * torch.exp(-0.5 * q)).clamp(0, 0.999)
            a = torch.where(a > min_alpha, a, torch.zeros_like(a))
            Tk = T[None] * torch.cumprod(1 - a, dim=0) / (1 - a)           # transmittance in front of each splat
            img = img + (Tk.unsqueeze(-1) * a.unsqueeze(-1) * col[sl, None, None, :]).sum(0)
            T = T * torch.prod(1 - a, dim=0)
    return (img + T.unsqueeze(-1) * torch.tensor(BG, device=dev)).clamp(0, 1).cpu().numpy()


def gaussians_to_metres(g, center, scale, index):
    """sample ``index`` of a batched decoder output, target frame -> metres."""
    out = {k: (v[index] if v is not None else None) for k, v in g.items()}
    out['mu'] = out['mu'] / scale + torch.as_tensor(center, dtype=out['mu'].dtype, device=out['mu'].device)
    out['log_s'] = out['log_s'] - float(np.log(scale))
    return out


def label(width, text, height=26):
    im = Image.new('RGB', (width, height), (22, 24, 31))
    ImageDraw.Draw(im).text((8, height // 2), text, fill=(232, 236, 244), anchor='lm')
    return np.asarray(im, np.float32) / 255.0


def tile(rows, pad=4, bg=0.16):
    """list of rows, each a list of (h, w, 3) images of equal height -> one image."""
    rows = [np.concatenate([np.pad(i, ((0, 0), (pad, pad), (0, 0)), constant_values=bg) for i in r], 1) for r in rows]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((pad, pad), (0, w - r.shape[1]), (0, 0)), constant_values=bg) for r in rows]
    return np.concatenate(rows, 0)

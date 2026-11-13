"""PAGE-4D [zhou2026page4d] (default): DINOv2 ViT-L, 24 alternating frame / global attention layers, DPT point-map
head; ~0.94B parameters, frozen.

``PAGE4DDecoder`` is the whole decoder with this backbone (set generator -> PAGE-4D's DINOv2 + aggregator + DPT point head,
frozen, + LoRA in stage B -> affine map, rgb branch, pixel selection): the recon branch's recon/decoders/page4d.py, moved
unchanged apart from paths. PAGE-4D's code: the clone at c4a39ac (JEPA4R_PAGE4D, default <repo>/third_party/PAGE4D);
relative ``weights`` / ``init_from`` resolve under JEPA4R_CHECKPOINTS (jepa4r.config.paths).
"""
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from jepa4r.config import paths

sys.path.insert(0, str(Path(os.environ.get('JEPA4R_PAGE4D') or Path(__file__).resolve().parents[3] / 'third_party' / 'PAGE4D') / 'model'))   # clone at c4a39ac

try:  # PAGE-4D's code is needed to build the decoder, not to import this module (tests import every impl module)
    from page.heads.head_act import inverse_log_transform
    from page.models.vggt import VGGT as PAGE4D
except ImportError:  # pragma: no cover
    inverse_log_transform = PAGE4D = None

from jepa4r.decoder.head import make_cam                                # noqa: E402
from jepa4r.decoder.lora import add_lora                                # noqa: E402
from jepa4r.decoder.losses import chamfer, sliced_wasserstein           # noqa: E402
from jepa4r.decoder.set_generator import SetDecoder                     # noqa: E402


# Moved from lewm-pcd-recon-page4d@d3c2d07:lewm_pcd/recon/decoders/page4d.py:PRETRAINED, ADAPTER_KEYS, checkpoint_path, load_page4d, PAGE4DDecoder (lines 43-200; verbatim except: relative weights resolve under JEPA4R_CHECKPOINTS instead of $STABLEWM_HOME/checkpoints)
PRETRAINED = ('dino', 'aggregator', 'head')                             # never updated; only adapters wrap them
ADAPTER_KEYS = ('.down.', '.up_q.', '.up_v.')


def checkpoint_path(path):
    path = Path(path)
    return path if path.is_absolute() else Path(paths()['checkpoints'], path)


def load_page4d(weights):
    """PAGE-4D with its point head only, released weights loaded."""
    m = PAGE4D(enable_camera=False, enable_depth=False, enable_track=False)
    sd = torch.load(checkpoint_path(weights), map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, f'PAGE-4D weights missing: {missing[:5]}'
    return m


class PAGE4DDecoder(nn.Module):
    def __init__(self, latent_dim=192, grid=16, num_points=1200, weights='page4d/checkpoint_nomask.pt',
                 train='frozen', lora_rank=4, lora_targets=('dino', 'aggregator'), init_from=None, color=True, color_weight=0.1,
                 sw_weight=1.0, sw_projections=100, gen_dim=512, gen_hidden=1024, gen_depth=3, query_dim=128,
                 target_center=(0.425, 0.0, 0.2), target_scale=4.0, view=(35.0, 30.0, 1.25)):
        super().__init__()
        assert train in ('frozen', 'lora'), train
        self.grid, self.num_points, self.color, self.train_mode = grid, num_points, color, train
        self.color_weight, self.sw_weight, self.sw_projections = color_weight, sw_weight, sw_projections

        m = load_page4d(weights)
        self.dino = m.aggregator.patch_embed
        self.dino.patch_embed = nn.Identity()                           # the pixel -> token conv: the only dropped part
        m.aggregator.patch_embed = None
        m.aggregator.spatial_mask_head = None                           # inert with the released schedule; never called on the spliced path
        self.aggregator, self.head = m.aggregator, m.point_head
        self.patch_size = self.aggregator.patch_size

        self.generator = SetDecoder(latent_dim=latent_dim, num_queries=grid * grid, out_dim=self.dino.embed_dim,
                                    dim=gen_dim, hidden=gen_hidden, depth=gen_depth, query_dim=query_dim)
        feat = self.head.scratch.output_conv2[0].in_channels
        self.rgb = nn.Sequential(nn.Conv2d(feat, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 3, 1)) if color else None

        cam = make_cam(az_deg=view[0], el_deg=view[1], dist=view[2], device='cpu')     # world -> camera, metres, OpenCV axes
        R, pos = cam['R'], -cam['R'].T @ cam['t']
        self.frame = nn.Linear(3, 3)                                    # PAGE-4D camera frame (unit scene scale) -> target frame
        with torch.no_grad():
            self.frame.weight.copy_(target_scale * view[2] * R.T)
            self.frame.bias.copy_(target_scale * (pos - torch.tensor(target_center)))

        self._frozen = [name + '.' for name in PRETRAINED]              # prefixes of pretrained modules that never change
        for name in PRETRAINED:
            getattr(self, name).requires_grad_(False)
        self.adapters = []
        if train == 'lora':                                             # small adapters on q / v of every attention of the chosen parts
            parts = {'dino': [self.dino.blocks], 'aggregator': [self.aggregator.frame_blocks, self.aggregator.global_blocks]}
            for name in lora_targets:
                for blocks in parts[name]:
                    self.adapters += add_lora(blocks, lora_rank)
        self._register_state_dict_hook(self._drop_frozen)
        self.register_load_state_dict_post_hook(self._allow_frozen_missing)
        if init_from:                                                   # LP-FT order: start from a run whose new parts are already trained
            sd = torch.load(checkpoint_path(init_from), map_location='cpu')
            own = {k for k, _ in self.named_parameters()} | {k for k, _ in self.named_buffers()}
            sd = {k: v for k, v in sd.items() if k in own and not self._is_frozen_key(k)}
            self.load_state_dict(sd, strict=False)
            print(f'[page4d] initialised {len(sd)} tensors from {init_from}')

    # ---- checkpoints hold what was trained; the frozen pretrained weights come back from ``weights`` -------------------
    def _is_frozen_key(self, key, prefix=''):
        key = key[len(prefix):]
        return any(key.startswith(f) for f in self._frozen) and not any(t in key for t in ADAPTER_KEYS)

    @staticmethod
    def _drop_frozen(module, state_dict, prefix, local_metadata):
        for k in [k for k in state_dict if k.startswith(prefix) and module._is_frozen_key(k, prefix)]:
            del state_dict[k]

    @staticmethod
    def _allow_frozen_missing(module, incompatible):
        incompatible.missing_keys[:] = [k for k in incompatible.missing_keys if not module._is_frozen_key(k)]

    def train(self, mode=True):                                         # frozen pretrained parts stay in eval (adapters have no dropout / norm state)
        super().train(mode)
        for name in self._frozen:
            getattr(self, name[:-1]).eval()
        return self

    # ---- forward ------------------------------------------------------------------------------------------------------
    @staticmethod
    def _run(block, *args):
        return checkpoint(block, *args, use_reentrant=False) if torch.is_grad_enabled() else block(*args)

    def tokens_to_map(self, tokens):
        """tokens (B, G*G, 1024) in place of the patch-projection output -> raw head features (B, C, 14G, 14G), xyz/conf map."""
        B, G, side = tokens.size(0), self.grid, self.grid * self.patch_size
        d = self.dino                                                    # DINOv2: prepare_tokens_with_masks without the conv
        x = torch.cat((d.cls_token.expand(B, -1, -1), tokens), dim=1)
        x = x + d.interpolate_pos_encoding(x, side, side)
        if d.register_tokens is not None:
            x = torch.cat((x[:, :1], d.register_tokens.expand(B, -1, -1), x[:, 1:]), dim=1)
        for blk in d.blocks:
            x = self._run(blk, x)
        patch_tokens = d.norm(x)[:, d.num_register_tokens + 1:]

        a = self.aggregator                                              # Aggregator.forward with S = 1, after its patch embed
        special = torch.cat([a.camera_token[:, 0], a.register_token[:, 0]], dim=1).expand(B, -1, -1)
        x = torch.cat([special, patch_tokens], dim=1)
        pos = a.position_getter(B, G, G, device=x.device) + 1
        pos = torch.cat([pos.new_zeros(B, a.patch_start_idx, 2), pos], dim=1)
        cached = getattr(a, 'cached_layer_indices', None) or set(range(a.depth))   # PAGE-4D's aggregator keeps every layer
        taps = []
        for i in range(a.depth):
            f = x = self._run(a.frame_blocks[i], x, pos)
            x = self._run(a.global_blocks[i], x, pos)
            taps.append(torch.cat([f, x], dim=-1)[:, None] if i in cached else None)

        h = self.head                                                    # DPTHead._forward_impl up to the fused feature map
        out = []
        for j, layer in enumerate(h.intermediate_layer_idx):
            t = h.norm(taps[layer][:, 0, a.patch_start_idx:])
            t = t.permute(0, 2, 1).reshape(B, t.size(-1), G, G)
            t = h.projects[j](t)
            if h.pos_embed:
                t = h._apply_pos_embed(t, side, side)
            out.append(h.resize_layers[j](t))
        feats = h.scratch_forward(out)
        feats = nn.functional.interpolate(feats, size=(side, side), mode='bilinear', align_corners=True)
        if h.pos_embed:
            feats = h._apply_pos_embed(feats, side, side)
        with torch.autocast(device_type=feats.device.type, enabled=False):   # coordinates in float32
            feats = feats.float()
            raw = h.scratch.output_conv2(feats)                          # (B, 4, side, side): xyz + confidence
            xyz = self.frame(inverse_log_transform(raw[:, :3]).flatten(2).transpose(1, 2))      # (B, side*side, 3), target frame
            if self.rgb is not None:
                xyz = torch.cat([xyz, torch.sigmoid(self.rgb(feats)).flatten(2).transpose(1, 2)], dim=-1)
        return xyz, raw[:, 3]                                            # (B, side*side, 3 or 6), conf logits

    def forward(self, latent):
        tokens = self.generator(latent)
        points, _ = self.tokens_to_map(tokens)
        return points                                                    # the dense map as a point list

    def _pick(self, dense, n, fixed):
        """n pixels of the map: a fresh random subset while training (every pixel gets supervised), a fixed one otherwise."""
        g = torch.Generator(device=dense.device).manual_seed(0) if fixed else None
        idx = torch.randperm(dense.size(1), device=dense.device, generator=g)[:n]
        return dense[:, idx]

    def loss(self, output, target):
        pred = self._pick(output, self.num_points, fixed=not self.training)
        loss, logs = chamfer(pred, target, self.color_weight)
        if self.sw_weight > 0:
            sw = sliced_wasserstein(pred, target, self.sw_projections)
            loss, logs['sw'] = loss + self.sw_weight * sw, sw.detach()
        return loss, logs

    def to_points(self, output, n=1200):
        return self._pick(output, n, fixed=True)


def build(bb_cfg, **kwargs):
    """The decoder with this backbone (kwargs: jepa4r.decoder.model.decoder_kwargs)."""
    return PAGE4DDecoder(weights=str(bb_cfg.weights), **kwargs)

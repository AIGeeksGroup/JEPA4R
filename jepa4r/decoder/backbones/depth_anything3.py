"""Depth Anything 3 [lin2026da3] backbone for Tab. 4a.

``DA3Decoder``: Depth Anything 3 Giant (depth-anything/DA3-GIANT): one DINOv2-g ViT with alternating local / global
attention from layer 13; its dual DPT head's depth and ray map; points = ray origin + depth x ray direction per pixel.
Generator, affine map, rgb branch, LoRA stage, losses and evaluation are PAGE4DDecoder's (``_BackboneDecoder``).
Moved unchanged apart from imports / paths from the untracked recon_backbones.py (GATES.md 1.2 item 6); DA3's code:
the clone at 3d835ec (JEPA4R_DA3, default <repo>/third_party/Depth-Anything-3).
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from jepa4r.decoder.backbones.page4d import PAGE4DDecoder, checkpoint_path
from jepa4r.decoder.head import make_cam
from jepa4r.decoder.lora import add_lora
from jepa4r.decoder.set_generator import SetDecoder

DA3_REPO = Path(
    os.environ.get('JEPA4R_DA3') or Path(__file__).resolve().parents[3] / 'third_party' / 'Depth-Anything-3'
)


# Moved from the untracked lewm_pcd/recon_ext/recon_backbones.py of the local cortexbench worktree (not in any branch;
# GATES.md 1.2 item 6; sha256 25d4bc5cf5d8ee600db02c2523f1a083d590aeaf7ae922213e8f366a310904be):_BackboneDecoder, _load
# (lines 62-111; verbatim)
# --------------------------------------------------------------------------------------- shared base, other models
class _BackboneDecoder(PAGE4DDecoder):
    """PAGE4DDecoder's construction with another backbone: ``build`` (the pretrained modules, named in PRETRAINED),
    ``make_rgb``, ``lora_blocks`` and ``tokens_to_map`` come from the subclass; forward, loss, to_points, the
    frozen-weight hooks and train() are PAGE4DDecoder's."""
    PRETRAINED = ()

    def __init__(self, latent_dim=192, grid=16, num_points=1200, weights=None, train='frozen', lora_rank=4,
                 lora_targets=None, init_from=None, color=True, color_weight=0.1, sw_weight=1.0, sw_projections=100,
                 gen_dim=512, gen_hidden=1024, gen_depth=3, query_dim=128, target_center=(0.425, 0.0, 0.2),
                 target_scale=4.0, view=(35.0, 30.0, 1.25)):
        nn.Module.__init__(self)
        assert train in ('frozen', 'lora'), train
        self.grid, self.num_points, self.color, self.train_mode = grid, num_points, color, train
        self.color_weight, self.sw_weight, self.sw_projections = color_weight, sw_weight, sw_projections

        embed_dim, feat_ch = self.build(weights)
        self.generator = SetDecoder(latent_dim=latent_dim, num_queries=grid * grid, out_dim=embed_dim,
                                    dim=gen_dim, hidden=gen_hidden, depth=gen_depth, query_dim=query_dim)
        self.rgb = self.make_rgb(feat_ch) if color else None

        cam = make_cam(az_deg=view[0], el_deg=view[1], dist=view[2], device='cpu')  # as PAGE4DDecoder
        R, pos = cam['R'], -cam['R'].T @ cam['t']
        self.frame = nn.Linear(3, 3)
        with torch.no_grad():
            self.frame.weight.copy_(target_scale * view[2] * R.T)
            self.frame.bias.copy_(target_scale * (pos - torch.tensor(target_center)))

        self._frozen = [name + '.' for name in self.PRETRAINED]
        for name in self.PRETRAINED:
            getattr(self, name).requires_grad_(False)
        self.adapters = []
        if train == 'lora':
            for blocks in self.lora_blocks():
                self.adapters += add_lora(blocks, lora_rank)
        self._register_state_dict_hook(self._drop_frozen)
        self.register_load_state_dict_post_hook(self._allow_frozen_missing)
        if init_from:
            sd = torch.load(checkpoint_path(init_from), map_location='cpu')
            own = {k for k, _ in self.named_parameters()} | {k for k, _ in self.named_buffers()}
            sd = {k: v for k, v in sd.items() if k in own and not self._is_frozen_key(k)}
            self.load_state_dict(sd, strict=False)
            print(f'[{type(self).__name__}] initialised {len(sd)} tensors from {init_from}')


def _load(module, sd, what):
    missing, unexpected = module.load_state_dict(sd, strict=False)
    assert not missing, f'{what}: weights missing {missing[:5]}'
    return unexpected


# Moved from the untracked lewm_pcd/recon_ext/recon_backbones.py of the local cortexbench worktree (not in any branch;
# GATES.md 1.2 item 6; sha256 25d4bc5cf5d8ee600db02c2523f1a083d590aeaf7ae922213e8f366a310904be):DA3Decoder (lines
# 194-274; verbatim except: DA3's clone at DA3_REPO instead of lewm_pcd/third_party)
# --------------------------------------------------------------------------------------------- Depth Anything 3
class DA3Decoder(_BackboneDecoder):
    PRETRAINED = ('dino', 'head')  # 'dino': the name the LoRA stage's optimizer matches (config/recon/exp/lora.yaml)

    def build(self, weights):
        sys.path.insert(0, str(DA3_REPO / 'src'))
        from depth_anything_3.model.dinov2.dinov2 import DinoV2
        from depth_anything_3.model.dualdpt import DualDPT
        from safetensors.torch import load_file
        path = checkpoint_path(weights)
        cfg = json.loads((path.parent / 'config.json').read_text())['config']
        args = lambda c: {key: val for key, val in c.items() if key != '__object__'}
        net, head = DinoV2(**args(cfg['net'])), DualDPT(**args(cfg['head']))
        sd = load_file(str(path))
        part = lambda prefix: {key[len(prefix):]: val for key, val in sd.items() if key.startswith(prefix)}
        _load(net, part('model.backbone.'), 'DA3 backbone')
        # the ray map's per-level LayerNorm affines
        missing, _ = head.load_state_dict(part('model.head.'), strict=False)
        # are absent from the release: identity, as
        assert all('output_conv2_aux' in key for key in missing), missing[:5]
        # the official loader (strict=False) has them
        self.dino, self.head = net.pretrained, head
        self.out_layers, self.cat_token = list(net.out_layers), net.cat_token
        return self.dino.embed_dim, self.head.scratch.output_conv2[0].in_channels

    def make_rgb(self, ch):  # on the head's fused full-resolution features, like its depth
        return nn.Sequential(nn.Conv2d(ch, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 3, 1))

    def lora_blocks(self):
        return [self.dino.blocks]

    def tokens_to_map(self, tokens):
        B, G = tokens.size(0), self.grid
        p = self.dino
        side = G * p.patch_size
        x = torch.cat([p.prepare_cls_token(B, 1), tokens], dim=1)       # prepare_tokens_with_masks without the conv
        x = x + p.interpolate_pos_encoding(x, side, side)
        if p.register_tokens is not None:
            x = torch.cat([x[:, :1], p.register_tokens.expand(B, -1, -1), x[:, 1:]], dim=1)
        pos, pos_nodiff = p._prepare_rope(B, 1, side, side, x.device)  # (B, 1, N, 2) or None
        outs, local_x = [], x
        for i, blk in enumerate(p.blocks):                              # _get_intermediate_layers_not_chunked, one view
            use = None if (i < p.rope_start or p.rope is None) else pos
            # the reference view's camera token in place of cls
            if p.alt_start != -1 and i == p.alt_start:
                x = torch.cat([p.camera_token[:, :1].expand(B, -1, -1), x[:, 1:]], dim=1)
            if p.alt_start != -1 and i >= p.alt_start and i % 2 == 1:  # "global" (with one view: the same tokens)
                x = self._run(blk, x, None if use is None else pos_nodiff[:, 0])
            else:                                                       # "local"
                x = self._run(blk, x, None if use is None else use[:, 0])
                local_x = x
            if i in self.out_layers:
                outs.append(torch.cat([local_x, x], dim=-1) if self.cat_token else x)
        C = p.embed_dim
        feats = [
            torch.cat([o[..., :C], p.norm(o[..., C:])], dim=-1) if o.size(-1) == 2 * C else p.norm(o) for o in outs
        ]
        feats = [f_[:, 1 + p.num_register_tokens:] for f_ in feats]

        # precision as PAGE4DDecoder (see VGGTOmegaDecoder)
        h = self.head
        ph = side // h.patch_size
        res = []
        # DualDPT._forward_impl: depth (main) and ray (aux)
        for s_i, take in enumerate(h.intermediate_layer_idx):
            t = h.norm(feats[take]).permute(0, 2, 1).reshape(B, -1, ph, ph)
            t = h.projects[s_i](t)
            if h.pos_embed:
                t = h._add_pos_embed(t, side, side)
            res.append(h.resize_layers[s_i](t))
        fused, aux = h._fuse(res)
        aux = aux[-1]                                                    # the ray map's level (only the last is output)
        out = int(ph * h.patch_size / h.down_ratio)
        fused = F.interpolate(fused, (out, out), mode='bilinear', align_corners=True)
        if h.pos_embed:
            fused = h._add_pos_embed(fused, side, side)
            aux = h._add_pos_embed(aux, side, side)
        with torch.autocast(device_type=x.device.type, enabled=False):
            fused, aux = fused.float(), aux.float()
            main = h.scratch.output_conv2(fused).permute(0, 2, 3, 1)
            depth = h._apply_activation_single(main[..., :-1], h.activation)[..., 0]           # (B, out, out)
            conf = h._apply_activation_single(main[..., -1], h.conf_activation)
            ray = h.scratch.output_conv2_aux[-1](aux)[:, :6]            # (B, 6, 8G, 8G): direction, origin (linear)
            ray = F.interpolate(ray, (out, out), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            # origin + depth x direction, the reference view's frame
            pts = ray[..., 3:] + depth[..., None] * ray[..., :3]
            xyz = self.frame(pts.flatten(1, 2))
            if self.rgb is not None:
                xyz = torch.cat([xyz, torch.sigmoid(self.rgb(fused)).flatten(2).transpose(1, 2)], dim=-1)
        return xyz, conf.flatten(1)


def build(bb_cfg, **kwargs):
    """The decoder with this backbone (kwargs: jepa4r.decoder.model.decoder_kwargs)."""
    return DA3Decoder(weights=str(bb_cfg.weights), **kwargs)

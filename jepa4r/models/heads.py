"""Projector (part of f_theta) and action encoder, inherited from LeWM (Sec. 3).

projector       encoder output -> z_t in R^192
action_encoder  action block (5 concatenated actions) -> embedding for the predictor's AdaLN-zero
``Embedder`` and ``MLP`` are taken unchanged from LeWM's ``module.py``; the predictor projection (``pred_proj``) is
the same MLP as the projector.
"""

from functools import partial

from torch import nn


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


_NORMS = {"batchnorm1d": nn.BatchNorm1d, "layernorm": nn.LayerNorm, None: None}


def _mlp(head_cfg, dim):
    norm = _NORMS[head_cfg.get("norm")]
    return MLP(
        input_dim=dim,
        hidden_dim=int(head_cfg.hidden_dim),
        output_dim=dim,
        norm_fn=partial(norm) if norm is not None else None,
    )


def build_projector(method_cfg):
    return _mlp(method_cfg.projector, int(method_cfg.latent_dim))


def build_pred_proj(method_cfg):
    return _mlp(method_cfg.pred_proj, int(method_cfg.latent_dim))


def build_action_encoder(method_cfg, action_dim):
    """``action_dim`` = dimension of one env action; the block concatenates ``frameskip`` of them (as the training
    code sets it:
    ``cfg.model.action_encoder.input_dim = frameskip * dataset.get_dim('action')``)."""
    return Embedder(
        input_dim=int(method_cfg.frameskip) * int(action_dim), emb_dim=int(method_cfg.action_encoder.emb_dim)
    )

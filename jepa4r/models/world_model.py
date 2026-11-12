"""JEPA world model (impl of method kind ``jepa_wm``): JEPA4R and LeWM, which differ only in the encoder.

f_theta = encoder (cfg.method.encoder) + projector;  g_phi = predictor;  action encoder   (Sec. 3, Sec. 4.2)
Training (LeWM recipe on stable-worldmodel [maes2026stableworldmodel]): clips of 4 frames 5 env steps apart,
teacher forcing, AdamW lr 5e-5, batch 128, 10 epochs, bf16, no stop-gradient / EMA target,
    L = 1/(3d) * sum_{t=1..3} ||z_hat_{t+1} - z_{t+1}||^2 + 0.09 * SIGReg(Z)        (Eq. 1)
Interface used by the rest of the code:
    encode(obs [B, ...]) -> z [B, 192]
    predict(z_hist [B, T, 192], a_hist [B, T, A*5]) -> z_next [B, 192]           (T <= history)
    rollout(z_ctx [B, 3, 192], action_blocks [B, H, A*5]) -> z_hat [B, H, 192]   (autoregressive; Sec. 4.4)
"""


def train(cfg, out_dir) -> dict:
    """Stage train_wm for kind jepa_wm: one checkpoint per epoch in out_dir; returns final losses."""
    raise NotImplementedError("TODO(integrate): JEPA4R / LeWM training on stable-worldmodel")


def build_world_model(method_cfg, action_dim):
    raise NotImplementedError("TODO(integrate)")


def load_world_model(env, method, tag="default", epoch=10):
    """Frozen world model from outputs/train_wm/<env>/<method>/<tag>/ (decoder, imagination)."""
    raise NotImplementedError("TODO(integrate)")


def load_frozen_encoder(cfg):
    """FrozenEncoder (jepa4r/policy/frozen.py) = f_theta of the epoch-10 checkpoint, for train_policy."""
    raise NotImplementedError("TODO(integrate)")

"""One entry per encoder row of Tab. 1 / Tab. 2 (``configs/method/*.yaml``), dispatched via ``cfg.method.impl``.

kind        trained by us  how                                         rows
jepa_wm     yes            LeWM recipe (Eq. 1) on stable-worldmodel    JEPA4R, LeWM, Tab. 4b encoders
pldm        yes            PLDM's own recipe on our paired images      PLDM
lejepa      yes            LeJEPA's own recipe on our paired images    LeJEPA
pretrained  no             authors' released weights                   VISReg, V-JEPA 2.1
random      no             random initialisation, never trained        Random (ViT-B)

Every ``impl`` module exposes
    train(cfg, out_dir) -> dict                 (trained kinds only; stage train_wm)
    load_frozen_encoder(cfg) -> FrozenEncoder   (all kinds; see jepa4r/policy/frozen.py)
"""
import importlib

from .config import stage_dir

TRAINED_KINDS = ("jepa_wm", "pldm", "lejepa")


def trained_by_us(method_cfg) -> bool:
    return method_cfg.kind in TRAINED_KINDS


def wm_dir(cfg):
    """Directory of the train_wm run that belongs to this config (same env, method and tag)."""
    return stage_dir("train_wm", cfg.env.name, cfg.method.name, None, str(cfg.tag))


def train(cfg, out_dir) -> dict:
    if not trained_by_us(cfg.method):
        raise SystemExit(f"method {cfg.method.name!r} ({cfg.method.kind}) is not trained by us; "
                         "go straight to train_policy")
    if cfg.env.suite in ("adroit", "rh56"):  # Relocate: the Relocate workflow trains all four methods
        from jepa4r.policy.bc import relocate_stage
        return {"harness_status": relocate_stage(cfg, "train_wm", out_dir)["status"]}
    from jepa4r.policy.gciql import is_cube_workflow
    if is_cube_workflow(cfg):  # Cube 1-4: the Cube workflow trains all four methods
        from jepa4r.policy.gciql import train_world_model
        return train_world_model(cfg, out_dir)
    return importlib.import_module(cfg.method.impl).train(cfg, out_dir)


def load_frozen_encoder(cfg):
    return importlib.import_module(cfg.method.impl).load_frozen_encoder(cfg)

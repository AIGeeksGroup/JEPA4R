"""Random (ViT-B), frozen (Tab. 2): a randomly initialised ViT-B that is never trained.

CortexBench's own construction of VC-1's ViT-B/16 without a checkpoint (vc_models, eai-vc 76fe35e):
configs/policy/bc/conf/model/rand_vitb.yaml (cortexbench@9331494), the embedding the CortexBench runs load. Needs the
eai-vc interpreter (JEPA4R_PY_CORTEXBENCH) where vc_models is installed.
"""
from pathlib import Path

CONF = Path(__file__).resolve().parents[2] / 'configs' / 'policy' / 'bc' / 'conf' / 'model' / 'rand_vitb.yaml'


def load_frozen_encoder(cfg):
    import hydra
    from omegaconf import OmegaConf

    from jepa4r.baselines import frozen

    return frozen(hydra.utils.call(OmegaConf.load(CONF)), cfg)

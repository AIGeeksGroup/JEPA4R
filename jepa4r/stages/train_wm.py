"""Stage ``train_wm`` -- train one world model / encoder per task.

Paper: Sec. 3 (LeWM recipe, Eq. 1), Sec. 4.2, Sec. 5.2, supp. sec:supp-wm and sec:supp-baselines.
in : data/wm/<env>/
out: outputs/train_wm/<env>/<method>/<tag>/ -- a checkpoint per epoch (the decoder reads epoch 10) and
     metrics.json with the final training losses.
Dispatch: jepa4r.methods.train -> cfg.method.impl
  jepa_wm  JEPA4R, LeWM, Tab. 4b encoders   jepa4r.models.world_model
  pldm     PLDM                             jepa4r.baselines.pldm
  lejepa   LeJEPA                           jepa4r.baselines.lejepa
VISReg / V-JEPA 2.1 (released weights) and Random (ViT-B) have nothing to train.
"""


def run(cfg, out_dir):
    from jepa4r.methods import train

    return train(cfg, out_dir)

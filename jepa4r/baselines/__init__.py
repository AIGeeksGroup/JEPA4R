"""2D baselines of Tab. 1 / Tab. 2 (supp. sec:supp-baselines). LeWM is not here: it is a ``jepa_wm`` method
trained with exactly the JEPA4R recipe (jepa4r/models/world_model.py).

trained by us on the paired images, 10 epochs, own recipes : pldm.py, lejepa.py
released weights, frozen                                    : visreg.py, vjepa21.py
never trained                                               : random_vit.py
Each module: load_frozen_encoder(cfg) -> FrozenEncoder (+ train(cfg, out_dir) for the trained ones).
"""

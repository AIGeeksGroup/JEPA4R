"""2D baselines of Tab. 1 / Tab. 2 (supp. sec:supp-baselines) trained by us on the paired images, 10 epochs, with their
own recipes: pldm.py, lejepa.py; cube/, relocate/ their training in the Cube 1-4 and Relocate workflows. LeWM is not
here: it is a ``jepa_wm`` method trained with exactly the JEPA4R recipe (jepa4r/models/world_model.py).
The released encoders (VISReg, V-JEPA 2.1) and the never-trained Random (ViT-B) are loaded by the policy harnesses
(jepa4r/policy/bc/jepa_baselines.py, configs/policy/bc/conf/model/rand_vitb.yaml; on Cube 1-4
cube/released_encoders.py).
"""

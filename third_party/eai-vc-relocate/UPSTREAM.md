# eai-vc (CortexBench) for Relocate and RH56: the train-loop patch

- Repository: https://github.com/facebookresearch/eai-vc
- Commit: `76fe35e87b1937168f1ec4b236e863451883eaf3`
- License: CC BY-NC 4.0 (`LICENSE`, eai-vc's; the patched file stays under it)

Not vendored: the behavior cloning on Adroit Relocate and the RM75-B + RH56 hand twin (`jepa4r/policy/bc/relocate`,
`jepa4r/envs/twins/rh56`) runs a checkout of eai-vc at the commit above with `eai-vc-train-loop.patch` applied
(`JEPA4R_EAI_VC_RELOCATE`; `configs/policy/bc/relocate/DEPENDENCIES.json`, entry `eai_vc`). The license is
eai-vc's.

```bash
git clone https://github.com/facebookresearch/eai-vc eai-vc-relocate && cd eai-vc-relocate
git checkout 76fe35e87b1937168f1ec4b236e863451883eaf3
patch -p1 < <this repository>/third_party/eai-vc-relocate/eai-vc-train-loop.patch
```

## Local changes

`cortexbench/mujoco_vc/visual_imitation/train_loop.py` only: a policy save also at each new best evaluation (as in the
paper's Relocate runs); the training and evaluation steps themselves are unchanged.
The MetaWorld, Scene and RM75-B MetaWorld runs use the unpatched checkout (`JEPA4R_EAI_VC`).

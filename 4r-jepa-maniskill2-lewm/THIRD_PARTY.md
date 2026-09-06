# Upstream sources

This adapter builds on the following projects. Exact revisions and source
hashes are recorded in `repro/sources.lock.json`; source checkouts are fetched
into ignored `vendor/` directories and are not included in this repository.

| Project | Role | Upstream license |
| --- | --- | --- |
| [ManiSkill2-Learn](https://github.com/haosulab/ManiSkill2-Learn) | PPO, PointNet, actor/critic, environment wrappers | Apache-2.0 |
| [ManiSkill2](https://github.com/haosulab/ManiSkill2) | Panda/PickCube physics and observation generation | See installed package/upstream license |
| [LeWorldModel](https://github.com/lucas-maes/le-wm) | JEPA, Transformer, SIGReg, training/evaluation configuration | MIT |
| [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) | Dataset, collector, expert-noise wrapper and planning implementations | MIT, as declared in the pinned package metadata |
| [PyTorch3D](https://github.com/facebookresearch/pytorch3d) | Pure PyTorch coordinate transforms | BSD |
| [OGBench](https://github.com/seohongpark/ogbench) | Original CubeMarkovOracle | See installed package/upstream license |

`physics_preflight.py` derives renderer-free initialization from ManiSkill2;
it retains upstream physics/controller/reward methods and is used solely for
state-based expert checks. Encoder subclasses inherit the original PointNet
forward and replace only the selected final activation. Upstream copyright
and license terms continue to apply to upstream and derivative portions.

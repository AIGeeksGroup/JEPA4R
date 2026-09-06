"""Current shared PPO/WM config: only the final PointNet activation is Identity."""

_base_ = "../vendor/ManiSkill2-Learn/configs/mfrl/ppo/maniskill2_pn.py"

custom_imports = dict(imports=["ms2_lewm.pointnet_identity"], allow_failed_imports=False)

agent_cfg = dict(
    actor_cfg=dict(
        nn_cfg=dict(visual_nn_cfg=dict(type="PointNetLastIdentity")),
    ),
)

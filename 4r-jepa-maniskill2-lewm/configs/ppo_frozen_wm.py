"""Stage 3 PPO: replace only the visual encoder and retain original current state."""

_base_ = "ppo_pointnet_identity.py"

custom_imports = dict(
    imports=["ms2_lewm.pointnet_identity", "ms2_lewm.frozen_ppo", "ms2_lewm.history"],
    allow_failed_imports=False,
)

agent_cfg = dict(
    actor_cfg=dict(nn_cfg=dict(
        type="FrozenWMVisuomotor",
        visual_nn_cfg=dict(_delete_=True, type="FrozenWorldModelEncoder", checkpoint=None),
    )),
    critic_cfg=dict(nn_cfg=dict(
        type="FrozenWMVisuomotor",
    )),
)

env_cfg = dict(extra_wrappers=[dict(type="PointCloudHistory", history_size=3, history_stride=1)])

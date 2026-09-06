"""Stage 3 PPO: frozen checkpoint encoder through latent2, fresh policy/value heads."""

_base_ = "ppo_pointnet_identity.py"

custom_imports = dict(
    imports=["ms2_lewm.pointnet_identity", "ms2_lewm.frozen_ppo", "ms2_lewm.history"],
    allow_failed_imports=False,
)

agent_cfg = dict(
    actor_cfg=dict(nn_cfg=dict(
        type="FrozenWMVisuomotor",
        visual_nn_cfg=dict(_delete_=True, type="FrozenWorldModelEncoder", checkpoint=None),
        mlp_cfg=dict(mlp_spec=[512, 256, 256, "action_shape"]),
    )),
    critic_cfg=dict(nn_cfg=dict(
        type="FrozenWMVisuomotor",
        mlp_cfg=dict(mlp_spec=[512, 256, 256, 1]),
    )),
)

env_cfg = dict(extra_wrappers=[dict(type="PointCloudHistory", history_size=3, history_stride=1)])

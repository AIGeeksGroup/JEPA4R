"""Load the selected upstream PPO policy on CPU without creating a second env."""
from pathlib import Path


def load_policy(checkpoint, env, config_path=None):
    """Rebuild PPO from its saved run configuration, including its activation.

    The supplied env must match PickCube-v0/pointcloud/1200 points/world frame/
    pd_ee_delta_pose. Caller retains ownership of its lifetime. An explicit
    config_path is required for checkpoints without an adjacent saved config;
    weight shapes alone cannot distinguish ReLU, tanh and Identity.
    """
    from maniskill2_learn.utils.meta import Config
    from maniskill2_learn.utils.data import GDict
    from maniskill2_learn.networks.utils import get_kwargs_from_shape, replace_placeholder_with_args
    from maniskill2_learn.methods.builder import build_agent
    from maniskill2_learn.utils.torch import load_checkpoint

    checkpoint = Path(checkpoint).resolve()
    if config_path is None:
        run_dir = checkpoint.parent.parent if checkpoint.parent.name == "models" else checkpoint.parent
        saved = sorted(run_dir.glob("*-train.py"), key=lambda p: p.stat().st_mtime)
        if not saved:
            raise ValueError("Checkpoint has no saved PPO config; supply config_path to identify the encoder activation.")
        config_path = saved[-1]
    config = Config.fromfile(str(config_path))
    observation = env.reset(seed=20260906)
    env_params = dict(obs_shape=GDict(observation).list_shape,
                      action_shape=env.action_space.shape[0],
                      action_space=env.action_space, is_discrete=False)
    config.agent_cfg["env_params"] = env_params
    config = replace_placeholder_with_args(config, **get_kwargs_from_shape(env_params["obs_shape"], env_params["action_shape"]))
    agent = build_agent(config.agent_cfg).float().cpu()
    load_checkpoint(agent, str(checkpoint), "cpu", strict=True)
    agent.eval()
    agent.set_mode("test")
    return agent

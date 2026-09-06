"""PPO 训练脚本，结构照 scripts/train_sac.py 和 parllel 的 examples/pointcloud_rl/train_ppo.py。
（train_ppo.py 的副本，仅增加 resume_model_path：从已保存的 model.pt 加载网络权重后继续训练。用法：+resume_model_path=<model.pt>）
encoder、环境、点云包装、评估与录像逻辑与 train_sac.py 相同；算法换成 parllel 的 PPO（可带 PPRL 重建辅助损失）。
用法示例：
    python scripts/train_ppo.py env=push_chair model=pointgpt_ppo env.name=LiftCube env.env_id=LiftCube-v0 ... env.continuous_task=True
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import gymnasium as gym
import gymnasium.spaces as spaces
import torch.nn as nn
import hydra
import parllel.logger as logger
import torch
import wandb
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from parllel import Array, ArrayDict, dict_map
from parllel.callbacks.recording_schedule import RecordingSchedule
from parllel.patterns import (
    add_advantage_estimation,
    add_agent_info,
    add_bootstrap_value,
    add_reward_clipping,
    add_reward_normalization,
    build_cages,
    build_sample_tree,
)
from parllel.replays import BatchedDataLoader
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler
from parllel.samplers.eval import EvalSampler
from parllel.torch.agents.gaussian import GaussianPgAgent
from parllel.torch.algos.ppo import build_loss_sample_tree
from pprl.models.pg.gaussian_fixed import GaussianFixed
from parllel.transforms.vectorized_video import RecordVectorizedVideo
from parllel.types import BatchSpec

from pprl.algos.truncation_bootstrap import BootstrapTruncated
from pprl.models.pg.pointcloud_pg_model import PointCloudPgModel
from pprl.utils.array_dict import build_obs_array


class ClippedEnvFactory:
    """环境工厂：高斯策略无界，送入环境前把动作裁剪到动作空间；回放里存的仍是未裁剪的采样动作。
    定义在模块顶层以便被多进程 cage 序列化。"""

    def __init__(self, env_factory):
        self.env_factory = env_factory

    def __call__(self, **kwargs):
        return gym.wrappers.ClipAction(self.env_factory(**kwargs))


@contextmanager
def build(config: DictConfig) -> Iterator[RLRunner]:
    parallel = config.parallel
    discount = config.algo.discount
    batch_spec = BatchSpec(config.batch_T, config.batch_B)
    storage = "shared" if parallel else "local"

    with open_dict(config.env):
        config.env.pop("name")
        traj_info = config.env.pop("traj_info")

    TrajInfoClass = get_class(traj_info)
    TrajInfoClass.set_discount(discount)

    env_factory = instantiate(config.env, _convert_="partial", _partial_=True)
    clipped_env_factory = ClippedEnvFactory(env_factory)

    cages, metadata = build_cages(
        EnvClass=clipped_env_factory,
        n_envs=batch_spec.B,
        TrajInfoClass=TrajInfoClass,
        parallel=parallel,
    )
    eval_cages, eval_metadata = build_cages(
        EnvClass=clipped_env_factory,
        n_envs=config.eval.n_eval_envs,
        env_kwargs={"add_rendering_to_info": True},
        TrajInfoClass=TrajInfoClass,
        parallel=parallel,
    )

    sample_tree, metadata = build_sample_tree(
        env_metadata=metadata,
        batch_spec=batch_spec,
        parallel=parallel,
        keys_to_skip=("obs", "next_obs"),
    )
    obs_space, action_space = metadata.obs_space, metadata.action_space

    sample_tree["observation"] = build_obs_array(
        metadata.example_obs,
        obs_space,
        batch_shape=tuple(batch_spec),
        storage=storage,
        padding=1,
    )
    sample_tree["next_observation"] = sample_tree["observation"].new_array(padding=0)
    sample_tree["observation"][0] = obs_space.sample()
    metadata.example_obs_batch = sample_tree["observation"][0]

    assert isinstance(action_space, spaces.Box)
    n_actions = action_space.shape[0]

    # encoder：与 train_sac.py 同一行代码实例化，保证 latent 定义一致
    with open_dict(config.model):
        config.model.pop("name")
    encoder = instantiate(config.model, _convert_="partial", obs_space=obs_space)
    # PPO 要求同一观测在采样与更新时输出一致：encoder 配置里不能有随机 FPS 起点和 dropout（见 conf/model/pointgpt_ppo.yaml）
    assert not config.model.tokenizer.get("random_start", True), "PPO 需要 model.tokenizer.random_start=False（用 model=pointgpt_ppo）"
    for name, module in encoder.named_modules():
        if name.startswith("gpt_decoder"):
            continue  # 解码器只用于重建损失，不影响策略输出
        if isinstance(module, nn.Dropout) and module.p > 0:
            raise ValueError(f"PPO 的 encoder 不能有 dropout：{name} p={module.p}（用 model=pointgpt_ppo）")
        if isinstance(module, nn.MultiheadAttention) and module.dropout > 0:
            raise ValueError(f"PPO 的 encoder 不能有 attention dropout：{name}（用 model=pointgpt_ppo）")

    model = PointCloudPgModel(
        encoder=encoder,
        action_size=n_actions,
        action_space=action_space,
        pi_hidden_sizes=config.pi_mlp_head.hidden_sizes,
        value_hidden_sizes=config.value_mlp_head.hidden_sizes,
        hidden_nonlinearity=config.pi_mlp_head.hidden_nonlinearity,
    )

    distribution = GaussianFixed(dim=n_actions, min_log_std=-5.0, deterministic_eval_mode=True)  # 修正似然比求和 bug；std 下限 e^-5，防止收缩到极小后比值溢出
    distribution.train()  # parllel 的 Distribution 不初始化 mode，先置为采样模式，评估时 agent 会切到 eval
    device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

    agent = GaussianPgAgent(
        model=model,
        distribution=distribution,
        example_obs=metadata.example_obs_batch,
        device=device,
    )

    sample_tree = add_agent_info(sample_tree, agent, metadata.example_obs_batch)
    sample_tree = add_bootstrap_value(sample_tree)

    batch_transforms, step_transforms = [], []
    if config.reward_normalization:
        sample_tree, batch_transforms = add_reward_normalization(sample_tree, batch_transforms, discount=discount)
    if config.reward_clip_min is not None or config.reward_clip_max is not None:
        sample_tree, batch_transforms = add_reward_clipping(
            sample_tree, batch_transforms, reward_clip_min=config.reward_clip_min, reward_clip_max=config.reward_clip_max
        )
    if config.bootstrap_truncated:
        # 时限截断不是终止：在截断步的奖励上加 γ·V(下一状态)，放在奖励归一化之后、优势估计之前
        batch_transforms.append(BootstrapTruncated(agent=agent, discount=discount))
    sample_tree, batch_transforms = add_advantage_estimation(
        sample_tree,
        batch_transforms,
        discount=discount,
        gae_lambda=config.algo.gae_lambda,
        normalize=config.algo.normalize_advantage,
    )

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_tree=sample_tree,
        max_steps_decorrelate=config.max_steps_decorrelate,
        get_bootstrap_value=True,
        step_transforms=step_transforms,
        batch_transforms=batch_transforms,
    )

    loss_sample_tree = build_loss_sample_tree(sample_tree)

    def batch_transform(tree: ArrayDict[Array]) -> ArrayDict[torch.Tensor]:
        tree = tree.to_ndarray()  # type: ignore
        tree = tree.apply(torch.from_numpy)
        return tree.to(device=device)

    dataloader = BatchedDataLoader(
        tree=loss_sample_tree,
        sampler_batch_spec=batch_spec,
        n_batches=config.algo.minibatches,
        batch_transform=batch_transform,
    )

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=config.algo.learning_rate)

    algorithm = instantiate(
        config.algo,
        _convert_="partial",
        agent=agent,
        dataloader=dataloader,
        optimizer=optimizer,
    )

    logger.debug("Allocating eval sample tree...")
    eval_tree_keys = ["action", "agent_info", "observation", "reward", "terminated", "truncated", "done"]
    eval_tree_example = ArrayDict({key: sample_tree[key] for key in eval_tree_keys})
    eval_sample_tree = eval_tree_example.new_array(batch_shape=(1, config.eval.n_eval_envs))
    eval_sample_tree["env_info"] = dict_map(
        Array.from_numpy,
        eval_metadata.example_info,
        batch_shape=(1, config.eval.n_eval_envs),
        storage=storage,
    )

    video_recorder = RecordVectorizedVideo(
        sample_tree=eval_sample_tree,
        buffer_key_to_record="env_info.rendering",
        env_fps=50,
        output_dir=Path(config.video_path),
        video_length=config.env.max_episode_steps,
        use_wandb=True,
    )
    recording_schedule = RecordingSchedule(video_recorder, trigger="on_eval")

    eval_sampler = EvalSampler(
        max_traj_length=config.env.max_episode_steps,
        max_trajectories=config.eval.max_trajectories,
        envs=eval_cages,
        agent=agent,
        sample_tree=eval_sample_tree,
        step_transforms=[video_recorder],
    )

    runner = RLRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        eval_sampler=eval_sampler,
        n_steps=config.runner.n_steps,
        log_interval_steps=config.runner.log_interval_steps,
        eval_interval_steps=config.runner.eval_interval_steps,
        callbacks=[recording_schedule],
    )

    # 续训：所有组件构建完之后再加载权重。parllel 的 add_agent_info 在构造阶段会用 obs_space.sample() 的随机点云
    # 以 train 模式前向一次，BatchNorm 的运行统计量会被这批假数据污染（离线测得运行方差最大变 30 倍），
    # 所以加载必须放在最后；优化器与奖励归一化统计量从零开始。
    resume_model_path = config.get("resume_model_path", None)
    if resume_model_path:
        agent.load_model(resume_model_path, device)
        agent.model.eval()
        logger.info(f"Loaded model weights from {resume_model_path} (after all components were built)")

    try:
        yield runner
    finally:
        eval_sampler.close()
        eval_sample_tree.close()
        sampler.close()
        sample_tree.close()
        agent.close()
        for eval_cage in eval_cages:
            eval_cage.close()
        for cage in cages:
            cage.close()


@hydra.main(version_base=None, config_path="../conf", config_name="train_ppo")
def main(config: DictConfig) -> None:
    with open_dict(config):
        wandb_config = config.pop("wandb", {})
        notes = wandb_config.pop("notes", None)
        tags = wandb_config.pop("tags", None)

    run = wandb.init(
        project="pprl",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),  # type: ignore
        sync_tensorboard=True,
        save_code=True,
        reinit=True,
        tags=tags,
        notes=notes,
    )

    logger.init(
        wandb_run=run,
        log_dir=Path(f"log_data/ppo/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        tensorboard=True,
        output_files={"txt": "log.txt"},  # type: ignore
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),  # type: ignore
        model_save_path=Path("model.pt"),
    )

    video_path = Path(config.video_path) / f"{datetime.now().strftime('%Y-%m-%d')}/{run.id}"  # type: ignore
    config.update({"video_path": video_path})

    with build(config) as runner:
        runner.run()

    logger.close()
    run.finish()  # type: ignore


if __name__ == "__main__":
    main()

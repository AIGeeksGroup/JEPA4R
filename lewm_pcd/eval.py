import os

os.environ.setdefault("MUJOCO_GL", "egl")  # upstream forces egl; MUJOCO_GL=osmesa lets eval render on CPU while the GPU trains

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
import envs  # noqa: F401  registers swm/OGBCubePCD-v0 (point-cloud env)

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    pre_wrappers = world_cfg.pop("pre_wrappers", None)  # e.g. the point-cloud wrapper (see config/eval/cube_pcd.yaml)
    if pre_wrappers:
        pre_wrappers = [hydra.utils.instantiate(w, _partial_=True) for w in pre_wrappers]
    world = swm.World(**world_cfg, image_shape=(224, 224), pre_wrappers=pre_wrappers)

    # create the transform
    obs_is_pcd = cfg.get("obs", "pixels") == "pcd"
    transform = {} if obs_is_pcd else {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    # the normalisation statistics are the training data's (dataset.stats): the eval dataset itself for the configs
    # that evaluate on it (cube, ...), the training set for MetaWorld, whose eval dataset is a held-out test set
    stats_name = cfg.dataset.get("stats") or cfg.eval.dataset_name
    stats_dataset = dataset if stats_name == cfg.eval.dataset_name else get_dataset(cfg, stats_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    if obs_is_pcd:  # same fixed affine as training (utils.PointCloudPreprocessor), via the policy's process hook
        from envs import PCDWorldModelPolicy, PointCloudScaler
        process["pcd"] = process["goal"] = PointCloudScaler(**OmegaConf.to_container(cfg.pcd_process, resolve=True))

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to(cfg.get("device", "cuda"))  # device=cpu keeps the GPU free for training (pair with solver.device=cpu)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, cost=model)  # swm HEAD: CEMSolver(cost=...)
        if obs_is_pcd:
            # the affine fed to the model must be the one it was trained with: compare with the checkpoint's config.yaml
            ckpt_cfg = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), cfg.policy).parent / "config.yaml"
            if ckpt_cfg.exists():
                trained = OmegaConf.load(ckpt_cfg).data.preprocessor
                for k in ("center", "scale", "use_rgb"):
                    a, b = OmegaConf.to_object(trained)[k], OmegaConf.to_object(cfg.pcd_process)[k]
                    assert list(a) == list(b) if isinstance(a, (list, tuple)) else a == b, f"pcd_process.{k}={b} differs from training ({a}) in {ckpt_cfg}"
            policy = PCDWorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform, use_rgb=cfg.pcd_process.use_rgb
            )
        elif int(cfg.plan_config.get("history_len", 1)) > 1 or int(cfg.eval.get("goal_window", 1)) > 1:
            # frame-window model on pixels (jepa_agg): the official policy plus goal window / history seed
            from envs import WindowedWorldModelPolicy
            policy = WindowedWorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform, obs_key="pixels"
            )
        else:
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform
            )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    block = int(cfg.plan_config.action_block)
    # frame-window world models (jepa_agg), any observation type: the World's goal key(s) and per-step obs key(s)
    # and the dataset columns that feed them
    win_goal = {"goal_pcd_xyz": "pcd_xyz", "goal_pcd_rgb": "pcd_rgb"} if obs_is_pcd else {"goal": "pixels"}
    win_obs = ("pcd_xyz", "pcd_rgb") if obs_is_pcd else ("pixels",)
    goal_window = int(cfg.eval.get("goal_window", 1))
    if goal_window > 1 and cfg.policy != "random":
        # the goal is the K frames ending at the goal frame, action_block env steps apart, taken from the same
        # episodes/starts sampled above
        dataset._open()
        goal_rows = random_episode_indices + cfg.eval.goal_offset_steps
        policy.goal_window = {
            key: np.stack([np.stack([dataset.h5_file[col][int(r) - (goal_window - 1 - k) * block] for k in range(goal_window)]) for r in goal_rows])
            for key, col in win_goal.items()
        }  # (E, K, ...) each

    history_len = int(cfg.plan_config.get("history_len", 1))  # absent in the 3D configs
    if history_len > 1 and cfg.policy != "random":
        # frame-window world models: seed the policy's history buffer with the (history_len - 1) * block dataset
        # steps preceding each start -- the episode's true past, so the first plan sees the frames
        # start-10, start-5, start like a training clip. A start closer than that to its episode's first
        # step keeps only whole blocks of real steps and gets the framework's own warm-up padding for the
        # rest (copies of the oldest real strided frame, zero actions; stable_worldmodel.buffer.HistoryBuffer).
        # Entry conventions follow HistoryBuffer: the action stored with a frame is the one that led into it;
        # the start frame's own entry (appended live by the policy) gets that action via 'last_action'.
        L = (history_len - 1) * block
        dataset._open()
        h5 = dataset.h5_file
        seed = {c: [] for c in (*win_obs, "action", "last_action")}
        for r, s0 in zip(random_episode_indices, eval_start_idx):
            r, s0 = int(r), int(s0)
            n = min(s0, L) // block * block                      # real prior steps, whole blocks only
            real = list(range(r - n, r))                         # rows of the real prior steps, oldest first
            rows = [real[0] if real else r] * (L - n) + real     # left-padded with the oldest real strided frame
            for c in win_obs:
                seed[c].append(np.stack([h5[c][k] for k in rows]))
            zero = np.zeros_like(h5["action"][r])
            seed["action"].append(np.stack([h5["action"][k - 1] if i >= L - n and k - 1 >= r - s0 else zero for i, k in enumerate(rows)]))
            seed["last_action"].append(h5["action"][r - 1] if s0 >= 1 else np.full_like(zero, np.nan))  # NaN = reset entry, as the World does
        policy.history_seed = {c: np.stack(v) for c, v in seed.items()}  # obs: (E, L, ...), action: (E, L, A), last_action: (E, A)
        n_short = int((eval_start_idx < L).sum())
        print(f"history seed: {L} steps before each of the {len(eval_start_idx)} starts ({n_short} starts padded)")
        if hasattr(model, "window"):  # the aggregator's window must be what the eval feeds it: history frames and goal frames
            assert history_len == goal_window == int(model.window), f"history_len {history_len} / goal_window {goal_window} / model.window {model.window} differ"

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path if cfg.get("video", True) else None,  # panel video needs dataset pixels
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()

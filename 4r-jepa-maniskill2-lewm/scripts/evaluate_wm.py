"""LeWM CEM/MPC evaluation, expressed in consecutive control-step model units.

The original five blocks of five actions become 25 single-action model steps.
The physical planning/execution horizon, goal offset, evaluation budget and
CEM hyperparameters are preserved. The temporal-unit conversion is reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for source in (
    ROOT / "src",
    ROOT / "vendor/le-wm",
    ROOT / "vendor/stable-worldmodel-lewm-compatible",
    ROOT / "vendor/ManiSkill2-Learn",
):
    sys.path.insert(0, str(source))


def make_eval_configs(device="cuda", smoke=False):
    """Convert model-step units while preserving physical control-step spans."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ROOT / "vendor/le-wm/config/eval/cube.yaml")
    solver_cfg = OmegaConf.load(ROOT / "vendor/le-wm/config/eval/solver/cem.yaml")
    solver_cfg.seed = cfg.seed
    solver_cfg.device = device
    official_plan = OmegaConf.to_container(cfg.plan_config, resolve=True)
    if smoke:
        cfg.eval.num_eval = 1
        cfg.eval.eval_budget = 5
        cfg.plan_config.horizon = 1
        cfg.plan_config.receding_horizon = 1
        solver_cfg.num_samples = 4
        solver_cfg.topk = 2
        solver_cfg.n_steps = 1
    source_plan = OmegaConf.to_container(cfg.plan_config, resolve=True)
    old_action_block = int(cfg.plan_config.action_block)
    cfg.plan_config.horizon *= old_action_block
    cfg.plan_config.receding_horizon *= old_action_block
    cfg.plan_config.action_block = 1
    from ms2_lewm.official import lewm_config
    cfg.plan_config.history_len = int(lewm_config().history_size)
    alignment = {
        "reason": "Consecutive-frame model predicts one native action step; preserve the original physical horizon.",
        "history_stride": 1,
        "model_action_dim": 7,
        "history_size": int(cfg.plan_config.history_len),
        "history_collection": "Every real control step, including cached MPC action execution; reset separately per environment.",
        "candidate_alignment": "Prepend the normalized executed H-1 context actions to future CEM candidates for the public JEPA rollout interface.",
        "official_plan": official_plan,
        "source_plan_after_smoke_overrides": source_plan,
        "adapted_plan": OmegaConf.to_container(cfg.plan_config, resolve=True),
        "original_control_step_horizon": int(source_plan["horizon"] * source_plan["action_block"]),
        "adapted_control_step_horizon": int(cfg.plan_config.horizon * cfg.plan_config.action_block),
        "original_control_steps_between_replans": int(source_plan["receding_horizon"] * source_plan["action_block"]),
        "adapted_control_steps_between_replans": int(cfg.plan_config.receding_horizon * cfg.plan_config.action_block),
    }
    return cfg, solver_cfg, alignment


def adapt_evaluation_cost(model):
    """Bridge future-only CEM actions to JEPA's context-inclusive rollout API."""
    import torch

    class ConsecutiveEvaluationCost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def get_cost(self, info, future_actions):
            lengths = info["_wm_history_length"][:, 0].long()
            context_actions = info["_wm_context_action"]
            costs = future_actions.new_empty(future_actions.shape[:2])
            # Reset environments can have shorter observed history than other
            # workers. Group equal lengths, omitting all padding from rollout.
            for length in lengths.unique().tolist():
                if not 1 <= length <= self.model.history_size:
                    raise ValueError("Invalid evaluation history length.")
                indices = torch.nonzero(lengths == length, as_tuple=True)[0]
                selected = {key: value.index_select(0, indices)
                            for key, value in info.items()
                            if key in ("pointcloud", "goal_pointcloud", "goal")}
                selected["pointcloud"] = selected["pointcloud"][:, :, -length:]
                past = context_actions.index_select(0, indices)
                past = past[:, :, -(length - 1):] if length > 1 else past[:, :, :0]
                future = future_actions.index_select(0, indices)
                aligned = torch.cat([past.to(future), future], dim=2)
                costs[indices] = self.model.get_cost(selected, aligned)
            return costs

    return ConsecutiveEvaluationCost()


def make_consecutive_policy(*, solver, config, process=None, transform=None, dataset_start_steps=None):
    """Collect actual online history before the unchanged official MPC policy."""
    from collections import deque
    import numpy as np
    from stable_worldmodel.policy import WorldModelPolicy

    class ConsecutiveWorldModelPolicy(WorldModelPolicy):
        def set_env(self, env):
            super().set_env(env)
            self._point_history = [deque(maxlen=self.cfg.history_len) for _ in range(env.num_envs)]
            self._executed_actions = [deque(maxlen=self.cfg.history_len - 1) for _ in range(env.num_envs)]
            self._last_step = [None] * env.num_envs
            self._flush_active = [False] * env.num_envs
            self._dataset_initial_pending = [dataset_start_steps is not None] * env.num_envs
            if dataset_start_steps is not None and len(dataset_start_steps) != env.num_envs:
                raise ValueError("Dataset start count must match the evaluation environment count.")

        def get_action(self, info_dict, **kwargs):
            info = dict(info_dict)
            current = np.asarray(info["pointcloud"])
            if current.shape[0] != self.env.num_envs or current.shape[1] != 1:
                raise ValueError("Official World must provide one current point-cloud frame per environment.")
            if "step_idx" not in info:
                raise ValueError("Evaluation history requires the official step_idx reset/step marker.")
            steps = np.asarray(info["step_idx"]).reshape(self.env.num_envs, -1)[:, -1]
            flush = np.asarray(info.get("_needs_flush", np.zeros(self.env.num_envs, dtype=bool))).reshape(-1)
            effective_flush = np.zeros(self.env.num_envs, dtype=bool)
            frame_batch, action_batch, lengths = [], [], []
            for i in range(self.env.num_envs):
                frames, actions = self._point_history[i], self._executed_actions[i]
                step = int(steps[i])
                if self._dataset_initial_pending[i]:
                    if step != int(dataset_start_steps[i]):
                        raise ValueError("Initial dataset step marker differs from the selected evaluation start.")
                    # Official dataset evaluation overwrites initial info with
                    # dataset step s, but the live wrapper next reports 1.
                    # This stored observation is local control step 0.
                    step = 0
                    self._dataset_initial_pending[i] = False
                reset_signal = bool(flush[i]) and not self._flush_active[i]
                self._flush_active[i] = bool(flush[i])
                reset_signal |= self._last_step[i] is not None and step < self._last_step[i]
                if reset_signal:
                    frames.clear()
                    actions.clear()
                    self._last_step[i] = None
                    effective_flush[i] = True
                if self._last_step[i] != step:
                    if frames:
                        if step != self._last_step[i] + 1:
                            raise ValueError("Evaluation must observe every consecutive control step.")
                        previous_action = np.asarray(info["action"])[i].reshape(-1, 7)[-1].copy()
                        if not np.isfinite(previous_action).all():
                            raise ValueError("Executed context actions must be finite.")
                        actions.append(previous_action)
                    frames.append(current[i, 0].copy())
                    self._last_step[i] = step
                lengths.append(len(frames))
                frame_batch.append(np.stack([frames[0]] * (self.cfg.history_len - len(frames)) + list(frames)))
                past = np.zeros((self.cfg.history_len - 1, 7), dtype=np.float32)
                if actions:
                    executed = np.stack(actions)
                    if "action" in self.process:
                        executed = self.process["action"].transform(executed)
                    past[-len(actions):] = executed
                action_batch.append(past)
            info["pointcloud"] = np.stack(frame_batch)
            info["_wm_context_action"] = np.stack(action_batch)
            info["_wm_history_length"] = np.asarray(lengths, dtype=np.int64)
            # EnvPool reuses its info dict; consume a persistent flush marker
            # only once, so cached actions still execute between MPC replans.
            info["_needs_flush"] = effective_flush
            # Padding only makes vectorized storage rectangular. The cost
            # adapter removes it, so no invented action reaches the model.
            return super().get_action(info, **kwargs)

    return ConsecutiveWorldModelPolicy(solver=solver, config=config, process=process, transform=transform)


def validate_evaluation_model_config(config):
    """Reject older block-action exports before instantiation or simulation."""
    if config.get("_target_") not in ("ms2_lewm.model.WorldModel", "ms2_lewm.model.PointCloudWorldModel"):
        raise ValueError("Evaluation requires this project's trained point-cloud WorldModel.")
    if config.get("history_stride") != 1 or config.get("action_dim") != 7:
        raise ValueError("Checkpoint must explicitly record history_stride=1 and action_dim=7; retrain older stride-5 models.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Official save_pretrained directory/checkpoint, or random")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    import hydra
    import numpy as np
    from omegaconf import OmegaConf
    from sklearn.preprocessing import StandardScaler
    import stable_worldmodel as swm
    from ms2_lewm.environment import dataset_eval_callables, ensure_renderer_headroom, make_world
    from ms2_lewm.official import LEWM, load_source

    cfg, solver_cfg, temporal_alignment = make_eval_configs(args.device, args.smoke)
    official_eval = load_source("_ms2_official_eval", LEWM / "eval.py")
    transforms = {
        "pixels": official_eval.img_transform(cfg),
        "goal": official_eval.img_transform(cfg),
    }
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=["action"])
    if dataset.get_dim("action") != 7:
        raise ValueError("Evaluation requires seven-dimensional native ManiSkill2 actions.")
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_col = dataset.get_col_data(col_name)
    step_col = dataset.get_col_data("step_idx")
    ep_indices = np.unique(ep_col)
    lengths = np.array([np.max(step_col[ep_col == i]) + 1 for i in ep_indices])
    max_start = dict(zip(ep_indices, lengths - cfg.eval.goal_offset_steps - 1))
    valid_indices = np.nonzero(step_col <= np.array([max_start[i] for i in ep_col]))[0]
    if len(valid_indices) - 1 < cfg.eval.num_eval:
        raise ValueError("Dataset has too few valid starts for the official goal offset / evaluation count")
    rng = np.random.default_rng(cfg.seed)
    # Preserve the public eval.py's exact endpoint-exclusive choice behavior.
    sample = rng.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    rows = np.sort(valid_indices[sample])
    starts = dataset.get_row_data(rows)["step_idx"].tolist()
    episodes = dataset.get_row_data(rows)[col_name].tolist()

    if args.model == "random":
        policy = swm.policy.RandomPolicy()
    else:
        # Use the pinned official resolver/loader, but validate the saved
        # temporal contract before any renderer or real environment is made.
        from stable_worldmodel.wm.utils import _resolve
        checkpoint_path, model_config = _resolve(args.model, swm.data.utils.get_cache_dir(sub_folder="checkpoints"))
        validate_evaluation_model_config(model_config)
        model = swm.wm.utils.load_pretrained(str(checkpoint_path))
        if model.history_stride != 1 or model.action_dim != 7:
            raise ValueError("Loaded model does not implement consecutive seven-dimensional action steps.")
        if args.device.startswith("cuda"):
            ensure_renderer_headroom(cfg.eval.num_eval, extra_model_mib=1024)
        model.to(args.device).eval().requires_grad_(False)
        actions = dataset.get_col_data("action")
        actions = actions[~np.isnan(actions).any(axis=1)]
        scaler = StandardScaler().fit(actions)
        solver = hydra.utils.instantiate(solver_cfg, model=adapt_evaluation_cost(model))
        plan_cfg = swm.PlanConfig(**OmegaConf.to_container(cfg.plan_config))
        policy = make_consecutive_policy(
            solver=solver,
            config=plan_cfg,
            dataset_start_steps=starts,
            process={"action": scaler},
            # Preserve the actual official RGB transforms even though this
            # pointcloud model consumes its separately added pointcloud key.
            transform=transforms,
        )

    world = make_world(
        num_envs=cfg.eval.num_eval,
        max_episode_steps=2 * cfg.eval.eval_budget,
        terminate_at_goal=True,
    )
    try:
        world.set_policy(policy)
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=starts,
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=episodes,
            callables=dataset_eval_callables(),
            video=args.output.parent / "videos" if (args.video or not args.smoke) else None,
        )
    finally:
        world.close()
    result = {
        "smoke": args.smoke,
        "dataset": str(args.dataset.resolve()),
        "model": args.model,
        "device": args.device,
        "stable_commit": "abdced49809d5eae38e24b27dc7b635c502c4812",
        "source_eval_config": "vendor/le-wm/config/eval/cube.yaml",
        "source_solver_config": "vendor/le-wm/config/eval/solver/cem.yaml",
        "eval": OmegaConf.to_container(cfg.eval),
        "plan": OmegaConf.to_container(cfg.plan_config),
        "temporal_alignment": temporal_alignment,
        "solver": OmegaConf.to_container(solver_cfg),
        "start_steps": starts,
        "episode_indices": episodes,
        "metrics": metrics,
    }
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=encode) + "\n")
    print(json.dumps(result, indent=2, default=encode))


if __name__ == "__main__":
    main()

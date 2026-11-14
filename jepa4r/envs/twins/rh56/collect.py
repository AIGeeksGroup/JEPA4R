"""Paired RGB / point-cloud collection on the RH56 twin, episode by episode as a plan lists them (split, reset seed,
coverage), with the feedback mix and the paired HDF5 writer of the Adroit collector.

coverage "protocol" draws the episode's plan with AdroitEnv._draw_plan (backbone / branch / random 0.5 / 0.3 / 0.2,
action noise up to 0.2), "expert" runs the noiseless expert. An episode has at most 200 steps (the native env may end
it earlier); every episode is saved as <split>_<seed>.npz (T actions, T+1 states), train / val episodes are also
appended to paired_rgb.h5 / paired_pcd.h5 (outgoing actions, the last row NaN), and manifest.json lists them. Success
is geometric: the object within 0.1 of the target for more than 25 steps, without overload. A bc_demo episode of the
expert must succeed."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from jepa4r.envs.twins.rh56._runtime import write_json


def read_plan(payload):
    return [(row["split"], int(row["seed"]), row["coverage"]) for row in payload["episodes"]]


def plan_config(coverage):
    return SimpleNamespace(
        force_type=None if coverage == "protocol" else 0,
        mix=(0.5, 0.3, 0.2),
        noise_max=0.2,
        episode_len=200,
        branch_tail_prob=0.5,
        burst_len=(10, 30),
    )


def draw_collection_plan(original_draw, coverage, rng):
    kind, noise, branch_t, branch_len = original_draw(plan_config(coverage), rng)
    if coverage == "expert":
        noise = 0.0
    actual = (
        "expert"
        if coverage == "expert"
        else "backbone"
        if kind == 0
        else "random"
        if kind == 2
        else "tail"
        if branch_len == 200
        else "burst"
    )
    return kind, noise, branch_t, branch_len, actual


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="JSON: episodes = [{split, seed, coverage}]")
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera", choices=("native", "opposite"), default="native")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    plan = read_plan(json.loads(args.plan.read_text(encoding="utf-8")))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    native = None
    try:
        from .native import NativeBridge, PACKAGE_ROOT, load_file

        factory = NativeBridge
        if args.camera == "opposite":
            from . import camera_preview

            factory = camera_preview.OppositeCameraBridge
        sys.path.insert(0, str(PACKAGE_ROOT.parent))
        from jepa4r.data.relocate.paired_writer import AdroitPairedWriter

        legacy_env = load_file("_rh56_adroit_plan", PACKAGE_ROOT / "envs/adroit/env.py")
        legacy_policy = load_file("_rh56_adroit_feedback", PACKAGE_ROOT / "envs/adroit/policy.py")
        manifest = dict(
            camera=args.camera,
            action_alignment="outgoing; last row NaN; NPZ T actions/T+1 states",
            episodes=[],
            auxiliary=[],
        )
        with AdroitPairedWriter(out / "paired_rgb.h5", out / "paired_pcd.h5") as writer:
            for split, seed, requested_coverage in plan:
                if native is None:
                    native = factory(seed)
                obs, _ = native.reset(seed)
                initial = native.get_state()
                rng = np.random.default_rng(np.random.SeedSequence([seed, 919]))
                kind, noise, branch_t, branch_len, coverage = draw_collection_plan(
                    legacy_env.AdroitEnv._draw_plan, requested_coverage, rng
                )
                policy = legacy_policy.AdroitFeedbackPolicy.__new__(legacy_policy.AdroitFeedbackPolicy)
                policy.expert = SimpleNamespace(action_dim=12, action=lambda env: env.expert_action(env.observation()))
                policy.rng = np.random.default_rng(np.random.SeedSequence([seed, 313]))
                policy.env = SimpleNamespace(
                    num_envs=1, envs=[SimpleNamespace(unwrapped=SimpleNamespace(native_env=native))]
                )
                frames = []
                observations = []
                props = []
                actions = []
                rewards = []
                traces = []
                states = []
                infos = []
                qpositions = []
                qvelocities = []
                for t in range(201):
                    frames.append(native.capture())
                    observations.append(obs.copy())
                    props.append(native.proprio())
                    states.append(native.get_state())
                    qpositions.append(native.env.data.qpos.copy())
                    qvelocities.append(native.env.data.qvel.copy())
                    if t == 200 or (infos and (infos[-1]["terminated"] or infos[-1]["truncated"])):
                        break
                    rand = kind == 2 or kind == 1 and branch_t <= t < branch_t + branch_len
                    info_for_policy = {"rand_action": np.array([[rand]]), "noise_std": np.array([[noise]])}
                    action = policy.get_action(info_for_policy)[0]
                    obs, reward, te, tr, info = native.step(action)
                    actions.append(action.copy())
                    rewards.append(reward)
                    infos.append(dict(info, terminated=bool(te), truncated=bool(tr), rand_action=bool(rand)))
                    if (t + 1) % 40 == 0:
                        print(f"{split} seed={seed} step={t + 1}", flush=True)
                T = len(actions)
                F = T + 1
                for i in range(T):
                    obj, tgt = frames[i + 1]["object_xyz"], frames[i + 1]["target_xyz"]
                    dist = float(np.linalg.norm(obj - tgt))
                    traces.append(np.r_[i, obj, tgt, dist, int(dist < 0.1)])
                tr = np.asarray(traces)
                goals = tr[:, 8].astype(bool)
                native_goals = np.array([i["goal_achieved"] for i in infos], dtype=bool)
                success = bool(goals.sum() > 25 and not native.env.overloaded)
                camera = dict(
                    mode=args.camera,
                    position=frames[0]["camera_xpos"].tolist(),
                    rotation=frames[0]["camera_xmat"].tolist(),
                    fovy=float(frames[0]["camera_fovy"]),
                    resolution=[256, 256],
                    rgb_msaa=8,
                    pcd_msaa=0,
                )
                episode_name = f"{split}_{seed}"
                arrays = dict(
                    pixels=np.stack([f["pixels"] for f in frames]),
                    xyz=np.stack([f["pcd_xyz"] for f in frames]),
                    rgb=np.stack([f["pcd_rgb"] for f in frames]),
                    label=np.stack([f["labels"] for f in frames]),
                    observations=np.asarray(observations),
                    proprio=np.asarray(props),
                    actions=np.asarray(actions),
                    rewards=np.asarray(rewards),
                    trace=tr,
                    integration=np.array([s["integration"] for s in states]),
                    qpos=np.asarray(qpositions),
                    qvel=np.asarray(qvelocities),
                    initial_state_json=np.array(json.dumps(initial, sort_keys=True)),
                    frame_stats=np.array(json.dumps([f["frame_stats"] for f in frames])),
                    seed=np.int64(seed),
                    dt=np.float64(0.01),
                    success=np.bool_(success),
                    native_goal_achieved=native_goals,
                    native_success=np.bool_(infos[-1]["success"]),
                    native_success_history=np.array([i["success"] for i in infos], dtype=bool),
                    rand_action_mask=np.array([i["rand_action"] for i in infos], dtype=bool),
                    sample_type=np.int8(kind),
                    noise_std=np.float64(noise),
                    branch_t=np.int32(branch_t),
                    branch_len=np.int32(branch_len),
                    camera_json=np.array(json.dumps(camera)),
                    type=np.array(coverage),
                    camera_xpos=frames[0]["camera_xpos"],
                    camera_xmat=frames[0]["camera_xmat"],
                    camera_fovy=frames[0]["camera_fovy"],
                )
                path = out / f"{episode_name}.npz"
                np.savez_compressed(path, **arrays)
                record = dict(
                    seed=seed,
                    split=split,
                    steps=T,
                    rows=F,
                    success=success,
                    final_distance=float(tr[-1, 7]),
                    overload=bool(native.env.overloaded),
                    npz=path.name,
                    sample_type=kind,
                    coverage=coverage,
                    noise_std=noise,
                    branch_t=branch_t,
                    branch_len=branch_len,
                    camera=camera,
                )
                if split in ("train", "val"):
                    record["episode_index"] = len(manifest["episodes"])
                    manifest["episodes"].append(record)
                    outgoing = np.r_[np.asarray(actions), np.full((1, 12), np.nan)].astype(np.float32)
                    episode = dict(
                        pixels=arrays["pixels"],
                        pcd_xyz=arrays["xyz"],
                        pcd_rgb=arrays["rgb"],
                        action=outgoing,
                        observation=arrays["observations"],
                        native_observation=arrays["observations"],
                        proprio=arrays["proprio"],
                        qpos=arrays["qpos"],
                        qvel=arrays["qvel"],
                        episode_seed=np.full(F, seed, np.int64),
                        step_idx=np.arange(F, dtype=np.int32),
                        sample_type=np.full(F, kind, np.int8),
                        noise_std=np.full(F, noise),
                        branch_t=np.full(F, branch_t, np.int32),
                        branch_len=np.full(F, branch_len, np.int32),
                        goal_achieved=np.r_[False, goals],
                        geometry_goal_achieved=np.r_[False, goals],
                        native_goal_achieved=np.r_[False, native_goals],
                        native_success=np.r_[False, [i["success"] for i in infos]],
                        reward=np.r_[0.0, rewards],
                        rand_action=np.r_[[i["rand_action"] for i in infos], False].astype(np.int8),
                        executed_action_valid=np.r_[np.ones(T), 0].astype(np.int8),
                        native_integration=arrays["integration"],
                        drive_p=np.array([s["p"] for s in states]),
                        drive_v=np.array([s["v"] for s in states]),
                        q_cmd=np.array([s["q_cmd"] for s in states]),
                    )
                    writer.write_episode(episode)
                else:
                    manifest["auxiliary"].append(record)
                    if split == "bc_demo" and not success:
                        write_json(out / "manifest.json", manifest)
                        raise RuntimeError(f"the expert failed the bc_demo episode {seed} (kept in {path})")
                write_json(out / "manifest.json", manifest)
                print(json.dumps(record), flush=True)
    finally:
        if native is not None:
            native.close()


if __name__ == "__main__":
    main()

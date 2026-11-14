"""Bounded native Relocate data checks and paired official demonstration replay."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(path, **report):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    artifacts = report.pop("artifacts", [])
    report.update(
        updated_at=time.time(),
        artifacts=[
            {"path": str(Path(p).resolve()), "bytes": Path(p).stat().st_size, "sha256": sha(p)} for p in artifacts
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def audit_pair(rgb_path, pcd_path, expected_episodes, episode_length=200):
    import h5py
    import numpy as np

    with h5py.File(rgb_path, "r") as a, h5py.File(pcd_path, "r") as b:
        required = {
            "ep_len",
            "ep_offset",
            "action",
            "observation",
            "next_observation",
            "proprio",
            "qpos",
            "qvel",
            "seed",
            "step_idx",
            "initial_state_sha256",
            "sample_type",
            "rand_action",
            "noise_std",
            "branch_t",
            "branch_len",
            "reward",
            "goal_achieved",
            "done",
            "time_limit",
            "target_pos",
        }
        for file in (a, b):
            missing = required - set(file)
            if missing:
                raise ValueError("missing required nonvisual fields: " + ",".join(sorted(missing)))
        lengths = a["ep_len"][:]
        offsets = a["ep_offset"][:]
        if expected_episodes < 1 or len(lengths) != expected_episodes or not np.all(lengths == episode_length):
            raise ValueError("episode count/length mismatch")
        if not np.array_equal(offsets, np.r_[0, np.cumsum(lengths)[:-1]]):
            raise ValueError("episode offsets mismatch")
        frames = int(lengths.sum())
        if not np.array_equal(a["ep_len"][:], b["ep_len"][:]) or not np.array_equal(offsets, b["ep_offset"][:]):
            raise ValueError("paired episode metadata mismatch")
        shared = set(a) - {"pixels"}
        if shared != set(b) - {"pcd_xyz", "pcd_rgb"}:
            raise ValueError("paired nonvisual field sets differ")
        for key in sorted(shared):
            if a[key].dtype != b[key].dtype:
                raise ValueError("paired dtype mismatch: " + key)
            if a[key].shape != b[key].shape or (key not in ("ep_len", "ep_offset") and len(a[key]) != frames):
                raise ValueError("shape mismatch: " + key)
            for start in range(0, len(a[key]), 64):
                x, y = a[key][start : start + 64], b[key][start : start + 64]
                if not np.array_equal(x, y):
                    raise ValueError("paired mismatch: " + key)
                if x.dtype.kind == "c" or (x.dtype.kind == "f" and not np.isfinite(x).all()):
                    raise ValueError("non-real/nonfinite " + key)
        dimensions = {
            "action": 30,
            "observation": 39,
            "next_observation": 39,
            "proprio": 30,
            "qpos": 36,
            "qvel": 36,
            "target_pos": 3,
        }
        for key, width in dimensions.items():
            if a[key].shape != (frames, width) or a[key].dtype.kind not in "fiu":
                raise ValueError("native shape/dtype mismatch: " + key)
        for key in required - set(dimensions) - {"ep_len", "ep_offset"}:
            if a[key].shape != (frames,):
                raise ValueError("metadata shape mismatch: " + key)
        for key in ("ep_len", "ep_offset", "seed", "step_idx", "sample_type", "branch_t", "branch_len"):
            if a[key].dtype.kind not in "iu":
                raise ValueError("integer metadata dtype required: " + key)
        seeds = a["seed"][offsets]
        initials = a["initial_state_sha256"][offsets]
        if len(set(seeds.tolist())) != expected_episodes:
            raise ValueError("duplicate seed")
        if len(set(initials.tolist())) != expected_episodes:
            raise ValueError("duplicate initial state")
        for episode, (offset, length) in enumerate(zip(offsets, lengths)):
            start, stop = int(offset), int(offset + length)
            if not np.all(a["seed"][start:stop] == seeds[episode]):
                raise ValueError("seed changed within an episode")
            if not np.all(a["initial_state_sha256"][start:stop] == initials[episode]):
                raise ValueError("initial state changed within an episode")
            if not np.array_equal(a["step_idx"][start:stop], np.arange(length)):
                raise ValueError("step_idx is not the native ordered sequence")
            if not np.array_equal(a["next_observation"][start : stop - 1], a["observation"][start + 1 : stop]):
                raise ValueError("next_observation is not aligned with the next pre-action observation")
        if a["pixels"].shape != (frames, 256, 256, 3) or a["pixels"].dtype != np.uint8:
            raise ValueError("RGB contract mismatch")
        if b["pcd_xyz"].shape != (frames, 1200, 3) or b["pcd_xyz"].dtype != np.float32:
            raise ValueError("PCD contract mismatch")
        if b["pcd_rgb"].shape != (frames, 1200, 3) or b["pcd_rgb"].dtype != np.uint8:
            raise ValueError("PCD RGB mismatch")
        blank = np.zeros(frames, dtype=bool)
        empty = np.zeros(frames, dtype=bool)
        for start in range(0, frames, 64):
            stop = min(start + 64, frames)
            actions = a["action"][start:stop]
            if np.any((actions < -1) | (actions > 1)):
                raise ValueError("mixed WM action is outside [-1,1]")
            if not np.array_equal(a["proprio"][start:stop], a["observation"][start:stop, :30]):
                raise ValueError("proprio differs from observation[:30]")
            cloud = b["pcd_xyz"][start:stop]
            if not np.isfinite(cloud).all():
                raise ValueError("nonfinite point cloud")
            pixels = a["pixels"][start:stop]
            blank[start:stop] = np.ptp(pixels.reshape(stop - start, -1), axis=1) == 0
            empty[start:stop] = np.all(np.ptp(cloud, axis=1) == 0, axis=1)
        for offset, length in zip(offsets, lengths):
            start, stop = int(offset), int(offset + length)
            if blank[start:stop].all():
                raise ValueError("entire episode contains blank RGB")
            if empty[start:stop].all():
                raise ValueError("entire episode contains empty/degenerate point clouds")
        return {
            "passed": True,
            "episodes": expected_episodes,
            "frames": frames,
            "seeds": seeds.tolist(),
            "initial_states": [x.decode() for x in initials],
            "blank_rgb_frames": int(blank.sum()),
            "empty_cloud_frames": int(empty.sum()),
        }


def audit(args):
    output = getattr(args, "report", None) or args.data_root / "audit.json"
    if Path(output).exists():
        raise FileExistsError(output)
    artifacts = [args.data_root / (s + "_" + r + ".h5") for s in ("train", "test") for r in ("rgb", "pcd")]
    common = dict(
        train_episodes=args.train_episodes,
        test_episodes=args.test_episodes,
        validator_sha256=sha(__file__),
        scope="data validation; original producer and prior audits unchanged",
    )
    try:
        reports = {
            split: audit_pair(args.data_root / (split + "_rgb.h5"), args.data_root / (split + "_pcd.h5"), count)
            for split, count in (("train", args.train_episodes), ("test", args.test_episodes))
        }
        if set(reports["train"]["seeds"]) & set(reports["test"]["seeds"]):
            raise ValueError("train/test seeds overlap")
        if set(reports["train"]["initial_states"]) & set(reports["test"]["initial_states"]):
            raise ValueError("train/test initial states overlap")
        return receipt(output, passed=True, actual_exit_code=0, reports=reports, artifacts=artifacts, **common)
    except Exception as error:
        receipt(
            output,
            passed=False,
            actual_exit_code=1,
            error=repr(error),
            artifacts=[p for p in artifacts if p.is_file()],
            **common,
        )
        raise


def rgb_demo(demo, env, capture):
    import numpy as np

    actions = np.asarray(demo["actions"])
    states = np.asarray(demo["state_obs"])
    if actions.shape != (200, 30) or states.shape != (200, 39):
        raise ValueError("official Relocate requires 200x30 actions and 200x39 states")
    env.set_env_state(deepcopy(demo["init_state_dict"]))
    frames = []
    for t, action in enumerate(actions):
        np.testing.assert_allclose(env.get_obs(), states[t], atol=1e-6, rtol=0)
        before = env.get_env_state()
        frame = capture()
        if frame.shape != (256, 256, 3) or frame.dtype != np.uint8 or frame.std() <= 1:
            raise ValueError("invalid/blank RGB frame")
        for key, value in before.items():
            np.testing.assert_array_equal(value, env.get_env_state()[key])
        frames.append(frame)
        _, reward, _, info = env.step(action.copy())
        np.testing.assert_allclose(reward, demo["rewards"][t], atol=1e-6, rtol=0)
        if bool(info["goal_achieved"]) != bool(demo["env_infos"]["goal_achieved"][t]):
            raise ValueError("official success replay differs")
    pixels = np.asarray(frames)
    result = dict(demo, images=pixels)
    for key in set(demo) - {"images"}:
        if pickle.dumps(result[key], protocol=4) != pickle.dumps(demo[key], protocol=4):
            raise ValueError("official field changed: " + key)
    return result


def validated_official_source(source, expected_sha, loader):
    """Validate all actual decoded demos before choosing the render budget."""
    from jepa4r.data.relocate import official_contract

    source = Path(source).expanduser().resolve()
    binding = source.parent / "manifest.json" if source.is_dir() else source
    count, read, digest = loader(source, str(expected_sha).lower())
    if count != 100:
        raise ValueError("require all 100 official source demonstrations before smoke/full replay")
    records = []
    artifacts = [binding, Path(official_contract.__file__)]
    for index in range(100):
        demo, origin = read(index)
        records.append(official_contract.demo_content_record(index, demo))
        artifacts.append(Path(origin["path"]).resolve())
    content = official_contract.validate_content_records(records)
    return dict(
        read=read, source_sha256=digest, content=content, records=records, artifacts=list(dict.fromkeys(artifacts))
    )


def official(args):
    from jepa4r.data.relocate.convert_adroit_demos_3d import load_sources, convert_demo

    source = validated_official_source(args.source, args.source_sha256, load_sources)
    if args.count not in (1, 100):
        raise ValueError("official render budget must be smoke1 or full100")
    read = source["read"]
    source_sha = source["source_sha256"]
    content = source["content"]
    args.output.mkdir(parents=True, exist_ok=False)
    validation_path = args.output / "content_validation.json"
    with validation_path.open("x", encoding="utf-8") as stream:
        json.dump(dict(content, records=source["records"]), stream, indent=2, allow_nan=False)
    from jepa4r.policy.bc.run_adroit_bc3d import build_point_clouds
    from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

    rgb = RelocateEnvV0()
    pcd = RelocateEnvV0()
    converted_rgb = []
    converted_pcd = []
    try:
        if int(rgb.model.vis.quality.offsamples) != 8:
            raise ValueError("official RGB requires native MSAA8")
        rgb.seed(12345)
        rgb.reset()
        rgb.sim.render(width=256, height=256, camera_name="vil_camera", device_id=0)

        def capture():
            return rgb.sim.render(width=256, height=256, camera_name="vil_camera", device_id=0)[::-1].copy()

        sampler = build_point_clouds(pcd.sim, "relocate-v0")
        pcd.seed(12345)
        pcd.reset()
        sampler.seed(12345)
        sampler()
        for index in range(args.count):
            demo, _ = read(index)
            sampler.seed(12345 + index)
            converted_rgb.append(rgb_demo(demo, rgb, capture))
            converted_pcd.append(convert_demo(demo, pcd, sampler))
            for key in ("actions", "state_obs", "rewards", "init_state_dict", "env_infos"):
                if pickle.dumps(converted_rgb[-1][key], protocol=4) != pickle.dumps(converted_pcd[-1][key], protocol=4):
                    raise ValueError("official paired nonvisual field differs: " + key)
        paths = []
        for route, demos in (("rgb", converted_rgb), ("pcd", converted_pcd)):
            path = args.output / route / "relocate-v0.pickle"
            path.parent.mkdir()
            with path.open("xb") as stream:
                pickle.dump(demos, stream, protocol=4)
            paths.append(path)
        return receipt(
            args.output / "report.json",
            passed=True,
            actual_exit_code=0,
            demos=args.count,
            frames=args.count * 200,
            source_sha256=source_sha,
            raw_actions_preserved=True,
            paired_nonvisual_equal=True,
            validated_source_count=content["count"],
            policy_critical_content_sha256=content["policy_critical_content_sha256"],
            rgb_msaa=8,
            pcd_msaa=0,
            artifacts=paths + [validation_path] + source["artifacts"],
        )
    finally:
        rgb.close()
        pcd.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-episodes", type=int, required=True)
    p.add_argument("--test-episodes", type=int, required=True)
    p.add_argument(
        "--report",
        type=Path,
        help="Fresh supplemental receipt outside the original dataset; never overwrite an existing audit.",
    )
    p = sub.add_parser("official")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    audit(args) if args.stage == "audit" else official(args)


if __name__ == "__main__":
    main()

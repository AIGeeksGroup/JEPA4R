"""Convert official Adroit Pen demos to a separately tracked PCD demo pickle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
LEWM_ROOT = HERE.parents[2]
CB_ROOT = LEWM_ROOT / "bench" / "cortexbench"
PINNED = Path(os.environ.get("CORTEXBENCH_PINNED_SRC", "/root/autodl-tmp/workspace/users/lewm/cortexbench-official-pinned-src"))
for path in (LEWM_ROOT, CB_ROOT, next(PINNED.glob("mj_envs-*"), PINNED / "mj_envs"), next(PINNED.glob("mjrl-*"), PINNED / "mjrl")):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", "/root/.mujoco/mujoco200")
os.environ.setdefault("MUJOCO_PY_MJKEY_PATH", "/root/.mujoco/mjkey.txt")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["LD_LIBRARY_PATH"] = f"{Path(os.environ['MUJOCO_PY_MUJOCO_PATH']) / 'bin'}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")

from bench.cortexbench.adroit.convert_demos_3d import convert_pen_demo  # noqa: E402
from bench.cortexbench.adroit.pcd import AdroitPCDConfig  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".partial.{os.getpid()}")
    with temp.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)
    temp.replace(path)


def pen_env():
    from mj_envs.envs.hand_manipulation_suite.pen_v0 import PenEnvV0

    return PenEnvV0()


def run(args: argparse.Namespace) -> dict:
    input_path = args.input_pickle.resolve()
    output_path = args.output_pickle.resolve()
    state_dir = args.state_dir.resolve()
    state_path = state_dir / "state.json"
    exit_path = state_dir / "exit_code"
    report_dir = state_dir / "reports"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    config = AdroitPCDConfig(image_size=args.image_size, num_points=args.num_points, voxel_size=args.voxel_size)
    input_hash = sha256(input_path)
    config_record = {
        "camera": config.camera,
        "image_size": config.image_size,
        "num_points": config.num_points,
        "voxel_size": config.voxel_size,
        "seed_base": args.seed_base,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = (str(input_path), input_hash, str(output_path), config_record, args.max_demos)
        recorded = (state.get("input_pickle"), state.get("input_sha256"), state.get("output_pickle"),
                    state.get("config"), state.get("requested_demos"))
        if recorded != expected:
            raise ValueError("existing conversion state does not match the requested input/output/config")
        if state.get("stage") == "finished" and (state_dir / "summary.json").exists() and output_path.exists():
            summary = json.loads((state_dir / "summary.json").read_text(encoding="utf-8"))
            if summary.get("status") == "PASS" and summary.get("output_sha256") == sha256(output_path):
                return summary
    else:
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
        state = {
            "stage": "loading",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "input_pickle": str(input_path),
            "input_sha256": input_hash,
            "output_pickle": str(output_path),
            "config": config_record,
            "total_demos": None,
            "requested_demos": args.max_demos,
            "completed_demos": 0,
            "failed_demos": [],
            "exit_code": None,
        }
        atomic_json(state_path, state)

    with input_path.open("rb") as handle:
        demos = pickle.load(handle)
    total = len(demos) if args.max_demos < 0 else min(len(demos), args.max_demos)
    if state.get("total_demos") not in (None, total):
        raise ValueError(f"resume demo count mismatch: state={state.get('total_demos')} current={total}")
    state.update(stage="converting", total_demos=total, updated_at=utc_now())
    atomic_json(state_path, state)

    converted = []
    partial_dir = state_dir / "demos"
    partial_dir.mkdir(parents=True, exist_ok=True)
    env = pen_env()
    try:
        for index in range(total):
            demo_path = partial_dir / f"demo_{index:04d}.pickle"
            report_path = report_dir / f"demo_{index:04d}.json"
            if demo_path.exists() and report_path.exists():
                with demo_path.open("rb") as handle:
                    converted_demo = pickle.load(handle)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if (report.get("pcd_finite") is not True or report.get("frames") != len(demos[index]["actions"])
                        or np.asarray(converted_demo.get("images")).shape != (report["frames"], config.num_points, 6)):
                    raise ValueError(f"invalid partial demo artifact at index {index}")
                converted.append(converted_demo)
                state.update(current_demo=index, completed_demos=index + 1, updated_at=utc_now())
                atomic_json(state_path, state)
                continue

            state.update(stage="converting", current_demo=index, updated_at=utc_now())
            atomic_json(state_path, state)
            converted_demo, report = convert_pen_demo(
                demos[index],
                env,
                rng=np.random.default_rng(args.seed_base + index),
                config=config,
                replay_atol=args.replay_atol,
            )
            report.update({"demo": index, "seed": args.seed_base + index})
            atomic_pickle(demo_path, converted_demo)
            atomic_json(report_dir / f"demo_{index:04d}.json", report)
            converted.append(converted_demo)
            state.update(completed_demos=index + 1, updated_at=utc_now())
            atomic_json(state_path, state)
            print(json.dumps({"demo": index, "completed": index + 1, "total": total, "report": report}, sort_keys=True), flush=True)
    finally:
        env.close()

    state.update(stage="writing", updated_at=utc_now())
    atomic_json(state_path, state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(output_path.name + f".partial.{os.getpid()}")
    with temp_output.open("wb") as handle:
        pickle.dump(converted, handle, protocol=4)
    if output_path.exists():
        temp_output.unlink()
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    temp_output.replace(output_path)
    output_hash = sha256(output_path)
    summary = {
        "status": "PASS",
        "task": "pen-v0",
        "demos": total,
        "input_pickle": str(input_path),
        "input_sha256": input_hash,
        "output_pickle": str(output_path),
        "output_sha256": output_hash,
        "config": state["config"],
        "state_replay": "per-frame state_obs replay checked",
        "pcd_contract": "images=(T,1200,6) float32, xyz metres + rgb 0..255",
        "created_at": utc_now(),
    }
    atomic_json(state_dir / "summary.json", summary)
    state.update(stage="finished", finished_at=utc_now(), updated_at=utc_now(), output_sha256=output_hash, exit_code=0)
    atomic_json(state_path, state)
    exit_path.write_text("0\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pickle", type=Path, required=True)
    parser.add_argument("--output-pickle", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--max-demos", type=int, default=-1)
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-points", type=int, default=1200)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--replay-atol", type=float, default=1e-6)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        state_dir = args.state_dir.resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        failure = {"stage": "failed", "finished_at": utc_now(), "error": repr(exc), "traceback": traceback.format_exc(), "exit_code": 1}
        atomic_json(state_dir / "state.json", failure)
        (state_dir / "exit_code").write_text("1\n", encoding="utf-8")
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

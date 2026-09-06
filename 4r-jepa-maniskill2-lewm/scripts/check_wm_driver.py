#!/usr/bin/env python
"""CPU integration test using an explicitly SYNTHETIC UNIT TEST FIXTURE.

This tests storage/Lightning/optimizer/checkpoint APIs, not expert collection,
learning quality, or task performance. It never calls a simulator, renderer,
or collector. All generated data and checkpoints remain under artifacts/tests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
NOTICE = (
    "SYNTHETIC UNIT TEST FIXTURE ONLY. NOT expert data; NOT a world-model "
    "quality result; NOT collected from a simulator. Never use as a training dataset."
)
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
SOURCE_PATHS = [ROOT / "src", ROOT / "vendor/ManiSkill2-Learn", ROOT / "vendor/pytorch3d",
                ROOT / "vendor/le-wm", ROOT / "vendor/stable-worldmodel-lewm-compatible"]
for path in SOURCE_PATHS:
    sys.path.insert(0, str(path))
os.environ["PYTHONPATH"] = os.pathsep.join(str(path) for path in SOURCE_PATHS)


def make_fixture(folder: Path) -> Path:
    import h5py
    import numpy as np
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    folder.mkdir(parents=True, exist_ok=False)
    (folder / "SYNTHETIC_UNIT_TEST_ONLY.txt").write_text(NOTICE + "\n")
    fixture = folder / "SYNTHETIC_UNIT_TEST_FIXTURE.h5"
    rng = np.random.default_rng(88127)
    with HDF5Writer(fixture, mode="error") as writer:
        for episode in range(2):
            n_frames = 40
            observation = rng.normal(size=(n_frames, 30)).astype(np.float32)
            writer.write_episode({
                "pixels": rng.integers(0, 256, size=(n_frames, 224, 224, 3), dtype=np.uint8),
                "pointcloud": rng.normal(size=(n_frames, 1200, 6)).astype(np.float32),
                "action": rng.uniform(-1, 1, size=(n_frames, 7)).astype(np.float32),
                "observation": observation,
                "proprio": observation.copy(),
                "episode_idx": np.full(n_frames, episode, dtype=np.int64),
                "step_idx": np.arange(n_frames, dtype=np.int64),
            })
    with h5py.File(fixture, "r+") as output:
        output.attrs["SYNTHETIC_UNIT_TEST_ONLY"] = True
        output.attrs["provenance"] = NOTICE
        output.attrs["random_seed"] = 88127
    return fixture


def run_driver(folder, fixture, run_name, *, resume=None, capture_mid=False):
    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(folder / "test_only_cache")
    env["LOCAL_DATASET_DIR"] = str(folder / "test_only_cache")
    env["XDG_CACHE_HOME"] = str(folder / "test_only_cache")
    env["TORCH_HOME"] = str(folder / "test_only_cache/torch")
    env["HF_HOME"] = str(folder / "test_only_cache/huggingface")
    env["SPT_CACHE_DIR"] = str(folder / "test_only_cache/stable-pretraining")
    command = [sys.executable, str(ROOT / "scripts/train_wm.py"), "--dataset", str(fixture),
               "--run-name", run_name, "--smoke"]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    log_path = folder / f"{run_name}.log"
    mid_checkpoint = folder / "actual_epoch_1_full_state.ckpt"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if capture_mid and not mid_checkpoint.exists():
                for candidate in folder.rglob("epoch=0-step=*.ckpt"):
                    shutil.copyfile(candidate, mid_checkpoint)
                    break
            time.sleep(0.1)
    if process.returncode:
        raise RuntimeError(f"Driver failed (exit {process.returncode}); see {log_path}\n" + log_path.read_text()[-14000:])
    summary_path = folder / "test_only_cache/checkpoints" / run_name / "summary.json"
    summary = json.loads(summary_path.read_text())
    return {"command": command, "log": str(log_path), "summary": summary}, mid_checkpoint


def check(args):
    import torch
    from stable_worldmodel.wm.utils import load_pretrained

    torch.set_num_threads(1)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = ROOT / "artifacts/tests" / f"wm_driver_fixture_{token}"
    fixture = make_fixture(folder)
    print(json.dumps({"notice": NOTICE, "fixture": str(fixture), "started": token}), flush=True)
    initial, middle = run_driver(folder, fixture, "wm_driver_fixture", capture_mid=True)
    if not middle.exists():
        raise RuntimeError(f"Actual first-epoch full checkpoint was not captured; inspect {folder}")
    mid_state = torch.load(middle, map_location="cpu", weights_only=False)
    assert mid_state["global_step"] == 2, mid_state["global_step"]
    assert mid_state["optimizer_states"] and mid_state["lr_schedulers"]
    first_summary = initial["summary"]
    assert first_summary["global_step"] == 4
    assert first_summary["action_dim"] == 7
    assert first_summary["history_stride"] == 1
    assert first_summary["data_frameskip"] == 1
    assert first_summary["finite_parameters"]
    assert all(change > 0 for change in first_summary["parameter_l1_change"].values())
    full_state = torch.load(first_summary["checkpoint"], map_location="cpu", weights_only=False)
    assert full_state["optimizer_states"] and full_state["lr_schedulers"]
    assert full_state["global_step"] == 4
    resumed, _ = run_driver(folder, fixture, "wm_driver_fixture_resume", resume=middle)
    assert resumed["summary"]["global_step"] == 4 > mid_state["global_step"]
    assert resumed["summary"]["finite_parameters"]
    assert "Restored all states" in Path(resumed["log"]).read_text()
    resumed_state = torch.load(resumed["summary"]["checkpoint"], map_location="cpu", weights_only=False)
    groups = {"frame_encoder": "model.encoder.frame_encoder.",
              "temporal_encoder": "model.encoder.temporal_encoder.", "predictor": "model.predictor."}
    resume_changes = {
        group: sum(float((value - mid_state["state_dict"][name]).abs().sum())
                   for name, value in resumed_state["state_dict"].items() if name.startswith(prefix))
        for group, prefix in groups.items()
    }
    assert all(change > 0 for change in resume_changes.values())
    def optimizer_step(checkpoint):
        return max(float(state["step"]) for optimizer in checkpoint["optimizer_states"]
                   for state in optimizer["state"].values() if "step" in state)
    before_step, after_step = optimizer_step(mid_state), optimizer_step(resumed_state)
    assert after_step > before_step

    model_file = folder / "test_only_cache/checkpoints/wm_driver_fixture/weights_epoch_2.pt"
    model = load_pretrained(str(model_file), cache_dir=str(folder / "test_only_cache")).eval()
    export_config = json.loads((model_file.parent / "config.json").read_text())
    assert export_config["history_stride"] == model.history_stride == 1
    assert export_config["action_dim"] == model.action_dim == 7
    points = torch.randn(2, 4, 24, 6)
    action = torch.randn(2, 4, 7)
    with torch.no_grad():
        output = model({"pointcloud": points, "action": action})
    assert output["prediction"].shape == (2, 512)
    assert torch.isfinite(output["loss"])
    assert all(param.device.type == "cpu" for param in model.parameters())
    exported = torch.load(model_file, map_location="cpu", weights_only=False)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, exported[name], rtol=0, atol=0)
    report = {
        "notice": NOTICE,
        "synthetic_unit_test_only": True,
        "real_expert_collection": False,
        "quality_or_success_evaluation": False,
        "passed": True,
        "fixture": str(fixture),
        "initial_training": initial,
        "resume_training": resumed,
        "captured_full_checkpoint": str(middle),
        "captured_checkpoint_step": mid_state["global_step"],
        "full_checkpoint_optimizer_and_scheduler_present": True,
        "resume_from_checkpoint_parameter_l1_change": resume_changes,
        "adam_step_before_resume": before_step,
        "adam_step_after_resume": after_step,
        "official_model_reload": str(model_file),
        "history_size": model.history_size,
        "history_stride": model.history_stride,
        "data_frameskip": first_summary["data_frameskip"],
        "action_dim": model.action_dim,
        "training_frame_offsets": [0, 1, 2, 3],
        "prediction_departure_action_offset": 2,
        "reloaded_prediction_shape": list(output["prediction"].shape),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "all_model_parameters_on_cpu": True,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "hardware_metrics_scope": "Upstream callback reports whole-host hardware, including unrelated existing GPU jobs; this test uses CPU only.",
    }
    (folder / "integration_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    check(parser.parse_args())

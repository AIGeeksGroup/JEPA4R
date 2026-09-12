"""Verify the isolated LiftCube MPC runtime and its frozen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_importable() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


ensure_repo_importable()


EXPECTED_CHECKPOINT_SHA256 = (
    "346ffd87ad74d9b55030258eb3c4cc66ab9b38a113d70e394b5679ab2976905d"
)
DEFAULT_VULKAN_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"
EXPECTED_SYSTEM_PACKAGES = {
    "libgl1": "1.4.0-1",
    "libglib2.0-0": "2.72.4-0ubuntu2.9",
    "libvulkan1": "1.3.204.1-2",
}
EXPECTED_RPM_SYSTEM_PACKAGES = {
    "libgl1": ("libglvnd-glx", "libglvnd-glx-1.3.2-1.el8.x86_64"),
    "libglib2.0-0": ("glib2", "glib2-2.56.4-9.el8.x86_64"),
    "libvulkan1": (
        "vulkan-loader",
        "vulkan-loader-1.2.162.0-1.el8.x86_64",
    ),
}
EXPECTED_VERSIONS = {
    "addict": "2.4.0",
    "einops": "0.8.2",
    "gymnasium": "1.3.0",
    "h5py": "3.11.0",
    "huggingface-hub": "1.29.0",
    "imageio": "2.37.4",
    "imageio-ffmpeg": "0.6.0",
    "mani-skill2": "0.5.3",
    "numpy": "1.23.5",
    "omegaconf": "2.3.1",
    "opencv-python": "4.11.0.86",
    "packaging": "24.1",
    "pytest": "8.3.5",
    "pyyaml": "6.0.3",
    "rtree": "1.4.1",
    "sapien": "2.2.2",
    "scipy": "1.12.0",
    "sonata": "1.0",
    "spconv-cu120": "2.3.6",
    "timm": "1.0.29",
    "torch": "2.5.1+cu121",
    "torch-scatter": "2.1.2+pt25cu121",
    "torchvision": "0.20.1+cu121",
    "tqdm": "4.66.5",
    "transforms3d": "0.4.2",
    "trimesh": "5.0.0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_system_package(package: str) -> tuple[str, str]:
    if shutil.which("dpkg-query"):
        expected = EXPECTED_SYSTEM_PACKAGES[package]
        command = ["dpkg-query", "-W", "-f=${Version}", package]
    elif shutil.which("rpm"):
        rpm_package, expected = EXPECTED_RPM_SYSTEM_PACKAGES[package]
        command = ["rpm", "-q", rpm_package]
    else:
        raise FileNotFoundError("neither dpkg-query nor rpm is available")
    actual = subprocess.check_output(command, text=True).strip()
    return actual, expected


def _patch_robot_link_ids(wrapper_base: type) -> None:
    if hasattr(wrapper_base, "robot_link_ids"):
        return

    def get_robot_link_ids(self):
        return getattr(self, "_robot_link_ids", self.unwrapped.robot_link_ids)

    def set_robot_link_ids(self, value):
        self._robot_link_ids = value

    wrapper_base.robot_link_ids = property(get_robot_link_ids, set_robot_link_ids)


def configure_vulkan_icd() -> None:
    """Select the NVIDIA Vulkan ICD when the host exposes an explicit file."""
    if not os.environ.get("VK_ICD_FILENAMES") and Path(DEFAULT_VULKAN_ICD).is_file():
        os.environ["VK_ICD_FILENAMES"] = DEFAULT_VULKAN_ICD


configure_vulkan_icd()


def _renderer_smoke() -> dict[str, Any]:
    configure_vulkan_icd()
    import gymnasium as gym
    import mani_skill2.envs  # noqa: F401
    import numpy as np
    from mani_skill2.utils.wrappers.observation import BaseGymObservationWrapper

    _patch_robot_link_ids(BaseGymObservationWrapper)
    env = gym.make(
        "LiftCube-v0",
        obs_mode="pointcloud_robot_seg",
        control_mode="pd_ee_delta_pose",
    )
    try:
        reset_result = env.reset(seed=4100)
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[-1] = 1.0
        step_result = env.step(action)
        stepped_observation = step_result[0]
        cloud = stepped_observation.get("pointcloud", stepped_observation)
        xyzw = np.asarray(cloud["xyzw"], dtype=np.float32)
        if xyzw.ndim != 2 or xyzw.shape[1] != 4 or not np.isfinite(xyzw).all():
            raise ValueError(f"invalid point cloud shape or values: {xyzw.shape}")
        return {
            "observation_points": int(xyzw.shape[0]),
            "observation_shape": list(xyzw.shape),
            "step_tuple_length": len(step_result),
        }
    finally:
        env.close()


def check_environment(
    checkpoint: Path,
    require_renderer: bool,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    report: dict[str, Any] = {
        "ok": False,
        "python": sys.version.split()[0],
        "packages": {},
        "system_packages": {},
        "checkpoint": {
            "path": str(checkpoint),
            "expected_sha256": expected_checkpoint_sha256,
        },
        "errors": [],
    }

    if sys.version_info[:2] != (3, 11):
        report["errors"].append(f"Python 3.11 required, got {sys.version.split()[0]}")

    for package in EXPECTED_SYSTEM_PACKAGES:
        try:
            actual, expected = _query_system_package(package)
            report["system_packages"][package] = actual
            if actual != expected:
                report["errors"].append(
                    f"{package} version must be {expected}, got {actual}"
                )
        except (OSError, subprocess.CalledProcessError) as exc:
            report["system_packages"][package] = None
            report["errors"].append(f"system package check failed for {package}: {exc!r}")

    for distribution, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
            report["packages"][distribution] = actual
            if actual != expected:
                report["errors"].append(
                    f"{distribution} version must be {expected}, got {actual}"
                )
        except importlib.metadata.PackageNotFoundError:
            report["packages"][distribution] = None
            report["errors"].append(f"missing package: {distribution}=={expected}")

    if checkpoint.is_file():
        actual_sha = _sha256(checkpoint)
        report["checkpoint"]["sha256"] = actual_sha
        if actual_sha != expected_checkpoint_sha256:
            report["errors"].append(
                f"checkpoint SHA-256 must be {expected_checkpoint_sha256}, got {actual_sha}"
            )
    else:
        report["errors"].append(f"missing checkpoint: {checkpoint}")

    try:
        import torch

        report["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "runtime": torch.version.cuda,
        }
        if torch.version.cuda != "12.1":
            report["errors"].append(f"CUDA runtime must be 12.1, got {torch.version.cuda}")
        if not torch.cuda.is_available():
            report["errors"].append("CUDA is unavailable")
        else:
            value = (torch.tensor([2.0], device="cuda") * 3.0).item()
            report["cuda"]["device"] = torch.cuda.get_device_name(0)
            report["cuda"]["tensor_result"] = value
            if value != 6.0:
                report["errors"].append(f"CUDA tensor result must be 6.0, got {value}")

        if report["checkpoint"].get("sha256") == expected_checkpoint_sha256:
            from scripts.eval_maniskill_group1_rollout import load_checkpoint_model

            model, state = load_checkpoint_model(checkpoint, torch.device("cuda"))
            report["checkpoint"]["step"] = int(state["step"])
            report["checkpoint"]["strict_model_load"] = True
            report["model"] = {
                "T": int(model.cfg.T),
                "K": int(model.cfg.K),
                "D": int(model.cfg.D),
                "skip_frame": int(model.cfg.skip_frame),
            }
    except Exception as exc:  # checker must emit one structured report on failure
        report["errors"].append(f"CUDA/project/checkpoint check failed: {exc!r}")

    try:
        import huggingface_hub
        import mani_skill2
        import sapien.core
        import sonata.model
        import torch_scatter

        report["imports"] = {
            "huggingface_hub": getattr(huggingface_hub, "__version__", "imported"),
            "mani_skill2": getattr(mani_skill2, "__version__", "imported"),
            "sapien": "imported",
            "sonata.model": "imported",
            "torch_scatter": getattr(torch_scatter, "__version__", "imported"),
        }
    except Exception as exc:
        report["errors"].append(f"MPC runtime import failed: {exc!r}")

    if require_renderer:
        try:
            report["renderer"] = _renderer_smoke()
        except Exception as exc:
            report["errors"].append(f"LiftCube renderer smoke failed: {exc!r}")

    report["ok"] = not report["errors"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=EXPECTED_CHECKPOINT_SHA256,
    )
    parser.add_argument("--require-renderer", action="store_true")
    args = parser.parse_args()
    report = check_environment(
        args.checkpoint,
        args.require_renderer,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

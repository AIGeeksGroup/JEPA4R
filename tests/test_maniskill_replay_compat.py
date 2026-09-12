import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_replay_entrypoint_patches_robot_link_ids_before_delegating() -> None:
    entrypoint = ROOT / "scripts" / "replay_maniskill_trajectory.py"
    assert entrypoint.is_file(), "project-owned ManiSkill replay entry point is missing"
    spec = importlib.util.spec_from_file_location("replay_maniskill_trajectory", entrypoint)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class WrapperBase:
        def __init__(self, unwrapped):
            self.unwrapped = unwrapped

    replayed = []

    def replay_main(args):
        wrapped = WrapperBase(SimpleNamespace(robot_link_ids=(3, 7)))
        replayed.append(args)
        assert args == "parsed-args"
        assert wrapped.robot_link_ids == (3, 7)

    module.run_replay("parsed-args", WrapperBase, replay_main)

    assert replayed == ["parsed-args"]


def test_generic_wrapper_forwarding_handles_replay_attributes() -> None:
    entrypoint = ROOT / "scripts" / "replay_maniskill_trajectory.py"
    spec = importlib.util.spec_from_file_location("replay_maniskill_trajectory", entrypoint)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Wrapper:
        def __init__(self, env):
            self.env = env

    base = SimpleNamespace(obs_mode="pointcloud_robot_seg", control_mode="pd_ee_delta_pose")
    module._patch_wrapper_attribute_forwarding(Wrapper)
    outer = Wrapper(Wrapper(base))

    assert outer.obs_mode == "pointcloud_robot_seg"
    assert outer.control_mode == "pd_ee_delta_pose"


def test_mpc_requirements_lock_mani_skill_replay_dependency() -> None:
    requirements = (ROOT / "requirements-mpc.txt").read_text(encoding="utf-8")
    assert "GitPython==3.1.45" in requirements.splitlines()

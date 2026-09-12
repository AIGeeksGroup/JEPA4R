"""Replay ManiSkill trajectories with the repository's wrapper compatibility fix."""

from scripts.check_liftcube_mpc_env import _patch_robot_link_ids, configure_vulkan_icd


def _patch_wrapper_attribute_forwarding(wrapper_base: type) -> None:
    """Restore attribute forwarding expected by ManiSkill2 0.5.3."""
    if "__getattr__" in wrapper_base.__dict__:
        return

    def get_forwarded_attribute(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    wrapper_base.__getattr__ = get_forwarded_attribute


def run_replay(
    args,
    wrapper_base: type,
    replay_main,
) -> None:
    """Patch the wrapper contract before ManiSkill constructs the environment."""
    _patch_robot_link_ids(wrapper_base)
    replay_main(args)


def main() -> None:
    from gymnasium import Wrapper

    configure_vulkan_icd()
    from mani_skill2.trajectory.replay_trajectory import main as replay_main
    from mani_skill2.trajectory.replay_trajectory import parse_args
    from mani_skill2.utils.wrappers.observation import BaseGymObservationWrapper

    _patch_wrapper_attribute_forwarding(Wrapper)
    run_replay(parse_args(), BaseGymObservationWrapper, replay_main)


if __name__ == "__main__":
    main()

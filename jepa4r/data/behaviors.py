"""Data-collection behaviours (Sec. 5.2, supp. sec:supp-data); proportions live in configs/env/*.yaml.

Cube (1-4 cubes)                         expert_low_noise 0.2, expert_high_noise 0.4,
                                         expert_then_random 0.2, random 0.2
Scene, MetaWorld (per task), Relocate    noisy_expert 0.5, expert_interrupted_by_random 0.3, random 0.2
Puzzle                                   OGBench's noisy recipe (wm_data.recipe = ogbench_noisy)

TODO(integrate): the expert controllers, noise scales, interruption / switch schedules, and how episodes are
assigned to behaviours (none of these is given in the paper).
"""
BEHAVIOURS = (
    "expert_low_noise", "expert_high_noise", "expert_then_random", "random",
    "noisy_expert", "expert_interrupted_by_random",
)


def make_behaviour(name, env, env_cfg, rng):
    """Return a callable obs -> action for one episode."""
    raise NotImplementedError(f"TODO(integrate): behaviour {name!r}")

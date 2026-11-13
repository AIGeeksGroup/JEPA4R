"""Data-collection behaviours (Sec. 5.2, supp. sec:supp-data); proportions live in configs/env/*.yaml.

Cube (1-4 cubes)                         expert_low_noise 0.2, expert_high_noise 0.4,
                                         expert_then_random 0.2, random 0.2
Scene, MetaWorld (per task), Relocate    noisy_expert 0.5, expert_interrupted_by_random 0.3, random 0.2
Puzzle                                   OGBench's noisy recipe (wm_data.recipe = ogbench_noisy)

In the integrated code each task's collection policy draws the behaviour of every episode itself (its expert, noise
scales and switch schedule moved unchanged from the source branches); cfg.env.wm_data.policy names it and the
collector (jepa4r/stages/collect_wm_data.py) instantiates it:
  MetaWorld   jepa4r/envs/metaworld/policy.py ThreePartPolicy (+ the task subclasses): a scripted expert's plan as
              backbone (0.5) / branch into random actions (0.3) / random (0.2), wm_data.env_kwargs.mix
  Scene       jepa4r/envs/ogbench/ogb_scene_env.py SceneThreePartPolicy (the same three-part plan)
  Puzzle      jepa4r/envs/ogbench/ogb_puzzle_env.py PuzzleNoisyPolicy (xi ~ U(0, 0.1) per episode, Gaussian noise
              [xi, xi, xi, 3 xi, 10 xi], uniform random action with probability 0.2)
  UR5e / RM75-B cube (reconstruction, twin)   stable-worldmodel's plan-oracle expert (jepa4r/envs/ogbench/expert_policy.py)
  Cube 1-4, Relocate   the OGBench-cube workflow's collectors (not integrated yet: see docs/reproduce.md, open items)
"""
BEHAVIOURS = (
    "expert_low_noise", "expert_high_noise", "expert_then_random", "random",
    "noisy_expert", "expert_interrupted_by_random",
)


def make_policy(env_cfg):
    """The collection policy of the task (cfg.env.wm_data.policy), as the collector builds it."""
    import hydra

    return hydra.utils.instantiate(env_cfg.wm_data.policy)

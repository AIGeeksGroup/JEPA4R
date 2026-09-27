"""MetaWorld tasks of CortexBench as stable-worldmodel envs: the env (top_cap2 camera, per-episode collection plan,
eval hooks), the single-depth-camera point cloud and the three-part collection policy (backbone / branch / random).
Tasks: bin-picking (env.MWBinPickEnv, policy.ThreePartPolicy), assembly (assembly.MWAssemblyEnv / AssemblyThreePartPolicy),
button-press-topdown (button_press.MWButtonPressEnv / ButtonThreePartPolicy).

Needs metaworld 3.0 (conda env ``lewm_mw``), so the parent package ``envs`` does not import this one. Importing it
registers the env ids; configs name an env as ``envs.metaworld:swm/MWBinPick-v0`` (gymnasium imports the module
first) or instantiate ``envs.metaworld.MWPointCloudWrapper``, which imports it too.
"""

from gymnasium.envs.registration import register

from .assembly import AssemblyThreePartPolicy, MWAssemblyEnv
from .button_press import ButtonThreePartPolicy, MWButtonPressEnv
from .env import SAMPLE_TYPES, MWBinPickEnv, MWTaskEnv
from .pcd import MWPointCloudWrapper
from .policy import ThreePartPolicy

register(id='swm/MWBinPick-v0', entry_point='envs.metaworld.env:MWBinPickEnv')
register(id='swm/MWAssembly-v0', entry_point='envs.metaworld.assembly:MWAssemblyEnv')
register(id='swm/MWButtonPress-v0', entry_point='envs.metaworld.button_press:MWButtonPressEnv')

__all__ = ['MWTaskEnv', 'MWBinPickEnv', 'MWAssemblyEnv', 'MWButtonPressEnv', 'MWPointCloudWrapper', 'ThreePartPolicy',
           'AssemblyThreePartPolicy', 'ButtonThreePartPolicy', 'SAMPLE_TYPES']

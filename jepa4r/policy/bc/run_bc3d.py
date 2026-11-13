# Moved from cortexbench@9331494:lewm_pcd/bench/cortexbench/run_bc3d.py (verbatim except: imports / paths adapted to the package)
"""One CortexBench MetaWorld imitation run with a point-cloud encoder (the 3D LeWM). This is run_bc.py (the official
hydra_launcher + train_loop at eai-vc 76fe35e, with its two logging-only patches), with the camera image replaced by
the same camera's point cloud and nothing else changed:

1. Demos: data_dir = $STABLEWM_HOME/datasets/cortexbench/metaworld-expert-v1.0-pcd (convert_demos_3d.py): the official
   demos replayed exactly, 'images' = the point cloud of every frame, (500, 1200, 6) = xyz (m) + rgb.
2. load_pretrained_model: for an embedding whose metadata says input 'pcd' (lewm_cortex.load_lewm_pcd), the transform
   is the model's own point-cloud preprocessing + unsqueeze(0) instead of the official PIL image wrapper.
3. Evaluation: MuJoCoPixelObsWrapper.get_image returns the frame's point cloud (pcd_fork.MWPointClouds, the 3D
   LeWM's training settings; rendered with multisampling off, which segmentation needs). Its point-sampling RNG is
   seeded through env.seed, which mjrl's do_rollout calls before every episode, so the evaluation is deterministic.
   Everything else (policy input = FLARE fusion of 3 embeddings + proprioception obs[:4], BC, 20 evaluations of 25
   episodes) is the official loop.

    source env.sh && $PY run_bc3d.py --config-name Metaworld_BC_config env=... embedding=lewm_mw_binpick_pcd seed=100 \
        data_dir=$STABLEWM_HOME/datasets/cortexbench/metaworld-expert-v1.0-pcd/ ...
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VI = os.path.join(os.environ['JEPA4R_EAI_VC'], 'cortexbench', 'mujoco_vc', 'visual_imitation')  # the eai-vc checkout
CONF_HOME = os.environ.get('JEPA4R_BC_CONF_HOME', HERE)  # <CONF_HOME>/conf/model/NAME.yaml: the embedding configs
sys.path.insert(0, VI)
sys.path.insert(0, HERE)

import hydra  # noqa: E402
import mujoco_vc.gym_wrapper as gw  # noqa: E402
import mujoco_vc.model_loading as ml  # noqa: E402
import numpy as np  # noqa: E402
import train_loop  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pcd_fork import MWPointClouds, no_multisampling  # noqa: E402

_official_load = ml.load_pretrained_model


def load_pretrained_model(embedding_name, input_type=np.ndarray, *args, **kwargs):
    config = OmegaConf.load(os.path.join(ml.vc_models_dir_path, 'conf/model', embedding_name + '.yaml'))
    model, embedding_dim, transforms, metadata = hydra.utils.call(config)
    if metadata.get('input') != 'pcd':
        return _official_load(embedding_name, input_type, *args, **kwargs)
    return model.eval(), embedding_dim, (lambda pcd: transforms(pcd).unsqueeze(0)), metadata


for mod in (ml, gw, train_loop):
    mod.load_pretrained_model = load_pretrained_model

SUFFIX = '-v2-goal-observable'


def get_image(self):
    """The frame's point cloud, (1200, 6) float32 = xyz (m) + rgb 0..255, instead of the RGB frame."""
    pcm = getattr(self, '_pcm', None)
    if pcm is None:
        if not self.spec.id.endswith(SUFFIX):
            raise RuntimeError(f'point clouds are implemented for MetaWorld only, not {self.spec.id}')
        no_multisampling(self.sim)  # asserts that nothing rendered before (the render context is created next)
        pcm = self._pcm = MWPointClouds(self.sim, self.spec.id[:-len(SUFFIX)])
        if getattr(self, '_pcd_seed', None) is not None:
            pcm.seed(self._pcd_seed)
    xyz, rgb = pcm()
    return np.concatenate([xyz, rgb.astype(np.float32)], axis=1)


def seed(self, seed=None):
    self._pcd_seed = seed
    if getattr(self, '_pcm', None) is not None:
        self._pcm.seed(seed)
    return self.env.seed(seed)


gw.MuJoCoPixelObsWrapper.get_image = get_image
gw.MuJoCoPixelObsWrapper.seed = seed

# run_bc.py: its config lookup and logging patches, then the official hydra_launcher with our argv
sys.argv = [os.path.join(HERE, 'run_bc.py')] + sys.argv[1:]
runpy.run_path(os.path.join(HERE, 'run_bc.py'), run_name='__main__')

"""On-disk layout of the paired datasets in data/wm/<env>/ and data/policy/<env>/.

TODO(integrate): keep the format of the existing collection code and document it here. Needed per step:
  rgb [224, 224, 3] uint8, pointcloud [1200, 6] float32, action [A], episode id and step index,
  train / validation split; optional state, proprio, goal state, segmentation ids.
Readers: data/clips.py (world-model clips), policy/features.py (latent caching),
decoder/train.py and eval/recon.py (point-cloud targets).
"""


def open_dataset(path, split="train"):
    raise NotImplementedError("TODO(integrate)")


class Writer:
    def __init__(self, path, env_cfg):
        raise NotImplementedError("TODO(integrate)")

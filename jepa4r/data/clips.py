"""World-model clips (Sec. 3): ``num_frames`` (= history + 1 = 4) frames spaced ``frameskip`` (= 5) env steps
apart; the action block between consecutive frames concatenates the 5 actions. Observation = point cloud
(JEPA4R) or image (LeWM, PLDM, LeJEPA) from the same rows.

The clip sampler is stable-worldmodel's HDF5 dataset (``swm.data.load_dataset(path, num_steps=history + num_preds,
frameskip, keys_to_load, keys_to_cache)``) with LeWM's transforms, moved here from ``utils.py``:
``get_pcd_preprocessor`` (xyz - center) * scale ++ rgb / 255, ``get_img_preprocessor`` (ImageNet normalisation, 224),
and a z-score normaliser per non-observation column fitted on the whole dataset file (NaN rows dropped).
"""
from pathlib import Path

import numpy as np
import torch

# Moved from cortexbench@9331494:lewm_pcd/utils.py (lines 6-32; verbatim except: stable_pretraining imported inside the
# functions)
def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    from stable_pretraining import data as dt

    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    from stable_pretraining import data as dt

    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

# Moved from cortexbench@9331494:lewm_pcd/utils.py:PointCloudPreprocessor, get_pcd_preprocessor (lines 62-93; verbatim)
class PointCloudPreprocessor:
    """Build the model input from the stored point-cloud columns.

    xyz (metres, world frame) -> (xyz - center) * scale ; rgb uint8 -> [0, 1].
    Output ``target``: float tensor (T, N, 3 or 6). Fixed affine on purpose: a
    z-score over a (frames, N, 3) column would be tied to point *order*, which is
    meaningless for an unordered set, and would need the whole 29 GB column.
    Picklable (class, not closure) so DataLoader workers can be spawned."""

    def __init__(self, source_xyz='pcd_xyz', source_rgb='pcd_rgb', target='pcd',
                 center=(0.425, 0.0, 0.2), scale=4.0, use_rgb=True, drop_sources=True):
        self.source_xyz, self.source_rgb, self.target = source_xyz, source_rgb, target
        self.center = torch.tensor([float(c) for c in center], dtype=torch.float32)
        self.scale = float(scale)
        self.use_rgb, self.drop_sources = use_rgb, drop_sources

    def __call__(self, x):
        xyz = (x[self.source_xyz].float() - self.center) * self.scale
        feats = [xyz]
        if self.use_rgb:
            feats.append(x[self.source_rgb].float() / 255.0)
        x[self.target] = torch.cat(feats, dim=-1)
        if self.drop_sources:
            for k in (self.source_xyz, self.source_rgb):
                if k != self.target:
                    x.pop(k, None)
        return x


def get_pcd_preprocessor(**kwargs):
    """Hydra entry point mirroring ``get_img_preprocessor``."""
    return PointCloudPreprocessor(**kwargs)


OBS_KEYS = {'pointcloud': ['pcd_xyz', 'pcd_rgb'], 'image': ['pixels']}


def dataset_kwargs(method_cfg, obs, num_preds=1):
    """The ``data.dataset`` block of LeWM's configs (cortexbench@9331494:lewm_pcd/config/train/data/ogb_pcd.yaml for
    point clouds, dmc.yaml for pixels): clip length history + num_preds, frameskip 5, the observation columns + action +
    observation (proprioception) loaded, action and observation cached."""
    return dict(num_steps=int(method_cfg.history) + int(num_preds), frameskip=int(method_cfg.frameskip),
                keys_to_load=OBS_KEYS[obs] + ['action', 'observation'], keys_to_cache=['action', 'observation'])


def preprocessor_kwargs(env_cfg, obs, img_size=224):
    """The ``data.preprocessor`` block: point clouds use the env's fixed affine (``env.pointcloud.affine``)."""
    if obs == 'pointcloud':
        a = env_cfg.pointcloud.affine
        return dict(_target_='jepa4r.data.clips.get_pcd_preprocessor', source_xyz='pcd_xyz', source_rgb='pcd_rgb',
                    target='pcd', center=[float(c) for c in a.center], scale=float(a.scale), use_rgb=True)
    return dict(_target_='jepa4r.data.clips.get_img_preprocessor', source='pixels', target='pixels', img_size=img_size)


class ClipDataset:
    def __init__(self, data_dir, method_cfg, split="train", env_cfg=None):
        """All clips of the env's world-model dataset (split='all'), or the train / validation part of LeWM's random
        0.9 / 0.1 clip split with ``torch.Generator().manual_seed(seed)`` (the order train.py draws them in)."""
        import hydra
        import stable_pretraining as spt
        import stable_worldmodel as swm

        obs = method_cfg.obs
        files = env_cfg.wm_data.files
        path = str(Path(data_dir) / files[obs])
        self.dataset = swm.data.load_dataset(path, transform=None, **dataset_kwargs(method_cfg, obs))
        transforms = [hydra.utils.instantiate(preprocessor_kwargs(env_cfg, obs))]
        for col in ('action', 'observation'):
            transforms.append(get_column_normalizer(self.dataset, col, col))
        self.dataset.transform = spt.data.transforms.Compose(*transforms)
        self.split = split
        self.data = self.dataset
        if split in ('train', 'val'):
            gen = torch.Generator().manual_seed(int(env_cfg.wm_data.get('split_seed', 3072)))
            tr = float(env_cfg.wm_data.get('train_split', 0.9))
            parts = spt.data.random_split(self.dataset, lengths=[tr, 1 - tr], generator=gen)
            self.data = parts[0 if split == 'train' else 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

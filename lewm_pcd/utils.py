import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
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
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )

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

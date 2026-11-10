"""Current manifest counts around the RGB16/A15 PLDM data path."""
import operator
from pathlib import Path



def data_profile(cube_name, data_binding=None):
    if data_binding is None:
        raise ValueError('current training requires an explicit data manifest')
    from ..fresh_data import profile
    value = profile(data_binding)
    if cube_name != 'cube' + str(value['cube_count']):
        raise ValueError('Cube selector differs from its actual data manifest')
    return value


def contract(profile_name, data_binding=None):
    profile = data_profile(profile_name, data_binding)
    frames = 16
    per = profile['view_rows'] - frames + 1
    train = profile['train_episodes'] * per
    val = profile['validation_episodes'] * per
    return dict(frames=frames, actions=15, per_episode=per,
                train_samples=train, validation_samples=val, train_batches=train // 128,
                validation_batches=(val + 127) // 128, validation_tail=val % 128,
                total_updates=10 * (train // 128))


def action_statistics(train_reader, profile_name, data_binding=None):
    """Exactly one train-only normalization; no image statistics or clipping."""
    import numpy as np
    import torch
    profile = data_profile(profile_name, data_binding)
    values = torch.from_numpy(np.asarray(train_reader.get_col_data('action')))
    expected = (profile['train_episodes'] * profile['view_rows'], 5)
    if tuple(values.shape) != expected or not bool(torch.isfinite(values).all()):
        raise ValueError('full finite train action-bearing rows required')
    return dict(mean=values.mean(0).tolist(), std=values.std(0, correction=1).tolist(),
                count=len(values), correction=1, epsilon=1e-6, source_split='train',
                manifest_sha256=profile['manifest_sha256'], normalizations_per_sample=1)


class PLDMPreprocess:
    """Original RGB domain seam; no ImageNet normalization or encoder call."""
    def __init__(self):
        self.last_contract = None

    def __call__(self, batch):
        import torch
        from torchvision.transforms import v2
        pixels = batch['pixels']
        actions = batch['actions']
        if pixels.dtype is not torch.uint8 or tuple(pixels.shape[1:]) != (16, 3, 224, 224):
            raise ValueError('one uint8 RGB16 preprocessing pass required')
        if tuple(actions.shape[1:]) != (15, 5) or not actions.is_floating_point():
            raise ValueError('normalized robot actions must be floating RGB16/A15 batches')
        batch['pixels'] = v2.Resize((65, 65), antialias=True)(pixels.float() / 255.0)
        if self.last_contract is None:
            self.last_contract = dict(raw_pixel_dtype='uint8', raw_pixel_shape=list(pixels.shape),
                model_pixel_dtype='float32', model_pixel_shape=list(batch['pixels'].shape),
                pixel_scale='divide_by_255', resize='torchvision v2 Resize65 antialias=True',
                action_dtype=str(actions.dtype), action_shape=list(actions.shape),
                action_normalization='train-only sample std + 1e-6, performed once in dataset')
        return batch


class VisualDataset:
    """Original scalar contiguous reads and single train-only normalization."""
    def __init__(self, reader, action_stats=None):
        self.reader = reader
        self.action_stats = action_stats
        self.gpu_transform = PLDMPreprocess()

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        sample = self.reader[index]
        pixels = sample['pixels']
        if self.action_stats is None:
            raise ValueError('shared train-only action statistics required')
        actions = sample['action'][:15]
        if tuple(actions.shape) != (15, 5) or pixels.shape[0] != 16:
            raise ValueError('RGB16/A15 contract differs')
        mean = actions.new_tensor(self.action_stats['mean'])
        std = actions.new_tensor(self.action_stats['std'])
        return dict(pixels=pixels, actions=((actions - mean) / (std + 1e-6)).float(),
                    sample_id=int(index))

    def close(self):
        self.reader.close()


def loaders(profile_name, data_root, method_module=None, workers=20, prefetch=2, data_binding=None):
    import torch
    data_profile(profile_name, data_binding)
    from ..fresh_data import make_reader
    open_reader = lambda split: make_reader(data_binding, split + '_rgb', num_steps=16,
                                            keys_to_load=['pixels', 'action'], keys_to_cache=['action'])
    train = VisualDataset(open_reader('train'))
    stats = action_statistics(train.reader, profile_name, data_binding)
    train.action_stats = stats
    val = VisualDataset(open_reader('validation'), stats)
    kwargs = dict(batch_size=128, num_workers=workers, pin_memory=True, persistent_workers=False)
    if workers > 0:
        kwargs['prefetch_factor'] = prefetch
    generator = torch.Generator().manual_seed(0)
    a = torch.utils.data.DataLoader(train, shuffle=True, drop_last=True, generator=generator, **kwargs)
    b = torch.utils.data.DataLoader(val, shuffle=False, drop_last=False, **kwargs)
    c = contract(profile_name, data_binding)
    if (len(train), len(val), len(a), len(b)) != (c['train_samples'], c['validation_samples'],
                                                c['train_batches'], c['validation_batches']):
        raise ValueError('actual complete epoch counts differ')
    return a, b, stats

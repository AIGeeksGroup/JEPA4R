"""Opt-in immutable FP32 GPU tables; sampling and RNG stay on official NumPy code."""
import smoke as s


class ResidentDataset(s.Dataset):
    def __init__(self, host):
        super().__init__(host)
        assert self['observations'].dtype == s.np.float32
        assert self['actions'].dtype == s.np.float32
        assert self['observations'].ndim == self['actions'].ndim == 2
        self.gpu_tables = {key: s.jax.device_put(self[key]) for key in ('observations', 'actions')}
        assert all(device.platform == 'gpu' for table in self.gpu_tables.values()
                   for device in table.devices())

    def get_subset(self, idxs):
        result = {key: self.gpu_tables[key][idxs] if key in self.gpu_tables else value[idxs]
                  for key, value in self._dict.items()}
        if 'next_observations' not in result:
            result['next_observations'] = self.gpu_tables['observations'][
                s.np.minimum(idxs + 1, self.size - 1)]
        return result


class ResidentGCDataset(s.GCDataset):
    def __post_init__(self):
        assert self.config['frame_stack'] is None and self.config['p_aug'] == 0
        assert 'next_observations' not in self.dataset
        super().__post_init__()

    def get_observations(self, idxs):
        return self.dataset.gpu_tables['observations'][idxs]


def dataset(out, cfg):
    host = s.dataset(out, cfg)
    return ResidentGCDataset(ResidentDataset(host.dataset), cfg)

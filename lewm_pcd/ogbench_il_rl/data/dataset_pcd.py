"""Original PCD pair reader, reusing the RGB compact-row contract."""
import torch
from . import dataset as base
from .dataset import IdentityAction

class _PCDSliceView:
    """The inherited pair checker uses 'pixels' as a generic observation alias.

    This alias contains the real processed point tensor, never RGB or features.
    """
    def __init__(self, reader):
        self.reader = reader

    def _load_slice(self, *args):
        result = self.reader._load_slice(*args)
        if not torch.equal(result['source_step'].reshape(-1).long(), torch.arange(args[1], args[2])):
            raise ValueError('PCD source local step differs from requested row')
        result['pixels'] = result['pcd']
        return result

    @property
    def h5_file(self):
        return self.reader.h5_file


class CubeOfficialPCDDataset(base.CubeOfficialDataset):
    def _reader(self, episode):
        from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
        from .transforms import DataTransform
        if episode in self.readers:
            self.readers.move_to_end(episode)
            return self.readers[episode]
        reader = HDF5Dataset(path=self.entries[episode]['path'], num_steps=1, frameskip=1,
            keys_to_load=['pcd_xyz', 'pcd_rgb', 'action', 'source_row', 'source_episode', 'source_step', 'source_terminal'],
            transform=DataTransform('pcd', self.preprocessor, IdentityAction()))
        if reader.lengths.tolist() != [self.rows] or reader.offsets.tolist() != [0]:
            raise ValueError('Original PCD local episode bounds differ')
        view = _PCDSliceView(reader)
        self.readers[episode] = view
        if len(self.readers) > self.open_files:
            _, retired = self.readers.popitem(last=False)
            if retired.h5_file is not None:
                retired.h5_file.close()
        return view

    def __getitem__(self, pair):
        result = super().__getitem__(pair)
        for key in ('current', 'goal'):
            value = result[key]
            if value.shape != (1200, 6) or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Real PCD single-frame1200x6 contract invalid')
        if (result['action'].abs() > 1).any():
            raise ValueError('Published normalized Box action outside[-1,1]')
        return result

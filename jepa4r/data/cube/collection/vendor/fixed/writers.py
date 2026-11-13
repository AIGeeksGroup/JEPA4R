import h5py
import numpy as np
from stable_worldmodel.data.formats.hdf5 import HDF5Writer, _is_string_col

class GzipHDF5Writer(HDF5Writer):
    """HDF5Writer whose datasets use multi-row chunks and gzip for the listed columns (schema creation only;
    episode writing, append validation and bookkeeping are inherited unchanged)."""

    def __init__(self, path, *, mode='append', chunk_rows=20, gzip_cols=('pixels',), gzip_chunk_rows=5, level=4):
        super().__init__(path, mode=mode)
        self.chunk_rows, self.gzip_cols, self.gzip_chunk_rows, self.level = (
            chunk_rows,
            set(gzip_cols),
            gzip_chunk_rows,
            level,
        )

    def _init_schema(self, sample_ep: dict) -> None:
        for col, vals in sample_ep.items():
            if _is_string_col(vals[0]):
                self._f.create_dataset(
                    col, shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(), chunks=(self.chunk_rows,)
                )
                continue
            sample = np.asarray(vals[0])
            if col in self.gzip_cols:
                kw = dict(
                    chunks=(self.gzip_chunk_rows, *sample.shape),
                    compression='gzip',
                    compression_opts=self.level,
                    shuffle=True,
                )
            else:
                kw = dict(chunks=(self.chunk_rows, *sample.shape))
            self._f.create_dataset(
                col, shape=(0, *sample.shape), maxshape=(None, *sample.shape), dtype=sample.dtype, **kw
            )
        self._f.create_dataset('ep_len', shape=(0,), maxshape=(None,), dtype=np.int32)
        self._f.create_dataset('ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64)

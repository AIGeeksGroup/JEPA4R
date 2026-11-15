"""Training views over the collected Cube data (views stage): virtual HDF5 datasets (VDS) over the collected files
that keep the first <horizon> rows of every episode (the training rows; the last raw row is the terminal one), one
per split and modality, listed in data_manifest.json; read through stable-worldmodel's HDF5 reader.
usage: python -m jepa4r.data.cube.fresh_data --collection-manifest <COLLECTION.json> --output <dir>"""

import argparse
import json
import operator
import os
from pathlib import Path
import sys

HORIZONS = {1: 200, 2: 500, 3: 1000, 4: 600}
VIEW_KEYS = ("train_rgb", "validation_rgb", "train_pcd", "validation_pcd")


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def _view(source, output, horizon, count, modality):
    import h5py
    import numpy as np

    keys = ["pixels", "action"] if modality == "rgb" else ["pcd_xyz", "pcd_rgb", "action"]
    with h5py.File(source, "r") as raw, h5py.File(output, "x", libver="latest") as view:
        raw_rows = int(raw["ep_len"][0])
        for key in keys:
            column = raw[key]
            layout = h5py.VirtualLayout((count * horizon, *column.shape[1:]), column.dtype)
            src = h5py.VirtualSource(str(source), key, shape=column.shape)
            for episode in range(count):
                layout[episode * horizon : (episode + 1) * horizon] = src[
                    episode * raw_rows : episode * raw_rows + horizon
                ]
            view.create_virtual_dataset(key, layout)
        view["ep_len"] = np.full(count, horizon, np.int32)
        view["ep_offset"] = np.arange(count, dtype=np.int64) * horizon
    return dict(path=str(output), episodes=count, rows_per_episode=horizon, keys=keys)


def create_binding(collection_manifest, output_root):
    """The four training views of a collection and their data_manifest.json."""
    source = Path(collection_manifest).resolve()
    collected = json.loads(source.read_text(encoding="utf8"))
    cubes = collected["cube_count"]
    horizon = HORIZONS[cubes]
    counts = {"train": collected["train_episodes"], "validation": collected["validation_episodes"]}
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    views = {}
    for key in VIEW_KEYS:
        target = output / (key + ".h5")
        if target.exists():
            target.unlink()
        split, modality = key.rsplit("_", 1)
        views[key] = _view(Path(collected["paths"][key]), target, horizon, counts[split], modality)
    result = dict(
        cube_count=cubes,
        raw_horizon=horizon,
        training_rows_per_episode=horizon,
        train_episodes=counts["train"],
        validation_episodes=counts["validation"],
        collection_manifest=str(source),
        views=views,
        training_seed=0,
        collection_seed=collected["collection_seed"],
    )
    save(output / "data_manifest.json", result)
    return output / "data_manifest.json"


def load_binding(path):
    """A views manifest (data_manifest.json)."""
    return json.loads(Path(path).read_text(encoding="utf8"))


def profile(path):
    value = load_binding(path)
    return dict(
        name="cube" + str(value["cube_count"]),
        cube_count=value["cube_count"],
        view_rows=value["training_rows_per_episode"],
        train_episodes=value["train_episodes"],
        validation_episodes=value["validation_episodes"],
    )


class ReadOnlyClipIndex:
    """Read-only (episode, start) index of the training clips, stored compactly; an index gives a tuple."""

    def __init__(self, pairs):
        import numpy as np

        self._pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self._pairs.setflags(write=False)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [tuple(map(int, row)) for row in self._pairs[key]]
        episode, start = self._pairs[operator.index(key)]
        return int(episode), int(start)

    def __getstate__(self):
        return self._pairs

    def __setstate__(self, pairs):
        self.__init__(pairs)


def reader_class():
    """stable-worldmodel's HDF5 reader (third_party/stable-worldmodel) as the training uses it: scalar slices, the
    actions normalized before they are reshaped."""
    if "FreshHDF5Dataset" in globals():
        return globals()["FreshHDF5Dataset"]
    from jepa4r.models.cube.training_runtime import code_root

    code = code_root()
    sys.path.insert(0, str(code / "third_party/stable-worldmodel"))
    import h5py
    from stable_worldmodel.data.formats import hdf5

    class FreshHDF5Dataset(hdf5.HDF5Dataset):
        def __init__(self, manifest, key, **kwargs):
            self.manifest, self.key = str(Path(manifest).resolve()), key
            path = load_binding(self.manifest)["views"][key]["path"]
            super().__init__(path=path, **kwargs)
            self.clip_indices = ReadOnlyClipIndex(self.clip_indices)

        def _open_h5(self):
            return h5py.File(self.h5_path, "r", swmr=True, rdcc_nbytes=8 << 20)

        def _open(self):
            if self.h5_file is not None and getattr(self, "_owner_pid", None) != os.getpid():
                self.close()
            super()._open()
            self._owner_pid = os.getpid()

        def close(self):
            if getattr(self, "h5_file", None) is not None:
                self.h5_file.close()
            self.h5_file = None
            self._owner_pid = None

        def __del__(self):
            self.close()

    FreshHDF5Dataset.__qualname__ = "FreshHDF5Dataset"
    globals()["FreshHDF5Dataset"] = FreshHDF5Dataset
    return FreshHDF5Dataset


def make_reader(manifest, key, *, num_steps=16, frameskip=1, **kwargs):
    return reader_class()(manifest, key, num_steps=num_steps, frameskip=frameskip, **kwargs)


def __getattr__(name):
    if name == "FreshHDF5Dataset":
        return reader_class()
    raise AttributeError(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(create_binding(args.collection_manifest, args.output))

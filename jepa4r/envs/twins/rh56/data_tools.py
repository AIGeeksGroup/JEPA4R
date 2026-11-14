"""Publish the RH56 collection batches as one paired dataset: per modality a virtual HDF5 dataset (VDS) over the
batches' paired_<rgb|pcd>.h5 (the data are not copied), and manifest.json with the episodes in order (episode_index)
and the batch each came from.
usage: python -m jepa4r.envs.twins.rh56.data_tools --batches <dir> ... --output <dir>"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from jepa4r.envs.twins.rh56._runtime import write_json


def publish_dataset(batches, output):
    """Compose the batches into one dataset without rewriting their HDF5 data."""
    import h5py

    batches = [Path(p).resolve() for p in batches]
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for batch in batches:
        manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        for row in manifest["episodes"]:
            records.append(
                dict(
                    row,
                    episode_index=len(records),
                    source_dir=str(batch),
                    source_episode_index=row["episode_index"],
                    npz=str((batch / row["npz"]).resolve()),
                )
            )
    published = []
    for route in ("rgb", "pcd"):
        columns = None
        shapes = {}
        dtypes = {}
        sources = []
        lengths = []
        offset = 0
        for batch in batches:
            path = batch / f"paired_{route}.h5"
            with h5py.File(path, "r") as f:
                columns = sorted(k for k in f if k not in ("ep_len", "ep_offset"))
                ln = f["ep_len"][:]
                n = int(ln.sum())
                lengths.extend(ln.tolist())
                for key in columns:
                    shapes[key] = f[key].shape[1:]
                    dtypes[key] = f[key].dtype
                sources.append(dict(path=os.path.relpath(path, out), rows=n, offset=offset))
                offset += n
        destination = out / f"paired_{route}.h5"
        with h5py.File(destination, "w", libver="latest") as f:
            for key in columns:
                layout = h5py.VirtualLayout(shape=(offset,) + shapes[key], dtype=dtypes[key])
                for source in sources:
                    layout[source["offset"] : source["offset"] + source["rows"]] = h5py.VirtualSource(
                        source["path"], key, shape=(source["rows"],) + shapes[key]
                    )
                f.create_virtual_dataset(key, layout)
            f["ep_len"] = np.asarray(lengths, np.int32)
            f["ep_offset"] = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
        published.append(dict(route=route, path=str(destination), rows=offset, sources=sources))
    manifest = dict(action_alignment="outgoing; last row NaN", episodes=records, datasets=published)
    write_json(out / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = publish_dataset(args.batches, args.output)
    counts = {}
    for row in manifest["episodes"]:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    print(json.dumps(dict(episodes=len(manifest["episodes"]), counts=counts), indent=2))


if __name__ == "__main__":
    main()

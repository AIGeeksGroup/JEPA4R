"""extract_feats.py for the JEPA-family baselines: OGBench's puzzle-<env-type>-play data -> every state re-rendered with
the LeWM data's camera (the image LeWM sees: front_pixels, 224) -> each given baseline encoder as CortexBench
loads it (wm_obs.BaselineEncoder: its own transforms, its frozen embedding) -> one latent per step, for run_dqc.py
--obs vjepa21 / lejepa / pldm. One rendering pass serves all the given encoders.
* Only the latents are stored (float32, N x D, row for row with the npz), written in chunks, restartable; each encoder
  has its own done file ({split}_done_<name>.npy), so a pass can be resumed or an encoder added later.
* At most 2 x workers rendered chunks wait for the encoders (Pool.imap alone renders ahead without bound: with V-JEPA
  2.1, slower than the rendering, the waiting 384 frames grew to 66 GB in five minutes).
    python extract_feats_baselines.py --env-type 4x5 --out $STABLEWM_HOME/datasets/ogbench/dqc_feats/puzzle_4x5 \\
        --encoder vjepa21=$STABLEWM_HOME/checkpoints/cortexbench_baselines/vjepa2_1_vitb_dist_vitG_384.pt \\
        [--encoder lejepa=.../checkpoints/lejepa_puzzle_4x5 --encoder pldm=.../checkpoints/pldm_puzzle_4x5] --workers 6
"""

import argparse
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import numpy as np

import extract_feats as ef  # this script's directory is on sys.path
from wm_obs import BaselineEncoder

_SIZES = {}


def _init_worker(env_type, sizes):
    ef._init_worker(env_type)
    _SIZES["sizes"] = sizes


def _render_chunk(args):
    """Set a run of states back into the simulator one by one and render each image; return it at every input size
    the encoders take (prep_frames: the PIL resize of their transforms)."""
    start, qpos, qvel, bstates = args
    pdat, u, cam = ef._W["pdat"], ef._W["u"], ef._W["cam"]
    pix = np.empty((len(qpos), pdat.IMG, pdat.IMG, 3), np.uint8)
    for k in range(len(qpos)):
        u.set_state(qpos[k].astype(np.float64), qvel[k].astype(np.float64), np.asarray(bstates[k]).astype(int))
        cam.r2d.update_scene(u._data, camera=pdat.CAMERA)
        pix[k] = cam.r2d.render()
    return start, {s: BaselineEncoder.prep_frames(pix, s) for s in _SIZES["sizes"]}


def main():
    home = Path(os.environ["STABLEWM_HOME"])
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoder", action="append", required=True, help="name=path, name in vjepa21 / visreg / lejepa / pldm"
    )
    ap.add_argument("--epoch", type=int, default=10, help="epoch of the trained baselines (lejepa, pldm)")
    ap.add_argument("--env-type", default="4x5", choices=["3x3", "4x4", "4x5"])
    ap.add_argument("--data", default=str(home / "datasets" / "ogbench" / "official"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    encs = {}
    for spec in a.encoder:
        name, path = spec.split("=", 1)
        encs[name] = BaselineEncoder(name, path, a.epoch)
    name = f"puzzle-{a.env_type}-play-v0" + ("-val" if a.split == "val" else "")
    f = np.load(Path(a.data) / f"{name}.npz")
    qpos, qvel, bst = f["qpos"], f["qvel"], f["button_states"]
    n = len(qpos) if a.limit is None else a.limit
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.split if a.limit is None else f"{a.split}_first{n}"
    starts = list(range(0, n, a.chunk))
    Z, done = {}, {}
    for k, enc in encs.items():
        zp, dp = out / f"{tag}_z{k}.npy", out / f"{tag}_done_{k}.npy"
        if dp.exists():
            Z[k], done[k] = np.load(zp, mmap_mode="r+"), np.load(dp)
        else:
            Z[k] = np.lib.format.open_memmap(zp, mode="w+", dtype=np.float32, shape=(n, enc.dim))
            done[k] = np.zeros(len(starts), bool)
    todo = [i for i in range(len(starts)) if not all(d[i] for d in done.values())]
    print(
        f"{name}: {n} rows, {len(starts)} chunks, {len(todo)} to do; encoders "
        f"{', '.join(f'{k} ({e.dim}-d at {e.size})' for k, e in encs.items())}",
        flush=True,
    )
    sizes = sorted({e.size for e in encs.values()})

    def sl(x, s):
        return x[s : min(s + a.chunk, n)]

    window = threading.BoundedSemaphore(2 * a.workers)  # chunks handed to the pool and not yet encoded

    def tasks():
        for i in todo:
            window.acquire()  # Pool.imap draws tasks in its own thread: this holds it back
            yield starts[i], sl(qpos, starts[i]), sl(qvel, starts[i]), sl(bst, starts[i])

    t0, n_done = time.time(), 0
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(a.env_type, sizes)) as pool:
        for j, (s, frames) in enumerate(pool.imap(_render_chunk, tasks())):
            for k, enc in encs.items():
                if not done[k][s // a.chunk]:
                    Z[k][s : s + len(frames[enc.size])] = enc.encode(frames[enc.size])
                    done[k][s // a.chunk] = True
            n_done += len(frames[sizes[0]])
            del frames
            window.release()
            if j % 20 == 0:
                for k in encs:
                    Z[k].flush()
                    np.save(out / f"{tag}_done_{k}.npy", done[k])
                print(
                    f"{min(int(d.sum()) for d in done.values())}/{len(starts)} chunks, "
                    f"{n_done / (time.time() - t0):.0f} frames/s",
                    flush=True,
                )
    for k in encs:
        Z[k].flush()
        np.save(out / f"{tag}_done_{k}.npy", done[k])
        assert done[k].all()
        z = np.load(out / f"{tag}_z{k}.npy", mmap_mode="r")
        print(k, z.shape, "nan" if np.isnan(z).any() else "ok", float(np.abs(z).mean()))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

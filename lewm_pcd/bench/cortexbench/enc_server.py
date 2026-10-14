"""Shared GPU process for the frozen encoders: every evaluation job (run_bc_scene.py with ENC_SOCKET set) sends its
preprocessed input here; the requests are batched, encoded in one CUDA context and returned to each job. Encoder,
preprocessing and weights are exactly the jobs' own (the same conf/model config, the same lewm_cortex loader); only the
dozens of per-job CUDA contexts are gone: with every job in its own context the GPU switches between them and a step
waits tens of ms (measured: 24 jobs at once, 90-110 ms per step, about 50 ms of it encoding).
Batched and one-by-one encoding differ by float rounding only (measured: at most 4e-6, against an embedding std of 0.85).
Protocol (Unix socket, '@name' = Linux abstract namespace): 8-byte little-endian length + pickle((encoder name, float32
array (B, ...))); reply: 8-byte length + pickle(float32 array (B, D)).
    source env.sh && $PY enc_server.py @lewm_scene_enc lewm_scene_pix,lewm_scene_pcd
"""
import os
import pickle
import selectors
import socket
import struct
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # lewm_cortex
CONFS = [HERE / 'conf' / 'model']
MAX_BATCH = 64


def conf(name):
    for d in CONFS:
        if (d / f'{name}.yaml').exists():
            return OmegaConf.load(d / f'{name}.yaml')
    raise FileNotFoundError(name)


def sock_addr(name):
    """'@name' = a socket in the Linux abstract namespace (no file path; paths on the data disk exceed AF_UNIX's 107 bytes)."""
    return '\0' + name[1:] if name.startswith('@') else name


def recv_exact(c, n):
    buf = bytearray()
    while len(buf) < n:
        b = c.recv(n - len(buf))
        if not b:
            raise ConnectionError
        buf += b
    return bytes(buf)


def main():
    path, names = sys.argv[1], sys.argv[2].split(',')
    models = {}
    for n in names:
        m, dim, _, meta = hydra.utils.call(conf(n))
        models[n] = m.eval().cuda()
        print(f'loaded {n}: dim {dim}, {meta.get("checkpoint")}', flush=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_addr(path))
    srv.listen(256)
    sel = selectors.DefaultSelector()
    sel.register(srv, selectors.EVENT_READ)
    n_req, n_batch, t_last = 0, 0, time.time()
    while True:
        reqs = []
        for key, _ in sel.select():
            if key.fileobj is srv:
                c, _ = srv.accept()
                sel.register(c, selectors.EVENT_READ)
                continue
            c = key.fileobj
            try:
                n = struct.unpack('<Q', recv_exact(c, 8))[0]
                name, x = pickle.loads(recv_exact(c, n))
                reqs.append((c, name, x))
            except (ConnectionError, OSError):
                sel.unregister(c)
                c.close()
        for name in {r[1] for r in reqs}:
            group = [r for r in reqs if r[1] == name]
            x = torch.from_numpy(np.concatenate([r[2] for r in group]))
            with torch.no_grad():  # at most MAX_BATCH at a time, to bound the GPU memory (the demo precompute sends 20 frames per request)
                out = np.concatenate([models[name](x[s: s + MAX_BATCH].cuda()).float().cpu().numpy()
                                      for s in range(0, len(x), MAX_BATCH)])
            i = 0
            for c, _, xi in group:
                b = pickle.dumps(out[i: i + len(xi)], protocol=4)
                i += len(xi)
                try:
                    c.sendall(struct.pack('<Q', len(b)) + b)
                except OSError:
                    pass
            n_req += len(group)
            n_batch += 1
        if time.time() - t_last > 600:
            print(f'{time.strftime("%F_%T")} {n_req} requests in {n_batch} batches (mean batch {n_req / max(n_batch, 1):.1f}), '
                  f'{len(sel.get_map()) - 1} clients', flush=True)
            n_req, n_batch, t_last = 0, 0, time.time()


if __name__ == '__main__':
    main()

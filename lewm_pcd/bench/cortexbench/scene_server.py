"""The scene CortexBench env (scene_env.py) as a subprocess of the harness (run_bc_scene.py's SceneProxy), as
rm75_server.py for the RM75: the harness runs in the cortexbench conda env (mujoco-py 2.0), the scene env needs MuJoCo 3
+ OGBench (lewm_mw). Protocol on stdin / a dup of stdout: 8-byte little-endian length + pickle of plain types (arrays as
(dtype, shape, bytes)); fd 1 is pointed at stderr so library prints cannot corrupt the stream.
    $MW_PY scene_server.py     (started by run_bc_scene.py)"""
import os
import pickle
import struct
import sys

out = os.fdopen(os.dup(1), 'wb')
os.dup2(2, 1)
sys.stdout = sys.stderr
inp = sys.stdin.buffer

import numpy as np  # noqa: E402

from scene_env import SceneCBEnv, proprio_of as scene_proprio  # noqa: E402


def send(obj):
    b = pickle.dumps(obj, protocol=4)
    out.write(struct.pack('<Q', len(b)))
    out.write(b)
    out.flush()


def recv():
    h = inp.read(8)
    if len(h) < 8:
        return None
    return pickle.loads(inp.read(struct.unpack('<Q', h)[0]))


def arr(a):
    a = np.ascontiguousarray(a)
    return (str(a.dtype), a.shape, a.tobytes())


env = None
while True:
    msg = recv()
    if msg is None:
        break
    cmd, args = msg
    if cmd == 'init':
        env = SceneCBEnv(args)
        send(None)
    elif cmd == 'seed':
        env.seed(args)
        send(None)
    elif cmd == 'reset':
        send(arr(env.reset()))
    elif cmd == 'step':
        o, r, d, info = env.step(np.frombuffer(args, dtype=np.float64))
        send((arr(o), r, d, info))
    elif cmd == 'obs':  # proprioception of the current state (the harness reads it once before the first reset)
        send(arr(scene_proprio(env.u)))
    elif cmd == 'rgb':
        send(arr(env.rgb()))
    elif cmd == 'pcd':
        send(arr(env.pcd()))
    elif cmd == 'close':
        env.close()
        send(None)
        break

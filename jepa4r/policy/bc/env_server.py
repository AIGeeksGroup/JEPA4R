"""The CortexBench env of a run_bc_proxy.py run as its subprocess: rm75_env.RM75CBEnv (the RM75-B twin's MetaWorld
tasks, MuJoCo 3 + metaworld 3.0) or scene_env.SceneCBEnv (OGBench Scene, MuJoCo 3 + OGBench), while the harness runs in
the cortexbench conda env (mujoco-py 2.0). Protocol on stdin / a dup of stdout: 8-byte little-endian length + pickle of
plain types (arrays as (dtype, shape, bytes)); fd 1 is pointed at stderr so library prints cannot corrupt the stream.
    $MW_PY env_server.py rm75|scene     (started by run_bc_proxy.py)"""

import os
import pickle
import struct
import sys

out = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr
inp = sys.stdin.buffer

import numpy as np

if sys.argv[1] == "scene":
    from scene_env import SceneCBEnv as Env, proprio_of

    def proprio(env):  # the 8 entries of the current state (the harness reads them once before the first reset)
        return proprio_of(env.u)

else:
    from rm75_env import RM75CBEnv as Env

    def proprio(env):  # metaworld's _get_obs() of the current state (the harness reads it before any reset)
        return env.u._mw._get_obs()


def send(obj):
    b = pickle.dumps(obj, protocol=4)
    out.write(struct.pack("<Q", len(b)))
    out.write(b)
    out.flush()


def recv():
    h = inp.read(8)
    if len(h) < 8:
        return None
    return pickle.loads(inp.read(struct.unpack("<Q", h)[0]))


def arr(a):
    a = np.ascontiguousarray(a)
    return (str(a.dtype), a.shape, a.tobytes())


env = None
while True:
    msg = recv()
    if msg is None:
        break
    cmd, args = msg
    if cmd == "init":
        env = Env(args)
        send(None)
    elif cmd == "seed":
        env.seed(args)
        send(None)
    elif cmd == "reset":
        send(arr(env.reset()))
    elif cmd == "step":
        o, r, d, info = env.step(np.frombuffer(args, dtype=np.float64))
        send((arr(o), r, d, info))
    elif cmd == "obs":
        send(arr(proprio(env)))
    elif cmd == "rgb":
        send(arr(env.rgb()))
    elif cmd == "pcd":
        send(arr(env.pcd()))
    elif cmd == "close":
        env.close()
        send(None)
        break

"""Spawned simulator workers with the pinned stable World pool contract.

Only reset/step/render run in workers. Policy/oracle calls and their random
draws stay in the parent in the original environment order. The original
World episode iterator, transition alignment and dataset writer are reused.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import wait
import time
import traceback

import numpy as np
from gymnasium.vector.utils import batch_space
from stable_worldmodel.world.env_pool import (
    EnvPool, _broadcast_arg, _stack_fresh, _write_env_info,
)
from stable_worldmodel.world.world import World


def _snapshot(env):
    base = env.unwrapped
    return {
        name: getattr(base, name)()
        for name in ("oracle_info", "action_adapter_state")
        if hasattr(base, name)
    }


def _worker(connection, env_fn):
    """One process owns one simulator and its renderer for its full lifetime."""
    env = None
    try:
        import torch

        torch.set_num_threads(1)
        env = env_fn()
        base = env.unwrapped
        metadata = {
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "spec": getattr(base, "spec", None),
            "variation_space": getattr(base, "variation_space", None),
            "_env_type": getattr(base, "_env_type", None),
            "_arm_sampling_bounds": getattr(base, "_arm_sampling_bounds", None),
        }
        connection.send((True, metadata, {}))
        while True:
            command, args, kwargs = connection.recv()
            if command == "close":
                break
            if command == "reset":
                # RL point sampling can use NumPy's process-global RNG. Each
                # worker gets a reproducible stream from its official reset
                # seed; it must not share/consume the parent oracle RNG.
                if kwargs.get("seed") is not None:
                    np.random.seed(int(kwargs["seed"]) % (2 ** 32))
                result = env.reset(*args, **kwargs)
            elif command == "step":
                result = env.step(*args, **kwargs)
            elif command == "call":
                method, method_args, method_kwargs = args
                if method not in {"set_new_target", "set_sim_state", "set_goal_cube_position"}:
                    raise ValueError(f"Unsupported environment RPC: {method}")
                result = getattr(base, method)(*method_args, **method_kwargs)
            else:
                raise ValueError(f"Unknown worker command: {command}")
            connection.send((True, result, _snapshot(env)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc(), {}))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class RemoteEnv:
    """Minimal oracle-facing environment with cached physical state."""

    def __init__(self, pool, index, metadata):
        self._pool = pool
        self._index = index
        self._snapshot = {}
        self.__dict__.update(metadata)

    @property
    def unwrapped(self):
        return self

    def oracle_info(self):
        return self._snapshot["oracle_info"]

    def action_adapter_state(self):
        return self._snapshot["action_adapter_state"]

    def _call(self, method, *args, **kwargs):
        return self._pool._exchange(
            {self._index: ("call", (method, args, kwargs), {})}
        )[self._index]

    def set_new_target(self, **kwargs):
        return self._call("set_new_target", **kwargs)

    def set_sim_state(self, *args, **kwargs):
        return self._call("set_sim_state", *args, **kwargs)

    def set_goal_cube_position(self, *args, **kwargs):
        return self._call("set_goal_cube_position", *args, **kwargs)


class ProcessEnvPool(EnvPool):
    """Dispatch all active environments before waiting for any result.

    Environment factories must be pickleable under ``spawn``. Workers never
    inherit a live SAPIEN/CUDA context from the parent. Pool state and stacked
    info formatting match the pinned serial ``EnvPool``.
    """

    def __init__(self, env_fns, *, start_method="spawn", timeout=120.0):
        if not env_fns:
            raise ValueError("At least one environment factory is required")
        if timeout <= 0:
            raise ValueError("Worker timeout must be positive")
        self.timeout = float(timeout)
        self._connections = []
        self.processes = []
        self.envs = []
        self._closed = False
        self._stacked_infos = None
        context = mp.get_context(start_method)
        try:
            for env_fn in env_fns:
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, env_fn), daemon=True)
                process.start()
                child.close()
                self._connections.append(parent)
                self.processes.append(process)
            metadata = self._gather(set(range(len(env_fns))), initialize=True)
            self.envs = [RemoteEnv(self, i, metadata[i]) for i in range(len(env_fns))]
            self._single_env = self.envs[0]
            self.seeds = np.zeros(len(self.envs), dtype=np.int64)
            self._action_space = batch_space(self.single_action_space, self.num_envs)
            self._observation_space = batch_space(self.single_observation_space, self.num_envs)
        except BaseException:
            self.close()
            raise

    def _gather(self, pending, *, initialize=False):
        deadline = time.monotonic() + self.timeout
        outputs = {}
        connection_ids = {self._connections[i]: i for i in pending}
        while pending:
            remaining = deadline - time.monotonic()
            ready = wait([self._connections[i] for i in pending], timeout=max(0.0, remaining))
            if not ready:
                raise TimeoutError(f"Environment workers timed out: {sorted(pending)}")
            for connection in ready:
                index = connection_ids[connection]
                try:
                    ok, result, snapshot = connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(
                        f"Environment worker {index} exited unexpectedly "
                        f"(exitcode={self.processes[index].exitcode})"
                    ) from exc
                if not ok:
                    raise RuntimeError(f"Environment worker {index} failed:\n{result}")
                if not initialize:
                    self.envs[index]._snapshot = snapshot
                outputs[index] = result
                pending.remove(index)
        return outputs

    def _exchange(self, requests):
        if self._closed:
            raise RuntimeError("Environment pool is closed")
        try:
            for index, request in requests.items():
                self._connections[index].send(request)
            return self._gather(set(requests))
        except BaseException:
            self.close()
            raise

    def reset(self, seed=None, options=None, mask=None):
        seeds = _broadcast_arg(seed, self.num_envs, increment=True)
        options = _broadcast_arg(options, self.num_envs)
        requests = {
            i: ("reset", (), {"seed": seeds[i], "options": options[i]})
            for i in range(self.num_envs) if mask is None or mask[i]
        }
        if self._stacked_infos is None and len(requests) != self.num_envs:
            raise ValueError("First reset must initialize every environment")
        results = self._exchange(requests)
        infos = [None] * self.num_envs
        for i, (_, info) in results.items():
            infos[i] = info
            if seeds[i] is not None:
                self.seeds[i] = seeds[i]
        if self._stacked_infos is None or mask is None:
            self._stacked_infos = _stack_fresh(infos)
        else:
            for i, info in enumerate(infos):
                if info is not None:
                    _write_env_info(self._stacked_infos, i, info)
        return None, self._stacked_infos

    def step(self, actions, mask=None):
        if self._stacked_infos is None:
            raise RuntimeError("Reset the environment pool before stepping")
        requests = {
            i: ("step", (actions[i],), {})
            for i in range(self.num_envs) if mask is None or mask[i]
        }
        results = self._exchange(requests)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        for i, (_, reward, terminated, truncated, info) in results.items():
            rewards[i], terminateds[i], truncateds[i] = reward, terminated, truncated
            _write_env_info(self._stacked_infos, i, info)
        return None, rewards, terminateds, truncateds, self._stacked_infos

    def close(self):
        """Close only this pool's workers; never signal any external process."""
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send(("close", (), {}))
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = time.monotonic() + 5.0
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=1.0)
        for connection in self._connections:
            connection.close()


class ProcessWorld(World):
    """The upstream World runner initialized with a process-backed pool."""

    def __init__(self, env_fns, **pool_options):
        self.envs = ProcessEnvPool(env_fns, **pool_options)
        self.policy = None
        self.infos = {}
        self.rewards = None
        self.terminateds = None
        self.truncateds = None

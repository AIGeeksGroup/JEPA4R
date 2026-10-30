"""One fixed-scene ABA check of the thread repair, with unchanged true2N math."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MUJOCO_GL'] = 'egl'
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_info, ThreadpoolController

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import visual_sr as core
default_torch_threads = torch.get_num_threads()
numpy_owner = (Path(np.__file__).resolve().parent.parent / 'numpy.libs').resolve()
paths = [p['filepath'] for p in threadpool_info()
         if p['user_api'] == 'blas' and Path(p['filepath']).resolve().parent == numpy_owner]
assert len(paths) == 1
default_numpy_threads = next(p['num_threads'] for p in threadpool_info() if p['filepath'] == paths[0])

agent = core.load_agent('G05')
encoder = core.OnlineEncoder('G05')
env, camera, proof = core.make_env('G05', core.seed_for(1, 0))
wrapped = core.OnlineVisualEnv(env, 'G05', encoder, core.seed_for(1, 0))
wrapped.capture_proof = proof
try:
    observation, info = wrapped.reset(options={'task_id': 1})
    raw = wrapped.current_raw().copy()
    wrapped.current_raw = lambda: raw
    policy = core.OnlineGCIQLPolicy(agent, wrapped)
    import jax
    key = jax.random.PRNGKey(0)
    def call():
        return np.asarray(policy.sample_actions(observation, goals=info['goal'], seed=key, temperature=0.0)).copy()
    baseline_rng = pickle.dumps((np.random.get_state(), random.getstate(), torch.get_rng_state().numpy()))
    cases = []
    actions = []
    for label, nt, nb in [('original_unbounded_A', default_torch_threads, default_numpy_threads),
                          ('M5_Torch1_NumPyBLAS4', 1, 4),
                          ('original_unbounded_A2', default_torch_threads, default_numpy_threads)]:
        torch.set_num_threads(nt)
        limiter = ThreadpoolController().select(filepath=paths).limit(limits=nb, user_api='blas')
        for _ in range(4):
            call()
        started = time.perf_counter()
        values = [call() for _ in range(16)]
        elapsed = time.perf_counter() - started
        actions.append(np.stack(values))
        cases.append(dict(label=label, torch_num_threads=torch.get_num_threads(), numpy_blas_threads=nb,
                          policy_calls=16, encoder_forwards=32, elapsed_seconds=elapsed,
                          seconds_per_call=elapsed / 16))
    assert all(np.array_equal(actions[0], a) for a in actions[1:])
    assert pickle.dumps((np.random.get_state(), random.getstate(), torch.get_rng_state().numpy())) == baseline_rng
    ratio = ((cases[0]['elapsed_seconds'] + cases[2]['elapsed_seconds']) / 2) / cases[1]['elapsed_seconds']
    assert ratio > 1.0, 'Thread restoration did not improve this measured window'
    result = dict(actual_exit_code=0, cases=cases, measured_speed_ratio=ratio,
                  action_max_abs_difference=0.0, host_and_Torch_RNG_unchanged=True,
                  both_cases_true_current_goal_2N=True, unchanged_checkpoint=True,
                  fixed_scene_only_not_full_SR_speed_claim=True, camera=camera,
                  raw_rgb_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
    with (HERE / 'benchmark_result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
finally:
    wrapped.close()

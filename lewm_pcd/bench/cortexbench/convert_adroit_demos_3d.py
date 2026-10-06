"""Official init-state/action replay; replace only images with visible PCD."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def convert_demo(demo, env, capture, atol=1e-6, rtol=1e-5):
    actions, states = np.asarray(demo['actions']), np.asarray(demo['state_obs'])
    assert actions.ndim == states.ndim == 2 and len(actions) == len(states) > 0
    assert actions.shape[1] in (24,30) and states.shape[1] == (45 if actions.shape[1] == 24 else 39)
    assert np.isfinite(actions).all() and np.isfinite(states).all()
    env.set_env_state(deepcopy(demo['init_state_dict']))
    frames = []
    for row, action in enumerate(actions):
        obs = env.get_obs()
        np.testing.assert_allclose(obs, states[row], atol=atol, rtol=rtol)
        if 'proprio' in demo['env_infos']:
            np.testing.assert_allclose(obs[:actions.shape[1]], demo['env_infos']['proprio'][row], atol=atol, rtol=rtol)
        before = env.get_env_state()
        xyz, rgb = capture()
        assert xyz.shape == rgb.shape == (1200,3) and xyz.dtype == np.float32 and rgb.dtype == np.uint8
        assert np.isfinite(xyz).all()
        after = env.get_env_state()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg='render changed native state')
        frames.append(np.concatenate([xyz, rgb.astype(np.float32)], axis=1))
        _, reward, _, _ = env.step(action.copy())
        if 'rewards' in demo:
            np.testing.assert_allclose(reward, demo['rewards'][row], atol=atol, rtol=rtol)
    return dict(demo, images=np.asarray(frames, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['pen-v0','relocate-v0'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--start-demo', type=int, default=0)
    parser.add_argument('--max-demos', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.state.json').exists():
        raise FileExistsError('use a fresh output shard')
    # Large official pickles require a separately qualified memory/runtime gate.
    payload = args.input.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != args.source_sha256:
        raise ValueError('official demo source SHA mismatch')
    from run_adroit_bc3d import load_backend
    if args.task == 'pen-v0':
        from mj_envs.envs.hand_manipulation_suite.pen_v0 import PenEnvV0 as Native
    else:
        from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0 as Native
    demos = pickle.loads(payload)
    del payload
    assert args.start_demo >= 0 and args.max_demos > 0 and args.start_demo + args.max_demos <= len(demos)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = dict(stage='converting', started_at=datetime.now(timezone.utc).isoformat(),
                 source=str(args.input.resolve()), source_sha256=digest, task=args.task,
                 config={'start_demo':args.start_demo,'max_demos':args.max_demos,'size':256,'atol':1e-6,'rtol':1e-5},
                 completed=0,failed=0,errors=[],exit_code=None)
    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        args.output.with_suffix('.state.json').write_text(json.dumps(state,indent=2))
    save()
    env = Native()
    try:
        pcm = load_backend().AdroitPointClouds(env.sim,args.task,size=256)
        converted=[]
        for index in range(args.start_demo,args.start_demo+args.max_demos):
            env.seed(12345+index); env.reset(); pcm.seed(12345+index)
            converted.append(convert_demo(demos[index],env,pcm))
            state['completed']+=1; save()
        with args.output.open('xb') as handle:
            pickle.dump(converted,handle,protocol=4)
        state.update(stage='finished_diagnostic',exit_code=0)
    except Exception as error:
        state.update(stage='failed',failed=1,errors=[repr(error)],exit_code=1)
        raise
    finally:
        env.close(); save()
        args.output.with_suffix('.exit_code').write_text(str(state['exit_code'])+'\n')


if __name__ == '__main__':
    main()

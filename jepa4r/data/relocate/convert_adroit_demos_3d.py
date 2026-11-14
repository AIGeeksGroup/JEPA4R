"""Official init-state/action replay; replace only images with visible PCD."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np


def load_sources(source, expected_sha256):
    """Read one bounded migrated official demo at a time, or retain the old pickle input."""
    binding = source.parent/'manifest.json' if source.is_dir() else source
    payload = binding.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError('official demo source SHA mismatch')
    if source.is_dir():
        records = json.loads(payload)['demo_files']
        assert [r['demo_index'] for r in records] == list(range(len(records)))
        def read(index):
            record = records[index]
            path = (source.parent/record['relative_path']).resolve()
            if path.parent != source.resolve():
                raise ValueError('demo path is outside the bound source directory')
            raw = path.read_bytes()
            if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise ValueError('official demo file SHA/bytes mismatch')
            return pickle.loads(raw), dict(index=index,path=str(path),sha256=record['sha256'])
        return len(records), read, digest
    demos = pickle.loads(payload)
    return len(demos), lambda index:(demos[index],dict(index=index,path=str(source.resolve()),sha256=digest)), digest


def convert_demo(demo, env, capture, atol=1e-6, rtol=1e-5):
    actions, states = np.asarray(demo['actions']), np.asarray(demo['state_obs'])
    assert actions.ndim == states.ndim == 2 and len(actions) == len(states) > 0
    assert actions.shape[1] == 30 and states.shape[1] == 39
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
    converted = dict(demo, images=np.asarray(frames, dtype=np.float32))
    for key in demo:
        if key != 'images':
            assert pickle.dumps(converted[key],protocol=4) == pickle.dumps(demo[key],protocol=4), key
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['relocate-v0'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--start-demo', type=int, default=0)
    parser.add_argument('--max-demos', type=int, default=1)
    args = parser.parse_args()
    if any(
        path.exists()
        for path in (args.output, args.output.with_suffix('.state.json'), args.output.with_suffix('.exit_code'))
    ):
        raise FileExistsError('use a fresh output shard')
    count, read_demo, digest = load_sources(args.input,args.source_sha256)
    from jepa4r.policy.bc.run_adroit_bc3d import build_point_clouds
    from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0 as Native
    assert args.start_demo >= 0 and args.max_demos > 0 and args.start_demo + args.max_demos <= count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = dict(
        stage='converting',
        started_at=datetime.now(timezone.utc).isoformat(),
        source=str(args.input.resolve()),
        source_sha256=digest,
        task=args.task,
        config={'start_demo': args.start_demo, 'max_demos': args.max_demos, 'atol': 1e-6, 'rtol': 1e-5},
        sampler_config=os.environ.get('ADROIT_PCD_CONFIG'),
        sampler_config_sha256=os.environ.get('ADROIT_PCD_CONFIG_SHA256'),
        source_demos=[],
        completed=0,
        failed=0,
        errors=[],
        exit_code=None,
    )
    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        args.output.with_suffix('.state.json').write_text(json.dumps(state,indent=2))
    save()
    env = None
    try:
        env = Native()
        pcm = build_point_clouds(env.sim,args.task)
        converted=[]
        for index in range(args.start_demo,args.start_demo+args.max_demos):
            env.seed(12345+index); env.reset(); pcm.seed(12345+index)
            demo, source_record = read_demo(index)
            converted.append(convert_demo(demo,env,pcm))
            state['source_demos'].append(source_record)
            state['completed']+=1; save()
        with args.output.open('xb') as handle:
            pickle.dump(converted,handle,protocol=4)
        state.update(stage='finished_diagnostic',exit_code=0)
    except Exception as error:
        state.update(stage='failed',failed=1,errors=[repr(error)],exit_code=1)
        raise
    finally:
        if env is not None: env.close()
        save()
        args.output.with_suffix('.exit_code').write_text(str(state['exit_code'])+'\n')


if __name__ == '__main__':
    main()

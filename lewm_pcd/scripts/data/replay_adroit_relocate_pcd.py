"""Bounded original-action replay shards; reuse the approved visible-point sampler."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'bench/cortexbench'))
from run_adroit_bc3d import build_point_clouds


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original nonvisual replay HDF5')
    parser.add_argument('--initials', type=Path, required=True)
    parser.add_argument('--sampling-config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Fresh shard HDF5')
    parser.add_argument('--start-episode', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, required=True)
    args = parser.parse_args()
    for path in (args.out, args.out.with_suffix('.state.json'), args.out.with_suffix('.exit_code')):
        if path.exists():
            raise FileExistsError(path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = args.sampling_config.resolve()
    os.environ['ADROIT_PCD_CONFIG'] = str(config)
    os.environ['ADROIT_PCD_CONFIG_SHA256'] = hashlib.sha256(config.read_bytes()).hexdigest()
    state = dict(stage='STARTED', completed=0, frames=0, started_at=time.time(), exit_code=None,
                 source=str(args.source.resolve()), initials=str(args.initials.resolve()),
                 source_sha256=sha(args.source), initials_sha256=sha(args.initials),
                 producer_sha256=sha(__file__), backend_sha256=sha(ROOT/'envs/adroit/pcd.py'),
                 sampler_sha256=os.environ['ADROIT_PCD_CONFIG_SHA256'],
                 start_episode=args.start_episode, num_episodes=args.num_episodes)
    def save():
        state['updated_at'] = time.time()
        args.out.with_suffix('.state.json').write_text(json.dumps(state,indent=2))
    save()
    env = None
    try:
        from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0
        spec = importlib.util.spec_from_file_location('relocate_replay_contract',ROOT/'envs/adroit/fragment.py')
        contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)
        env = RelocateEnvV0()
        assert not env.sim.render_contexts and env.sim._render_context_offscreen is None
        env.model.vis.quality.offsamples = 0
        capture = build_point_clouds(env.sim,'relocate-v0')
        restore = contract.RelocateFragment(env)
        with h5py.File(args.source,'r') as source, np.load(args.initials,allow_pickle=False) as initial:
            lengths, offsets = source['ep_len'][:], source['ep_offset'][:]
            assert np.all(lengths > 0) and np.array_equal(offsets,np.r_[0,np.cumsum(lengths)[:-1]])
            assert args.start_episode >= 0 and args.num_episodes > 0
            stop = args.start_episode + args.num_episodes
            assert stop <= len(lengths) and len(initial['seed']) == len(lengths)
            seeds = source['seed'][offsets]
            assert len(np.unique(seeds)) == len(lengths) and np.array_equal(seeds,initial['seed'])
            assert np.array_equal(initial['initial_state_sha256'],source['initial_state_sha256'][offsets])
            indices = range(args.start_episode,stop)
            lens = lengths[args.start_episode:stop]
            columns = [k for k in source if k not in ('ep_len','ep_offset')]
            assert all(isinstance(source[k],h5py.Dataset) and source[k].shape[0] == sum(lengths) for k in columns)
            assert not any(k.startswith(('pixels','pcd_xyz','pcd_rgb')) for k in columns), 'nonvisual source required'
            with h5py.File(args.out,'x') as out:
                for key,value in source.attrs.items():out.attrs[key]=value
                out.attrs['sampling_config'] = config.read_text()
                out.create_dataset('ep_len',data=lens,maxshape=(None,))
                out.create_dataset('ep_offset',data=np.r_[0,np.cumsum(lens)[:-1]],maxshape=(None,))
                total = int(sum(lens))
                for key in columns:
                    out.create_dataset(key,shape=(total,*source[key].shape[1:]),dtype=source[key].dtype,
                                       maxshape=(None,*source[key].shape[1:]),chunks=True)
                out.create_dataset('pcd_xyz',shape=(total,1200,3),dtype='float32',chunks=(1,1200,3),maxshape=(None,1200,3))
                out.create_dataset('pcd_rgb',shape=(total,1200,3),dtype='uint8',chunks=(1,1200,3),maxshape=(None,1200,3))
                written = 0
                for episode in indices:
                    start, length = int(offsets[episode]), int(lengths[episode])
                    values = {key:source[key][start:start+length] for key in columns}
                    seed = int(seeds[episode]); env.seed(seed); env.reset(); capture.seed(seed)
                    restore._restore_initial({key:initial[key][episode] for key,_ in contract.RELOCATE_STATE_FIELDS})
                    np.testing.assert_allclose(env.get_obs(),initial['reset_observation'][episode],atol=2e-6,rtol=0)
                    for step,action in enumerate(values['action']):
                        np.testing.assert_allclose(env.get_obs(),values['observation'][step],atol=2e-6,rtol=0)
                        for key in ('qpos','qvel'):
                            np.testing.assert_allclose(env.get_env_state()[key],values[key][step],atol=1e-8,rtol=0)
                        assert action.shape == (30,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1
                        before = env.get_env_state(); xyz,rgb = capture()
                        assert xyz.shape == rgb.shape == (1200,3) and np.isfinite(xyz).all()
                        assert xyz.dtype == np.float32 and rgb.dtype == np.uint8
                        for key,value in before.items():np.testing.assert_array_equal(value,env.get_env_state()[key])
                        out['pcd_xyz'][written+step], out['pcd_rgb'][written+step] = xyz,rgb
                        obs,reward,done,info = env.step(action.copy())
                        np.testing.assert_allclose(obs,values['next_observation'][step],atol=2e-6,rtol=0)
                        np.testing.assert_allclose(reward,values['reward'][step],atol=2e-6,rtol=0)
                        assert bool(done)==bool(values['done'][step])
                        assert bool(info['goal_achieved'])==bool(values['goal_achieved'][step])
                    for key,value in values.items():out[key][written:written+length]=value
                    written += length; state['completed']+=1; state['frames']=written
                    out.flush();save()
                assert written==total
        state.update(stage='COMPLETE',exit_code=0)
    except BaseException as error:
        state.update(stage='FAILED',exit_code=1,error=repr(error));raise
    finally:
        if env is not None:env.close()
        save()
        args.out.with_suffix('.exit_code').write_text(str(state.get('exit_code',1))+'\n')


if __name__ == '__main__':
    main()

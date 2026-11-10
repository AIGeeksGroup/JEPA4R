"""Re-render only the mainline's selected held-out windows; no source HDF5 writes."""
import argparse
import hashlib
import json
import importlib.util
from pathlib import Path
import time

import h5py
import numpy as np


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def hdf_dependencies(path):
    helper = Path(__file__).resolve().parents[2] / 'envs/adroit/hdf_binding.py'
    specification = importlib.util.spec_from_file_location('_relocate_hdf_binding', helper)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.hdf_dependencies(path)


def selected_rows(lengths, steps, mode='full'):
    if mode not in ('full', 'smoke'):
        raise ValueError('RGB window mode must be full or smoke')
    maximum = np.repeat(np.asarray(lengths) - 25 - 1, lengths)
    valid = np.flatnonzero(np.asarray(steps) <= maximum)
    return np.sort(valid[np.random.default_rng(42).choice(len(valid) - 1,
                   size=50 if mode == 'full' else 1, replace=False)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--initials', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('full', 'smoke'), default='full',
                        help='Match the independent CEM full or one-fragment diagnostic mode')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='STARTED', started_at=time.time(), completed_frames=0,
                  mode=args.mode, diagnostic=args.mode == 'smoke',
                  camera='vil_camera', width=256, height=256, seed=42)
    def save():
        report['updated_at'] = time.time()
        (args.out/'report.json').write_text(json.dumps(report, indent=2))
    save()
    env = None
    try:
        source_binding = hdf_dependencies(args.test)
        initial_sha = file_sha(args.initials)
        from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0
        env = RelocateEnvV0()
        assert not env.sim.render_contexts and env.sim._render_context_offscreen is None
        env.model.vis.quality.offsamples = 0  # Original SHA724f worker's RGB capture.
        report['offsamples'] = int(env.model.vis.quality.offsamples)
        with h5py.File(args.test, 'r') as source, np.load(args.initials, allow_pickle=False) as initial:
            lengths, offsets = source['ep_len'][:], source['ep_offset'][:]
            starts = selected_rows(lengths, source['step_idx'][:], args.mode)
            rows = np.unique((starts[:, None] + np.arange(26)[None]).ravel())
            episodes = np.repeat(np.arange(len(lengths)), lengths)
            report.update(sampled_start_rows=starts.tolist(), cached_rows=len(rows))
            cache = args.out/'rgb_windows.h5'
            with h5py.File(cache, 'x') as output:
                output.create_dataset('row_index', data=rows)
                images = output.create_dataset('pixels', shape=(len(rows),256,256,3), dtype='uint8',
                                               chunks=(1,256,256,3), compression='gzip', compression_opts=1)
                row_map = {int(row): i for i, row in enumerate(rows)}
                for episode in np.unique(episodes[rows]):
                    offset, length = int(offsets[episode]), int(lengths[episode])
                    stop = int(max(rows[episodes[rows] == episode]) - offset)
                    values = {key: source[key][offset:offset+length]
                              for key in ('action','observation','qpos','qvel','next_observation','reward','seed')}
                    seed = int(values['seed'][0])
                    assert seed == int(initial['seed'][episode])
                    env.seed(seed); env.reset()
                    for key in ('qpos','qvel','hand_qpos','obj_pos','palm_pos','target_pos'):
                        np.testing.assert_array_equal(env.get_env_state()[key], initial[key][episode])
                    np.testing.assert_allclose(env.get_obs(), initial['reset_observation'][episode], atol=2e-6, rtol=0)
                    for step in range(stop+1):
                        np.testing.assert_allclose(env.get_obs(), values['observation'][step], atol=2e-6, rtol=0)
                        for key in ('qpos','qvel'):
                            np.testing.assert_allclose(env.get_env_state()[key], values[key][step], atol=1e-8, rtol=0)
                        row = offset + step
                        if row in row_map:
                            before = env.get_env_state()
                            image = env.sim.render(width=256,height=256,camera_name='vil_camera',device_id=0)[::-1].copy()
                            assert image.shape == (256,256,3) and image.dtype == np.uint8
                            for key, value in before.items():
                                np.testing.assert_array_equal(value, env.get_env_state()[key])
                            images[row_map[row]] = image
                            report['completed_frames'] += 1
                        action = values['action'][step]
                        assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1
                        obs, reward, _, _ = env.step(action.copy())
                        np.testing.assert_allclose(obs, values['next_observation'][step], atol=2e-6, rtol=0)
                        np.testing.assert_allclose(reward, values['reward'][step], atol=2e-6, rtol=0)
                    output.flush(); save()
            assert report['completed_frames'] == len(rows)
            report.update(status='COMPLETE', exit_code=0,
                          inputs={str(args.test.resolve()):source_binding['root_sha256'],
                                  str(args.initials.resolve()):initial_sha,
                                  str(Path(__file__).resolve()):file_sha(__file__)},
                          input_dependencies={'test':source_binding},
                          dependency_binding_source_sha256=file_sha(Path(__file__).resolve().parents[2] / 'envs/adroit/hdf_binding.py'),
                          cache_sha256=file_sha(cache))
    except BaseException as error:
        report.update(status='FAILED', exit_code=130 if isinstance(error, KeyboardInterrupt) else 1,
                      error=repr(error))
        raise
    finally:
        if env is not None:
            env.close()
        save()
        (args.out/'exit_code').write_text(str(report.get('exit_code',1))+'\n')


if __name__ == '__main__':
    main()

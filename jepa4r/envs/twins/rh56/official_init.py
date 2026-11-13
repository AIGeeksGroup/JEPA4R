# Moved from OGBench-cube@4fbb0c6:lewm_pcd/rh56_adapter/official_init.py (verbatim except: imports / paths adapted to the package)
"""Opt-in task-initial-state adapter; delegate collection to rh56_adapter.collect.

Only Object/target positions are imported. RH56 reset, feedback expert, actions,
physics, renderer, camera, sampling and the original collection loop are reused.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

from jepa4r.envs.twins.rh56._runtime import (RUNTIME_HELPERS_SHA256, atomic_json,
                                   file_sha256 as sha, temporary_bindings)

def write_json(path, value, *, new=False):
    return atomic_json(path, value, new=new, sort_keys=True, allow_nan=False)


def load_initials(path, seed_base):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema') != 'official-task-init-v1':
        raise ValueError('expected official-task-init-v1 schema')
    source = Path(payload['source_manifest'])
    if not source.is_absolute() or sha(source) != payload['source_sha']:
        raise ValueError('official source manifest changed or is not absolute')
    records = payload['records']
    if (len(records) != 100 or any(type(r['demo_index']) is not int for r in records)
            or [r['demo_index'] for r in records] != list(range(100))):
        raise ValueError('require the unchanged official demo order 0..99')
    if type(seed_base) is not int or not 64000000 <= seed_base <= 64999900:
        raise ValueError('100 fresh seeds must stay in the formal BC namespace')
    sources = [r['source_id'] for r in records]
    if any(not isinstance(s, str) or not s for s in sources) or len(set(sources)) != 100:
        raise ValueError('official source IDs must be nonempty and unique')
    for record in records:
        if not re.fullmatch('[0-9a-f]{64}', record['original_init_sha256']):
            raise ValueError('invalid original initialization SHA256')
        for key in ('obj_pos', 'target_pos'):
            value = np.asarray(record[key], dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f'{key} must contain three finite coordinates')
    return payload, {seed_base + r['demo_index']: r for r in records}


def make_bridge(base_class, records, forward, on_reset):
    class OfficialTaskBridge(base_class):
        def reset(self, seed=None):
            if seed not in records:
                raise ValueError(f'seed {seed} is not in the fixed official initialization plan')
            _, info = super().reset(seed=seed)
            row = records[seed]
            env = self.env
            model, data = env.model, env.data
            native = {key: np.asarray(getattr(data, key)).copy()
                      for key in ('qpos', 'qvel', 'ctrl', 'act')}
            drive = {key: np.asarray(getattr(env, key)).copy() for key in ('p', 'v', 'q_cmd')}
            body_pos, site_pos = model.body_pos.copy(), model.site_pos.copy()
            body_pos[env.obj_bid] = row['obj_pos']
            site_pos[env.target_sid] = row['target_pos']
            model.body_pos[env.obj_bid] = row['obj_pos']
            model.site_pos[env.target_sid] = row['target_pos']
            forward(model, data)
            for key, value in native.items():
                np.testing.assert_array_equal(getattr(data, key), value, err_msg=f'RH56 {key} changed')
            for key, value in drive.items():
                np.testing.assert_array_equal(getattr(env, key), value, err_msg=f'RH56 drive {key} changed')
            np.testing.assert_array_equal(model.body_pos, body_pos)
            np.testing.assert_array_equal(model.site_pos, site_pos)
            np.testing.assert_allclose(data.xpos[env.obj_bid], row['obj_pos'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(data.site_xpos[env.target_sid], row['target_pos'], atol=1e-10, rtol=0)
            on_reset(dict(seed=int(seed), **row, robot_and_drive_unchanged=True,
                          actual_obj_pos=data.xpos[env.obj_bid].tolist(),
                          actual_target_pos=data.site_xpos[env.target_sid].tolist()))
            return self.observation(), info

    return OfficialTaskBridge


def summarize_attempts(records, expected_seeds):
    if [int(r['seed']) for r in records] != list(expected_seeds):
        raise ValueError('attempted source order differs from the frozen plan')
    failed = [int(r['seed']) for r in records if not r['success']]
    return dict(attempted=len(records), failed_seeds=failed,
                all_requested_sources_succeeded=not failed,
                selection='all requested official indices retained; no resampling or filtering')


def build_bc_manifest(attempts, expected_seeds, episode_root, source_records, *,
                      init_source, init_source_sha256, producer_chain):
    """Describe all fixed-source attempts; a failed set has no trainable selection."""
    summary = summarize_attempts(attempts, expected_seeds)
    episode_root = Path(episode_root).resolve()
    all_attempts = []
    for record in attempts:
        source = source_records[int(record['seed'])]
        path = (episode_root / record['npz']).resolve()
        if not path.is_relative_to(episode_root):
            raise ValueError('demonstration payload must stay inside its collection output')
        all_attempts.append(dict(record, npz=str(path),
            official_demo_index=source['demo_index'], source_id=source['source_id'],
            source_init_sha256=source['original_init_sha256']))
    succeeded = summary['all_requested_sources_succeeded']
    count = len(all_attempts)
    status = ('REJECTED_FAILED_OFFICIAL_SOURCE' if not succeeded else
              'READY_FORMAL_BC' if count == 100 else
              'READY_PROFILE_BC' if count == 1 else 'PARTIAL_NOT_FORMAL_READY')
    return dict(schema='rh56-official-init-bc-demos-v1', status=status,
                selected=all_attempts if succeeded else [], all_attempts=all_attempts,
                requested_count=count, all_requested_sources_succeeded=succeeded,
                formal_bc_ready=succeeded and count == 100,
                profile_bc_ready=succeeded and count == 1,
                selection=summary['selection'], init_source=str(Path(init_source).resolve()),
                init_source_sha256=init_source_sha256, producer_chains=[dict(producer_chain)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-source', required=True, type=Path)
    parser.add_argument('--init-source-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed-base', type=int, default=64500000)
    parser.add_argument('--start-demo', type=int, default=0)
    parser.add_argument('--count', type=int, default=100)
    args = parser.parse_args(argv)
    if sha(args.init_source) != args.init_source_sha256:
        raise ValueError('frozen task initialization manifest SHA256 differs')
    payload, all_records = load_initials(args.init_source, args.seed_base)
    if not (0 <= args.start_demo < 100 and 1 <= args.count <= 100 - args.start_demo):
        raise ValueError('requested contiguous official demo range is invalid')
    seeds = list(range(args.seed_base + args.start_demo, args.seed_base + args.start_demo + args.count))
    records = {seed: all_records[seed] for seed in seeds}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(phase='preparing', started=time.time(), completed=0, failed_episodes=0, exit_code=None)

    def save():
        state['updated'] = time.time()
        write_json(out / 'wrapper_state.json', state)

    save()
    try:
        code = Path(__file__).resolve().parents[4]
        sys.path.insert(0, str(code))
        import mujoco
        from jepa4r.envs.twins.rh56 import camera_preview, collect, native
        plan = dict(episodes=[dict(split='bc_demo', seed=seed, coverage='expert') for seed in seeds])
        write_json(out / 'init_source.json', payload, new=True)
        write_json(out / 'plan.json', plan, new=True)
        chain = dict(schema='official-task-init-producer-chain-v1', wrapper=str(Path(__file__).resolve()),
                     wrapper_sha256=sha(__file__), runtime_helpers_sha256=RUNTIME_HELPERS_SHA256,
                     init_source_path=str(args.init_source.resolve()),
                     init_source_sha256=args.init_source_sha256,
                     source_manifest=payload['source_manifest'], source_sha=payload['source_sha'],
                     original_collect_sha256=sha(collect.__file__), native_sha256=sha(native.__file__),
                     camera_sha256=sha(camera_preview.__file__), seed_base=args.seed_base,
                     requested_indices=[r['demo_index'] for r in records.values()],
                     imported_fields=['obj_pos', 'target_pos'],
                     original_producer_fields='unchanged; this chain additionally binds the reset adapter')
        write_json(out / 'producer_chain.json', chain, new=True)
        resets = {}

        def receipt(value):
            resets[str(value['seed'])] = value
            write_json(out / 'reset_receipts.json', resets)

        bridge = make_bridge(camera_preview.OppositeCameraBridge, records, mujoco.mj_forward, receipt)
        state['phase'] = 'collecting_original_loop'
        save()
        with temporary_bindings((camera_preview, 'OppositeCameraBridge', bridge)):
            collect.main(['--plan', str(out / 'plan.json'), '--output', str(out / 'episodes'),
                          '--camera', 'opposite', '--allow-expert-failure'])
        manifest = json.loads((out / 'episodes/manifest.json').read_text())
        summary = summarize_attempts(manifest['auxiliary'], seeds)
        write_json(out / 'attempt_summary.json', summary, new=True)
        bc_manifest = build_bc_manifest(manifest['auxiliary'], seeds, out / 'episodes', records,
            init_source=args.init_source, init_source_sha256=args.init_source_sha256,
            producer_chain=dict(path=str(out / 'producer_chain.json'),
                                sha256=sha(out / 'producer_chain.json')))
        bc_path = out / 'bc_demos_manifest.json'
        write_json(bc_path, bc_manifest, new=True)
        state.update(phase='complete', completed=summary['attempted'],
                     failed_episodes=len(summary['failed_seeds']), exit_code=0,
                     all_requested_sources_succeeded=summary['all_requested_sources_succeeded'],
                     bc_demos_manifest=str(bc_path), bc_manifest_status=bc_manifest['status'])
        save()
        print(json.dumps(dict(status=bc_manifest['status'], bc_demos_manifest=str(bc_path),
                              selected=len(bc_manifest['selected']), attempts=summary['attempted'])), flush=True)
    except BaseException:
        state.update(phase='failed', exit_code=1, error=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()

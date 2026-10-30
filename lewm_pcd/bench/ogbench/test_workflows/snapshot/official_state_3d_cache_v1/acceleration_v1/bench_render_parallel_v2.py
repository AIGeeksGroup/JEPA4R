"""Same-source renderer capacity only; rerendered colors are not guaranteed bitexact."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
CORE = HERE.parent
STRICT_TOP = ('sampling_seed', 'PCD_xyz_sha256', 'camera', 'render_geometry',
              'canonical_raw_ids_sha256')
STRICT_BOUNDARY_KEYS = ('phase', 'capture_index', 'sampling_seed', 'rng_before_sha256',
    'rng_after_sha256', 'shape', 'source_UV_sha256', 'source_geom_sha256',
    'core_source_sha256', 'provider_source_sha256', 'geometry_offsamples', 'camera_K', 'world_from_cv')
COLOR_KEYS = ('PCD_rgb_sha256', 'RGB_sha256', 'capture_receipt_stream_sha256')
CHECK_KEYS = STRICT_TOP + COLOR_KEYS + ('capture_boundaries',)


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_render(result, expected):
    for key in STRICT_TOP:
        assert result[key] == expected[key], ('strict_source_geometry_failed', key)
    pose_differences, boundary_colors = {}, {}
    assert set(result['capture_boundaries']) == set(expected['capture_boundaries']) == {'0','999','1000'}
    for row, before in expected['capture_boundaries'].items():
        after = result['capture_boundaries'][row]
        for key in STRICT_BOUNDARY_KEYS:
            assert after[key] == before[key], ('strict_boundary_failed', row, key)
        for key in ('qpos','qvel'):
            assert after['source_pose_sha256'][key] == before['source_pose_sha256'][key], ('strict_pose_failed',row,key)
        pose_differences[row] = {key: dict(old=value,new=after['source_pose_sha256'][key],
            equal=value == after['source_pose_sha256'][key])
            for key,value in before['source_pose_sha256'].items() if key not in ('qpos','qvel')}
        boundary_colors[row] = {key:dict(old=before[key],new=after[key],equal=before[key]==after[key])
            for key in ('cloud_sha256','RGB_AA4_sha256')}
    return dict(numeric_equivalence_verified=False, native_color_bitexact_not_guaranteed=True,
        boundary_missing_ctrl_mocap=pose_differences, boundary_color_hashes=boundary_colors,
        **{key:dict(old=expected[key],new=result[key],equal=expected[key]==result[key]) for key in COLOR_KEYS})


def init_renderer(plan):
    sys.path.insert(0, str(CORE))
    from pcd_renderer import init_worker
    init_worker(plan['workspace'], str(Path(plan['shared_state_root']) / 'source_arrays.json'),
                plan['camera'], plan['pcd'])
    if 'torch' in sys.modules:
        sys.modules['torch'].set_num_threads(1)


def render_metadata(episode):
    import resource
    from pcd_renderer import render_episode
    before = resource.getrusage(resource.RUSAGE_SELF)
    value = render_episode(('train', episode))
    after = resource.getrusage(resource.RUSAGE_SELF)
    result = {key: value[key] for key in CHECK_KEYS if key != 'canonical_raw_ids_sha256'}
    assert value['raw_ids'].shape == (1001,)
    result['canonical_raw_ids_sha256'] = hashlib.sha256(value['raw_ids'].astype('<i8',copy=False).tobytes()).hexdigest()
    result.update(episode=episode, worker_pid=os.getpid(),
        render_started=value['render_started'], render_completed=value['render_completed'],
        user_CPU_seconds=after.ru_utime-before.ru_utime,
        system_CPU_seconds=after.ru_stime-before.ru_stime,
        worker_maxrss_kib=after.ru_maxrss)
    return result


def cg_snapshot():
    root = Path('/sys/fs/cgroup')
    names = ('cpu.max', 'cpu.stat', 'cpu.pressure', 'memory.current', 'memory.max',
             'memory.events', 'memory.pressure')
    return {name: (root / name).read_text() for name in names}


def stage(args, output):
    plan = json.loads(args.plan.read_text())
    expected = {}
    for episode in range(16):
        path = Path(plan['cache_dir']) / 'train' / f'episode_{episode:06d}/receipt.json'
        record = json.loads(path.read_text())
        assert record['actual_exit_code'] == 0 and record['raw_row_count'] == 1001
        assert record['checkpoint_sha256'] == plan['checkpoint_sha256']
        expected[episode] = dict(record=record, path=str(path), sha256=sha(path))
    for relative in ('official_state_3d_cache_v1/pcd_renderer.py',
                     'official_state_3d_cache_v1/common.py'):
        assert sha(Path(plan['workspace']) / relative) == plan['source_pins'][relative]
    arrays_receipt = Path(plan['shared_state_root']) / 'source_arrays.json'
    assert sha(arrays_receipt) == plan['source_arrays_receipt_sha256']
    arrays = json.loads(arrays_receipt.read_text())
    sys.path.insert(0, str(CORE))
    from common import legacy
    original = legacy(plan['workspace'])
    for split in arrays['splits'].values():
        assert original.raw_stat(split['source_npz']) == split['accepted_stat']
    before = cg_snapshot()
    pool_started = time.time()
    with ProcessPoolExecutor(max_workers=args.stage_workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=init_renderer, initargs=(plan,)) as pool:
        warmup_start = time.time()
        warmup = list(pool.map(render_metadata, range(args.stage_workers), chunksize=1))
        save(output / 'warmup_render_metadata.json', dict(results=warmup))
        for item in warmup:
            item['scope_readback'] = check_render(item, expected[item['episode']]['record'])
        warmup_end = time.time()
        started = time.time()
        results = list(pool.map(render_metadata, range(16), chunksize=1))
        ended = time.time()
        save(output / 'timed_render_metadata.json', dict(results=results))
        for item in results:
            item['scope_readback'] = check_render(item, expected[item['episode']]['record'])
    pool_closed = time.time()
    for item in expected.values():
        assert sha(Path(item['path'])) == item['sha256'], 'Original receipt changed'
    report = dict(actual_exit_code=0, workers=args.stage_workers,
        benchmark_scope='same_source_resource_parallel_throughput_capacity_only',
        numeric_equivalence_verified=False, native_color_bitexact_not_guaranteed=True,
        data_only=True, neural_forward_calls=0, training_updates=0, SR_episodes=0,
        full_PCD_or_RGB_persisted=False, plan_sha256=sha(args.plan),
        benchmark_source_sha256=sha(Path(__file__)),
        warmup=dict(episodes=len(warmup), rows=len(warmup)*1001,
            wall_seconds=warmup_end-warmup_start,
            first_renderer_started_seconds=min(x['render_started'] for x in warmup)-pool_started,
            startup_included=True, results=warmup),
        timed=dict(episodes=16, rows=16016, wall_seconds=ended-started,
            frames_per_second=16016/(ended-started),
            renderer_span_seconds=max(x['render_completed'] for x in results)-min(x['render_started'] for x in results),
            parent_receives_only_hash_metadata=True, results=results),
        pool_shutdown_seconds=pool_closed-ended,
        cgroup_before=before, cgroup_after=cg_snapshot(),
        source_receipts=[{key: value for key,value in x.items() if key != 'record'} for x in expected.values()],
        limitations='Data-only capacity excludes full PCD IPC, encoder work and production writes; not an end-to-end speed claim')
    save(output / 'report.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--stage-workers', type=int, choices=(8,16), required=True)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='prepared_only_no_render_or_GPU', total_rows_if_executed=(args.stage_workers+16)*1001)))
        return
    assert os.name == 'posix' and not sys.flags.optimize
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save(args.output_dir / 'actual_argv.json', dict(argv=sys.argv,
        cwd=os.getcwd(), started=time.time(), affinity=sorted(os.sched_getaffinity(0))))
    code = 1
    try:
        stage(args, args.output_dir)
        code = 0
    except BaseException as error:
        save(args.output_dir / 'failure.json', dict(error=repr(error), actual_exit_code=1))
        raise
    finally:
        (args.output_dir / 'actual_exit_code').write_text(str(code)+'\n')


if __name__ == '__main__':
    main()

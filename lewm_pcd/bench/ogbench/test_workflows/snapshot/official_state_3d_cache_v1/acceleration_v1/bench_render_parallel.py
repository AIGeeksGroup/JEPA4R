"""Data-only 8/16 renderer capacity check; original capture and private RNG unchanged."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CORE = HERE.parent
CHECK_KEYS = ('sampling_seed', 'PCD_xyz_sha256', 'PCD_rgb_sha256', 'RGB_sha256',
              'capture_receipt_stream_sha256', 'camera', 'render_geometry')


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_render(result, expected):
    for key in CHECK_KEYS:
        assert result[key] == expected[key], ('renderer_equivalence_failed', key)


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
    result = {key: value[key] for key in CHECK_KEYS}
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


def pg_members(pgid):
    members = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != 'Z':
                    members.append(int(path.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    return members


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
        for item in warmup:
            check_render(item, expected[item['episode']]['record'])
        warmup_end = time.time()
        started = time.time()
        results = list(pool.map(render_metadata, range(16), chunksize=1))
        ended = time.time()
        for item in results:
            check_render(item, expected[item['episode']]['record'])
    pool_closed = time.time()
    for item in expected.values():
        assert sha(Path(item['path'])) == item['sha256'], 'Original receipt changed'
    report = dict(actual_exit_code=0, workers=args.stage_workers,
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
    parser.add_argument('--stage-workers', type=int, choices=(8,16))
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status='prepared_only_no_render_or_GPU', total_rows_if_executed=56056)))
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
        if args.stage_workers:
            stage(args, args.output_dir)
        else:
            records = []
            for workers in (8,16):
                out = args.output_dir / f'workers{workers}'
                argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--plan', str(args.plan),
                    '--output-dir', str(out), '--execute', '--stage-workers', str(workers)]
                started = time.time()
                with (args.output_dir / f'workers{workers}.log').open('xb') as log:
                    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    exit_code = child.wait()
                residual = pg_members(child.pid)
                record = dict(workers=workers, argv=argv, PID_locator=child.pid,
                    actual_exit_code=exit_code, started=started, ended=time.time(),
                    owned_group_residual_nonZ=residual)
                save(args.output_dir / f'workers{workers}_exit.json', record)
                assert exit_code == 0 and not residual, record
                records.append(record)
            save(args.output_dir / 'report.json', dict(actual_exit_code=0,
                stages=records, rendered_rows=56056, no_NN=True,
                source_sha256=sha(Path(__file__)), comparison=[
                    json.loads((args.output_dir/f'workers{x}'/'report.json').read_text())['timed']
                    for x in (8,16)]))
        code = 0
    except BaseException as error:
        save(args.output_dir / 'failure.json', dict(error=repr(error), actual_exit_code=1))
        raise
    finally:
        (args.output_dir / 'actual_exit_code').write_text(str(code)+'\n')


if __name__ == '__main__':
    main()

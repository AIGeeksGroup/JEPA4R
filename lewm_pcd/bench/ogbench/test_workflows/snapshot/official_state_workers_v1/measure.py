"""Bounded real-data input equivalence and timing; never a formal 1M/SR run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np

from ordered_input import MODES, OrderedInput


HERE = Path(__file__).resolve().parent
DATA_NAMES = ('cube-triple-play-v0.npz', 'cube-triple-play-v0-val.npz')
PINS = {'agents/gciql.py': '05a0b5d9b987d3049df8a1158bbfbaba2ca2611604d24f9e484f8fb6d0cd92bc',
        'utils/datasets.py': '52c373922a87e1ba8ad8b3b96c3474d5e6385e2ea2c7206930f7a8a315816609'}
ADAPTER_SHA = '4508ce847694976f8af924215d11985ac4d795093b22d62e8424928346fbdda2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def mark(out, phase, **fields):
    value = dict(phase=phase, updated=time.time(), **fields)
    for name in ('phase.json', 'status.json'):
        (out / name).write_text(json.dumps(value, indent=2) + '\n')


def same_rng(left, right):
    assert left[0] == right[0] and left[2:] == right[2:]
    assert np.array_equal(left[1], right[1])


def rng_summary(state):
    return dict(generator=state[0], keys_sha256=hashlib.sha256(state[1].tobytes()).hexdigest(),
                position=int(state[2]), has_gauss=int(state[3]), cached_gaussian=float(state[4]))


def save_rng(path, state):
    np.savez(path, generator=np.array(state[0]), keys=state[1], position=np.array(state[2]),
             has_gauss=np.array(state[3]), cached_gaussian=np.array(state[4]))


def load_rng(path):
    with np.load(path, allow_pickle=False) as z:
        return (str(z['generator'].item()), z['keys'].copy(), int(z['position']),
                int(z['has_gauss']), float(z['cached_gaussian']))


def batch_summary(batch):
    return {key: dict(shape=list(value.shape), dtype=value.dtype.str, bytes=value.nbytes,
                     sha256=hashlib.sha256(value.tobytes()).hexdigest())
            for key, value in batch.items()}


def exact(left, right, label, out):
    leaves_a, structure_a = s.jax.tree_util.tree_flatten(left)
    leaves_b, structure_b = s.jax.tree_util.tree_flatten(right)
    assert structure_a == structure_b and len(leaves_a) == len(leaves_b), label
    for number, (a, b) in enumerate(zip(leaves_a, leaves_b)):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            np.savez(out / 'first_mismatch.npz', original=a, candidate=b)
            raise AssertionError(f'{label}: leaf{number} shape/dtype/bytes differ')
        assert np.isfinite(a).all(), (label, number, 'nonfinite')


def _loss_and_gradient(agent, batch):
    # The exact official update split and unchanged total_loss; diagnostic only.
    _, rng = s.jax.random.split(agent.rng)
    return s.jax.value_and_grad(lambda params: agent.total_loss(batch, params, rng=rng),
                              has_aux=True)(agent.network.params)


def file_stats(paths):
    result = {}
    for path in paths:
        st = path.stat()
        result[path.name] = dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns,
                                 ctime_ns=st.st_ctime_ns, device=st.st_dev, inode=st.st_ino)
    return result


def read_threads():
    names = [thread.name for thread in threading.enumerate()]
    rows = sorted(name for name in names if name.startswith('ordered-rows'))
    coordinators = sorted(name for name in names if name.startswith('ordered-materialize'))
    return dict(row_names=rows, row_count=len(rows), coordinator_names=coordinators,
                coordinator_count=len(coordinators))


def load_data(args, cfg, out):
    receipt_path = Path(args.data_receipt).resolve()
    receipt = json.loads(receipt_path.read_text())
    assert receipt['stage'] == 'complete' and receipt['actual_exit_code'] == 0
    freeze_path = Path(args.data_stat_freeze).resolve()
    freeze = json.loads(freeze_path.read_text())
    assert freeze['actual_exit_code'] == 0 and freeze['acceptance_sha256'] == sha(receipt_path)
    paths = [(Path(args.data_dir) / name).resolve() for name in DATA_NAMES]
    before = file_stats(paths)
    datasets, metadata = [], {}
    loader_sha = sha(s.ROOT / 'source/ogbench/ogbench/utils.py')
    for split, path in zip(('train', 'validation'), paths):
        accepted = next(item for item in receipt['datasets']
                        if Path(item['schema_loader']['path']).resolve() == path)
        raw, schema = accepted['download'], accepted['schema_loader']
        frozen = freeze['datasets'][split]
        assert Path(frozen['path']).resolve() == path and frozen['raw_sha256'] == raw['sha256']
        assert raw['actual_exit_code'] == schema['actual_exit_code'] == 0
        assert path.stat().st_size == raw['bytes'] == schema['raw_bytes']
        assert all(frozen['accepted_stat'][key] == before[path.name][key]
                   for key in ('size', 'mtime_ns', 'ctime_ns', 'device', 'inode'))
        assert raw['sha256'] == schema['raw_sha256'] and schema['utils_sha256'] == loader_sha
        assert schema['all_raw_fields_finite'] and schema['all_compact_fields_finite']
        assert schema['compact_rows_exactly_match_raw'] and schema['compact_terminal_semantics_exact']
        mark(out, 'loading_original_dataset', dataset=path.name)
        # No environment creation, downloading, qpos/qvel loading, or data rewrite.
        data = s.GCDataset(s.Dataset.create(**s.ogbench.load_dataset(
            str(path), compact_dataset=True, add_info=False)), cfg)
        assert data.size == schema['rows'] and int(data.dataset['valids'].sum()) == schema['compact_valid_rows']
        assert set(data.dataset._dict) == {'observations', 'actions', 'terminals', 'valids'}
        assert data.dataset['observations'].shape == (data.size, 46)
        assert data.dataset['actions'].shape == (data.size, 5)
        metadata[path.name] = dict(raw_sha256=raw['sha256'], raw_bytes=raw['bytes'], rows=data.size,
            episodes=schema['episodes'], compact_valid_rows=schema['compact_valid_rows'],
            accepted_stat=frozen['accepted_stat'],
            fields={key: dict(shape=list(value.shape), dtype=value.dtype.str, bytes=value.nbytes)
                    for key, value in data.dataset._dict.items()})
        datasets.append(data)
    assert file_stats(paths) == before
    return datasets, dict(data_receipt=dict(path=str(receipt_path), sha256=sha(receipt_path)),
                         data_stat_freeze=dict(path=str(freeze_path), sha256=sha(freeze_path)),
                         datasets=metadata, raw_npz_stats_before=before), paths


def identity(args, cfg):
    sources = {name: sha(s.IMPL / name) for name in (*PINS, 'utils/flax_utils.py', 'main.py')}
    assert all(sources[name] == digest for name, digest in PINS.items())
    assert sha(HERE / 'ordered_input.py') == ADAPTER_SHA
    return dict(stage=args.stage, mode=args.mode, official_commit='a977253ce5b88434f4e4f531c50047cc95993a51',
        source_sha256=sources, harness_sha256=sha(__file__), adapter_sha256=ADAPTER_SHA,
        old_smoke_sha256=sha(s.__file__), official_config=cfg.to_dict(),
        Python=sys.version, NumPy=np.__version__, JAX=s.jax.__version__, Flax=s.flax.__version__,
        JAX_default_matmul_precision=str(s.jax.config.jax_default_matmul_precision),
        devices=[dict(platform=d.platform, kind=d.device_kind, id=d.id) for d in s.jax.devices()],
        current_dim=46, goal_dim=46, action_dim=5, batch_size=1024, input_dtype='float32',
        fixture_only=False, artificial_terminal=False, formal_training=False, SR=False,
        official_agent_sampler_save_restore_unmodified=True, enabled_by_default=False,
        CPU_affinity=sorted(os.sched_getaffinity(0)),
        environment={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS',
            'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})


def create_initial(train, cfg):
    np.random.seed(0)
    example = train.sample(1)
    agent = s.GCIQLAgent.create(0, example['observations'], example['actions'], cfg)
    assert int(agent.network.step) == 1 and int(agent.network.opt_state[0].count) == 0
    return example, agent


def correctness(out, train, val, cfg):
    gold, modes = [], {}
    initial_example, initial_agent = create_initial(train, cfg)
    after_example = np.random.get_state()
    reference_state = None
    for mode in MODES:
        mark(out, 'strict_numerical_gate', mode=mode, completed_updates=0, actual_exit_code=None)
        example, fresh_agent = create_initial(train, cfg)
        exact(initial_example, example, mode + ':example', out)
        same_rng(after_example, np.random.get_state())
        exact(s.flax.serialization.to_state_dict(initial_agent),
              s.flax.serialization.to_state_dict(fresh_agent), mode + ':initial153', out)
        agent = initial_agent
        updates, rngs, validation, threads = [], [], None, None
        with OrderedInput(train, mode=mode) as inputs:
            for step in range(1, 4):
                batch = inputs.sample(prefetch_next=step == 1)
                s.check_batch(batch)
                logical_rng = inputs.committed_rng_state
                raw = loss_and_gradient(agent, batch)
                if mode == 'baseline':
                    gold.append(dict(batch=batch, rng=logical_rng, raw=s.jax.device_get(raw)))
                    np.savez(out / f'gold_train_{step}.npz', **batch)
                else:
                    exact(gold[step - 1]['batch'], batch, f'{mode}:batch{step}', out)
                    same_rng(gold[step - 1]['rng'], logical_rng)
                    exact(gold[step - 1]['raw'], raw, f'{mode}:raw_loss_gradient{step}', out)
                agent, info = agent.update(batch)
                observed = s.metrics(info)
                assert observed['grad/norm'] > 0
                if mode == 'baseline':
                    gold[step - 1]['info'] = s.jax.device_get(info)
                else:
                    exact(gold[step - 1]['info'], info, f'{mode}:update_info{step}', out)
                updates.append(observed)
                rngs.append(rng_summary(logical_rng))
                threads = read_threads()
                if step == 2:
                    inputs.barrier()
                    same_rng(gold[1]['rng'], np.random.get_state())
                    validation_batch = val.sample(1024)  # Official default evaluation=False.
                    s.check_batch(validation_batch)
                    validation_rng = np.random.get_state()
                    validation_raw = agent.total_loss(validation_batch, grad_params=None)
                    validation = s.metrics(validation_raw[1])
                    if mode == 'baseline':
                        val_gold = dict(batch=validation_batch, rng=validation_rng,
                                        raw=s.jax.device_get(validation_raw))
                        np.savez(out / 'gold_validation.npz', **validation_batch)
                    else:
                        exact(val_gold['batch'], validation_batch, mode + ':validation_batch', out)
                        same_rng(val_gold['rng'], validation_rng)
                        exact(val_gold['raw'], validation_raw, mode + ':validation_loss_info', out)
                    inputs.barrier()  # Drained save boundary; unique actual save occurs at final step3.
                    same_rng(val_gold['rng'], np.random.get_state())
                mark(out, 'strict_numerical_gate', mode=mode, completed_updates=step, actual_exit_code=None)
            inputs.barrier()
            same_rng(gold[2]['rng'], np.random.get_state())
        assert read_threads()['row_count'] == read_threads()['coordinator_count'] == 0
        same_rng(gold[2]['rng'], np.random.get_state())
        assert int(agent.network.step) == 4 and int(agent.network.opt_state[0].count) == 3
        state = s.flax.serialization.to_state_dict(agent)
        if mode == 'baseline':
            reference_state = s.jax.device_get(state)
            saved = s.fingerprint(agent)
            (out / 'checkpoint').mkdir()
            s.save_agent(agent, str(out / 'checkpoint'), 3)
            np.savez(out / 'reload_batch.npz', **batch)
            save_rng(out / 'numpy_tail_aux.npz', np.random.get_state())
            tail = np.random.get_state()
            next_batch = train.sample(1024)
            next_batch_summary = batch_summary(next_batch)
            next_rng = rng_summary(np.random.get_state())
            np.random.set_state(tail)
        else:
            exact(reference_state, state, mode + ':final_full153', out)
            assert s.fingerprint(agent) == saved
        modes[mode] = dict(updates=updates, consumed_training_rng=rngs, validation=validation,
            validation_rng=rng_summary(val_gold['rng']), final_rng=rng_summary(np.random.get_state()),
            final_state=s.fingerprint(agent), Adam_count=3, network_step=4,
            all_batch_keys_dtype_shape_bytes_exact=True, raw_loss_gradients_exact=True,
            update_info_exact=True, boundary_NumPy_RNG_exact=True, final_full153_exact=True,
            threads_during=threads, threads_after_close=read_threads())
    return dict(actual_updates_per_mode=3, modes=modes, saved_state=saved,
        golden_batches={f'train{number + 1}': batch_summary(item['batch']) for number, item in enumerate(gold)},
        golden_validation_batch=batch_summary(val_gold['batch']), next_training_batch=next_batch_summary,
        next_training_rng=next_rng, final_NumPy_RNG=rng_summary(gold[2]['rng']),
        unique_checkpoint_step=3, original_checkpoint_contains_host_NumPy_RNG=False,
        host_NumPy_RNG_auxiliary_file='numpy_tail_aux.npz',
        fresh_reload_verified=False, numerical_scope='3 updates per mode, real validation, final153; no tolerance')


def fresh(out, train, cfg, args):
    reference = Path(args.reference_dir).resolve()
    report = json.loads((reference / 'report.json').read_text())
    assert report['actual_exit_code'] == 0 and report['stage'] == 'correctness'
    assert report['source_sha256'] == identity(args, cfg)['source_sha256']
    assert report['official_config'] == json.loads(json.dumps(cfg.to_dict()))
    assert report['data_receipt']['sha256'] == sha(args.data_receipt)
    assert report['data_stat_freeze']['sha256'] == sha(args.data_stat_freeze)
    with np.load(reference / 'reload_batch.npz', allow_pickle=False) as z:
        batch = dict(z)
    s.check_batch(batch)
    agent = s.GCIQLAgent.create(0, batch['observations'][:1], batch['actions'][:1], cfg)
    agent = s.restore_agent(agent, str(reference / 'checkpoint'), 3)
    assert s.fingerprint(agent) == report['saved_state']
    assert int(agent.network.step) == 4 and int(agent.network.opt_state[0].count) == 3
    actions = np.asarray(agent.sample_actions(batch['observations'], batch['actor_goals'],
        seed=s.jax.random.PRNGKey(0), temperature=0))
    assert actions.shape == (1024, 5) and np.isfinite(actions).all()
    np.random.set_state(load_rng(reference / 'numpy_tail_aux.npz'))
    assert rng_summary(np.random.get_state()) == report['final_NumPy_RNG']
    assert batch_summary(train.sample(1024)) == report['next_training_batch']
    assert rng_summary(np.random.get_state()) == report['next_training_rng']
    return dict(reference_dir=str(reference), reference_report_sha256=sha(reference / 'report.json'),
        full153_restored_exact=s.fingerprint(agent), finite_actions=True, Adam_count=3, network_step=4,
        restored_agent_JAX_RNG=True, original_checkpoint_contains_host_NumPy_RNG=False,
        host_NumPy_RNG_restored_from_separate_aux=True, next_training_batch_digest_exact=True)


def benchmark(out, train, cfg, mode):
    _, agent = create_initial(train, cfg)
    windows = []
    with OrderedInput(train, mode=mode) as inputs:
        mark(out, 'warmup', completed_updates=0, target_updates=64, actual_exit_code=None)
        for step in range(64):
            agent, info = agent.update(inputs.sample(prefetch_next=step < 63))
        inputs.barrier()
        s.jax.block_until_ready((agent, info))
        for number in range(3):
            mark(out, 'timed_window', window=number + 1, completed_updates=64 + number * 2048,
                 target_window_updates=2048, actual_exit_code=None)
            sample_seconds = 0.
            wall_started, started = time.time(), time.perf_counter()
            for step in range(2048):
                sample_started = time.perf_counter()
                batch = inputs.sample(prefetch_next=step < 2047)
                sample_seconds += time.perf_counter() - sample_started
                agent, info = agent.update(batch)
            inputs.barrier()
            s.jax.block_until_ready((agent, info))
            elapsed, wall_finished = time.perf_counter() - started, time.time()
            observed = s.metrics(info)
            assert observed['grad/norm'] > 0
            windows.append(dict(window=number + 1, updates=2048, seconds=elapsed,
                Hz=2048 / elapsed, host_sample_seconds=sample_seconds,
                wall_started=wall_started, wall_finished=wall_finished,
                last_metrics=observed, last_batch=batch_summary(batch),
                boundary_NumPy_RNG=rng_summary(np.random.get_state()), threads=read_threads()))
    assert read_threads()['row_count'] == read_threads()['coordinator_count'] == 0
    assert int(agent.network.step) == 6209 and int(agent.network.opt_state[0].count) == 6208
    return dict(warmup_updates=64, timed_updates=6144, total_updates=6208, windows=windows,
        median_Hz=float(np.median([window['Hz'] for window in windows])),
        Adam_count=6208, network_step=6209, final_state=s.fingerprint(agent),
        final_NumPy_RNG=rng_summary(np.random.get_state()), threads_after_close=read_threads(),
        full_agent_and_info_tail_synchronized=True, sample_wait_and_dispatch_included=True,
        formal_periodic_validation_evaluation_logging_excluded=True,
        GPU_resources='external root supervisor; not inferred from throughput',
        scope='bounded6208-update diagnostic; not formal1M or SR or official completion ETA')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('correctness', 'fresh', 'benchmark'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=tuple(MODES), default='baseline')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--data-receipt', required=True)
    parser.add_argument('--data-stat-freeze', required=True)
    parser.add_argument('--reference-dir')
    args = parser.parse_args()
    if args.stage == 'fresh' and not args.reference_dir:
        parser.error('fresh requires --reference-dir from the correctness invocation')
    if args.stage != 'benchmark' and args.mode != 'baseline':
        parser.error('correctness runs all modes once; fresh restores its unique baseline checkpoint')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'argv.json', dict(argv=[sys.executable, *sys.argv], pid=os.getpid(), started=time.time()))
    try:
        global s, loss_and_gradient
        sys.path.insert(0, str(HERE.parent / 'official_state_v1'))
        import smoke as s
        assert s.jax.default_backend() == 'gpu'
        loss_and_gradient = s.jax.jit(_loss_and_gradient)
        cfg = s.config()
        bindings = identity(args, cfg)
        write(out / 'config_and_source.json', bindings)
        (train, val), data_binding, paths = load_data(args, cfg, out)
        bindings.update(data_binding)
        write(out / 'bindings.json', bindings)
        if args.stage == 'correctness':
            result = correctness(out, train, val, cfg)
        elif args.stage == 'fresh':
            result = fresh(out, train, cfg, args)
        else:
            result = benchmark(out, train, cfg, args.mode)
        after = file_stats(paths)
        assert after == bindings['raw_npz_stats_before']
        write(out / 'report.json', dict(**bindings, **result, raw_npz_stats_after=after,
                                      actual_exit_code=0, completed=time.time()))
        mark(out, 'complete', actual_exit_code=0)
    except BaseException as error:
        write(out / 'failure.json', dict(actual_exit_code=1, error=repr(error),
            traceback=traceback.format_exc(), failed=time.time(), tolerance_relaxed=False))
        mark(out, 'failed', actual_exit_code=1, error=repr(error))
        raise


if __name__ == '__main__':
    main()

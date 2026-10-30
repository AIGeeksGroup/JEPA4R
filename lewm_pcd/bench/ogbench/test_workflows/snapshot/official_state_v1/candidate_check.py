"""Strict candidate-only equivalence gate; benchmark requires a separate invocation."""
import argparse
import json
from pathlib import Path
import time

import smoke as s
from resident_inputs import dataset as resident_dataset


def same_rng(left, right):
    assert left[0] == right[0] and s.np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def exact(left, right, label, out):
    a, ta = s.jax.tree_util.tree_flatten(left)
    b, tb = s.jax.tree_util.tree_flatten(right)
    assert ta == tb and len(a) == len(b), label
    for index, (x, y) in enumerate(zip(a, b)):
        x, y = s.np.asarray(x), s.np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
            s.np.savez(out / 'first_mismatch.npz', original=x, candidate=y)
            raise AssertionError(f'{label}: leaf {index} dtype/shape/bytes differ')


@s.jax.jit
def loss_and_gradient(agent, batch):
    # Validation only: call the unchanged official loss with its official update RNG.
    _, rng = s.jax.random.split(agent.rng)
    return s.jax.value_and_grad(
        lambda params: agent.total_loss(batch, params, rng=rng), has_aux=True)(agent.network.params)


def paired_sample(host, resident, size, out, label):
    before = s.np.random.get_state()
    original = host.sample(size)
    expected_rng = s.np.random.get_state()
    s.np.random.set_state(before)
    candidate = resident.sample(size)
    same_rng(expected_rng, s.np.random.get_state())
    exact(original, candidate, label, out)
    return original, candidate


def correctness(out):
    assert s.jax.default_backend() == 'gpu'
    out.mkdir(parents=True, exist_ok=False)
    cfg = s.config()
    for name in ('host_fixture', 'resident_fixture'):
        (out / name).mkdir()
    host = s.dataset(out / 'host_fixture', cfg)
    resident = resident_dataset(out / 'resident_fixture', cfg)
    s.np.random.seed(0)
    example, _ = paired_sample(host, resident, 1, out, 'initial_example')
    baseline = s.GCIQLAgent.create(0, example['observations'], example['actions'], cfg)
    candidate = baseline
    updates = []
    s.phase(out, 'candidate_exact_two_updates')
    for index in range(2):
        a, b = paired_sample(host, resident, 1024, out, f'batch{index + 1}')
        s.check_batch(a)
        s.check_batch(b)
        exact(loss_and_gradient(baseline, a), loss_and_gradient(candidate, b),
              f'raw_loss_and_gradient{index + 1}', out)
        baseline, original_info = baseline.update(a)
        candidate, candidate_info = candidate.update(b)
        exact(original_info, candidate_info, f'update_info{index + 1}', out)
        exact(s.flax.serialization.to_state_dict(baseline),
              s.flax.serialization.to_state_dict(candidate), f'full_state{index + 1}', out)
        observed = s.metrics(candidate_info)
        assert observed['grad/norm'] > 0
        updates.append(observed)
    va, vb = paired_sample(host, resident, 1024, out, 'validation_batch')
    original_val = baseline.total_loss(va, grad_params=None)[1]
    candidate_val = candidate.total_loss(vb, grad_params=None)[1]
    exact(original_val, candidate_val, 'validation_metrics', out)
    assert int(candidate.network.step) == 3 and int(candidate.network.opt_state[0].count) == 2
    saved = s.fingerprint(candidate)
    assert saved == s.fingerprint(baseline)
    (out / 'checkpoint').mkdir()
    s.save_agent(candidate, str(out / 'checkpoint'), 2)
    s.np.savez(out / 'reload_batch.npz', **{key: s.np.asarray(value) for key, value in b.items()})
    sources = ('agents/gciql.py', 'utils/datasets.py', 'utils/flax_utils.py', 'main.py')
    s.write_report(out / 'report.json', dict(actual_exit_code=0, completed=time.time(),
        source_sha256={name: s.sha(s.IMPL / name) for name in sources},
        official_config=cfg.to_dict(), candidate_source_sha256=s.sha(Path(__file__)),
        resident_source_sha256=s.sha(Path(__file__).with_name('resident_inputs.py')),
        input_dtype='float32', current_dim=46, goal_dim=46, batch_size=1024,
        continuous_two_batches_all_keys_dtype_shape_bytes_exact=True,
        host_NumPy_RNG_after_each_sample_exact=True, initial_example_exact=True,
        raw_loss_and_gradients_exact=True, both_updated_full153_states_exact=True,
        validation=s.metrics(candidate_val), two_update_metrics=updates,
        saved_state=saved, network_step=3, Adam_count=2,
        resident_table_bytes=sum(table.nbytes for table in resident.dataset.gpu_tables.values()),
        fixture_only=True, artificial_terminal=True, full_data_throughput_proven=False,
        formal_training_started=False, SR_started=False, enabled_by_default=False))
    s.phase(out, 'candidate_exact_complete_waiting_fresh_restore')


def benchmark(out):
    import benchmark_profile
    benchmark_profile.worker(out=out, dataset_factory=resident_dataset)
    s.write_report(out / 'candidate_binding.json', dict(actual_exit_code=0,
        input_route='official NumPy sampler and FP32 immutable resident observation/action tables',
        resident_source_sha256=s.sha(Path(__file__).with_name('resident_inputs.py')),
        profile_source_sha256=s.sha(Path(benchmark_profile.__file__)),
        default_route_changed=False, all_results_fixture_only=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--bench', action='store_true')
    args = parser.parse_args()
    destination = Path(args.out).resolve()
    try:
        benchmark(destination) if args.bench else correctness(destination)
    except BaseException as exc:
        if destination.exists():
            with (destination / 'failure.json').open('x') as stream:
                json.dump(dict(actual_exit_code=1, error=repr(exc), time=time.time(),
                    candidate_stopped=True, tolerance_relaxed=False), stream, indent=2)
        raise

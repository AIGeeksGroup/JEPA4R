"""State-only priority: real widths, original ordered prefetch, sparse host sync.

GCIQL.update is unchanged. Only host metric reads move from every update to
the existing 100-step status boundary. Diagnostic gates explicitly inspect all
their metrics; formal runs do not claim an every-update finite trace.
"""
import argparse
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import trainer_runtime as runtime
from state_inputs import StateInputs

ORIGINAL_PREPARE = runtime.prepare
ORIGINAL_UPDATE = runtime.core.update
ORIGINAL_SAVE = runtime.core.save_boundary


def install(input_mode='prefetch1', host_metrics='async100'):
    if input_mode not in ('baseline','prefetch1') or host_metrics not in ('sync_reference','async100'):
        raise ValueError('Only the frozen baseline/prefetch1 and explicit host-metric modes')

    def make_input(dataset, cfg, *, cubes, mode, features, workers):
        assert mode == 'state' and features is None and workers == 0
        split_mode = input_mode if not getattr(make_input, 'made_train', False) else 'baseline'
        make_input.made_train = True
        return StateInputs(dataset, cfg, cubes=cubes, mode=split_mode,
            ordered_input_path=runtime.ROOT/'official_state_workers_v1/ordered_input.py')

    def prepare(config, shared_datasets=None):
        assert config['modality'] == 'state' and config['save_step1'] is True
        make_input.made_train = False
        samplers, resources, binding = ORIGINAL_PREPARE(config, shared_datasets)
        resources.actual_update_count = resources.train_batches = 0
        def barrier():
            for sampler in samplers.values():
                sampler.barrier()
            assert not any(sampler.pending for sampler in samplers.values())
        resources.barrier = barrier
        binding['runtime'].update(input_mode=input_mode, host_metrics=host_metrics,
            formal_metric_finite_scope='every100 recorded updates and each5000 validation',
            no_every_update_finite_trace_claim=True, ordered_prefetch_barrier_before_checkpoints=True)
        binding['code'].update({str(p):runtime.utility.sha(p) for p in
            [Path(__file__), HERE/'state_inputs.py', runtime.ROOT/'official_state_workers_v1/ordered_input.py']})
        return samplers, resources, binding

    def get_batch(samplers, resources, split, config):
        started = time.perf_counter()
        if split == 'train':
            following = resources.train_batches+1
            boundary = (following == 1 or following % config['validation_interval'] == 0 or
                following % config['save_interval'] == 0 or following == config['total_updates'])
            batch = samplers[split].sample(config['batch_size'], prefetch_next=not boundary)
            resources.train_batches = following
        else:
            resources.barrier()
            batch = samplers[split].sample(config['batch_size'], prefetch_next=False)
        resources.last_batch_timing = dict(input_seconds=time.perf_counter()-started)
        return batch

    def update(agent, batch, resources):
        if host_metrics == 'sync_reference':
            return ORIGINAL_UPDATE(agent,batch,resources)
        resources.guard()
        started = time.perf_counter()
        agent, info = agent.update(batch)  # Original, unmodified GCIQL computation.
        dispatched = time.perf_counter()
        resources.actual_update_count += 1
        # Monitoring is host-only: no extra optimizer, RNG or numeric operation.
        if resources.actual_update_count % 100 == 0:
            info = {key:float(value) for key,value in runtime.jax.device_get(info).items()}
            assert all(runtime.np.isfinite(value) for value in info.values())
            assert info['grad/norm'] > 0
        resources.last_update_timing = dict(jax_dispatch_seconds=dispatched-started,
            loss_sync_seconds=time.perf_counter()-dispatched)
        return agent, info

    def save(agent, resources, binding, output, counter, validation_count, budget_total):
        resources.barrier()
        return ORIGINAL_SAVE(agent,resources,binding,output,counter,validation_count,budget_total)

    runtime.DistanceInputs = make_input
    runtime.prepare = prepare
    runtime.get_batch = get_batch
    runtime.core.update = update
    runtime.core.save_boundary = save
    return runtime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--input-mode', choices=('baseline','prefetch1'), default='prefetch1')
    args = parser.parse_args()
    raise SystemExit(install(args.input_mode).run(args.config))

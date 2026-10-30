"""Bounded state-policy equivalence/throughput and checkpoint boundary checks."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time

import state_runtime


def rng_fingerprint(runtime):
    state = runtime.np.random.get_state()
    return dict(generator=state[0], keys=runtime.core.digest_array(state[1]), tail=list(state[2:]))


def make_context(config, mode, metrics, shared):
    runtime = state_runtime.install(mode, metrics)
    samplers, resources, binding = runtime.prepare(config, shared)
    agent, _ = runtime.initial_agent(config, samplers)
    return runtime, samplers, resources, binding, agent


def numeric_case(config, mode, metrics, shared):
    runtime, samplers, resources, binding, agent = make_context(config,mode,metrics,shared)
    hashes = []
    try:
        for _ in range(32):
            batch = runtime.get_batch(samplers,resources,'train',config)
            digest = hashlib.sha256()
            for key,value in sorted({**batch, **samplers['train'].last_ids}.items()):
                digest.update(key.encode());digest.update(runtime.core.digest_array(value).encode())
            hashes.append(digest.hexdigest())
            agent,info = runtime.core.update(agent,batch,resources)
            values = {key:float(value) for key,value in info.items()}
            assert all(runtime.np.isfinite(value) for value in values.values()) and values['grad/norm']>0
        resources.barrier()
        return dict(batch_and_ID_SHA=hashes, full_state=runtime.fingerprint(agent), hostNP=rng_fingerprint(runtime))
    finally:
        resources.close()


def timed_case(config,mode,shared):
    runtime,samplers,resources,binding,agent = make_context(config,mode,'async100',shared)
    try:
        for _ in range(64):
            agent,_ = runtime.core.update(agent,runtime.get_batch(samplers,resources,'train',config),resources)
        runtime.jax.block_until_ready(agent)
        started = time.perf_counter()
        for _ in range(1024):
            agent,_ = runtime.core.update(agent,runtime.get_batch(samplers,resources,'train',config),resources)
        resources.barrier()
        runtime.jax.block_until_ready(agent)
        elapsed = time.perf_counter()-started
        assert int(agent.network.step)==1089 and int(agent.network.opt_state[0].count)==1088
        return dict(mode=mode,warmup_updates=64,timed_updates=1024,seconds=elapsed,
            updates_per_second=1024/elapsed, host_metrics='same async100 as formal',
            includes_sample_reward_projection_update=True, excludes_periodic_val_checkpoint_initialization=True)
    finally:
        resources.close()


def run_gate(config,output):
    assert not output.exists()
    output.parent.mkdir(parents=True,exist_ok=True)
    runtime = state_runtime.install('baseline','sync_reference')
    samplers, resources, _ = runtime.prepare(config)
    shared = {split:sampler.dataset for split,sampler in samplers.items()}
    resources.close()
    reference = numeric_case(config,'baseline','sync_reference',shared)
    candidate = numeric_case(config,'prefetch1','async100',shared)
    runtime.utility.atomic_json(output.parent/'numerical_comparison.json',dict(
        actual_exit_code=0 if reference==candidate else 1, reference=reference,candidate=candidate,
        scope='32 realB1024 updates, all inputs/IDs, full153 and committed hostNP; no tolerance relaxation'))
    assert reference == candidate, 'Real32-update input/RNG/full153 equivalence failed; preserve diagnostics'
    timings = [timed_case(config,mode,shared) for mode in ('baseline','prefetch1','baseline')]
    controls = [timings[0]['updates_per_second'],timings[2]['updates_per_second']]
    faster = max(controls)
    drift = (max(controls)-min(controls))/faster
    gain = timings[1]['updates_per_second']/faster-1
    selected = 'prefetch1' if drift<=.05 and gain>=.03 else 'baseline'
    selected_hz = timings[1]['updates_per_second'] if selected=='prefetch1' else min(controls)
    selection = dict(actual_exit_code=0,group_id=config['group_id'],role=config['role'],
        numerical_reference='original per-step sync and original GCDataset',
        reference=reference,candidate_exact=True,timings=timings,control_drift=drift,
        prefetch_gain_vs_faster_control=gain, engineering_rule='prefetch if >=3% and controls drift<=5%; otherwise async baseline',
        selected_input_mode=selected,selected_steady_updates_per_second=selected_hz,
        conditional_1M_train_seconds=1000000/selected_hz,
        target_1M_seconds=3600, validation_save_startup_extra=True,
        diagnostic_updates_discarded=64+3*1088+2,formal_training_started=False)
    runtime.utility.atomic_json(output.parent/'throughput_selection.json',selection)
    runtime = state_runtime.install(selected,'async100')
    boundary = importlib.import_module('gciql_boundary')
    # Reuse the same readonly real data for the small configured-batch gate.
    boundary.prepare = lambda cfg:runtime.prepare(cfg,shared)
    boundary.get_batch = runtime.get_batch
    boundary.gate(config,output)
    runtime.utility.atomic_json(output/'selection.json',selection)


def accept_priority(config,output,report):
    runtime = state_runtime.install('baseline','async100')
    boundary = importlib.import_module('gciql_boundary')
    boundary.accept(config,output,report)
    result = json.loads(report.read_text())
    assert len(result['checkpoint_transactions'])==11
    result.update(checkpoint_count=11, includes_step1=True, group_id=config['group_id'],
        finite_training_scope='recorded100-step status and200 train/val CSV points, not every update',
        input_mode=json.loads((output/'resolved.json').read_text())['binding']['runtime']['input_mode'])
    runtime.utility.atomic_json(report,result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('gate','reload','accept'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--input-mode',choices=('baseline','prefetch1'))
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True)
    if args.action=='gate':
        run_gate(config,args.output)
    elif args.action=='reload':
        selection=json.loads((args.output/'selection.json').read_text())
        assert selection['actual_exit_code']==0
        runtime=state_runtime.install(selection['selected_input_mode'],'async100')
        boundary=importlib.import_module('gciql_boundary')
        boundary.reload_gate(config,args.output,args.report)
    else:
        accept_priority(config,args.output,args.report)

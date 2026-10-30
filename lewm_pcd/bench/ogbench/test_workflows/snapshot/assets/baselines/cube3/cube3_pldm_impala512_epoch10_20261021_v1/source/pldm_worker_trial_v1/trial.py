"""Isolated W20/24/20 trial. Defaults to prepare; never resumes formal training."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

BASE_FREEZE = 'cdfd14b894fda7bf7f068c489bc550e819ade12ca16f36250ebf0dda2cd374d7'
MANIFEST = '98dde8e9a49bfdd3c18591939cab9208bd9d3a2af7ae095b47e020cd02ba9c29'
PARENT_UPDATES, TOTAL_UPDATES = 45703, 457030
WARM, WINDOW, TIMED = 64, 256, 768
CASES = [('A1_native20', 20), ('B_native24', 24), ('A2_native20', 20)]
HERE = Path(__file__).resolve().parent


def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    p = Path(p); tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(p)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def resources(workers):
    if workers not in (20, 24): raise ValueError('only approved train W20/24')
    return dict(train_workers=workers, validation_workers=20, batch_size=128,
                prefetch_factor=2, pin_memory=True, persistent_workers=False)


def validate_parent(proof):
    expected = dict(stage='pldm_epoch1_verified', actual_exit_code=0,
        global_step=PARENT_UPDATES, Adam_steps=[float(PARENT_UPDATES)],
        native_schedule_index=PARENT_UPDATES-1, model_bare_exact=True,
        model_optimizer_finite=True)
    if any(proof.get(k) != v for k, v in expected.items()) or proof.get('epoch') not in (0, 1):
        raise ValueError('complete protected PLDM epoch1 parent required')
    binding = proof['baseline_binding']
    if (binding.get('method'), binding.get('manifest_sha256'), binding.get('source_freeze_sha256')) != ('pldm', MANIFEST, BASE_FREEZE):
        raise ValueError('parent method/data/source differs')
    return proof


def compare_cohort(rows):
    keys = ('ids', 'first2_inputs', 'RNG_before', 'RNG_after', 'start_global_step',
            'end_global_step', 'native_schedule_index')
    for row in rows:
        if (row['start_global_step'], row['end_global_step'], row['native_schedule_index']) != (PARENT_UPDATES, PARENT_UPDATES+WARM+TIMED, PARENT_UPDATES+WARM+TIMED-1):
            raise ValueError('short trial changed update/schedule budget')
        if any(row[k] != rows[0][k] for k in keys): raise ValueError('consumed cohort/input/RNG differs')
    return True


def recommend_workers(rates):
    A1, B, A2 = rates
    drift = max(A1, A2)/min(A1, A2)-1
    gain = B/max(A1, A2)-1
    return dict(controls_drift=drift, candidate_gain_over_faster_control=gain,
                recommended_train_workers=24 if drift<=.05 and gain>=.05 else 20)


def source_guard(root):
    frozen = read(HERE/'source_freeze_v1.json')
    if str(root) != frozen['root']: raise ValueError('different base root')
    for name, h in frozen['files'].items():
        if sha(HERE/name) != h: raise ValueError('trial source differs: '+name)
    for name, h in frozen['base_files'].items():
        if sha(root/name) != h: raise ValueError('original base differs: '+name)
    sys.path.insert(0, str(root))
    import shared_data
    data_root = Path(frozen['data_root'])
    shared_data.require_sources(root, data_root)
    return data_root


def require_runtime(a):
    data_root = source_guard(a.root); release = read(a.release)
    out = a.output.parent if a.mode == 'case' else a.output
    if (release.get('stage'), release.get('source_freeze_sha256'), release.get('output'),
        release.get('parent_proof'), release.get('parent_proof_sha256')) != (
        'approved_pldm_worker_trial', sha(HERE/'source_freeze_v1.json'), str(out),
        str(a.parent_proof), sha(a.parent_proof)):
        raise ValueError('matching Root runtime release required')
    proof = validate_parent(read(a.parent_proof))
    for record in proof['files'].values():
        s = Path(record['path']).stat()
        if (s.st_size, s.st_mtime_ns) != (record['bytes'], record['mtime_ns']):
            raise ValueError('protected parent changed')
    stop = Path(release['retirement_receipt'])
    if sha(stop) != release['retirement_receipt_sha256'] or read(stop).get('actual_exit_code') != 0:
        raise ValueError('actual owned retirement receipt0 required')
    if Path('/proc/120578').exists(): raise ValueError('original trainer still present')
    original=a.root/'outputs/baseline_chain_v1'
    for path in (original/'formal/pldm/actual_exit_code',original/'actual_exit_code',Path(str(original)+'.outer_exit_code')):
        if not path.exists(): raise ValueError('original process exits not yet recorded')
    if Path('/proc/115392').exists() or (original/'gate/lejepa').exists():
        raise ValueError('original chain not closed before new trial')
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if gpu.stdout.strip(): raise ValueError('GPU not empty; no concurrent trial')
    if int(os.environ.get('SLURM_RESTART_COUNT', '0')) != 0: raise ValueError('fresh isolated ownrun required')
    return data_root, proof


def fingerprint(value):
    """Exact values/layout without consuming RNG. Used outside timed windows."""
    h = hashlib.sha256()
    def visit(x):
        if hasattr(x, 'detach'):
            h.update(str((str(x.dtype), tuple(x.shape), tuple(x.stride()))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str): h.update(repr(k).encode()); visit(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for y in x: visit(y)
        elif hasattr(x, 'tobytes'):
            h.update(str((str(x.dtype), x.shape)).encode()); h.update(x.tobytes())
        else: h.update(repr(x).encode())
    visit(value); return h.hexdigest()


def run_case(a):
    data_root, proof = require_runtime(a)
    if (a.output.name, a.workers) not in CASES: raise ValueError('case/worker binding differs')
    os.environ['SPT_CACHE_DIR'] = str(a.output/'_spt_cache')
    os.environ['STABLEWM_HOME'] = str(a.root/'outputs')
    import torch
    import lightning as pl
    import stable_pretraining as spt
    import inspect
    import random
    import numpy as np
    entry = load('pldm_frozen_train', a.root/'train.py')
    state = dict(stage='resuming_trial_parent', actual_exit_code=None, workers=resources(a.workers),
                 parent_proof_sha256=sha(a.parent_proof), argv=sys.argv, ids=[], first2_inputs=[])
    result = a.output/'result'; result.mkdir(exist_ok=False)
    def save(): write(result/'trial_result.json', state)
    native_loaders, native_binding, native_options, native_manager = entry.loaders, entry.binding, entry.trainer_options, spt.Manager
    if sha(inspect.getfile(native_manager)) != read(HERE/'source_freeze_v1.json')['runtime_manager_sha256']:
        raise ValueError('actual SPT Manager source differs from accepted runtime')
    queued = {}; manager_ref = {}; loader_ref = {}
    def rng(): return fingerprint((random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all(), loader_ref['train'].generator.get_state()))
    def loaders(*args, **kwargs):
        train, val, stats = native_loaders(*args, **kwargs)
        if stats != proof['baseline_binding']['action_statistics']:
            raise ValueError('parent train-only action statistics differ')
        train.num_workers = a.workers
        if val.num_workers != 20 or train.prefetch_factor != 2 or not train.pin_memory:
            raise ValueError('loader resource contract differs')
        base = type(train)
        class TraceLoader(base):
            def __iter__(self):
                for index, batch in enumerate(super().__iter__()):
                    # IDs are CPU/pinned here. No GPU .cpu()/tolist() in the timed loop.
                    if batch['sample_id'].device.type!='cpu': raise ValueError('IDs no longer CPU')
                    row = dict(ids=batch['sample_id'].tolist())
                    if index < 2: row['inputs'] = fingerprint({k:batch[k] for k in ('pixels','actions')})
                    queued[index] = row; batch['_trial_trace_index'] = index
                    yield batch
        train.__class__ = TraceLoader; loader_ref['train'] = train
        return train, val, stats
    def binding(*args):
        return {**native_binding(*args), 'train_workers':a.workers,
                'worker_trial_source_freeze_sha256':sha(HERE/'source_freeze_v1.json'),
                'parent_full_sha256':proof['files']['full']['sha256'],
                'diagnostic_updates_discarded':True}
    def options(phase, callbacks):
        out = native_options(phase, callbacks); out['limit_val_batches'] = 2
        # The complete457030-update native schedule and max_epochs10 stay unchanged.
        return out
    class Probe(pl.pytorch.callbacks.Callback):
        def on_train_start(self, trainer, module):
            own = Path(manager_ref['manager']._run_dir).resolve()
            if own == Path(proof['own_run']).resolve() or not own.is_relative_to(a.output):
                raise ValueError('trial would write original ownrun')
            for cb in trainer.callbacks:
                if isinstance(cb,pl.pytorch.callbacks.ModelCheckpoint) and not Path(cb.dirpath).resolve().is_relative_to(own):
                    raise ValueError('checkpoint callback escaped new ownrun')
            if trainer.global_step != PARENT_UPDATES or trainer.current_epoch != 1:
                raise ValueError('native epoch/global restore differs')
            steps = [float(x['step']) for opt in trainer.optimizers for x in opt.state.values() if 'step' in x]
            if not steps or set(steps) != {float(PARENT_UPDATES)}: raise ValueError('Adam restore differs')
            evidence=[cb for cb in trainer.callbacks if hasattr(cb,'last_schedule_index')]
            if len(evidence)!=1 or evidence[0].last_schedule_index!=PARENT_UPDATES-1:
                raise ValueError('native Evidence callback state restore differs')
            if fingerprint(module.model.state_dict()) != proof['model_fingerprint'] or fingerprint(trainer.optimizers[0].state_dict()) != proof['optimizer_fingerprint']:
                raise ValueError('full model/optimizer restore differs')
            state.update(own_run=str(own), start_global_step=int(trainer.global_step),
                         epoch_index=int(trainer.current_epoch), RNG_before=rng(), timing_windows=[])
            self.done=0; self.start=None; save()
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            index = batch['_trial_trace_index']
            if not isinstance(index, int): raise ValueError('CPU trace index was changed')
            row = queued.pop(index); state['ids'].append(row['ids'])
            if self.done < 2: state['first2_inputs'].append(row['inputs'])
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            self.done += 1
            if trainer.global_step != PARENT_UPDATES+self.done: raise ValueError('unexpected global update')
            if self.done == WARM or self.done in (WARM+WINDOW,WARM+2*WINDOW,WARM+TIMED):
                torch.cuda.synchronize(); stamp = time.perf_counter()
                if self.done == WARM: self.start=stamp
                else:
                    state['timing_windows'].append(dict(updates=WINDOW, seconds=stamp-self.start,
                        updates_per_second=WINDOW/(stamp-self.start)))
                    self.start=stamp
            if self.done == WARM+TIMED:
                state.update(end_global_step=int(trainer.global_step), RNG_after=rng())
                trainer.should_stop=True; save()
        def on_train_end(self, trainer, module):
            evidence=[cb for cb in trainer.callbacks if hasattr(cb,'last_schedule_index')]
            if len(evidence)!=1: raise ValueError('native before-step evidence missing')
            state['native_schedule_index']=evidence[0].last_schedule_index
    def manager(**kwargs):
        kwargs.update(ckpt_path=proof['files']['full']['path'], weights_only=False)
        kwargs['trainer'].callbacks.append(Probe())
        value=native_manager(**kwargs); manager_ref['manager']=value; return value
    entry.loaders, entry.binding, entry.trainer_options, spt.Manager = loaders, binding, options, manager
    try:
        args=SimpleNamespace(root=a.root, data_root=data_root, output=result, method='pldm',
                             phase='formal', run_tag=a.output.parent.name+'_'+a.output.name)
        save(); entry.run(args)
        if len(state['ids']) != WARM+TIMED or len(state['first2_inputs']) != 2: raise ValueError('consumed cohort incomplete')
        compare_cohort([state])
        summary=read(Path(state['own_run'])/'summary.json')
        if any(not math.isfinite(z[k]) for name,z in summary['metrics'].items() if 'loss' in name for k in ('min','max','last')):
            raise ValueError('recorded native loss nonfinite')
        seconds=sum(x['seconds'] for x in state['timing_windows'])
        state.update(stage='worker_trial_case_complete', actual_exit_code=0,
                     measured_updates=TIMED, timed_seconds=seconds, updates_per_second=TIMED/seconds,
                     native_summary=summary, diagnostic_weights_not_for_formal=True)
    except BaseException:
        state.update(stage='worker_trial_case_failed', actual_exit_code=1, error=traceback.format_exc())
    finally:
        spt.Manager=native_manager; save()
    return state['actual_exit_code']


def run_aba(a):
    data_root, proof = require_runtime(a)
    a.output.mkdir(parents=True, exist_ok=False)
    old=load('worker_trial_processes',data_root/'training_serial_compact_relay_v1/serial_relay.py')
    legacy=old.load_frozen(data_root/'training_relay_v1/relay.py',old.LEGACY_RELAY_SHA,'worker_trial_resources')
    state=dict(stage='worker_ABA_running', started_at=time.time(), exit_code=None, cases={})
    args=SimpleNamespace(root=data_root, code_root=data_root/'code_front600_v1/lewm_pcd', output=a.output)
    try:
        for label, workers in CASES:
            old.gpu_empty(); directory=a.output/label
            cmd=[sys.executable,'-B',str(HERE/'trial.py'),'--mode','case','--root',str(a.root),
                 '--output',str(directory),'--workers',str(workers),'--parent-proof',str(a.parent_proof),
                 '--release',str(a.release)]
            old.run_owned(cmd,directory,args,state,legacy,gate=True)
            row=read(directory/'result/trial_result.json')
            if row['actual_exit_code']!=0: raise ValueError('own case failed')
            state['cases'][label]=row; write(a.output/'status.json',state)
        rows=[state['cases'][name] for name,_ in CASES]; compare_cohort(rows)
        state.update(stage='worker_ABA_complete_awaiting_Root_formal_release',exit_code=0,
            **recommend_workers([x['updates_per_second'] for x in rows]),
            all_diagnostic_updates_discarded=True, formal_started=False,
            formal_resume_policy='Protected epoch1 + remaining9epochs; new seed0 loader, not exact original sample/RNG continuation',
            next_method_policy='No automatic old LeJEPA CPU launch; separate isolated uint8 source/gate/release')
    except BaseException: state.update(stage='worker_ABA_failed_stopped',exit_code=1,error=traceback.format_exc())
    state['finished_at']=time.time();write(a.output/'status.json',state)
    (a.output/'actual_exit_code').write_text(str(state['exit_code'])+'\n');return state['exit_code']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','aba','case'),default='prepare')
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent-proof',type=Path);parser.add_argument('--release',type=Path)
    parser.add_argument('--workers',type=int,choices=(20,24),default=20)
    a=parser.parse_args();a.root=a.root.resolve();a.output=a.output.resolve()
    if a.mode=='prepare':
        source_guard(a.root)
        print(json.dumps(dict(stage='prepared_only_no_workload',cases=CASES,warm=WARM,timed=TIMED,
              requires=['protected_parent_CPU0','owned_retirement0','Root_runtime_release','GPUempty'],
              full_training_started=False)));return 0
    if not a.parent_proof or not a.release: raise ValueError('explicit parent and Root release required')
    a.parent_proof=a.parent_proof.resolve();a.release=a.release.resolve()
    return run_case(a) if a.mode=='case' else run_aba(a)


if __name__=='__main__':raise SystemExit(main())

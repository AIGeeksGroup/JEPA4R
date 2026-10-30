"""Preserve and CPU-verify first full epoch. Never sends signals or launches trials."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from trial import BASE_FREEZE, PARENT_UPDATES, read, sha, write, source_guard, fingerprint


def stamp(path):
    s=path.stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def file_state(path):
    if not path.is_file() or path.stat().st_size==0:return None
    if list(path.parent.glob('.'+path.name+'.*.tmp')) or list(path.parent.glob(path.name+'.tmp*')):return None
    return dict(path=str(path),**stamp(path))


def ready(summary, files, previous, entered_next_epoch):
    metrics=summary.get('metrics',{})
    return bool(all(files) and files==previous and entered_next_epoch and summary.get('epoch',-1)>=1
        and metrics.get('validate/loss_step',{}).get('count',0)>=4571
        and metrics.get('validate/loss_epoch',{}).get('count',0)>=1)


def identity(root):
    p=Path('/proc/120578');z=(p/'stat').read_text().split(') ',1)[1].split()
    argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1]
    F=root/'outputs/baseline_chain_v1'
    if (int(z[2]),int(z[19]),argv)!=(120578,1772603084,read(F/'formal/pldm/actual_argv.json')):
        raise ValueError('original PLDM trainer identity differs')
    if read(F/'status.json').get('active_owned_pid')!=120578:raise ValueError('original phase changed')
    return dict(pid=120578,pgrp=int(z[2]),birth=int(z[19]),state=z[0],argv=argv)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--release',type=Path);p.add_argument('--mode',choices=('prepare','watch'),default='prepare')
    a=p.parse_args();a.root=a.root.resolve();a.output=a.output.resolve();source_guard(a.root)
    if a.mode=='prepare':print(json.dumps(dict(stage='prepared_only_no_signal_or_model_read',full_updates=PARENT_UPDATES,validation_batches=4571)));return 0
    release=read(a.release)
    if (release.get('stage'),release.get('output'),release.get('source_freeze_sha256'))!=('approved_pldm_epoch1_preservation',str(a.output),sha(Path(__file__).with_name('source_freeze_v1.json'))):
        raise ValueError('explicit Root boundary release required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('watch/verify must hide CUDA')
    a.output.mkdir(parents=True,exist_ok=False)
    F=a.root/'outputs/baseline_chain_v1';D=F/'formal/pldm'
    RUN=a.root/'outputs/checkpoints/_spt_cache/runs/20261017/231033/9222fffa9218'
    BARE=a.root/'outputs/checkpoints/triple600_pldm_formal_seed0_baseline_chain_v1/weights_epoch_1.pt'
    FULL=RUN/'checkpoints/last.ckpt'
    state=dict(stage='waiting_original_full_epoch1_save',started_at=time.time(),actual_exit_code=None,signals_sent=[])
    try:
        previous=None
        while True:
            state['identity']=identity(a.root);summary=read(RUN/'summary.json')
            files=[file_state(FULL),file_state(BARE)]
            log=(D/'stdout.log').read_text(errors='replace')
            entered=bool(re.search(r'\[Epoch\s+1/10\]\s+step\s+',log))
            state.update(updated_at=time.time(),summary=summary,files=files,entered_second_epoch=entered)
            write(a.output/'status.json',state)
            if ready(summary,files,previous,entered):break
            previous=files;time.sleep(5)
        target=a.output/'preserved_epoch1';target.mkdir()
        raw={};preserved={}
        for name,source in [('full',FULL),('bare',BARE),('protocol',D/'result/actual_protocol.json')]:
            before=stamp(source);payload=source.read_bytes()
            if before!=stamp(source):raise ValueError('parent changed during copy')
            destination=target/('last_epoch1.ckpt' if name=='full' else 'weights_epoch_1.pt' if name=='bare' else 'actual_protocol.json')
            with destination.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
            if sha(destination)!=hashlib.sha256(payload).hexdigest():raise ValueError('preserved copy differs')
            preserved[name]=dict(path=str(destination),source=str(source),sha256=sha(destination),source_stat=before,**stamp(destination))
            raw[name]=payload
        import torch
        torch.set_num_threads(1)
        full=torch.load(io.BytesIO(raw['full']),map_location='cpu',weights_only=False)
        bare=torch.load(io.BytesIO(raw['bare']),map_location='cpu',weights_only=True)
        model={k[6:]:v for k,v in full['state_dict'].items() if k.startswith('model.')}
        binding=read(target/'actual_protocol.json')['binding']
        steps=[float(v['step']) for opt in full['optimizer_states'] for v in opt['state'].values() if 'step' in v]
        def finite(x):
            if isinstance(x,torch.Tensor):return not x.is_floating_point() or bool(torch.isfinite(x).all())
            if isinstance(x,dict):return all(finite(v) for v in x.values())
            if isinstance(x,(tuple,list)):return all(finite(v) for v in x)
            return True
        if full['global_step']!=PARENT_UPDATES or full['epoch'] not in (0,1) or set(steps)!={float(PARENT_UPDATES)} or not steps:
            raise ValueError('parent global/epoch/Adam counters differ')
        if full['spt_run_dir']!=str(RUN) or full['baseline_binding']!=binding or binding['source_freeze_sha256']!=BASE_FREEZE:
            raise ValueError('parent ownrun/data/model/source binding differs')
        if full['native_method_schedule']['last_before_optimizer_step']!=PARENT_UPDATES-1:
            raise ValueError('native before-opt schedule index differs')
        if full['loops']['fit_loop']['epoch_loop.val_loop.batch_progress']['total']['completed']!=4571:
            raise ValueError('parent full validation incomplete')
        if model.keys()!=bare.keys() or any(not torch.equal(model[k],bare[k]) for k in model):raise ValueError('full/bare differ')
        if not any(k.startswith('idm.') for k in model):raise ValueError('IDM state missing')
        if not finite(full['state_dict']) or not finite(full['optimizer_states']):raise ValueError('parent nonfinite')
        if torch.cuda.is_initialized():raise ValueError('CPU preservation initialized CUDA')
        proof=dict(stage='pldm_epoch1_verified',actual_exit_code=0,files=preserved,
            own_run=str(RUN),global_step=full['global_step'],epoch=full['epoch'],Adam_steps=sorted(set(steps)),
            native_schedule_index=PARENT_UPDATES-1,model_bare_exact=True,model_optimizer_finite=True,
            model_fingerprint=fingerprint(model),optimizer_fingerprint=fingerprint(full['optimizer_states'][0]),
            baseline_binding=binding,pause_or_termination_performed=False,cuda_initialized=False,
            bare_save_semantics='Original torch.save direct write; complete next-epoch boundary, stable stat and full/bare exact CPU verification required',
            full_save_semantics='Original SPT temp+fsync+atomic replace; original bytes preserved')
        write(target/'status.json',proof);(target/'actual_exit_code').write_text('0\n')
        state.update(stage='protected_parent_CPU0_awaiting_Root_owned_termination',actual_exit_code=0,
                     parent_proof_path=str(target/'status.json'),parent_proof_sha256=sha(target/'status.json'),
                     identity_after_copy=identity(a.root))
    except BaseException:state.update(stage='epoch1_preservation_failed',actual_exit_code=1,error=traceback.format_exc())
    state['finished_at']=time.time();write(a.output/'status.json',state)
    (a.output/'actual_exit_code').write_text(str(state['actual_exit_code'])+'\n');return state['actual_exit_code']


if __name__=='__main__':raise SystemExit(main())

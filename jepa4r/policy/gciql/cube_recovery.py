# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/cube_recovery.py (verbatim except: imports / paths adapted to the package)
# ruff: noqa: F401  (moved verbatim: unused imports)
"""Opt-in Cube T4 executor around original loops and the shared sealed loader."""
from contextlib import contextmanager,redirect_stdout,redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import argparse,copy,hashlib,json,os,sys,traceback

SHARED_LOADER_SHA256='01b81d49014a1f9e7c27f245a5e9a48153f9fcfccc892814b0ad0c3302200d23'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def recovery_identity(plan):
    from .fresh_data import load_binding
    from .fresh_train import _source_binding
    data=load_binding(plan['data_manifest'])
    if(plan['mode']!='smoke'or plan['method']not in('lewm_pcd','lewm_rgb','pldm')or plan['workers']!=0):
        raise ValueError('Cube recovery is explicit workers0 bounded smoke only')
    if sha(plan['data_manifest'])!=plan['data_manifest_sha256']:
        raise ValueError('Cube recovery data manifest changed after planning')
    if plan['method'] in ('lewm_pcd','lewm_rgb'):
        trainer=plan['config']['trainer']
        if (trainer.get('max_epochs'),trainer.get('limit_train_batches'),trainer.get('limit_val_batches'))!=(1,2,1):
            raise ValueError('Cube recovery requires the original two-update smoke budget')
        if plan['config']['loader'].get('num_workers')!=0:
            raise ValueError('Cube recovery requires an unprefetched workers0 loader')
    config=copy.deepcopy(plan.get('config',plan['recipe']))
    return dict(model=dict(method=plan['method'],cube_count=plan['cube_count'],
                    recovery_executor_sha256=sha(__file__),shared_loader_sha256=SHARED_LOADER_SHA256,
                    source=_source_binding(plan['method'],plan['cube_count'])),
                data=dict(manifest_sha256=plan['data_manifest_sha256'],original_producer=data['producer_binding']),
                config=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                split='episode_heldout_v1',action_dim=5,training_seed=0)

@contextmanager
def training_context(identity,directory,*,pause_after_batches=None,resume_checkpoint=None,aliases=()):
    import torch,stable_pretraining as spt
    from jepa4r.policy.bc.relocate import recovery
    from jepa4r.policy.bc.relocate.wm_recovery import validate_resume
    if sha(recovery.__file__)!=SHARED_LOADER_SHA256:
        raise ValueError('shared loader/callback source changed; freeze/review the new contract')
    original_loader,original_manager=torch.utils.data.DataLoader,spt.Manager
    wrapped_loader=recovery.stateful_loader_class(original_loader)
    changed=[]
    def manager(*args,**kwargs):
        trainer=kwargs.get('trainer')
        if trainer is None or kwargs.get('seed')!=0:
            raise ValueError('Cube guide retains the original named Manager and explicit seed0')
        callback=recovery.make_lightning_recovery_callback(identity,directory,pause_after_batches=pause_after_batches)
        trainer.callbacks.append(callback)
        if resume_checkpoint is not None:
            kwargs['ckpt_path']=str(validate_resume(resume_checkpoint,identity))
        elif kwargs.get('ckpt_path') is not None:
            raise ValueError('fresh Cube recovery cannot silently reuse a pre-existing checkpoint')
        kwargs['weights_only']=False
        return original_manager(*args,**kwargs)
    torch.utils.data.DataLoader=wrapped_loader;spt.Manager=manager
    for module in aliases:
        if getattr(module,'DataLoader',None)is original_loader:
            changed.append(module);module.DataLoader=wrapped_loader
    try:yield
    finally:
        for module in changed:module.DataLoader=original_loader
        torch.utils.data.DataLoader=original_loader;spt.Manager=original_manager

def execute(plan,*,pause_after_batches=None,resume_checkpoint=None):
    from . import fresh_train,fresh_data
    from .training_runtime import build_environment,pinned_native
    identity=recovery_identity(plan)
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    fresh_data.save(output/'plan.json',plan)
    fresh_data.save(output/'RECOVERY_IDENTITY.json',dict(identity=identity,startup_seed=0,
        scope='original Cube model/forward/optimizer/scheduler; shared original-loader recovery'))
    state=dict(stage='starting',actual_exit_code=None,completed_count=0,failed_count=0,identity=identity,
               started_at=datetime.now(timezone.utc).isoformat())
    fresh_data.save(output/'status.json',state);previous=dict(os.environ);code=1
    with(output/'stdout_stderr.log').open('x',encoding='utf8',buffering=1)as log,redirect_stdout(log),redirect_stderr(log):
        try:
            os.environ.update(build_environment(Path(plan['data_manifest']).parent,output/'native'))
            from jepa4r.policy.bc.relocate.recovery import RecoveryPause
            try:
                if plan['method']in('lewm_pcd','lewm_rgb'):
                    from omegaconf import OmegaConf
                    with pinned_native()as native, fresh_train.route_loader(native,plan), fresh_train._record_loaders(native,output), \
                            training_context(identity,output/'recovery',pause_after_batches=pause_after_batches,resume_checkpoint=resume_checkpoint,aliases=(native,)):
                        native.run(OmegaConf.create(plan['config']))
                else:
                    from .pldm import train,shared_data
                    native_output=output/'native';native_output.mkdir()
                    args=SimpleNamespace(method='pldm',root=Path(train.__file__).parent,profile='cube'+str(plan['cube_count']),
                        data_binding=Path(plan['data_manifest']),data_root=Path(plan['data_manifest']).parent,
                        phase='gate',output=native_output,run_tag='guide',workers=0,prefetch=plan['prefetch']or 1)
                    train.require_sources(args.root,args.data_binding)
                    with training_context(identity,output/'recovery',pause_after_batches=pause_after_batches,
                            resume_checkpoint=resume_checkpoint,aliases=(train,shared_data)):
                        train.run(args)  # Original implementation; do not copy its training loop.
                final=output/'recovery/final.ckpt'
                if not final.is_file():raise ValueError('original fit did not seal the complete training state')
                import torch
                checkpoint=torch.load(final,map_location='cpu',weights_only=False)
                if checkpoint['global_step']!=2:raise ValueError('Cube small recovery must complete exactly2 updates')
                state.update(stage='complete',actual_exit_code=0,completed_count=2,final_checkpoint=dict(path=str(final),sha256=sha(final)))
                code=0
            except RecoveryPause as event:
                checkpoint=Path(str(event));meta=json.loads(checkpoint.with_suffix('.ckpt.json').read_text())
                if meta['actual_exit_code']!=75 or meta['global_step']!=1:raise ValueError('pause was not a sealed one-update handoff')
                state.update(stage='paused_for_independent_resume',actual_exit_code=75,completed_count=1,
                             checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint)));code=75
        except BaseException:
            state.update(stage='failed',actual_exit_code=1,failed_count=1,error=traceback.format_exc());traceback.print_exc()
        finally:
            os.environ.clear();os.environ.update(previous)
            state['finished_at']=state['updated_at']=datetime.now(timezone.utc).isoformat()
            fresh_data.save(output/'status.json',state);(output/'actual_exit_code').write_text(str(code)+'\n')
    return code

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=('lewm_pcd','lewm_rgb','pldm'),required=True)
    parser.add_argument('--data-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pause-after-batches',type=int,choices=(1,));parser.add_argument('--resume-checkpoint',type=Path)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    if args.pause_after_batches and args.resume_checkpoint:raise ValueError('pause and resume are separate independent runs')
    from .fresh_train import describe_training
    plan=describe_training(args.method,args.data_manifest,args.output,phase='smoke',workers=0,prefetch=1)
    recovery_identity(plan)
    print(json.dumps(plan,indent=2))
    return execute(plan,pause_after_batches=args.pause_after_batches,resume_checkpoint=args.resume_checkpoint)if args.execute else 0
if __name__=='__main__':raise SystemExit(main())

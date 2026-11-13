# Moved from OGBench-cube@4fbb0c6:lewm_pcd/rh56_adapter/wm_recovery.py (verbatim except: imports / paths adapted to the
# package; lines wrapped at 120 columns, unused imports removed)
"""Bounded RH56 state hooks around the original Trainer/Manager and data loop."""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

EPOCH_FIELDS=('epoch_sum','epoch_batches','val_sum','val_batches')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def epoch_progress(callback,epoch):
    if not all(hasattr(callback, name) for name in EPOCH_FIELDS):
        raise ValueError('missing original RH56 epoch statistics')
    return dict(epoch=int(epoch),**{name:getattr(callback,name) for name in EPOCH_FIELDS})


def restore_epoch_progress(callback,saved,epoch):
    if saved['epoch']!=epoch:return False
    if (set(saved)!=set(EPOCH_FIELDS)|{'epoch'} or any(saved[name]<0 for name in ('epoch_batches','val_batches'))):
        raise ValueError('invalid restored original RH56 epoch statistics')
    for name in EPOCH_FIELDS:setattr(callback,name,saved[name])
    return True


class EvidenceState:
    """Retain only deterministic original bookkeeping plus its real reload batch."""
    def __init__(self,evidence,capture,output):self.evidence=evidence;self.capture=capture;self.output=Path(output)
    def state_dict(self):
        callback=self.capture.get('evidence_callback')
        stats = (
            None
            if callback is None or not all(hasattr(callback, n) for n in EPOCH_FIELDS)
            else epoch_progress(callback, self.capture['epoch_index'])
        )
        return dict(evidence=copy.deepcopy({k:v for k,v in self.evidence.items() if k!='fit_wall_seconds'}),
                    epoch_progress=stats,real_reload_batch=self.capture.get('batch'))
    def load_state_dict(self,saved):
        import torch
        self.evidence.clear();self.evidence.update(copy.deepcopy(saved['evidence']))
        self.capture['restored_epoch_progress']=copy.deepcopy(saved['epoch_progress'])
        stats=saved['epoch_progress']
        if stats is not None:
            callback=self.capture.get('evidence_callback')
            if callback is None:raise ValueError('original RH56 evidence callback is not constructed')
            restore_epoch_progress(callback,stats,stats['epoch'])
            self.capture['epoch_index']=stats['epoch']
        batch=saved['real_reload_batch']
        if batch is not None:
            self.capture['batch']=batch
            with (self.output/'real_training_batch.pt').open('xb') as stream:torch.save(batch,stream)


def recovery_identity(args,cfg,producer_identity):
    import torch
    from omegaconf import OmegaConf
    from jepa4r.models.relocate import recovery
    from jepa4r.utils import training_state
    baseline=Path(args.baseline_root).resolve();path=baseline/'envs/adroit/hdf_binding.py'
    spec = importlib.util.spec_from_file_location('_rh56_recovery_hdf_binding', path)
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    content=binding.hdf_dependencies(args.dataset);config=OmegaConf.to_container(cfg,resolve=True)
    config['data']['dataset']['name']='$BOUND_TRAIN_CONTENT'
    if config['model']['action_encoder']['input_dim'] != 60:
        raise ValueError('RH56 recovery requires the real twelve-action five-step block')
    return dict(
        model=dict(
            route=args.route,
            baseline_tree_sha256=producer_identity['baseline_tree_sha256'],
            adapter_sha256=producer_identity['adapter_sha256'],
            recovery_sha256=sha(__file__),
            loader_sha256=sha(recovery.__file__),
            state_saver_sha256=sha(training_state.__file__),
            hdf_binder_sha256=sha(path),
        ),
        data=dict(content_sha256=content['content_sha256'], manifest_sha256=producer_identity['manifest_sha256']),
        config=hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        split='original RH56 episode-heldout manifest and train-row-only original normalizer',
        action_dim=12,
        numerical_profile=dict(
            torch=torch.__version__,
            precision=config['trainer']['precision'],
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        ),
    )


@contextmanager
def training_context(train,identity_factory,directory,*,seed=3072,pause_after_batches=None,resume_checkpoint=None,
                     evidence_provider=None,evidence_restorer=None):
    import torch
    from jepa4r.models.relocate.recovery import stateful_loader_class,make_lightning_recovery_callback
    from jepa4r.models.relocate.wm_recovery import validate_resume
    if seed!=3072:raise ValueError('guide RH56 WM recovery seed must be3072')
    directory=Path(directory)
    if directory.exists():raise FileExistsError(directory)
    original_loader=torch.utils.data.DataLoader;original_manager=train.spt.Manager
    def manager(*args,**kwargs):
        trainer=kwargs.get('trainer')
        if trainer is None:raise ValueError('RH56 recovery requires the original named Trainer/Manager')
        identity=identity_factory();        callback = make_lightning_recovery_callback(
            identity,
            directory,
            pause_after_batches=pause_after_batches,
            evidence_provider=evidence_provider,
            evidence_restorer=evidence_restorer,
        )
        (directory/'IDENTITY.json').write_text(json.dumps(dict(identity=identity,seed=seed,
            scope='original RH56 loop, complete state, workers0 bounded smoke/profile'),indent=2))
        kwargs['seed']=seed;kwargs['weights_only']=False
        if resume_checkpoint is not None:
            kwargs['ckpt_path'] = str(validate_resume(resume_checkpoint, identity))
        elif kwargs.get('ckpt_path') is not None:
            raise ValueError('fresh RH56 recovery cannot reuse an implicit checkpoint')
        trainer.callbacks.append(callback);return original_manager(*args,**kwargs)
    torch.utils.data.DataLoader=stateful_loader_class(original_loader);train.spt.Manager=manager
    try:yield
    finally:torch.utils.data.DataLoader=original_loader;train.spt.Manager=original_manager

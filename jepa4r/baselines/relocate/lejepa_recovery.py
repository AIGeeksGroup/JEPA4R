# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/relocate/lejepa_recovery.py (verbatim except: imports / paths adapted
# to the package; lines wrapped at 120 columns, unused imports removed)
"""State-only hook for the original Native LeJEPA manual training loop."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

class LejepaRecovery:
    def __init__(
        self,
        directory,
        *,
        resume_from=None,
        pause_after_updates=None,
        identity=None,
        evidence_provider=None,
        evidence_restorer=None,
    ):
        self.directory=Path(directory);self.resume_from=None if resume_from is None else Path(resume_from)
        self.pause_after_updates = pause_after_updates
        self.identity = identity
        self.evidence_provider = evidence_provider
        self.evidence_restorer = evidence_restorer
        self.global_step = 0
        self.start_epoch = 0
        self.start_batch = 0
        self.records = []
        self.last_checkpoint = None
        self.input_sha = None
    def _production_identity(self,args):
        import torch
        from jepa4r.models.relocate import recovery
        from jepa4r.utils import training_state

        source = ROOT / 'envs/adroit/hdf_binding.py'
        spec = importlib.util.spec_from_file_location('_lejepa_hdf_binding', source)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        bound=m.hdf_dependencies(args.dataset);cfg=dict(vars(args));cfg['dataset']='$BOUND_TRAIN_CONTENT'
        return dict(
            model={
                'method': 'native-lejepa',
                'native_script_sha256': _sha(ROOT / 'baselines/relocate/train_lejepa_relocate.py'),
                'hook_sha256': _sha(__file__),
                'loader_sha256': _sha(recovery.__file__),
                'state_saver_sha256': _sha(training_state.__file__),
            },
            data={
                'content_sha256': bound['content_sha256'],
                'audit_sha256': _sha(Path(args.dataset).parent / 'audit.json'),
            },
            config=hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            split='original-all-visual-frames',
            action_dim=30,
            numerical_profile=dict(
                torch=torch.__version__,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                precision='original BF16 autocast and original GradScaler',
            ),
        )
    def attach(self,net,sigreg,optimizer,scheduler,scaler,loader,args):
        import torch
        from jepa4r.utils.training_state import FixedBatchCursor,restore_torch_state,save_torch_state

        if args.workers != 0 or args.max_steps < 1:
            raise ValueError('LeJEPA recovery requires explicit workers=0 bounded smoke')
        if not hasattr(loader, 'commit_batch'):
            raise ValueError('LeJEPA recovery needs the original explicitly stateful DataLoader')
        if self.pause_after_updates is not None and not 0<self.pause_after_updates<args.max_steps:
            raise ValueError('LeJEPA pause must precede the bounded final budget')
        if self.identity is None:
            if args.seed!=3072 or torch.initial_seed()!=3072:raise ValueError('guide LeJEPA seed3072 not applied')
            self.identity=self._production_identity(args)
        self.args = args
        self.loader = loader
        self.model = torch.nn.ModuleDict({'encoder': net, 'sigreg': sigreg})
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.cursor = FixedBatchCursor([f'epoch{e}/batch{b}' for e in range(args.epochs) for b in range(len(loader))])
        self.directory.mkdir(parents=True, exist_ok=False)
        if self.resume_from is not None:
            manifest=json.loads(self.resume_from.with_suffix(self.resume_from.suffix+'.json').read_text())
            if manifest['progress'].get('phase') != 'committed_after_update':
                raise ValueError('LeJEPA resume requires a committed update, not a starting reference')
            progress = restore_torch_state(
                self.resume_from,
                model=self.model,
                optimizers={'main': optimizer},
                schedulers={'main': scheduler},
                scaler=scaler,
                identity=self.identity,
                cursor=self.cursor,
            )
            if manifest['progress'] != progress:
                raise ValueError('LeJEPA resume sidecar progress differs from sealed main state')
            if progress['global_step']>=args.max_steps:raise ValueError('completed budget cannot resume training')
            relative=Path(progress['loader_file'])
            if relative.name!=str(relative):raise ValueError('loader state must remain beside checkpoint')
            path=self.resume_from.parent/relative
            if _sha(path)!=progress['loader_sha256']:raise ValueError('loader state checksum differs')
            loader.load_state_dict(torch.load(path,map_location='cpu',weights_only=False))
            self.global_step = progress['global_step']
            self.start_epoch = progress['epoch']
            self.start_batch = progress['next_batch_index']
            self.records = list(progress['batch_records'])
            if self.evidence_restorer is not None:self.evidence_restorer(progress['optimizer_observers'])
        else:
            save_torch_state(self.directory/'starting.pt',model=self.model,optimizers={'main':optimizer},
                schedulers={'main':scheduler},scaler=scaler,identity=self.identity,cursor=self.cursor,
                progress=dict(phase='unstarted_reference_not_resumable',global_step=0))
        return self.start_epoch
    def next_batch_index(self,epoch):return self.start_batch if epoch==self.start_epoch else 0
    def before_input(self,vs):
        import numpy as np
        import torch
        a=vs.detach().cpu().numpy();h=hashlib.sha256((str(a.dtype)+str(a.shape)).encode())
        for i in range(0,len(a),8):h.update(np.ascontiguousarray(a[i:i+8]).tobytes())
        self.input_sha=h.hexdigest();path=self.directory/f'input_batch_{self.global_step:04d}.pt.gz'
        with gzip.open(path,'xb',compresslevel=1) as f:torch.save(vs.detach().cpu(),f)
    def after_step(self,epoch,batch_index,*,loss):
        from jepa4r.models.relocate.recovery import RecoveryPause
        expected=f'epoch{epoch}/batch{batch_index}'
        if self.cursor.take()!=expected:raise ValueError('LeJEPA actual batch cursor differs')
        ids=self.loader.last_batch_ids.copy();self.loader.commit_batch();self.global_step+=1
        self.records.append(
            dict(
                epoch=int(epoch),
                batch_index=int(batch_index),
                global_step=self.global_step,
                sample_ids=ids,
                input_sha256=self.input_sha,
                loss=float(loss.detach().cpu()),
            )
        )
        self.last_checkpoint=self.save(epoch,batch_index+1)
        if self.pause_after_updates is not None and self.global_step == self.pause_after_updates:
            raise RecoveryPause(str(self.last_checkpoint))
    def save(self,epoch,next_batch):
        import torch
        from jepa4r.utils.training_state import save_torch_state
        loader_path=self.directory/f'step_{self.global_step:04d}.loader.pt'
        with loader_path.open('xb') as f:torch.save(self.loader.state_dict(),f)
        progress = dict(
            phase='committed_after_update',
            epoch=int(epoch),
            next_batch_index=int(next_batch),
            global_step=self.global_step,
            loader_file=loader_path.name,
            loader_sha256=_sha(loader_path),
            batch_records=self.records.copy(),
            optimizer_observers={} if self.evidence_provider is None else self.evidence_provider(),
        )
        path=self.directory/f'step_{self.global_step:04d}.pt'
        save_torch_state(
            path,
            model=self.model,
            optimizers={'main': self.optimizer},
            schedulers={'main': self.scheduler},
            scaler=self.scaler,
            identity=self.identity,
            cursor=self.cursor,
            progress=progress,
        )
        return path

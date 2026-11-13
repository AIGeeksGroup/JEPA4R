# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/relocate/recovery.py (verbatim except: imports / paths adapted to the package)
"""Explicit workers=0 recovery around the original DataLoader/Lightning loop."""
import hashlib
import json
from pathlib import Path


class RecoveryPause(RuntimeError):
    """Intentional sealed-checkpoint handoff; caller records process exit75."""


def stateful_loader_class(original):
    import torch  # noqa: F401  (moved verbatim)
    from torch.utils.data import RandomSampler,SequentialSampler,BatchSampler
    from jepa4r.utils.training_state import FixedBatchCursor,capture_rng,restore_rng

    class RecordedBatches:
        def __init__(self,source,require_commit):
            self.source=source;self.require_commit=require_commit
            self.batches=None;self.cursor=None;self.last_ids=None;self.pending=None
            self.plan_rng=None;self.plan_generator=None;self.source_iterator=None;self.continue_once=False
        def __len__(self):return len(self.source)
        @property
        def generator(self):return getattr(self.source.sampler,'generator',None)
        def materialize(self):
            self.pending=None;self.last_ids=None
            self.plan_rng=capture_rng()
            self.plan_generator=self.generator.get_state() if self.generator is not None else None
            self.source_iterator=iter(self.source)
            # Pull exactly N batches, leaving the original sampler's postamble
            # suspended until the caller actually exhausts this iterator.
            self.batches=[[int(x) for x in next(self.source_iterator)] for _ in range(len(self.source))]
            if not self.batches:raise ValueError('recovery needs a nonempty real batch stream')
            self.cursor=FixedBatchCursor(list(range(len(self.batches))))
        def __iter__(self):
            if self.continue_once:self.continue_once=False
            else:self.materialize()
            for index in range(self.cursor.position,len(self.batches)):
                if self.pending is not None:raise RuntimeError('commit the actual batch before fetching another')
                self.pending=index;self.last_ids=self.batches[index].copy()
                yield self.last_ids
                if not self.require_commit:self.commit()
                if self.pending is not None:raise RuntimeError('actual batch was fetched without a completed commit')
            try:next(self.source_iterator)
            except StopIteration:return
            raise RuntimeError('original sampler exceeded its declared batch count')
        def commit(self):
            if self.pending is None or self.pending!=self.cursor.position:raise RuntimeError('batch commit cursor differs')
            self.cursor.take();self.pending=None
        def state_dict(self):
            return dict(batches=self.batches,cursor=None if self.cursor is None else self.cursor.state_dict(),
                plan_rng=self.plan_rng,plan_generator=self.plan_generator,require_commit=self.require_commit)
        def load_state_dict(self,state):
            batches=state['batches']
            if (state['require_commit']!=self.require_commit or batches is None or len(batches)!=len(self.source)
                    or any(not row or any(type(i) is not int or not 0<=i<len(self.source.sampler.data_source) for i in row) for row in batches)):
                raise ValueError('recovery batch stream/configuration differs')
            cursor=FixedBatchCursor(list(range(len(batches))));cursor.load_state_dict(state['cursor'])
            live_rng=capture_rng();live_generator=self.generator.get_state() if self.generator is not None else None
            try:
                restore_rng(state['plan_rng'])
                if self.generator is not None:self.generator.set_state(state['plan_generator'])
                source_iterator=iter(self.source)
                replay=[[int(x) for x in next(source_iterator)] for _ in range(len(self.source))]
                if replay!=batches:raise ValueError('original sampler no longer produces the saved sample order')
            finally:
                restore_rng(live_rng)
                if self.generator is not None:self.generator.set_state(live_generator)
            self.batches=[row.copy() for row in batches];self.cursor=cursor
            self.plan_rng=state['plan_rng'];self.plan_generator=state['plan_generator']
            self.source_iterator=source_iterator;self.pending=None;self.continue_once=True

    class StatefulLoader(original):
        def __init__(self,*args,**kwargs):
            if len(args)>1:raise ValueError('recovery loader requires named loader options')
            if kwargs.get('num_workers',0)!=0:raise ValueError('exact recovery currently requires workers=0')
            dataset=args[0] if args else kwargs.pop('dataset')
            if kwargs.get('batch_sampler') is not None:raise ValueError('custom batch samplers need their own recovery contract')
            shuffle=kwargs.pop('shuffle',False);sampler=kwargs.pop('sampler',None)
            require_commit=kwargs.pop('_recovery_commit',shuffle)
            if sampler is None:sampler=RandomSampler(dataset,generator=kwargs.get('generator')) if shuffle else SequentialSampler(dataset)
            if not isinstance(sampler,(RandomSampler,SequentialSampler)) or getattr(sampler,'replacement',False):
                raise ValueError('unsupported stochastic sampler recovery contract')
            if isinstance(sampler,RandomSampler) and sampler.num_samples!=len(dataset):raise ValueError('nondefault sample count is unsupported')
            size=kwargs.pop('batch_size',1);drop=kwargs.pop('drop_last',False)
            self.recovery_batches=RecordedBatches(BatchSampler(sampler,size,drop),require_commit)
            self.resume_pending=None;self.resume_after_iterator_rng=None
            super().__init__(dataset,batch_sampler=self.recovery_batches,**kwargs)
            # Retain the original public loader contract used by SPT/schedulers.
            object.__setattr__(self,'batch_size',size)
            object.__setattr__(self,'drop_last',drop)
            object.__setattr__(self,'sampler',sampler)
        @property
        def last_batch_ids(self):return self.recovery_batches.last_ids
        def commit_batch(self):self.recovery_batches.commit()
        def state_dict(self):
            return dict(schema='original-loader-recovery-v1',batch_sampler=self.recovery_batches.state_dict(),
                generator=self.generator.get_state() if self.generator is not None else None,rng=capture_rng())
        def load_state_dict(self,state):
            if state.get('schema')!='original-loader-recovery-v1' or (self.generator is None)!=(state['generator'] is None):
                raise ValueError('loader state/generator contract differs')
            self.recovery_batches.load_state_dict(state['batch_sampler'])
            if self.generator is not None:self.generator.set_state(state['generator'])
            self.resume_pending=state
        def __iter__(self):
            pending=self.resume_pending
            if pending is not None:restore_rng(pending['rng'])
            iterator=super().__iter__()
            if pending is not None:
                pos=self.recovery_batches.cursor.position
                if 0<pos<len(self.recovery_batches):
                    # Reconstructing a mid-epoch iterator must not draw a second
                    # worker base seed from the live model/loader RNG streams.
                    restore_rng(pending['rng'])
                    if self.generator is not None:self.generator.set_state(pending['generator'])
                else:self.resume_after_iterator_rng=capture_rng()
                self.resume_pending=None
            return iterator
    return StatefulLoader


def _train_loader(trainer):
    loader=trainer.train_dataloader
    while isinstance(loader,(dict,list,tuple)):
        values=list(loader.values()) if isinstance(loader,dict) else list(loader)
        if len(values)!=1:raise ValueError('recovery currently requires one training DataLoader')
        loader=values[0]
    if not hasattr(loader,'commit_batch'):raise ValueError('training DataLoader is not explicitly resumable')
    return loader


def make_lightning_recovery_callback(identity,directory,*,pause_after_batches=None,evidence_provider=None,evidence_restorer=None):
    import torch
    from jepa4r.utils.training_state import lightning_state_callback,restore_rng
    base=type(lightning_state_callback(identity))
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    class Recovery(base):
        def __init__(self):super().__init__();self.records=[];self.loaded_evidence=None;self.loaded_loader=None;self.input_sha=None
        @property
        def state_key(self):return 'AdroitT4Recovery-v1'
        def state_dict(self):return {'records':self.records.copy()}
        def load_state_dict(self,state):self.records=list(state['records'])
        def on_save_checkpoint(self,trainer,module,checkpoint):
            super().on_save_checkpoint(trainer,module,checkpoint)
            checkpoint['adroit_train_loader_state']=_train_loader(trainer).state_dict()
            if evidence_provider is not None:checkpoint['adroit_optimizer_observers']=evidence_provider()
        def on_load_checkpoint(self,trainer,module,checkpoint):
            super().on_load_checkpoint(trainer,module,checkpoint)
            self.loaded_loader=checkpoint['adroit_train_loader_state']
            self.loaded_evidence=checkpoint.get('adroit_optimizer_observers')
        def on_fit_start(self,trainer,module):
            if self.loaded_evidence is not None and evidence_restorer is not None:evidence_restorer(self.loaded_evidence)
        def on_train_start(self,trainer,module):
            loader=_train_loader(trainer)
            if self.loaded_loader is not None:
                # SPT's scheduler estimates stepping batches during optimizer
                # construction, before Lightning restores loop state. Its
                # already-created loader would otherwise miss the deferred
                # CombinedLoader restore. Load the public state explicitly.
                loader.load_state_dict(self.loaded_loader)
                loader.resume_pending=None
                loader.resume_after_iterator_rng=None
                self.loaded_loader=None
            super().on_train_start(trainer,module)
            if loader.resume_after_iterator_rng is not None:
                restore_rng(loader.resume_after_iterator_rng);loader.resume_after_iterator_rng=None
            path=directory/'starting.ckpt'
            if path.exists():raise FileExistsError(path)
            trainer.save_checkpoint(str(path))
            with path.with_suffix('.ckpt.json').open('x') as stream:json.dump(dict(
                schema='sealed-lightning-recovery-v1',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                identity=identity,global_step=int(trainer.global_step),phase='process_start_reference',
                actual_exit_code=0),stream,indent=2)
        def on_train_batch_start(self,trainer,module,batch,batch_idx):
            def cpu(value):
                if torch.is_tensor(value):return value.detach().cpu().clone()
                if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
                if isinstance(value,(list,tuple)):return type(value)(cpu(v) for v in value)
                return value
            path=directory/f'input_batch_{len(self.records):04d}.pt'
            with path.open('xb') as stream:torch.save(cpu(batch),stream)
            self.input_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            loader=_train_loader(trainer);ids=loader.last_batch_ids.copy();loader.commit_batch()
            loss=outputs.get('loss') if isinstance(outputs,dict) else outputs
            self.records.append(dict(epoch=int(trainer.current_epoch),batch_idx=int(batch_idx),global_step=int(trainer.global_step),
                sample_ids=ids,input_sha256=self.input_sha,loss=None if loss is None else float(loss.detach().cpu())))
            if pause_after_batches is not None and len(self.records)>=pause_after_batches:
                path=directory/f'pause_batch_{len(self.records):04d}.ckpt'
                if path.exists():raise FileExistsError(path)
                trainer.save_checkpoint(str(path))
                with path.with_suffix('.ckpt.json').open('x') as stream:json.dump(dict(schema='sealed-lightning-recovery-v1',
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),identity=identity,global_step=int(trainer.global_step),
                    committed_batches=len(self.records),actual_exit_code=75,phase='paused_for_independent_resume'),stream,indent=2)
                raise RecoveryPause(str(path))
        def on_fit_end(self,trainer,module):
            path=directory/'final.ckpt'
            if path.exists():raise FileExistsError(path)
            trainer.save_checkpoint(str(path))
            with path.with_suffix('.ckpt.json').open('x') as stream:json.dump(dict(schema='sealed-lightning-recovery-v1',
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),identity=identity,global_step=int(trainer.global_step),
                committed_batches=len(self.records),actual_exit_code=0,phase='complete'),stream,indent=2)
    return Recovery()

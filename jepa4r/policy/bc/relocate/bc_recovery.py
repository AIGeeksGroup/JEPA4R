"""Default-off state hooks for the original EAI direct BC loop.

No optimizer/loss/training loop is implemented here. The approved function AST
retains its original training nodes; only state/control metadata hooks are added.
"""
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType

from jepa4r.models.relocate.recovery import RecoveryPause

FUNCTION='bc_pvr_train_loop'
TOKEN='__jepa4r_bc_recovery_hook__'
METRICS=('highest_tr_score','highest_score','highest_tr_success','highest_success','head_audited',
         'mean_score','success_percentage','tr_score','tr_success','previous_highest_score','epoch_log')


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):digest.update(block)
    return digest.hexdigest()


def _json(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write(chr(10))
        stream.flush();os.fsync(stream.fileno())


def _recipe(config):
    from omegaconf import OmegaConf
    if OmegaConf.is_config(config):
        config=OmegaConf.to_container(config,resolve=True)
    keys=('seed','epochs','num_demos','device','bc_kwargs','eval_frequency','save_frequency','eval_num_traj','num_cpu')
    result={key:config[key] for key in keys}
    result['environment'] = {
        key: config['env_kwargs'].get(key) for key in ('suite', 'history_window', 'proprio_key', 'embedding_name')
    }
    return _json(result)


def _one(values,label):
    if len(values)!=1:raise ValueError('frozen BC AST anchor must be unique: '+label)
    return values[0]


def _instrument(official):
    source=Path(official.__file__)
    original=getattr(official,FUNCTION)
    if original.__globals__ is not official.__dict__ or Path(original.__code__.co_filename).resolve()!=source.resolve():
        raise ValueError('BC function is not the original module function')
    tree=ast.parse(source.read_text(encoding='utf-8'),filename=original.__code__.co_filename)
    function=_one([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==FUNCTION],FUNCTION)
    trusted={**official.__dict__}
    exec(
        compile(
            ast.Module(body=[copy.deepcopy(function)], type_ignores=[]),
            original.__code__.co_filename,
            'exec',
            dont_inherit=True,
        ),
        trusted,
    )
    if (
        original.__code__.co_code != trusted[FUNCTION].__code__.co_code
        or original.__code__.co_consts != trusted[FUNCTION].__code__.co_consts
        or original.__code__.co_names != trusted[FUNCTION].__code__.co_names
    ):
        raise ValueError('live BC function bytecode differs from frozen source')
    calls=list(ast.walk(function))
    for spelling in ('DataLoader','torch.optim.Adam','torch.nn.MSELoss'):
        _one([n for n in calls if isinstance(n,ast.Call) and ast.unparse(n.func)==spelling],spelling)
    epoch = _one(
        [
            n
            for n in function.body
            if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch'
        ],
        'epoch',
    )
    batch = _one(
        [
            n
            for n in epoch.body
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Tuple)
            and [x.id for x in n.target.elts] == ['mb_idx', 'batch']
        ],
        'minibatch',
    )
    if ast.unparse(batch.iter)!='enumerate(dataloader)':raise ValueError('frozen enumerate contract differs')
    train = _one(
        [
            n
            for n in epoch.body
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == 'policy.model.train'
        ],
        'model.train',
    )
    initial_loss = _one(
        [
            n
            for n in epoch.body
            if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == ['running_loss']
        ],
        'epoch loss',
    )
    _one(
        [
            n
            for n in batch.body
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == 'optimizer.step'
        ],
        'optimizer.step',
    )
    accumulation=batch.body[-1]
    if (
        not isinstance(accumulation, ast.Assign)
        or [ast.unparse(t) for t in accumulation.targets] != ['running_loss']
        or not isinstance(accumulation.value, ast.BinOp)
    ):
        raise ValueError('frozen original running-loss boundary differs')
    protected = [
        n
        for n in function.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in ('optimizer', 'loss_func') for t in n.targets)
    ] + list(batch.body)
    before=[ast.dump(n) for n in protected]
    def insert(text,anchor):
        nodes=ast.parse(text).body
        for node in nodes:ast.increment_lineno(node,anchor.lineno-1)
        return nodes
    function.body[:0]=insert(TOKEN+'.begin(config)',function.body[0])
    index=function.body.index(epoch)
    metric_nodes=[]
    for name in METRICS:
        metric_nodes += insert(
            "if "
            + TOKEN
            + ".has_metric("
            + repr(name)
            + "):\n    "
            + name
            + ' = '
            + TOKEN
            + '.metric('
            + repr(name)
            + ')',
            epoch,
        )
    function.body[index:index]=metric_nodes
    epoch.body[:0]=insert('if not '+TOKEN+'.visit_epoch(epoch):\n    continue',epoch)
    index=epoch.body.index(train)+1;epoch.body[index:index]=insert(TOKEN+'.epoch_start(locals())',train)
    index = epoch.body.index(initial_loss) + 1
    epoch.body[index:index] = insert('running_loss = ' + TOKEN + '.epoch_loss(epoch, running_loss)', initial_loss)
    batch.iter.args.append(ast.parse(TOKEN+'.batch_start(epoch)',mode='eval').body)
    batch.body+=insert(TOKEN+'.after_batch(locals())',accumulation)
    epoch.body+=insert(TOKEN+'.after_epoch(locals())',epoch.body[-1])
    if before!=[ast.dump(n) for n in protected]:raise AssertionError('original BC training math AST changed')
    result=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    return (
        original,
        result,
        dict(
            source_sha256=_sha(source),
            original_math_ast_sha256=hashlib.sha256(json.dumps(before).encode()).hexdigest(),
            original_math_nodes_unchanged=True,
        ),
    )


class BCRecovery:
    def __init__(self,identity,out,*,resume_from,pause_after_updates,observer,source_evidence):
        import numpy as np
        import torch
        from jepa4r.models.relocate import recovery
        from jepa4r.utils import training_state
        training_state.validate_identity(identity,identity)
        if not {'encoder','source'}<=set(identity) or not identity['encoder']:
            raise ValueError('BC identity requires encoder/source bindings')
        declared=identity['source']
        if not (isinstance(declared, dict) and declared.get('eai_train_loop_sha256')) and not isinstance(declared, str):
            raise ValueError('BC identity source must bind the EAI train loop')
        self.identity=_json(identity)
        self.identity['bc_recovery_runtime'] = dict(
            eai=declared.get('eai_train_loop_sha256') if isinstance(declared, dict) else declared,
            hook=_sha(__file__),
            loader=_sha(recovery.__file__),
            state_saver=_sha(training_state.__file__),
            python=sys.version.split()[0],
            torch=torch.__version__,
            numpy=np.__version__,
        )
        if pause_after_updates is not None and (type(pause_after_updates) is not int or pause_after_updates < 1):
            raise ValueError('pause_after_updates must be positive')
        self.directory=Path(out).resolve();self.resume_from=None if resume_from is None else Path(resume_from).resolve()
        self.pause_after_updates=pause_after_updates;self.observer=observer;self.evidence=source_evidence
        self.records = []
        self.epoch_records = []
        self.updates = 0
        self.last_checkpoint = None
        self.loaded = None
        self.progress = None
        self.started = False
        self.restored = False
        self.pending_pause = False
        self.recipe = None
        if self.resume_from is not None:
            meta=json.loads(self.resume_from.with_suffix(self.resume_from.suffix+'.json').read_text(encoding='utf-8'))
            training_state.validate_identity(self.identity,meta['identity'])
            if meta.get('actual_exit_code') != 0 or _sha(self.resume_from) != meta['sha256']:
                raise ValueError('BC checkpoint source/checksum differs')
            progress=meta['progress'];name=progress['extra_file']
            if Path(name).name != name or '/' in name or chr(92) in name or ':' in name:
                raise ValueError('unsafe BC auxiliary state path')
            extra=self.resume_from.parent/name
            if _sha(extra)!=progress['extra_sha256']:raise ValueError('BC loader/loop auxiliary SHA differs')
            loaded=torch.load(extra,map_location='cpu',weights_only=False)
            training_state.validate_identity(self.identity,loaded['identity'])
            if (
                loaded.get('schema') != 'original-eai-bc-extra-v1'
                or loaded['loader']['batch_sampler']['cursor'] != meta['cursor']
            ):
                raise ValueError('BC loader cursor and model checkpoint differ')
            self.loaded = loaded
            self.progress = progress
            self.recipe = loaded['recipe']
            self.records = copy.deepcopy(loaded['records'])
            self.epoch_records = copy.deepcopy(loaded['epoch_records'])
            self.updates = progress['committed_updates']
            if len(self.records)!=self.updates:raise ValueError('BC observer history/update count differs')
        self.directory.mkdir(parents=True,exist_ok=False)
        _write(
            self.directory / 'IDENTITY.json',
            dict(
                identity=self.identity,
                source=source_evidence,
                scope='state hooks around original EAI loop; no training math copy',
            ),
        )

    def begin(self,config):
        if self.started:raise ValueError('BC recovery context executes exactly one original loop')
        self.started=True;actual=_recipe(config)
        if self.progress is not None and self.progress['epoch']>=actual['epochs']:
            raise ValueError(
                'BC recovery checkpoint has no remaining training budget; use strict completed-policy reload'
            )
        if self.recipe is not None and self.recipe!=actual:raise ValueError('actual BC recipe differs from saved state')
        self.recipe=actual

    def has_metric(self,name):return self.loaded is not None and name in self.loaded['metrics']
    def metric(self,name):return self.loaded['metrics'][name]
    def visit_epoch(self,epoch):return self.progress is None or epoch>=self.progress['epoch']

    def batch_start(self, epoch):
        return self.progress['next_mb_idx'] if self.progress is not None and epoch == self.progress['epoch'] else 0
    def epoch_loss(self,epoch,initial):
        return (
            self.loaded['running_loss']
            if self.progress is not None and epoch == self.progress['epoch'] and self.progress['boundary'] == 'batch'
            else initial
        )

    def _notify(self,event,live):
        if self.observer is None:return
        from jepa4r.utils.training_state import capture_rng,restore_rng
        state=capture_rng()
        try:self.observer(copy.deepcopy(event),MappingProxyType(dict(live)))
        finally:restore_rng(state)

    def epoch_start(self,live):
        loader=live['dataloader']
        if loader.num_workers != 0 or not hasattr(loader, 'commit_batch'):
            raise ValueError('BC recovery requires explicit workers=0 stateful loader')
        if self.pause_after_updates is not None and self.pause_after_updates>=len(loader)*self.recipe['epochs']:
            raise ValueError('BC pause must precede final budget; RH T4 needs its explicit two-epoch configuration')
        if self.loaded is None or self.restored:return
        from jepa4r.utils.training_state import restore_torch_state
        loader.load_state_dict(self.loaded['loader'])
        if self.progress['boundary']=='epoch_end':
            # load_state_dict rebuilds an old iterator at its final yield. Retire
            # its empty postamble, then restore the checkpoint RNG below.
            try:next(iter(loader))
            except StopIteration:pass
            else:raise ValueError('completed BC epoch unexpectedly yielded another batch')
            if loader.generator is not None:loader.generator.set_state(self.loaded['loader']['generator'])
        parameter_ids=[id(x) for x in live['policy'].model.parameters()]
        restored_progress = restore_torch_state(
            self.resume_from,
            model=live['policy'].model,
            optimizers={'bc_adam': live['optimizer']},
            schedulers={},
            scaler=None,
            identity=self.identity,
            cursor=loader.recovery_batches.cursor,
        )
        if restored_progress!=self.progress:raise ValueError('BC main checkpoint progress differs from sidecar')
        if parameter_ids != [id(x) for x in live['policy'].model.parameters()]:
            raise AssertionError('BC restore replaced original parameters')
        best=self.loaded['best_policy'];path=Path('iterations/best_policy.pickle')
        if best is not None:
            if path.exists() and _sha(path) != best['sha256']:
                raise ValueError('existing BC best policy differs from resumed history')
            if not path.exists():
                with path.open('xb') as stream:stream.write(best['bytes'])
        elif path.exists():raise ValueError('BC output contains a best policy outside saved history')
        self.restored=True
        self._notify(
            dict(
                kind='resume_loaded',
                committed_updates=self.updates,
                batch_records=self.records,
                epoch_records=self.epoch_records,
            ),
            live,
        )

    def after_batch(self,live):
        loader=live['dataloader'];ids=loader.last_batch_ids.copy();loader.commit_batch();self.updates+=1
        cursor=loader.recovery_batches.cursor.state_dict();batches=loader.recovery_batches.batches
        next_ids=None if cursor['position']==len(batches) else batches[cursor['position']].copy()
        event = dict(
            kind='batch_committed',
            epoch=int(live['epoch']),
            mb_idx=int(live['mb_idx']),
            update=self.updates,
            batch_size=len(ids),
            sample_ids=ids,
            next_sample_ids=next_ids,
            next_epoch_order_from_saved_rng=next_ids is None,
            loss=float(live['loss'].detach().cpu()),
            running_loss=float(live['running_loss']),
            learning_rates=[float(g['lr']) for g in live['optimizer'].param_groups],
        )
        self.records.append(event);self._notify(event,live)
        if self.pause_after_updates==self.updates:
            if live['mb_idx'] + 1 < len(loader):
                self._seal(live, 'batch', pause=True)
                raise RecoveryPause('BC real update committed; independent resume requested')
            self.pending_pause=True

    def after_epoch(self,live):
        event = dict(
            kind='epoch_finished',
            epoch=int(live['epoch']),
            committed_updates=self.updates,
            running_loss=float(live['running_loss']),
            minibatches=int(live['mb_idx']) + 1,
        )
        if 'epoch_log' in live and live['epoch_log'].get('eval/epoch')==live['epoch']:
            event['evaluation']={k:float(v) if hasattr(v,'item') else v for k,v in live['epoch_log'].items()}
        self.epoch_records.append(event);self._notify(event,live)
        if self.pending_pause:
            self._seal(live, 'epoch_end', pause=True)
            raise RecoveryPause('BC complete epoch committed; independent resume requested')
        if live['epoch']==self.recipe['epochs']-1:self._seal(live,'epoch_end',pause=False)

    def _seal(self,live,boundary,*,pause):
        import torch
        from jepa4r.utils.training_state import save_torch_state

        loader = live['dataloader']
        stem = 'update_%06d' % self.updates
        extra_path = self.directory / (stem + '.extra.pt')
        path = self.directory / (stem + '.pt')
        best=Path('iterations/best_policy.pickle')
        best_state=None if not best.exists() else dict(bytes=best.read_bytes(),sha256=_sha(best))
        extra = dict(
            schema='original-eai-bc-extra-v1',
            identity=self.identity,
            recipe=self.recipe,
            loader=loader.state_dict(),
            metrics={name: copy.deepcopy(live[name]) for name in METRICS if name in live},
            running_loss=copy.deepcopy(live['running_loss']),
            records=copy.deepcopy(self.records),
            epoch_records=copy.deepcopy(self.epoch_records),
            best_policy=best_state,
        )
        with extra_path.open('xb') as stream:torch.save(extra,stream);stream.flush();os.fsync(stream.fileno())
        progress = dict(
            schema='original-eai-bc-progress-v1',
            committed_updates=self.updates,
            boundary=boundary,
            epoch=int(live['epoch']) + (boundary == 'epoch_end'),
            next_mb_idx=0 if boundary == 'epoch_end' else int(live['mb_idx']) + 1,
            extra_file=extra_path.name,
            extra_sha256=_sha(extra_path),
            next_sample_ids=self.records[-1]['next_sample_ids'],
            expected_caller_process_exit_code=75 if pause else 0,
        )
        save_torch_state(
            path,
            model=live['policy'].model,
            optimizers={'bc_adam': live['optimizer']},
            schedulers={},
            scaler=None,
            identity=self.identity,
            cursor=loader.recovery_batches.cursor,
            progress=progress,
        )
        self.last_checkpoint=path
        _write(
            self.directory / (stem + '.handoff.json'),
            dict(
                checkpoint=path.name,
                state_write_actual_exit_code=0,
                expected_caller_process_exit_code=75 if pause else 0,
                actual_caller_process_exit_code=None,
                phase='checkpoint_sealed_for_pause' if pause else 'original_training_budget_complete',
                progress=progress,
            ),
        )


@contextmanager
def install_bc_recovery(official,identity,out,*,enabled=False,resume_from=None,pause_after_updates=None,observer=None):
    """Install only in this context; disabled leaves the original object/path untouched.

    observer(event, live) receives a passive event and read-only locals mapping.
    resume_loaded contains prior batch/epoch records for caller-owned counters.
    The caller catches RecoveryPause and records actual process exit75 separately
    from the shared checkpoint manifest's successful state-write exit0.
    """
    if not enabled:
        yield None;return
    from jepa4r.models.relocate.recovery import stateful_loader_class
    original,tree,evidence=_instrument(official)
    namespace=official.__dict__
    if TOKEN in namespace:raise ValueError('BC recovery contexts may not overlap')
    controller = BCRecovery(
        identity,
        out,
        resume_from=resume_from,
        pause_after_updates=pause_after_updates,
        observer=observer,
        source_evidence=evidence,
    )
    original_loader=official.DataLoader;previous_cwd=Path.cwd()
    namespace[TOKEN]=controller
    try:
        official.DataLoader=stateful_loader_class(original_loader)
        exec(compile(tree,original.__code__.co_filename,'exec',dont_inherit=True),namespace)
        yield controller
    finally:
        setattr(official,FUNCTION,original);official.DataLoader=original_loader
        namespace.pop(TOKEN,None);os.chdir(previous_cwd)

"""Real original-Trainer WM gates; imports are CPU-only until run() requests GPU.

No loss/optimizer/scheduler implementation is copied. install_recorder only
appends observation callbacks; original train.run and SaveCkptCallback execute.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

PROTOCOL = {'epochs':10,'batch':128,'precision':'bf16','history':3,'frameskip':5,
            'num_preds':1,'action':24,'input':120,'pcd_dim':6,'img':224}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda:f.read(1<<20), b''): h.update(value)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_batch_shapes(route, shapes, batch):
    key = 'pixels' if route == 'rgb' else 'pcd'
    wanted = (batch,4,3,224,224) if route == 'rgb' else (batch,4,1200,6)
    assert tuple(shapes[key]) == wanted, (key, shapes[key], wanted)
    assert tuple(shapes['action']) == (batch,4,120), 'expected grouped original 5x24 actions'


def validate_action_mask(nonfinite, nan):
    for bad, is_nan in zip(nonfinite, nan):
        assert not any(any(row) for row in bad[:3]), 'nonfinite consumed action history (possibly zero std)'
        assert not any(bad[3][:96]), 'nonterminal nonfinite action in unused block'
        tail = bad[3][96:]
        assert not any(tail) or (all(tail) and all(is_nan[3][96:])), 'only complete native terminal24 NaNs are allowed'


def configure_spt_cache(train, path, evidence_dir=None):
    resolved = str(Path(path).resolve())
    cleanup = {'checkpoints':False,'logs':False,'callback_artifacts':False}
    train.spt.set(cache_dir=resolved, cleanup=cleanup)
    actual = train.spt.get_config()
    assert Path(actual.cache_dir).resolve()==Path(resolved)
    assert all(actual.cleanup[key] is False for key in cleanup)
    if evidence_dir is not None:
        save(Path(evidence_dir)/'effective_spt_config.json',dict(cache_dir=actual.cache_dir,cleanup=dict(actual.cleanup)))
    return resolved


def trainer_progress(checkpoint, expected_epochs):
    epochs = int(checkpoint['epoch'])+1
    steps = int(checkpoint['global_step'])
    assert epochs == expected_epochs and steps > 0, 'incomplete actual trainer epoch/optimizer evidence'
    optimizers = checkpoint['optimizer_states']
    assert optimizers and all(item.get('state') for item in optimizers), 'missing actual optimizer state'
    return epochs, steps


def tensor_sha(state):
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def all_finite(value):
    import torch
    if torch.is_tensor(value): return bool(torch.isfinite(value).all())
    if isinstance(value, dict): return all(all_finite(v) for v in value.values())
    if isinstance(value, (list,tuple)): return all(all_finite(v) for v in value)
    return True


def install_recorder(train_module, evidence_dir, route, dataset=None):
    """Read-only numerical callbacks plus a fresh full-state evidence checkpoint."""
    import torch
    from lightning.pytorch.callbacks import Callback
    evidence_dir = Path(evidence_dir).resolve(); evidence_dir.mkdir(parents=True, exist_ok=True)
    BaseTrainer = train_module.pl.Trainer

    class Recorder(Callback):
        def __init__(self):
            self.result = dict(gate='RUNNING', finite=True, epochs_completed=0, optimizer_steps=0,
                               backward_calls=0, loss_count=0, losses=[], grad_norms=[], shapes={},
                               dataset=str(dataset) if dataset is not None else None)
            self.model = self.trainer = self.batch = None
        def persist(self): save(evidence_dir/'training_evidence.json', self.result)
        def on_fit_start(self, trainer, module):
            self.trainer, self.model = trainer, module.model
            plugin = trainer.precision_plugin
            self.result.update(precision_plugin=type(plugin).__module__+'.'+type(plugin).__name__,
                               actual_precision=str(getattr(plugin,'precision',None)),
                               model_parameter_dtype=str(next(module.model.parameters()).dtype),
                               automatic_optimization=bool(module.automatic_optimization),
                               trainer_gradient_clip_val=trainer.gradient_clip_val)
            self.result['source_hashes'] = source_hashes(Path(train_module.__file__).parent,train_module)
            if dataset is not None and Path(dataset).is_file(): self.result['dataset_sha256'] = sha(dataset)
            self.persist()
        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            shapes = {k:list(v.shape) for k,v in batch.items() if torch.is_tensor(v)}
            b = int(batch['action'].shape[0]); validate_batch_shapes(route, shapes, b)
            self.result['shapes'] = shapes
            key = 'pixels' if route == 'rgb' else 'pcd'
            assert torch.isfinite(batch[key]).all(), 'nonfinite encoder input'
            bad = ~torch.isfinite(batch['action'])
            if bool(bad.any()): validate_action_mask(bad.detach().cpu().tolist(),torch.isnan(batch['action']).detach().cpu().tolist())
            if self.batch is None:
                self.batch = {k:v[:min(4,b)].detach().cpu().clone() for k,v in batch.items() if torch.is_tensor(v)}
                torch.save(self.batch, evidence_dir/'reference_batch.pt')
                self.result.update(reference_batch_sha256=sha(evidence_dir/'reference_batch.pt'),
                                   reference_batch_dtype={k:str(v.dtype) for k,v in self.batch.items()},
                                   actual_training_batch=b)
        def on_after_backward(self, trainer, module):
            grads = [p.grad.detach() for p in module.model.parameters() if p.grad is not None]
            assert grads and all_finite(grads), 'missing/nonfinite actual backward gradients'
            norms = torch.stack([g.float().norm() for g in grads])
            self.result['grad_norms'] = (self.result['grad_norms']+[float(norms.norm())])[-20:]
            self.result['backward_calls'] += 1
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            assert isinstance(outputs,dict), 'original SPT training_step returned no state'
            values = {k:float(outputs[k].detach()) for k in ('loss','pred_loss','sigreg_loss')}
            assert all_finite({k:outputs[k] for k in values}), 'nonfinite original loss'
            assert all_finite(module.model.state_dict()), 'nonfinite model state after optimizer'
            assert tuple(outputs['emb'].shape) == (batch['action'].shape[0],4,192)
            assert tuple(outputs['act_emb'].shape) == (batch['action'].shape[0],4,192)
            self.result.update(loss_count=self.result['loss_count']+1,
                               losses=(self.result['losses']+[values])[-20:],
                               latent_shape=list(outputs['emb'].shape),
                               optimizer_steps=int(trainer.global_step),
                               manual_optimizer_gradient_clip_val=dict(module._optimizer_gradient_clip_val),
                               manual_optimizer_gradient_clip_algorithm=dict(module._optimizer_gradient_clip_algorithm),
                               peak_cuda_allocated=int(torch.cuda.max_memory_allocated()),
                               peak_cuda_reserved=int(torch.cuda.max_memory_reserved()))
        def on_train_epoch_end(self, trainer, module):
            self.result['epochs_completed'] += 1
            self.result['last_completed_trainer_epoch'] = int(trainer.current_epoch)
            if int(trainer.current_epoch)+1 == int(trainer.max_epochs):
                path = evidence_dir/'trainer_final_evidence.ckpt'
                assert not path.exists(), 'fresh trainer evidence required'
                trainer.save_checkpoint(str(path), weights_only=False)
                self.result.update(trainer_checkpoint=str(path), trainer_checkpoint_sha256=sha(path))
            self.persist()
        def on_train_end(self, trainer, module):
            assert self.result['loss_count'] > 0 and self.result['backward_calls'] > 0 and int(trainer.global_step)>0
            self.result.update(gate='PASS', finite=True, optimizer_steps=int(trainer.global_step),
                               final_model_sha256=tensor_sha(module.model.state_dict()))
            self.persist()
        def on_exception(self, trainer, module, error):
            self.result.update(gate='FAIL', finite=False, error=repr(error)); self.persist()

    recorder = Recorder()
    class RecordingTrainer(BaseTrainer):
        def __init__(self,*args,**kwargs):
            kwargs['callbacks'] = list(kwargs.get('callbacks') or [])+[recorder]
            super().__init__(*args,**kwargs)
            recorder.result.update(precision_plugin=type(self.precision_plugin).__name__,
                                   actual_precision=str(getattr(self.precision_plugin,'precision',None)),
                                   trainer_gradient_clip_val=self.gradient_clip_val)
            recorder.persist()
    train_module.pl.Trainer = RecordingTrainer
    return recorder, BaseTrainer


def source_hashes(code, train):
    import stable_pretraining as spt
    import stable_worldmodel.data.formats.hdf5 as hdf
    import stable_worldmodel.data.dataset as ds
    from lightning.pytorch.trainer.connectors.accelerator_connector import _AcceleratorConnector
    paths = {str(p.relative_to(code)):p for p in code.glob('*.py') if p.name in ('train.py','jepa.py','module.py','utils.py')}
    for folder in ('encoders','config/train'):
        for p in (code/folder).rglob('*'):
            if p.suffix in ('.py','.yaml'): paths[str(p.relative_to(code))]=p
    paths['recorded_train_wrapper'] = Path(__file__).with_name('wm_recorded_train.py')
    paths.update(gate=Path(__file__), spt_module=Path(inspect.getfile(spt.Module)),
                 spt_manager=Path(inspect.getfile(spt.Manager)), spt_split=Path(inspect.getfile(spt.data.random_split)),
                 hdf_reader=Path(inspect.getfile(hdf)), swm_dataset=Path(inspect.getfile(ds)),
                 precision_connector=Path(inspect.getfile(_AcceleratorConnector)))
    return {name:sha(path) for name,path in paths.items()}


def outputs(model,batch,plugin):
    import torch
    model = model.to('cuda')  # Original Trainer teardown moves its model back to CPU.
    batch = {k:v.to('cuda') for k,v in batch.items()}
    batch['action'] = torch.nan_to_num(batch['action'],0.)  # Exact original forward boundary handling.
    model.eval()
    with torch.no_grad(), plugin.forward_context():
        value = model.encode(dict(batch))
        prediction = model.predict(value['emb'][:,:3],value['act_emb'][:,:3])
    assert tuple(value['emb'].shape)==(batch['action'].shape[0],4,192)
    assert tuple(prediction.shape)==(batch['action'].shape[0],3,192)
    assert all_finite((value['emb'],value['act_emb'],prediction)), 'nonfinite actual encoder/predictor'
    return tuple(v.detach().float().cpu() for v in (value['emb'],value['act_emb'],prediction))


def strict_reload(model, batch, plugin, checkpoint_dir, weights):
    import torch
    from stable_worldmodel.wm.utils import load_pretrained
    path=Path(checkpoint_dir)/weights
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert set(saved)==set(model.state_dict()) and all_finite(saved)
    assert tensor_sha(saved)==tensor_sha(model.state_dict()), 'saved whole-model state differs'
    reference=outputs(model,batch,plugin)
    loaded=load_pretrained(str(path.resolve()))
    loaded=plugin.convert_module(loaded).to('cuda')
    assert tensor_sha(loaded.state_dict())==tensor_sha(saved), 'strict config/state reload changed dtype/value'
    current=outputs(loaded,batch,plugin)
    maximum=[]
    for a,b in zip(reference,current):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        maximum.append(float((a-b).abs().max()))
    return dict(strict_reload='PASS',finite=True,checkpoint_sha256=sha(path),
                latent_shape=list(current[0].shape),prediction_shape=list(current[2].shape),
                reload_max_abs=maximum)


def gpu_gate(args,code,train,report):
    import torch
    import hydra
    from omegaconf import OmegaConf
    results=[]
    for batch,steps in ((4,2),(128,2)):
        stage=report/('batch%d'%batch); stage.mkdir()
        run='gate_%s_b%d'%(args.route,batch)
        os.environ['STABLEWM_HOME']=str(stage/'wm-cache'); os.environ['SPT_CACHE_DIR']=str(stage/'spt-cache')
        configure_spt_cache(train,stage/'spt-cache',stage)
        overrides=['data=adroit_pen'+('_pcd' if args.route=='pcd' else ''),
                   'model='+('lewm' if args.route=='rgb' else 'lewm_pcd_dexart'),
                   'data.dataset.name='+str(args.dataset.resolve()),'subdir='+run,'output_model_name='+run,
                   'trainer.max_epochs=1','trainer.devices=1','+trainer.limit_train_batches='+str(steps),
                   '+trainer.limit_val_batches=1','loader.batch_size='+str(batch),'num_workers=0',
                   'loader.persistent_workers=false','loader.prefetch_factor=null']
        with hydra.initialize_config_dir(version_base=None,config_dir=str(code/'config/train')):
            cfg=hydra.compose(config_name='lewm',overrides=overrides)
        assert cfg.trainer.precision=='bf16' and cfg.trainer.gradient_clip_val==1.0
        assert cfg.data.dataset.frameskip==5 and cfg.history_size==3 and cfg.num_preds==1
        OmegaConf.save(cfg,stage/'requested_config.yaml')
        recorder,Base=install_recorder(train,stage,args.route,args.dataset)
        oldcwd=Path.cwd(); os.chdir(stage)
        torch.cuda.reset_peak_memory_stats()
        try:
            train.run.__wrapped__(cfg)  # Original dataset transforms, Module, forward, optimizer, scheduler and callback.
            torch.cuda.synchronize()
        finally:
            os.chdir(oldcwd); train.pl.Trainer=Base
        assert recorder.result['gate']=='PASS' and recorder.result['actual_training_batch']==batch
        ckdir=stage/'wm-cache/checkpoints'/run
        final=OmegaConf.load(ckdir/'config.yaml')
        assert final.model.action_encoder.input_dim==120
        reloaded=strict_reload(recorder.model,recorder.batch,recorder.trainer.precision_plugin,ckdir,'weights_epoch_1.pt')
        ckpt=torch.load(stage/'trainer_final_evidence.ckpt',map_location='cpu',weights_only=False)
        epochs,optimsteps=trainer_progress(ckpt,1)
        assert all_finite(ckpt['optimizer_states'])
        results.append(dict(batch=batch,epochs_completed=epochs,optimizer_steps=optimsteps,
                            evidence=recorder.result,config_sha256=sha(ckdir/'config.yaml'),**reloaded))
        del ckpt,recorder
        torch.cuda.empty_cache()
    return dict(gate='PASS',status='PASS',finite=True,strict_reload='PASS',batch128_feasibility='PASS',stages=results)


def verify_full(args,code,train):
    import torch
    import hydra
    from omegaconf import OmegaConf
    from stable_worldmodel.wm.utils import load_pretrained
    from lightning.pytorch.trainer.connectors.accelerator_connector import _AcceleratorConnector
    ckdir=args.checkpoint_root.resolve()
    evidence=json.loads((ckdir/'training_evidence.json').read_text())
    assert evidence['gate']=='PASS' and evidence['finite'] and evidence['epochs_completed']==10
    assert evidence['actual_training_batch']==128 and evidence['loss_count']>0 and evidence['backward_calls']>0
    assert Path(evidence['dataset']).resolve()==args.dataset.resolve(), 'evidence dataset differs'
    cfg=OmegaConf.load(ckdir/'config.yaml')
    assert cfg.trainer.max_epochs==10 and cfg.loader.batch_size==128 and cfg.trainer.precision=='bf16'
    assert cfg.model.action_encoder.input_dim==120 and cfg.data.dataset.frameskip==5
    assert cfg.history_size==3 and cfg.num_preds==1 and cfg.img_size==224 and cfg.pcd_in_dim==6
    assert cfg.optimizer.type=='AdamW' and cfg.optimizer.lr==5e-5 and cfg.optimizer.weight_decay==1e-3
    assert cfg.loss.sigreg.weight==.09
    assert evidence['source_hashes']==source_hashes(code,train), 'training sources changed since actual run'
    assert evidence['dataset_sha256']==sha(args.dataset), 'training dataset header changed since actual run'
    cp=Path(evidence['trainer_checkpoint']); assert sha(cp)==evidence['trainer_checkpoint_sha256']
    checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
    epochs,steps=trainer_progress(checkpoint,10); assert all_finite(checkpoint['optimizer_states'])
    path=ckdir/args.weights
    saved=torch.load(path,map_location='cpu',weights_only=True)
    prefixes=[k[:-len(next(iter(saved)))] for k in checkpoint['state_dict'] if k.endswith(next(iter(saved)))]
    matches=[p for p in prefixes if all(p+k in checkpoint['state_dict'] for k in saved)
             and tensor_sha({k:checkpoint['state_dict'][p+k] for k in saved})==tensor_sha(saved)]
    assert matches, 'epoch10 .pt is not the actual final Trainer model state'
    batchfile=ckdir/'reference_batch.pt'; assert sha(batchfile)==evidence['reference_batch_sha256']
    batch=torch.load(batchfile,map_location='cpu',weights_only=True)
    validate_batch_shapes(args.route,{k:v.shape for k,v in batch.items()},batch['action'].shape[0])
    bad = ~torch.isfinite(batch['action'])
    if bool(bad.any()): validate_action_mask(bad.tolist(),torch.isnan(batch['action']).tolist())
    connector=_AcceleratorConnector(accelerator='gpu',devices=1,precision=cfg.trainer.precision)
    plugin=connector.precision_plugin
    model=plugin.convert_module(load_pretrained(str(path))).to('cuda')
    result=strict_reload(model,batch,plugin,ckdir,args.weights)
    assert tensor_sha(saved)==evidence['final_model_sha256'], 'evidence final model differs'
    return dict(gate='PASS',status='PASS',epochs_completed=epochs,optimizer_steps=steps,
                trainer_checkpoint_sha256=sha(cp),config_sha256=sha(ckdir/'config.yaml'),
                evidence_sha256=sha(ckdir/'training_evidence.json'),precision_plugin=type(plugin).__name__,**result)


def run(args):
    report=args.report.resolve(); assert not report.exists(), 'fresh report directory required'
    report.mkdir(parents=True)
    state=dict(stage='preflight',gate='UNVERIFIED',started_at=datetime.now(timezone.utc).isoformat(),exit_code=None)
    save(report/'state.json',state); exit_code=1
    with (report/'stdout.log').open('w') as stdout,(report/'stderr.log').open('w') as stderr:
        with redirect_stdout(stdout),redirect_stderr(stderr):
            try:
                code=args.code_root.resolve(); code=code/'lewm_pcd' if (code/'lewm_pcd').is_dir() else code
                sys.path.insert(0,str(code)); import train
                import torch
                assert torch.cuda.is_available(), 'real GPU is unavailable'
                versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','lightning','stable-pretraining','transformers','stable-worldmodel','h5py')}
                common=dict(protocol=PROTOCOL,source_hashes=source_hashes(code,train),versions=versions,
                            train_rgb_sha=sha(args.train_rgb),train_pcd_sha=sha(args.train_pcd),
                            active_plan_sha256=sha(args.active_plan),route=args.route,dataset=str(args.dataset.resolve()),
                            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),scope='diagnostic_gate_not_formal_training')
                state.update(stage=args.mode,**common); save(report/'state.json',state)
                result=gpu_gate(args,code,train,report) if args.mode=='gpu-gate' else verify_full(args,code,train)
                assert sha(args.train_rgb)==common['train_rgb_sha'] and sha(args.train_pcd)==common['train_pcd_sha']
                save(report/'report.json',dict(common,**result)); exit_code=0; state.update(stage='complete',gate='PASS')
            except BaseException as error:
                state.update(stage='failed',gate='FAIL',error=repr(error),finite=False)
                (report/'failure.log').write_text(traceback.format_exc(),encoding='utf-8'); traceback.print_exc()
            finally:
                state['exit_code']=exit_code; save(report/'state.json',state); (report/'exit_code').write_text(str(exit_code)+'\n')
    return exit_code


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('gpu-gate','verify-full'),required=True)
    p.add_argument('--route',choices=('rgb','pcd'),required=True)
    for name in ('code-root','dataset','report','train-rgb','train-pcd','active-plan'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-root',type=Path); p.add_argument('--weights',default='weights_epoch_10.pt')
    raise SystemExit(run(p.parse_args()))

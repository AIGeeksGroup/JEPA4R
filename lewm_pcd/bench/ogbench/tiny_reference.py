"""External passive recorder for the frozen old Tiny main and its real step."""
import argparse,ast,copy,hashlib,importlib.util,inspect,json,os,sys
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr

OLD_TINY_SHA256='7d6e15821eb2d5aa976b5cc5f86b2e258c7212dc36d3fcad9c8febf177ab8ff1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def step_markers(path):
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    pre=[n.lineno for n in ast.walk(main)if isinstance(n,ast.With)]
    post=[]
    for node in ast.walk(main):
        body=getattr(node,'body',None)
        if not isinstance(body,list):continue
        for index,item in enumerate(body[:-1]):
            call=item.value if isinstance(item,ast.Expr)else None
            if (isinstance(call,ast.Call)and isinstance(call.func,ast.Attribute)and
                    isinstance(call.func.value,ast.Name)and call.func.value.id=='scheduler'and call.func.attr=='step'):
                post.append(body[index+1].lineno)
    if len(pre)!=1 or len(post)!=1:raise ValueError('old native step markers are ambiguous')
    return pre[0],post[0]

def _cpu(value):
    if hasattr(value,'detach'):return value.detach().cpu().numpy()
    if isinstance(value,dict):return {k:_cpu(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(_cpu(v)for v in value)
    return value

class ReferenceBudgetComplete(Exception):pass

class PassiveTinyRecorder:
    def __init__(self,module,config,output,*,enforce_native=True):
        self.module,self.config,self.output=module,config,Path(output)
        self.source=Path(inspect.getfile(module.main)).resolve()
        if inspect.signature(module.main).parameters:raise ValueError('old recorder never calls new main(gate)')
        self.enforce_native=enforce_native
        self.actual_batch_size=256
        if config.get('numerical_fixture')is not None:
            helper=_module('_tiny_reference_cpu_fixture_validation',Path(__file__).with_name('tiny_cpu_fixture.py'))
            self.actual_batch_size=helper.validate_config(config)
        if enforce_native and sha(self.source)!=OLD_TINY_SHA256:raise ValueError('reference is not frozen f437 Tiny source')
        self.pre,self.post=step_markers(self.source)
        self.state=_module('_tiny_reference_training_state',Path(__file__).with_name('training_state.py'))
        self.compare=_module('_tiny_reference_array_compare',Path(__file__).parents[1]/'gates/arrays.py')
        self.entries=config['batches']
        if len(self.entries)<3 or config['stop_after']not in(1,2):raise ValueError('two updates plus explicit next batch are required')
        self.cursor=self.state.FixedBatchCursor([row['id']for row in self.entries])
        import numpy as np
        data=Path(config['batch_arrays']['path'])
        if sha(data)!=config['batch_arrays']['sha256']:raise ValueError('real augmented input SHA differs')
        self.data=np.load(data,allow_pickle=False)
        self.fetch_position=self.loader_epoch=0
        self.records=[];self.net=self.container=None

    def close(self):self.data.close()

    def loader_factory(self,original):
        owner=self
        def factory(dataset,*args,**kwargs):
            native=original(dataset,*args,**kwargs)
            if len(dataset)!=owner.config['loader_samples']or len(native)!=owner.config['loader_batches_per_epoch']:
                raise ValueError('fixed input stream changed the native loader/scheduler counts')
            owner.loader_metadata=dict(samples=len(dataset),batches=len(native),batch_size=native.batch_size,
                                       drop_last=native.drop_last,num_workers=native.num_workers)
            class Replay:
                def __len__(self):return len(native)
                def __getattr__(self,name):return getattr(native,name)
                def __iter__(self):
                    import numpy as np,torch
                    epoch=owner.loader_epoch;owner.loader_epoch+=1
                    while owner.fetch_position<len(owner.entries):
                        row=owner.entries[owner.fetch_position]
                        if row['epoch']!=epoch:break
                        value=np.array(owner.data[row['array_key']],copy=True)
                        if value.dtype!=np.float32 or(owner.enforce_native and value.shape!=(owner.actual_batch_size,4,3,224,224)):
                            raise ValueError('reference requires actual FP32 augmented native input, not features')
                        owner.fetch_position+=1
                        yield torch.from_numpy(value)
            return Replay()
        return factory

    def _save(self,path,value):
        with Path(path).open('x',encoding='utf8')as stream:json.dump(value,stream,indent=2,allow_nan=False)

    def _snapshot(self,path,progress):
        before=copy.deepcopy(self.state.capture_rng())
        self.state.save_torch_state(path,model=self.container,optimizers={'main':self.opt},
            schedulers={'main':self.scheduler},scaler=self.scaler,identity=self.config['identity'],cursor=self.cursor,progress=progress)
        if not self.compare.compare_nested(_cpu(self.state.capture_rng()),_cpu(before),exact=True)['passed']:
            raise ValueError('passive checkpoint recording altered the model RNG stream')

    def _before(self,local):
        import torch
        row=self.entries[self.cursor.position]
        if (row['epoch'],row['batch_index'])!=(local['epoch'],local['step']):
            raise ValueError('old main loop and frozen real batch event IDs differ')
        if self.container is None:
            self.net,self.opt,self.scheduler,self.scaler=local['net'],local['opt'],local['scheduler'],local['scaler']
            self.container=torch.nn.ModuleDict(dict(encoder=self.net,sigreg=local['sigreg']))
            if self.enforce_native:
                args=local['args']
                if(args.bs,args.V,args.epochs,args.lr,args.lamb,args.proj_dim,args.max_steps)!=(256,4,10,.002,.02,16,0):
                    raise ValueError('old native Tiny recipe differs from the guide reference')
            self._snapshot(self.output/'initial_state.pt',dict(step=0,epoch=row['epoch'],batch_index=row['batch_index']))
        self.before={name:p.detach().cpu().clone().numpy()for name,p in self.net.named_parameters()}

    def _after(self,local):
        import numpy as np
        consumed=self.cursor.take();count=self.cursor.position
        steps=sorted({int(v['step'])for v in self.opt.state.values()if 'step'in v})
        if steps!=[count]:raise ValueError('old optimizer did not perform the counted effective update')
        arrays={'output/'+key:local[key].detach().float().cpu().numpy()for key in
                ['emb','proj','inv_loss','sigreg_loss','loss']}
        for name,param in self.net.named_parameters():
            arrays['before/'+name]=self.before[name]
            arrays['after/'+name]=param.detach().cpu().numpy()
            if param.grad is not None:arrays['gradient/'+name]=param.grad.detach().float().cpu().numpy()
        if not all(np.isfinite(value).all()for value in arrays.values()):raise ValueError('old reference produced nonfinite arrays')
        if not any(np.any(arrays['after/'+name]!=value)for name,value in self.before.items()):
            raise ValueError('old reference has no actual parameter update')
        with(self.output/f'step_{count}_arrays.npz').open('xb')as stream:np.savez(stream,**arrays)
        next_row=self.entries[count]
        self._snapshot(self.output/f'step_{count}.pt',dict(step=count,epoch=next_row['epoch'],batch_index=next_row['batch_index']))
        record=dict(step=count,batch_id=consumed,next_batch_id=next_row['id'],epoch=local['epoch'],batch_index=local['step'],
                    actual_optimizer_steps=steps,lr_after_step=[p['lr']for p in self.opt.param_groups],
                    state_sha256=sha(self.output/f'step_{count}.pt'),arrays_sha256=sha(self.output/f'step_{count}_arrays.npz'))
        self.records.append(record);self._save(self.output/f'step_{count}.json',record)
        if count>=self.config['stop_after']:raise ReferenceBudgetComplete

    def _trace(self,frame,event,arg):
        if Path(frame.f_code.co_filename).resolve()!=self.source or frame.f_code.co_name!='main':return None
        if event=='line':
            if frame.f_lineno==self.pre:self._before(frame.f_locals)
            elif frame.f_lineno==self.post:self._after(frame.f_locals)
        return self._trace

    def run(self):
        self.output.mkdir(parents=True,exist_ok=False)
        if sys.gettrace()is not None:raise RuntimeError('passive reference requires a fresh untraced process')
        original=self.module.DataLoader;self.module.DataLoader=self.loader_factory(original)
        stopped=False
        try:
            sys.settrace(self._trace)
            self.module.main()  # Frozen original: no gate parameter, no new training loop.
        except ReferenceBudgetComplete:stopped=True
        finally:
            sys.settrace(None);self.module.DataLoader=original;self.close()
        if not stopped or self.cursor.position!=self.config['stop_after']:raise ValueError('reference budget did not really complete')
        result=dict(schema='old-tiny-passive-reference-v1',actual_exit_code=0,optimizer_updates=self.cursor.position,
            new_gate_interface_used=False,old_main_called=True,old_main_normal_ten_epoch_completion=False,
            original_main_source=dict(path=str(self.source),sha256=sha(self.source)),observer_source_sha256=sha(__file__),
            state_helper_sha256=sha(Path(__file__).with_name('training_state.py')),loader_metadata=self.loader_metadata,
            fixed_augmented_inputs=self.config['batch_arrays'],identity=self.config['identity'],records=self.records,
            next_batch_id=self.cursor.state_dict()['next_batch_id'],original_arithmetic_optimizer_scheduler_used=True,
            scope='fixed real augmented batch stream; arbitrary worker queue resume is not claimed')
        self._save(self.output/'RESULT.json',result)
        (self.output/'actual_exit_code').write_text('0\n')
        return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    cfg=json.loads(args.config.read_text())
    numerical=cfg.get('numerical_fixture')is not None
    if cfg.get('mode')!='smoke'or(cfg.get('precision')!='bf16'and not numerical)or cfg.get('model')!='lejepa':
        raise ValueError('old Tiny production reference is explicit smoke/BF16/LeJEPA only')
    print(json.dumps(dict(stage='old_tiny_passive_reference',executed=False,updates=cfg['stop_after'],source_root=cfg['source_root'])))
    if not args.execute:return 0
    source=Path(cfg['source_root']).resolve()
    if sha(source/'SOURCE_MANIFEST.json')!=cfg['source_manifest_sha256']:raise ValueError('f437 source manifest differs')
    existing=sys.modules.get('bench.ogbench')
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(source):
        raise RuntimeError('run the old observer as a direct script in a fresh process, not current-package -m')
    sys.path[:0]=[str(source/'lewm_pcd'),str(source/'lewm_pcd/third_party/stable-worldmodel'),str(source/'lewm_pcd/third_party/ogbench')]
    from bench.ogbench.training_runtime import load_tiny_trainer,build_environment
    original=load_tiny_trainer()
    if Path(original.__file__).resolve()!=source/'lewm_pcd/bench/ogbench/native/tiny_train.py':raise ValueError('old Tiny imported from another source tree')
    dataset=Path(cfg['dataset_binding']['path']).resolve()
    if sha(dataset)!=cfg['dataset_binding']['sha256']:raise ValueError('real original RGB view identity differs')
    os.environ.update(build_environment(dataset.parent,args.output.resolve()/'native'))
    values=[str(original.__file__),'--task','cube1_fixed_real_reference','--dataset',str(dataset),'--name','old_f437_reference',
            '--epochs','10','--bs','256','--V','4','--workers',str(cfg.get('workers',1))]
    previous=sys.argv;sys.argv=values
    try:
        if numerical:
            helper=_module('_tiny_reference_cpu_fixture',Path(__file__).with_name('tiny_cpu_fixture.py'))
            with helper.cpu_fp32(original,cfg)as metadata:
                PassiveTinyRecorder(original,cfg,args.output).run()
                write=args.output/'NUMERICAL_FIXTURE.json'
                with write.open('x',encoding='utf8')as stream:json.dump(metadata,stream,indent=2)
        else:PassiveTinyRecorder(original,cfg,args.output).run()
    finally:sys.argv=previous
    return 0

if __name__=='__main__':raise SystemExit(main())

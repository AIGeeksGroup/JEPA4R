"""Actual frozen EAI loop CPU unit fixtures; not real T3/T4 or control evaluation."""
import ast
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

LEWM=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(LEWM))
try:
    import numpy as np
    import torch
except ImportError:
    torch=None
try:
    from jepa4r.policy.bc.relocate import bc_recovery as recovery
except ImportError:
    recovery=None

FROZEN='fixture-eai-train-loop'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actual_module(source,dimensions,access,policies,logs,workers=None):
    """Compile original function/class AST; replace only non-training I/O factories."""
    from torch.utils.data import Dataset,DataLoader
    # Unit fixture bypasses only package-level environment registration.
    # Actual policy/network source files and all forward/optimizer math stay original.
    package=types.ModuleType('mjrl');package.__path__=[str(Path(os.environ['JEPA4R_MJRL_ROOT'])/'mjrl')]
    package.__file__=str(Path(package.__path__[0])/'__init__.py');sys.modules['mjrl']=package
    from mjrl.policies.gaussian_mlp import BatchNormMLP

    names = {
        'set_seed',
        'configure_cluster_GPUs',
        'bc_pvr_train_loop',
        'compute_metrics_from_paths',
        'FrozenEmbeddingDataset',
    }
    tree=ast.parse(source.read_text(),filename=str(source))
    selected=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    module=types.ModuleType('original_eai_unit_fixture');module.__file__=str(source)
    module.__dict__.update(
        np=np,
        torch=torch,
        os=os,
        pickle=pickle,
        gc=gc,
        Dataset=Dataset,
        DataLoader=DataLoader,
        GymEnv=object,
        tqdm=lambda values: values,
        tabulate=lambda values: str(values),
        _device_audit=lambda *a, **k: None,
        fuse_embeddings_flare=lambda values: values,
    )
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),module.__dict__)
    original_dataset=module.FrozenEmbeddingDataset
    class ObservedDataset(original_dataset):
        def __getitem__(self,index):
            access.append(int(index));return super().__getitem__(index)
    module.FrozenEmbeddingDataset=ObservedDataset
    def policy_factory(**kwargs):
        result=BatchNormMLP(**kwargs);policies.append(result);return result
    module.BatchNormMLP=policy_factory
    class Environment:
        def __init__(self):
            self.spec=types.SimpleNamespace(observation_dim=dimensions[0],action_dim=dimensions[1])
            self.horizon=200
            self.env = types.SimpleNamespace(
                embedding=torch.nn.Linear(1, 1), unwrapped=types.SimpleNamespace(evaluate_success=lambda paths: 40.0)
            )
    module.env_constructor=lambda **kwargs:Environment()
    module.compute_embeddings=lambda paths,**kwargs:paths
    module.precompute_features=lambda paths,**kwargs:paths
    class Logger:
        def log(self,data,**kwargs):logs.append(copy.deepcopy(dict(data=data,kwargs=kwargs)))
    def setup(config):random.random();return Logger()
    module.setup_wandb=setup
    def paths(policy,num_traj=1,**kwargs):
        counters=[int(v) for k,v in policy.model.state_dict().items() if k.endswith('num_batches_tracked')]
        count=max(counters);random.random();np.random.random();torch.rand(1)
        return [dict(rewards=np.array([100.-count]),env_infos={}) for _ in range(num_traj)]
    module.sample_paths=paths
    module.rollout_from_init_states=lambda states,env,policy,**kwargs:paths(policy,len(states))
    if workers is not None:
        class WrongWorkers(DataLoader):
            def __init__(self,*args,**kwargs):kwargs['num_workers']=workers;super().__init__(*args,**kwargs)
        module.DataLoader=WrongWorkers
    return module


def worker(spec_path):
    """Independent process executes one phase of the original BC function."""
    torch.set_num_threads(1)
    spec=json.loads(Path(spec_path).read_text());root=Path(spec['root']);root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,os.environ['JEPA4R_MJRL_ROOT'])
    dimensions=spec.get('dimensions',[606,30]);batch=spec.get('batch',128);epochs=spec.get('epochs',1)
    source=Path(spec.get('source',os.environ['JEPA4R_EAI_LOOP']))
    data=root/'data';data.mkdir();rng=np.random.default_rng(710)
    features=rng.normal(size=(200,dimensions[0])).astype(np.float32)
    actions=(2*rng.normal(size=(200,dimensions[1]))).astype(np.float32)
    demos = [
        dict(
            features=features,
            actions=actions,
            embeddings=np.zeros((200, 1), np.float32),
            rewards=np.zeros(200),
            init_state_dict={'fixture': True},
        )
    ]
    with (data/'fixture-v0.pickle').open('xb') as stream:pickle.dump(demos,stream,protocol=4)
    access,policies,logs=[],[],[]
    module=actual_module(source,dimensions,access,policies,logs,spec.get('workers'))
    original=module.bc_pvr_train_loop;original_loader=module.DataLoader;original_cwd=Path.cwd()
    config = dict(
        seed=100,
        data_dir=str(data),
        num_demos=1,
        device='cpu',
        epochs=epochs,
        job_name=str(root / 'job'),
        env_kwargs=dict(
            render_gpu_id=0,
            env_name='fixture-v0',
            suite='adroit',
            embedding_name='synthetic_unit_fixture',
            history_window=3,
            proprio_key='proprio',
        ),
        bc_kwargs=dict(batch_size=batch, lr=1e-3, hidden_sizes='(256,256,256)', nonlinearity='relu', dropout=0.0),
        eval_frequency=spec.get('eval_frequency', 100),
        save_frequency=100,
        eval_num_traj=1,
        num_cpu=1,
    )
    identity = dict(
        model=dict(kind='real-MJRL-BatchNormMLP', input=dimensions[0], output=dimensions[1]),
        data=dict(
            unit_fixture=True,
            features_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
            actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        ),
        config=dict(epochs=epochs, batch=batch, lr=1e-3, seed=100),
        split='synthetic-unit-only',
        action_dim=dimensions[1],
        encoder='synthetic-already-fused-features-no-real-encoder',
        source=FROZEN,
    )
    for key,value in spec.get('identity_changes',{}).items():identity[key]=value
    if 'lr_override' in spec:config['bc_kwargs']['lr']=spec['lr_override']
    captured={};controller=None;observations=[];status=0;error=None
    def profile(frame,event,arg):
        if event=='return' and frame.f_code.co_name=='bc_pvr_train_loop':captured.update(frame.f_locals)
    def observer(event,live):
        observations.append(copy.deepcopy(event))
        if 'policy' in live and policies:assert live['policy'] is policies[0]
        # Instrumentation RNG must not leak into the original loop.
        random.random();np.random.random();torch.rand(1)
    random.seed(192)
    previous_profile=sys.getprofile();sys.setprofile(profile)
    try:
        with recovery.install_bc_recovery(module,identity,root/'recovery',enabled=spec.get('enabled',True),
                resume_from=spec.get('resume'),pause_after_updates=spec.get('pause'),observer=observer) as controller:
            module.bc_pvr_train_loop(config)
    except recovery.RecoveryPause:status=75
    except ValueError as exception:status=2;error=str(exception)
    finally:
        sys.setprofile(previous_profile)
        restored=module.bc_pvr_train_loop is original and module.DataLoader is original_loader
        if spec.get('enabled',True):assert Path.cwd()==original_cwd
        os.chdir(original_cwd)
    from jepa4r.utils.training_state import capture_rng
    snapshot = dict(
        rng=capture_rng(),
        access=access,
        logs=logs,
        model=None,
        optimizer=None,
        records=[] if controller is None else controller.records,
        epoch_records=[] if controller is None else controller.epoch_records,
    )
    if 'policy' in captured:snapshot['model']=copy.deepcopy(captured['policy'].model.state_dict())
    if 'optimizer' in captured:snapshot['optimizer']=copy.deepcopy(captured['optimizer'].state_dict())
    checkpoint=None if controller is None or controller.last_checkpoint is None else str(controller.last_checkpoint)
    torch.save(snapshot,root/'snapshot.pt')
    best=root/'job/iterations/best_policy.pickle'
    result = dict(
        actual_exit_code=status,
        error=error,
        checkpoint=checkpoint,
        source_restored=restored,
        environment_registration_stubbed=True,
        policy_source_sha256=sha(Path(sys.modules['mjrl.policies.gaussian_mlp'].__file__)),
        cpu_threads=torch.get_num_threads(),
        policies_created=len(policies),
        gpu_initialized=torch.cuda.is_initialized(),
        fixture_only=True,
        best_policy_sha256=sha(best) if best.exists() else None,
        observer_kinds=[e['kind'] for e in observations],
    )
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');return status


@unittest.skipIf(torch is None,'Pinned PyTorch CPU runtime required')
class BCRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(recovery,'BC recovery hook is not implemented')
        if not os.environ.get('JEPA4R_EAI_LOOP') or not os.environ.get('JEPA4R_MJRL_ROOT'):
            self.skipTest('exact original EAI/MJRL source paths required')
        artifact=os.environ.get('BC_RECOVERY_ARTIFACT_ROOT')
        if artifact:
            self.root=Path(artifact)/self._testMethodName;self.root.mkdir(parents=True,exist_ok=False)
        else:
            temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def phase(self,name,expected=0,**options):
        root = self.root / name
        spec = dict(root=str(root), **options)
        path = self.root / (name + '.json')
        path.write_text(json.dumps(spec))
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES='',
            PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1',
        )
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(path)]
        process=subprocess.run(command,capture_output=True,text=True,env=env)
        (self.root/(name+'.log')).write_text(process.stdout+process.stderr,encoding='utf-8')
        self.assertEqual(process.returncode,expected,process.stdout+process.stderr)
        result=json.loads((root/'result.json').read_text());self.assertEqual(result['actual_exit_code'],expected)
        self.assertTrue(result['source_restored']);self.assertFalse(result['gpu_initialized'])
        return result,torch.load(root/'snapshot.pt',map_location='cpu',weights_only=False)

    def equal(self,left,right):
        if isinstance(left,torch.Tensor):self.assertTrue(torch.equal(left,right));return
        if isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right);return
        self.assertEqual(type(left),type(right))
        if isinstance(left,dict):
            self.assertEqual(set(left),set(right))
            for key in left:self.equal(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):self.equal(a,b)
        else:self.assertEqual(left,right)

    def compare_resume(self,dimensions,batch,epochs):
        opts=dict(dimensions=dimensions,batch=batch,epochs=epochs)
        _,plain=self.phase('plain',enabled=False,**opts)
        _,continuous=self.phase('continuous',**opts)
        for key in ('model','optimizer','rng','access','logs'):self.equal(plain[key],continuous[key])
        paused,prefix=self.phase('paused',expected=75,pause=1,**opts)
        self.assertIsNotNone(paused['checkpoint'])
        meta=json.loads(Path(paused['checkpoint']+'.json').read_text());self.assertEqual(meta['actual_exit_code'],0)
        resumed,suffix=self.phase('resumed',resume=paused['checkpoint'],**opts)
        for key in ('model','optimizer','rng'):self.equal(continuous[key],suffix[key])
        self.assertEqual(prefix['access']+suffix['access'],continuous['access'])
        self.equal(prefix['logs']+suffix['logs'],continuous['logs'])
        self.assertEqual([r['batch_size'] for r in suffix['records']],[128,72] if batch==128 else [200,200])
        self.assertIn('resume_loaded',resumed['observer_kinds'])

    def test_native_original_606_to30_128_plus72_independent_resume(self):self.compare_resume([606,30],128,1)

    def test_rh56_original_589_to12_two_epochs_one_batch_each(self):self.compare_resume([589,12],256,2)

    def test_identity_and_actual_recipe_reject_before_update(self):
        paused,_=self.phase('paused',expected=75,pause=1)
        for key in ('data','config','encoder','source'):
            result,state=self.phase('wrong_'+key,expected=2,resume=paused['checkpoint'],identity_changes={key:'wrong'})
            self.assertEqual(result['policies_created'],0)
            self.assertEqual(state['access'],[])
        result,state=self.phase('wrong_recipe',expected=2,resume=paused['checkpoint'],lr_override=2e-3)
        self.assertEqual(state['access'],[])

    def test_unsupported_worker_rejected(self):
        _,state=self.phase('wrong_workers',expected=2,workers=1);self.assertEqual(state['access'],[])

    def test_evaluated_best_policy_history_survives_epoch_boundary(self):
        opts=dict(dimensions=[589,12],batch=256,epochs=3,eval_frequency=1)
        _,continuous=self.phase('continuous',**opts)
        paused,prefix=self.phase('paused',expected=75,pause=2,**opts)
        resumed,suffix=self.phase('resumed',resume=paused['checkpoint'],**opts)
        for key in ('model','optimizer','rng'):self.equal(continuous[key],suffix[key])
        self.equal(prefix['logs']+suffix['logs'],continuous['logs'])
        self.assertEqual(paused['best_policy_sha256'],resumed['best_policy_sha256'])

    def test_completed_budget_checkpoint_is_not_a_fresh_random_head(self):
        finished,_=self.phase('finished')
        result,state=self.phase('completed_resume',expected=2,resume=finished['checkpoint'])
        self.assertEqual(result['policies_created'],0)
        self.assertEqual(state['access'],[])

    def test_main_payload_progress_must_match_the_sidecar(self):
        paused,_=self.phase('paused',expected=75,pause=1)
        original=Path(paused['checkpoint']);copied=self.root/'tampered';copied.mkdir()
        for path in original.parent.iterdir():
            if path.is_file():shutil.copyfile(path,copied/path.name)
        target=copied/original.name;sidecar=target.with_suffix(target.suffix+'.json')
        value=json.loads(sidecar.read_text());value['progress']['next_mb_idx']=0
        sidecar.write_text(json.dumps(value))
        _,state=self.phase('wrong_progress',expected=2,resume=str(target))
        self.assertEqual(state['access'],[])

    def guard_identity(self):
        return dict(model='fixture-head',data='fixture-data',config='fixture-config',
                    split='synthetic-unit-only',action_dim=30,encoder='fixture-encoder',source=FROZEN)

    def test_live_co_names_mutation_rejected_with_unchanged_code_constants_and_file(self):
        sys.path.insert(0,os.environ['JEPA4R_MJRL_ROOT'])
        source=Path(os.environ['JEPA4R_EAI_LOOP']);before_file=sha(source)
        official=actual_module(source,[606,30],[],[],[])
        original=official.bc_pvr_train_loop;code=original.__code__
        self.assertIn('MSELoss',code.co_names)
        names=tuple('L1Loss' if name=='MSELoss' else name for name in code.co_names)
        changed = types.FunctionType(
            code.replace(co_names=names),
            original.__globals__,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        official.bc_pvr_train_loop=changed
        self.assertEqual(changed.__code__.co_code,code.co_code)
        self.assertEqual(changed.__code__.co_consts,code.co_consts)
        self.assertEqual(changed.__code__.co_filename,code.co_filename)
        self.assertIs(changed.__globals__,original.__globals__)
        self.assertEqual(sha(source),before_file)
        with self.assertRaisesRegex(ValueError,'bytecode'):
            with recovery.install_bc_recovery(official,self.guard_identity(),self.root/'rejected',enabled=True):pass
        self.assertIs(official.bc_pvr_train_loop,changed)
        self.assertFalse((self.root/'rejected').exists())

    def test_frozen_original_function_still_installs_and_restores_without_training(self):
        sys.path.insert(0,os.environ['JEPA4R_MJRL_ROOT'])
        source=Path(os.environ['JEPA4R_EAI_LOOP']);before_file=sha(source)
        official=actual_module(source,[606,30],[],[],[])
        original,loader=official.bc_pvr_train_loop,official.DataLoader
        with recovery.install_bc_recovery(official,self.guard_identity(),self.root/'installed',enabled=True) as control:
            self.assertIsNot(official.bc_pvr_train_loop,original)
            self.assertFalse(control.started);self.assertEqual(control.records,[])
        self.assertIs(official.bc_pvr_train_loop,original);self.assertIs(official.DataLoader,loader)
        self.assertEqual(sha(source),before_file)
        self.assertFalse(torch.cuda.is_initialized())

    def test_disabled_does_not_inspect_or_patch_official_function(self):
        original = lambda config: None
        loader = object()
        official = types.SimpleNamespace(bc_pvr_train_loop=original, DataLoader=loader)
        with recovery.install_bc_recovery(official,{},None,enabled=False) as controller:
            self.assertIsNone(controller)
            self.assertIs(official.bc_pvr_train_loop, original)
            self.assertIs(official.DataLoader, loader)


class RecipeConfigTests(unittest.TestCase):
    def config(self):
        return dict(
            seed=100,
            epochs=1,
            num_demos=1,
            device='cuda',
            eval_frequency=5,
            save_frequency=10,
            eval_num_traj=25,
            num_cpu=1,
            embedding='actual-model',
            bc_kwargs=dict(hidden_sizes='(256,256,256)', batch_size=128, lr=0.001, dropout=0),
            env_kwargs=dict(suite='adroit', history_window=3, proprio_key='proprio', embedding_name='${embedding}'),
        )

    def test_real_DictConfig_and_plain_dict_have_same_recipe_and_hash(self):
        from omegaconf import OmegaConf
        config=OmegaConf.create(self.config());before=OmegaConf.to_container(config,resolve=False)
        expected=OmegaConf.to_container(config,resolve=True)
        actual=recovery._recipe(config);plain=recovery._recipe(expected)
        self.assertEqual(actual,plain)
        self.assertEqual(hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest(),
                         hashlib.sha256(json.dumps(plain,sort_keys=True).encode()).hexdigest())
        self.assertEqual(actual['environment']['embedding_name'],'actual-model')
        self.assertEqual(OmegaConf.to_container(config,resolve=False),before)

    def test_actual_bc_kwargs_change_rejects_before_any_update(self):
        from omegaconf import OmegaConf
        controller=object.__new__(recovery.BCRecovery)
        controller.started=False;controller.progress=None
        controller.recipe=recovery._recipe(OmegaConf.to_container(OmegaConf.create(self.config()),resolve=True))
        changed=OmegaConf.create(self.config());changed.bc_kwargs.lr=.002
        with self.assertRaisesRegex(ValueError,'actual BC recipe differs'):
            controller.begin(changed)

    def test_actual_frozen_function_receives_original_DictConfig_object(self):
        from omegaconf import OmegaConf
        if not os.environ.get('JEPA4R_EAI_LOOP') or not os.environ.get('JEPA4R_MJRL_ROOT'):
            self.skipTest('exact original EAI/MJRL source paths required')
        source=Path(os.environ['JEPA4R_EAI_LOOP']);before_source=sha(source)
        policies=[];official=actual_module(source,[606,30],[],policies,[])
        original,loader=official.bc_pvr_train_loop,official.DataLoader
        config=OmegaConf.create(self.config());unresolved=OmegaConf.to_container(config,resolve=False)
        identity=dict(model='fixture-head',data='fixture-data',config='fixture-config',
                      split='synthetic-unit-only',action_dim=30,encoder='fixture-encoder',source=FROZEN)
        class StopAfterRecipe(Exception):pass
        received=[]
        with tempfile.TemporaryDirectory() as directory:
            with recovery.install_bc_recovery(official,identity,Path(directory)/'hook',enabled=True) as controller:
                begin=controller.begin
                def observe(value):
                    received.append(value);begin(value);raise StopAfterRecipe()
                controller.begin=observe
                with self.assertRaises(StopAfterRecipe):official.bc_pvr_train_loop(config)
        self.assertIs(received[0],config)
        self.assertEqual(OmegaConf.to_container(config,resolve=False),unresolved)
        self.assertEqual(policies,[])
        self.assertIs(official.bc_pvr_train_loop,original);self.assertIs(official.DataLoader,loader)
        self.assertEqual(sha(source),before_source)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':raise SystemExit(worker(sys.argv[2]))
    unittest.main()

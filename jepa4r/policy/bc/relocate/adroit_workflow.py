"""Native Adroit Relocate stage connections; planning never executes a process."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULES = {'adroit_collect': 'jepa4r.data.relocate.adroit_collect',   # stage modules outside this package
           'adroit_data': 'jepa4r.data.relocate.adroit_data'}


def plan(args, assets):
    output = Path(args.output).resolve()
    smoke = args.mode == 'smoke'
    model = args.model
    if model not in ('jepa4r','lewm','pldm','lejepa','visreg','vjepa'):
        raise ValueError('unsupported native Relocate model: '+model)
    common={str(key):str(value) for key,value in assets.get('environment',{}).items()}
    for key,value in {'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl',
        'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}.items():common.setdefault(key,value)
    paths=[ROOT.parent,ROOT/'policy/bc',assets.get('stable_worldmodel_root',''),
        assets.get('mjrl_root',''),assets.get('mj_envs_root','')]
    if assets.get('eai_vc_root'):
        eai = Path(assets['eai_vc_root'])
        paths.extend(
            [eai / 'cortexbench/mujoco_vc/src', eai / 'vc_models/src', eai / 'cortexbench', eai / 'vc_models', eai]
        )
    paths.extend(assets.get('pythonpath',[]))
    paths.extend(common.get('PYTHONPATH','').split(os.pathsep))
    paths.extend(os.environ.get('PYTHONPATH','').split(os.pathsep))
    unique=[];seen=set()
    for path in paths:
        value=str(path)
        if not value:continue
        key=os.path.normcase(os.path.normpath(value))
        if key not in seen:seen.add(key);unique.append(value)
    common['PYTHONPATH']=os.pathsep.join(unique)
    sampling=ROOT.parent/'configs/env/relocate/adroit_relocate_sampling.json'
    common.update(ADROIT_FRESH_EVAL='1',ADROIT_PCD_CONFIG=str(sampling),
        ADROIT_PCD_CONFIG_SHA256=hashlib.sha256(sampling.read_bytes()).hexdigest())
    for key, env in (('eai_vc_root','CORTEXBENCH_EAI_VC_ROOT'),
                     ('mujoco_path','MUJOCO_PY_MUJOCO_PATH'),('mujoco_key','MUJOCO_PY_MJKEY_PATH')):
        if assets.get(key): common[env]=str(assets[key])
    if assets.get('mujoco_path'):
        common.setdefault(
            'LD_LIBRARY_PATH',
            str(Path(assets['mujoco_path']) / 'bin') + os.pathsep + os.environ.get('LD_LIBRARY_PATH', ''),
        )
    if assets.get('wm_extra_site'): common['ADROIT_WM_EXTRA_SITE']=str(assets['wm_extra_site'])
    stages=[]
    def add(name, python, module, arguments, receipt, **expected):
        stages.append(
            {
                'name': name,
                'command': [
                    str(python),
                    '-B',
                    '-m',
                    MODULES.get(module, 'jepa4r.policy.bc.relocate.' + module),
                    *map(str, arguments),
                ],
                'cwd': str(ROOT.parent),
                'environment': dict(common),
                'receipt': str(receipt),
                'expected': expected,
            }
        )
    data=output/'data'; train=2 if smoke else 10000; test=1 if smoke else 150
    add(
        'collect',
        args.sim_python,
        'adroit_collect',
        [
            '--output',
            data,
            '--expert',
            assets.get('adroit_expert', 'REQUIRED_ADROIT_EXPERT'),
            '--train-episodes',
            train,
            '--test-episodes',
            test,
            '--seed',
            args.seed,
        ],
        data / 'collection.json',
        train_episodes=train,
        test_episodes=test,
    )
    add('data-audit',args.sim_python,'adroit_data',['audit','--data-root',data,
        '--train-episodes',train,'--test-episodes',test],data/'audit.json',train_episodes=train,test_episodes=test)
    checkpoint=output/'wm/checkpoints/relocate'; epoch=1 if smoke else 10
    if model in ('lewm','jepa4r','pldm','lejepa'):
        add('wm-train',args.wm_python,'adroit_learning',['train','--model',model,'--data-root',data,
            '--output',output/'wm','--mode',args.mode],output/'wm/train.json',minimum_updates=2)
        add('wm-reload',args.wm_python,'adroit_learning',['reload-wm','--model',model,
            '--data-root',data,'--output',output/'wm','--epoch',epoch],output/'wm/reload.json',strict_reload=True)
    else:
        checkpoint=Path(assets.get(model+'_checkpoint','REQUIRED_'+model.upper()+'_CHECKPOINT'))
    official=output/'official'
    add('official-demos',args.sim_python,'adroit_data',['official','--source',
        assets.get('adroit_official_demos','REQUIRED_OFFICIAL_DEMOS'),'--source-sha256',
        assets.get('adroit_official_sha256','REQUIRED_OFFICIAL_SOURCE_SHA256'),'--output',official,
        '--count',1 if smoke else 100],official/'report.json',demos=1 if smoke else 100,validated_source_count=100)
    for seed in ((100,) if smoke else (100,200,300)):
        bc=output/('bc-seed'+str(seed))
        add(
            'bc-seed' + str(seed),
            args.sim_python,
            'adroit_learning',
            [
                'bc',
                '--model',
                model,
                '--checkpoint',
                checkpoint,
                '--epoch',
                epoch,
                '--output',
                bc,
                '--',
                '--config-name',
                'Adroit_BC_config',
                'env=relocate-v0',
                'seed=' + str(seed),
                'data_dir=' + str(official / ('pcd' if model == 'jepa4r' else 'rgb')),
                'num_demos=' + str(1 if smoke else 100),
                'epochs=' + str(1 if smoke else 100),
                'eval_num_traj=' + str(1 if smoke else 25),
                'eval_frequency=5',
                'save_frequency=10',
                'num_cpu=1',
                'bc_kwargs.batch_size=' + str(128 if smoke else 256),
                'hydra.run.dir=' + str(bc / 'run'),
                'hydra.job.chdir=true',
                'job_name=policy',
                'wandb.project=local',
                'wandb.entity=local',
            ],
            bc / 'report.json',
            epochs=1 if smoke else 100,
            rollouts=1 if smoke else 500,
        )
        add('policy-reload-seed'+str(seed),args.sim_python,'adroit_learning',['reload-policy',
            '--model',model,'--output',bc],bc/'reload.json',strict_reload=True)
    if assets.get('policy_evidence',False):
        for stage in stages:
            if stage['name'].startswith(('bc-seed','policy-reload-seed')):
                command=stage['command']
                command.insert(command.index('--') if '--' in command else len(command),'--policy-evidence')
    if assets.get('stable_worldmodel_root'):
        for s in stages: s['environment']['ADROIT_FRAMEWORK_ROOT']=str(assets['stable_worldmodel_root'])
    for key in ('visreg_checkpoint','vjepa_checkpoint','vjepa_root'):
        if assets.get(key):
            for s in stages: s['environment']['ADROIT_'+key.upper()]=str(assets[key])
    paired=[data/(split+'_'+route+'.h5') for split in ('train','test') for route in ('rgb','pcd')]
    train_input=data/('train_pcd.h5' if model=='jepa4r' else 'train_rgb.h5')
    weight=checkpoint/f'weights_epoch_{epoch}.pt' if model in ('jepa4r','lewm','pldm','lejepa') else checkpoint
    checkpoint_files=[checkpoint/'config.json',weight]
    if model!='lejepa':checkpoint_files.append(checkpoint/'config.yaml')
    source=Path(assets.get('adroit_official_demos','REQUIRED_OFFICIAL_DEMOS')).expanduser()
    if not source.is_absolute():source=ROOT.parent/source
    if source.is_dir():
        binding=source.parent/'manifest.json';metadata=json.loads(binding.read_text(encoding='utf-8'))
        official_inputs = [
            binding,
            *[(source.parent / record['relative_path']).resolve() for record in metadata['demo_files']],
        ]
    else:official_inputs=[source]
    dependencies=ROOT.parent/'configs/policy/bc/relocate/DEPENDENCIES.json'
    for stage in stages:
        name=stage['name']
        if name=='collect':
            required=paired+[data/('initial_states_'+split+'.npz') for split in ('train','test')]+[
                data/(split+'_nonvisual.h5') for split in ('train','test')]+[
                Path(assets.get('adroit_expert','REQUIRED_ADROIT_EXPERT')),dependencies,sampling]
        elif name=='data-audit':required=paired
        elif name in ('wm-train','wm-reload'):
            required=checkpoint_files+[train_input,data/'audit.json']
            if name=='wm-reload':required=required+[output/'wm/train.json']
        elif name=='official-demos':
            required = (
                [official / route / 'relocate-v0.pickle' for route in ('rgb', 'pcd')]
                + [official / 'content_validation.json', ROOT / 'data/relocate/official_contract.py']
                + official_inputs
            )
        elif name.startswith('bc-seed'):
            bc=output/name
            required=[bc/'run/metrics.jsonl',bc/'run/policy/iterations/best_policy.pickle',
                official/('pcd' if model=='jepa4r' else 'rgb')/'relocate-v0.pickle',official/'report.json',weight]
            if model in ('jepa4r','lewm','pldm','lejepa'):required.append(checkpoint/'config.json')
            else:required.append(dependencies)
        else:
            bc=output/('bc-'+name.replace('policy-reload-',''))
            required=[bc/'run/policy/iterations/best_policy.pickle',bc/'report.json']
        stage['required_artifacts']=[str(path) for path in required]
    return stages


def verify_stage(stage,args,assets):
    """Require semantic evidence plus hashes of the actual stage artifacts."""
    path=Path(stage['receipt'])
    try:
        report=json.loads(path.read_text(encoding='utf-8'))
        if report.get('passed') is not True or report.get('actual_exit_code') != 0:
            raise ValueError('stage did not pass its content checks')
        for key,value in stage.get('expected',{}).items():
            if key=='minimum_updates':
                if report.get('optimizer_updates',0)<value: raise ValueError('no actual optimizer updates')
            elif report.get(key)!=value: raise ValueError('receipt contract mismatch: '+key)
        records=report.get('artifacts',[])
        if not records: raise ValueError('stage has no bound artifacts')
        def resolved(value):
            value=Path(value).expanduser()
            return (value if value.is_absolute() else Path(stage['cwd'])/value).resolve()
        required=stage.get('required_artifacts')
        if not required:raise ValueError('stage has no required artifact contract')
        covered={resolved(record['path']) for record in records}
        missing={resolved(value) for value in required}-covered
        if missing:raise ValueError('receipt omitted required artifacts: '+','.join(map(str,sorted(missing))))
        for record in records:
            artifact=resolved(record['path']); digest=hashlib.sha256()
            with artifact.open('rb') as stream:
                for block in iter(lambda:stream.read(1<<20),b''): digest.update(block)
            if artifact.stat().st_size!=record['bytes'] or digest.hexdigest()!=record['sha256']:
                raise ValueError('artifact changed: '+str(artifact))
        return {'passed':True,'details':report}
    except (OSError,ValueError,KeyError,TypeError) as error:
        return {'passed':False,'details':str(error)}

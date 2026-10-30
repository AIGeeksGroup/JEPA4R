"""Stdlib-only, single-owner PCD closure -> prepared G12 -> original runner."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
DEFAULT_Q=ROOT/'cube4_visual_gciql_v1'
PCD_PHASES=('00_wait_G11_acceptance','01_reference','02_fresh','03_GPU_preprocess','04_profile','05_pilot','06_full','07_audit')
PINS={'run_role.py':'266beaac57b529effc5bcdb33ff7fe7a7a7fb5464067450e67c447713212395a',
      'run_G11.sh':'48868a0631df4fd374c09bb389904a1f096c0e12f9f4aaf280387b9d029789d3',
      'G11/launch_plan.json':'8dfc75e9d8f31c45c514bef608c7e72366f3a72e6772fe921aafc379fd9a47ea'}
PREPARE_SHA='cd76e395ed8ec4fa8020ba06ae43ec684b63f63fe37ad3acb6f48a0f13ea4c15'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def write_new(path,value):
    data=value if isinstance(value,str) else json.dumps(value,indent=2)+'\n'
    with path.open('x') as stream:stream.write(data)


def cache_ready(q,pcd):
    exits=[pcd/'runtime_v1/outer_exit_code',q/'control_v1/PCD/wrapper_exit_code']
    for path in exits:
        if path.exists() and path.read_text().strip()!='0':raise RuntimeError('PCD failed; no automatic fallback: '+str(path))
    status=pcd/'runtime_v1/status.json'
    if status.exists() and read(status).get('actual_exit_code') not in (None,0):raise RuntimeError('PCD runtime failed')
    if not all(path.exists() for path in exits):return False
    state=read(status);assert state['actual_exit_code']==0 and state['stage']=='all_embedding_phases_completed'
    closed=read(pcd/'runtime_v1/closure.json');assert closed['actual_exit_code']==0 and closed['completed_phases']==8
    for name in PCD_PHASES:
        value=read(pcd/'runtime_v1'/f'{name}_exit.json')
        assert value['actual_exit_code']==0 and value['residual_owned_members']==[]
    return True


def activate(q,pcd,runtime,invoke=subprocess.run):
    assert cache_ready(q,pcd),'PCD not closed: do not prepare/launch G12'
    control=q/'control_v1';functional=q/'functional_v1'
    if (control/'G12').exists() or (control/'run_G12.sh').exists():raise FileExistsError('Existing G12 namespace is preserved')
    for name,expected in PINS.items():assert sha(control/name)==expected,name
    assert sha(functional/'prepare_G12.py')==PREPARE_SHA
    def phase(name,argv,env=None):
        folder=runtime/name;folder.mkdir()
        write_new(folder/'actual_argv.json',dict(argv=argv,cwd=str(q.parent),started=time.time(),environment=env))
        with (folder/'run.log').open('x') as stream:
            result=invoke(argv,cwd=q.parent,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=False)
        write_new(folder/'actual_exit_code',str(result.returncode)+'\n')
        write_new(folder/'exit.json',dict(actual_exit_code=result.returncode,completed=time.time()))
        assert result.returncode==0,name
    save(runtime/'status.json',dict(stage='preparing_G12_config',actual_exit_code=None,updated=time.time()))
    phase('01_prepare_G12',[str(q.parent/'env_runtime/bin/python'),'-B',str(functional/'prepare_G12.py'),'--q',str(q),'--pcd-case',str(pcd)],
          dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu'))
    prepared_path=functional/'G12_CONFIG_PREPARED_v2.json';prepared=read(prepared_path)
    config=functional/'configs/G12_cube4_pcd.v2.json';cfg=read(config)
    assert prepared['actual_exit_code']==0 and Path(prepared['config_path']).resolve()==config
    assert prepared['config_sha256']==sha(config) and cfg['group_id']=='G12' and cfg['modality']=='pcd' and cfg['executable'] is True
    g11=read(functional/'configs/G11_cube4_rgb.json')
    for key in ('agent_config','reward','input_contract','batch_size','total_updates','checkpoint_steps'):
        assert cfg[key]==g11[key],key
    assert cfg['output']==str(q/'runs/G12_cube4_pcd')
    assert all('source_rebind' not in arguments for arguments in cfg['feature_loader_arguments'].values())
    role=control/'G12';role.mkdir(exist_ok=False)
    original=read(control/'G11/launch_plan.json');plan=copy.deepcopy(original)
    assert plan['groups']==['G11'] and len(plan['phases'])==4
    plan.update(groups=['G12'],G12_executable=True,G12_waits_actual_PCD_cache=False)
    def move(value):
        if isinstance(value,str):return value.replace(DEFAULT_Q.as_posix(),str(q)).replace('G11','G12').replace('cube4_rgb','cube4_pcd')
        if isinstance(value,list):return [move(item) for item in value]
        if isinstance(value,dict):return {key:move(item) for key,item in value.items()}
        return value
    plan['phases']=move(plan['phases'])
    for index,stage in enumerate(plan['phases']):
        assert stage['group_id']=='G12' and stage['action']==('gate','reload','train','accept')[index]
        stage['argv'][stage['argv'].index('--config')+1]=str(config)
        assert 'TEMPLATE' not in ' '.join(stage['argv'])
    text=(control/'run_G11.sh').read_text()
    before='RUN="$CASE/control_v1/G11"';assert text.count(before)==1
    script=text.replace(before,'RUN="$CASE/control_v1/G12"')
    baseline='ROOT='+ROOT.as_posix();assert script.count(baseline)==1
    script=script.replace(baseline,'ROOT='+shlex.quote(str(q.parent)))
    shell=control/'run_G12.sh';write_new(shell,script)
    plan['source_pins'].update({str(config):sha(config),str(prepared_path):sha(prepared_path),str(shell):sha(shell),str(functional/'prepare_G12.py'):PREPARE_SHA})
    plan['source_pins'].update({path:item['sha256'] for path,item in prepared['metadata_input_files'].items()})
    plan['activation_lineage']=dict(prepared_config=str(prepared_path),prepared_config_sha256=sha(prepared_path),
        PCD_closed_before_prepare=True,PCD_closure_path=str(pcd/'runtime_v1/closure.json'),
        PCD_closure_sha256=sha(pcd/'runtime_v1/closure.json'),parent_G12_waits_actual_PCD_cache=original.get('G12_waits_actual_PCD_cache'))
    write_new(role/'launch_plan.json',plan)
    save(runtime/'status.json',dict(stage='G12_original_runner',config_sha256=sha(config),launch_plan_sha256=sha(role/'launch_plan.json'),actual_exit_code=None,updated=time.time()))
    phase('02_run_G12',['bash',str(shell)])
    assert (role/'wrapper_exit_code').read_text().strip()=='0' and (role/'runtime/outer_exit_code').read_text().strip()=='0'
    result=read(role/'runtime/status.json')
    assert result['actual_exit_code']==0 and result['completed_phases']==4 and result['trained_groups']==result['accepted_groups']==1
    return dict(actual_exit_code=0,completed=time.time(),config_sha256=sha(config),launch_plan_sha256=sha(role/'launch_plan.json'),G12_four_phases_accepted=True)


def run(q,pcd,poll_seconds=30,invoke=subprocess.run):
    q,pcd=Path(q).resolve(),Path(pcd).resolve()
    assert q.name=='cube4_visual_gciql_v1' and pcd.name=='pcd_training_camera_v1'
    runtime=q/'control_v1/after_cache_G12_runtime_v1'
    try:runtime.mkdir(exist_ok=False)
    except FileExistsError:
        write_new(q/'control_v1'/f'after_cache_G12_refused_{time.time_ns()}_{os.getpid()}.json',
            dict(stage='duplicate_runtime_refused',actual_exit_code=1,runtime=str(runtime),old_runtime_not_modified=True,at=time.time()))
        raise
    write_new(runtime/'owner.json',dict(pid=os.getpid(),started=time.time(),argv=__import__('sys').argv,single_instance_directory_lock=True))
    code=1
    try:
        while not cache_ready(q,pcd):
            save(runtime/'status.json',dict(stage='waiting_PCD_actual_closure',actual_exit_code=None,updated=time.time(),NN_imported=False))
            time.sleep(poll_seconds)
        result=activate(q,pcd,runtime,invoke)
        save(runtime/'status.json',dict(stage='complete',**result));code=0
        return result
    except BaseException:
        save(runtime/'status.json',dict(stage='failed',actual_exit_code=1,error=traceback.format_exc(),updated=time.time(),automatic_restart=False))
        raise
    finally:write_new(runtime/'outer_exit_code',str(code)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--q',type=Path,default=DEFAULT_Q)
    parser.add_argument('--pcd-case',type=Path,default=ROOT/'cube4_official_embeddings_v1/pcd_training_camera_v1')
    args=parser.parse_args();run(args.q,args.pcd_case)

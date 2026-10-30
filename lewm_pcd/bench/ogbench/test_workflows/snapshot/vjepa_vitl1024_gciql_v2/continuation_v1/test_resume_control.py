"""Continuation metadata only; no subprocess, model, remote access or old writes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class ResumeTests(unittest.TestCase):
    def setUp(self):
        path=HERE/'resume_control.py';self.api=None
        if path.exists():
            spec=importlib.util.spec_from_file_location('_resume_test',path)
            self.api=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.api)

    def fixture(self):
        self.assertIsNotNone(self.api,'Missing continuation metadata entry')
        scratch=HERE/'scratch';scratch.mkdir(exist_ok=True)
        root=Path(tempfile.mkdtemp(dir=scratch,prefix='resume_'))
        case=root/'case_v2';control=case/'continuation_v1'
        patcher=patch.multiple(self.api,ROOT=root,CASE=case,CONTROL=control)
        patcher.start();self.addCleanup(patcher.stop)
        case.mkdir();control.mkdir()
        source=HERE.parent/'release_v2/run.py'
        if not source.exists():source=HERE.parent/'run.py'
        (case/'run.py').write_bytes(source.read_bytes())
        frozen=root/'frozen.py';frozen.write_text('# fixture\n')
        config_paths=[]
        for purpose in ('pilot','formal'):
            for n in (1,2,3,4):
                g=f'vjepa21_vitl1024_cube{n}';p=case/'configs'/purpose/(g+'.json')
                cfg=dict(group_id=g,purpose=purpose,input_contract=dict(current_dim=1043,goal_dim=1024+3*n),execution_root=str(case/'pilot' if purpose=='pilot' else case),durability=dict(group_id=g,pilot_output=str(case/'pilot/gates'/g),formal_output=str(case/'runs'/g),ack_root=str(case/'control/backup_acks'),failure_root=str(case/'control/backup_failures')))
                write(p,cfg);config_paths.append(str(p))
        receiver=dict(case_root=str(control),config_paths=config_paths,remote_config=str(control/'backup_config.json'),remote_script=str(case/'backup_watch.py'),local_root='D:/unchanged-fixture')
        write(control/'backup_config.json',receiver)
        original_receiver=dict(receiver,case_root=str(case),remote_config=str(case/'control/backup_config.json'))
        write(case/'control/backup_config.json',original_receiver)
        plans={}
        for mode,labels in [('smoke',self.api.SMOKE_LABELS),('full',self.api.FULL_LABELS)]:
            phases=[]
            for n in (1,2,3,4):
                g=f'vjepa21_vitl1024_cube{n}';cfg=case/'configs'/('pilot' if mode=='smoke' else 'formal')/(g+'.json')
                for i,label in enumerate(labels,1):
                    argv=['python','-B',str(case/'backup_gate.py') if label=='backup_gate' else '/frozen/functional.py','--config',str(cfg)]
                    if label=='backup_gate':argv+=['--report',str(case/'control'/mode/'durability_reports'/(g+'.json'))]
                    phases.append(dict(name=f'{g}_{i:02d}_{label}',argv=argv,cwd=str(root),environment={}))
            plan=dict(queue='vj_downstream',mode=mode,phases=phases,source_pins={str(frozen):ref(frozen)['sha256']},training_updates=0 if mode=='smoke' else 4000000,SR_episodes=0 if mode=='smoke' else 1000,families=['vjepa21_vitl1024'],cube_counts=[1,2,3,4],expected_features=11011000)
            path=case/'control'/mode/'launch_plan.json';write(path,plan);plans[mode]=plan
        oldrun=case/'control/smoke/runtime';write(oldrun/'failure.json',dict(actual_exit_code=1));write(oldrun/'status.json',dict(stage='failed',actual_exit_code=1))
        (oldrun/'outer_exit_code').write_text('1\n');(oldrun.parent/'wrapper_exit_code').write_text('1\n')
        phases=[]
        for step in plans['smoke']['phases'][:2]:
            name=step['name'];exitp=oldrun/(name+'_exit.json');argvp=oldrun/(name+'_argv.json');logp=oldrun/(name+'.log')
            write(exitp,dict(actual_exit_code=0,residual_owned_members=[]));write(argvp,dict(argv=step['argv'],cwd=step['cwd']));logp.write_text('fixture successful phase\n')
            phases.append(dict(name=name,exit_file=ref(exitp),argv_file=ref(argvp),log_file=ref(logp)))
        write(oldrun/(plans['smoke']['phases'][2]['name']+'_exit.json'),dict(actual_exit_code=1,residual_owned_members=[]))
        cfg=Path(config_paths[0]);g='vjepa21_vitl1024_cube1';report=case/'pilot/gates'/g/'report.json'
        write(report,dict(actual_exit_code=0,actual_backend='gpu',updates=2,batch_size=1024,network_step=3,adam_count=2,encoder_calls=0,WM_dynamics_calls=0,saved_state={'leaves':153},first_layer_kernels={'v':[2070,512],'q':[2,2075,512]}))
        cps=[]
        for counter in (1,2):
            manifest=report.parent/'checkpoints'/f'update_{counter:09d}/manifest.json';write(manifest,dict(counter=counter,agent_sha256='a'*64,sidecar_sha256='b'*64))
            ack=case/'control/backup_acks/pilot'/g/f'update_{counter:09d}.json'
            write(ack,dict(actual_exit_code=0,counter=counter,stage='pilot',group_id=g,config_sha256=ref(cfg)['sha256'],manifest_sha256=ref(manifest)['sha256'],local_receipt_sha256='d'*64,scp_exit_codes=[0,0,0,0]))
            cps.append(dict(counter=counter,manifest=ref(manifest),ack=ref(ack),local_receipt_sha256='d'*64))
        adoption=dict(schema_version=1,actual_exit_code=0,old_smoke_plan=ref(case/'control/smoke/launch_plan.json'),old_full_plan=ref(case/'control/full/launch_plan.json'),adopted_phases=phases,gate=dict(config=ref(cfg),report=ref(report),checkpoints=cps))
        write(control/'ADOPTED_V2_PHASES.json',adoption)
        evidence=control/'evidence'
        failure=evidence/'failure.json';write(failure,dict(actual_exit_code=1,error='fixture transport timeout'))
        timeout=evidence/'timeout.json';write(timeout,dict(actual_exit_code=None,error='TimeoutExpired'))
        retry=evidence/'retry.json';write(retry,dict(checkpoints=[],logs=[]));retryexit=evidence/'retry_exit';retryexit.write_text('0\n')
        manual=evidence/'manual.json';write(manual,dict(actual_exit_code=0))
        archive=case/'control/backup_failures_resolved_v1/global.json';write(archive,dict(actual_exit_code=1))
        resolution=dict(schema_version=1,actual_exit_code=0,classification='transport_timeout',failure_record=ref(failure),timeout_command=ref(timeout),retry_record=dict(**ref(retry),actual_exit_code=0,exit_file=ref(retryexit)),archived_sticky=[dict(original_path=str(case/'control/backup_failures/global.json'),archive_path=str(archive),sha256=ref(archive)['sha256'])],manual_backup_record=dict(**ref(manual),actual_exit_code=0),old_control_failure_preserved=True,experiment_sources_configs_unchanged=True)
        write(control/'TRANSPORT_RESOLUTION.json',resolution)
        return case,control,plans

    def test_exact_remaining_plan_reuses_nn_argv_and_keeps_old_failure(self):
        case,control,old=self.fixture()
        for mode,count in [('smoke',22),('full',20)]:
            result=self.api.derived_plan(mode)
            original=old[mode]['phases'][2:] if mode=='smoke' else old[mode]['phases']
            self.assertEqual(len(result['phases']),count)
            for step,before in zip(result['phases'],original):
                self.assertEqual(step['name'],'cont1_'+before['name'])
                if 'backup_gate' not in before['name']:self.assertEqual(step['argv'],before['argv'])
                else:
                    oldargv=before['argv'][:];newargv=step['argv'][:];i=oldargv.index('--report')+1
                    self.assertEqual(newargv[:i]+newargv[i+1:],oldargv[:i]+oldargv[i+1:])
                    self.assertTrue(Path(newargv[i]).is_relative_to(control))
        self.assertEqual((case/'control/smoke/runtime/outer_exit_code').read_text(),'1\n')
        self.assertFalse((case/'control/full/runtime').exists())

    def test_bad_adoption_and_still_active_failure_cannot_release(self):
        case,control,_=self.fixture()
        failure=case/'control/backup_failures/global.json';write(failure,dict(actual_exit_code=1))
        with self.assertRaises(RuntimeError):self.api.derived_plan('smoke')
        self.assertTrue(failure.exists())
        self.assertEqual((case/'control/smoke/runtime/outer_exit_code').read_text(),'1\n')

    def test_effective_smoke_requires_real_new22_not_old_outer_pass(self):
        case,control,_=self.fixture()
        with self.assertRaises((FileNotFoundError,AssertionError)):self.api.require_effective_smoke()
        self.assertEqual((case/'control/smoke/runtime/outer_exit_code').read_text(),'1\n')

    def test_nonzero_adopted_exit_cannot_be_declared_accepted(self):
        case,control,_=self.fixture();path=control/'ADOPTED_V2_PHASES.json'
        adoption=json.loads(path.read_text());p=Path(adoption['adopted_phases'][1]['exit_file']['path'])
        write(p,dict(actual_exit_code=1,residual_owned_members=[]));adoption['adopted_phases'][1]['exit_file']=ref(p);write(path,adoption)
        with self.assertRaises(AssertionError):self.api.derived_plan('smoke')
        self.assertEqual((case/'control/smoke/runtime/outer_exit_code').read_text(),'1\n')

    def test_effective24_accepts_adopted2_and_new22_then_rejects_failed_exit(self):
        case,control,_=self.fixture();plan=self.api.derived_plan('smoke');runtime=control/'control/smoke/runtime'
        write(runtime.parent/'launch_plan.json',plan)
        write(runtime/'closure.json',dict(actual_exit_code=0,completed_phases=22,cube_counts=[1,2,3,4],families=['vjepa21_vitl1024'],training_updates=0,SR_episodes=0))
        (runtime/'outer_exit_code').write_text('0\n');(runtime.parent/'wrapper_exit_code').write_text('0\n')
        for step in plan['phases']:write(runtime/(step['name']+'_exit.json'),dict(actual_exit_code=0,residual_owned_members=[]))
        for n in (1,2,3,4):
            group=f'vjepa21_vitl1024_cube{n}';cfgpath=case/'configs/pilot'/(group+'.json');cfg=json.loads(cfgpath.read_text());home=Path(cfg['execution_root'])
            if n!=1:write(home/'gates'/group/'report.json',dict(actual_exit_code=0,actual_backend='gpu',updates=2,batch_size=1024,network_step=3,adam_count=2,encoder_calls=0,WM_dynamics_calls=0))
            write(home/'gates'/(group+'_reload.json'),dict(actual_exit_code=0,full153_state_exact=True,finite_actions=True))
            entries=[]
            for counter in (1,2):
                manifest=home/'gates'/group/'checkpoints'/f'update_{counter:09d}/manifest.json'
                ack=Path(cfg['durability']['ack_root'])/'pilot'/group/f'update_{counter:09d}.json'
                if n!=1:
                    write(manifest,dict(counter=counter,agent_sha256='a'*64,sidecar_sha256='b'*64))
                    write(ack,dict(actual_exit_code=0,counter=counter,stage='pilot',group_id=group,config_sha256=ref(cfgpath)['sha256'],manifest_sha256=ref(manifest)['sha256'],local_receipt_sha256='d'*64,scp_exit_codes=[0,0,0,0]))
                entries.append(dict(counter=counter,ack_path=str(ack),ack_sha256=ref(ack)['sha256'],manifest_sha256=ref(manifest)['sha256'],local_receipt_sha256='d'*64))
            write(runtime.parent/'durability_reports'/(group+'.json'),dict(actual_exit_code=0,stage='gate',backup_stage='pilot',group_id=group,config_sha256=ref(cfgpath)['sha256'],checkpoint_count=2,checkpoints=entries))
            write(home/'native_sr'/group/'report.json',dict(actual_exit_code=0,episodes=1,current_dim=1043,goal_dim=1024+3*n,max_episode_steps=4,encoding_protocol='fresh_current_and_goal_each_policy_call_2N'))
            write(home/'native_sr'/group/'independent_audit.json',dict(actual_exit_code=0,episodes=1,config_sha256=ref(cfgpath)['sha256'],actual_step_arrays_checked=True,diagnostic_excluded_from_formal_SR=True))
        self.assertEqual(self.api.require_effective_smoke()['effective_completed_phases'],24)
        self.assertEqual((case/'control/smoke/runtime/outer_exit_code').read_text(),'1\n')
        write(runtime/(plan['phases'][-1]['name']+'_exit.json'),dict(actual_exit_code=1,residual_owned_members=[]))
        with self.assertRaises(AssertionError):self.api.require_effective_smoke()


if __name__=='__main__':unittest.main()

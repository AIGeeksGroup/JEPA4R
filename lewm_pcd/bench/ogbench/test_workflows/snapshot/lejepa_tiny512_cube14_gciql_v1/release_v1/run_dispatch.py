"""Exclusive attempt/worker admission around original serial; no phase loop."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import traceback
import prepare_control as m


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def fresh_dispatch():
    t=Path(str(m.CASE/'control/full'))
    for name in ['runtime','wrapper_exit_code','dispatch.lock','dispatch_receipt.json','dispatch.stdout.log','dispatch.stderr.log','worker_claim.json','worker_failure.json','worker_outcome.json']:
        assert not (t/name).exists(),'Existing attempt must be inspected, never duplicated: '+name
    return t


def claim_worker(t,expected,attempt_token):
    lockpath=t/'dispatch.lock';assert lockpath.is_file(),'Execute requires an existing legitimate dispatch attempt'
    lock=m.read(lockpath)
    assert lock['schema_version']==1 and lock['launch_plan_SHA256']==expected
    assert isinstance(attempt_token,str) and len(attempt_token)==64 and m.digest(attempt_token.encode())==lock['worker_token_sha256'],'Worker attempt token differs'
    assert isinstance(lock['attempt_id'],str) and len(lock['attempt_id'])==32
    assert lock['worker_script_sha256']==m.sha(Path(__file__)),'Worker source differs from reserved attempt'
    for name in ['runtime','wrapper_exit_code','worker_claim.json','worker_failure.json','worker_outcome.json']:
        assert not (t/name).exists(),'Existing claimed/active/finished worker must not be touched: '+name
    claim=dict(schema_version=1,attempt_id=lock['attempt_id'],worker_pid=os.getpid(),claimed=time.time(),
        launch_plan_SHA256=expected,worker_token_sha256=lock['worker_token_sha256'])
    write(t/'worker_claim.json',claim)  # atomic exclusive owner claim, independent of parent receipt/PPID
    return claim


def load_original_serial():
    path=Path(str(m.F/'control/serial.py'));assert m.sha(path)==m.SOURCE_SERIAL_SHA
    sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location('_tiny_original_serial',path)
    serial=importlib.util.module_from_spec(spec);spec.loader.exec_module(serial)
    return serial


def run(expected,attempt_token):
    t=Path(str(m.CASE/'control/full'));claim=claim_worker(t,expected,attempt_token)
    oldargv=list(sys.argv);oldpath=list(sys.path);serial=None;validate=None;code=1;called=False;failure=None
    try:
        proof=m.installed_preflight(expected);assert proof['ready'],'Space became insufficient; no NN phase started'
        plan=m.read(t/'launch_plan.json');serial=load_original_serial();validate=serial.validate_plan
        def checked(value):assert value==plan;m.validate_launch(value)
        serial.validate_plan=checked
        sys.argv=[str(m.F/'control/serial.py'),'--plan-directory',str(t),'--execute']
        called=True;result=serial.main();assert result==0;code=0;return 0
    except BaseException as error:
        code=(error.code if type(error.code) is int else (0 if error.code is None else 1)) if isinstance(error,SystemExit) else (130 if isinstance(error,KeyboardInterrupt) else 1)
        failure=dict(actual_exit_code=code,attempt_id=claim['attempt_id'],worker_pid=os.getpid(),failed=time.time(),
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),original_serial_called=called,
            runtime_created=(t/'runtime').exists(),automatic_restart=False)
        write(t/'worker_failure.json',failure)
        raise  # preserve the original admission/import/serial traceback in detached stderr
    finally:
        if serial is not None and validate is not None:serial.validate_plan=validate
        sys.argv[:]=oldargv;sys.path[:]=oldpath
        outer=t/'runtime/outer_exit_code';outer_code=None;outer_error=None
        try:
            if outer.is_file():outer_code=int(outer.read_text())
        except (OSError,ValueError) as error:outer_error=repr(error)
        outcome=dict(actual_exit_code=code,attempt_id=claim['attempt_id'],worker_pid=os.getpid(),completed=time.time(),
            launch_plan_SHA256=expected,original_serial_called=called,runtime_created=(t/'runtime').exists(),
            original_outer_exit_code=outer_code,original_outer_read_error=outer_error,
            worker_failure_record='worker_failure.json' if failure is not None else None,
            scientific_phase_success_not_inferred_from_worker_entry=True,automatic_restart=False)
        write(t/'worker_outcome.json',outcome)
        with (t/'wrapper_exit_code').open('x') as stream:stream.write(str(code)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--expected-plan-sha256',required=True)
    choice=p.add_mutually_exclusive_group();choice.add_argument('--dispatch',action='store_true');choice.add_argument('--execute',action='store_true')
    p.add_argument('--attempt-token');a=p.parse_args()
    if a.execute:return run(a.expected_plan_sha256,a.attempt_token)
    assert a.attempt_token is None,'Attempt token belongs only to the reserved child worker'
    t=fresh_dispatch();proof=m.installed_preflight(a.expected_plan_sha256)
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    assert gpu.returncode==0
    proof['GPU_compute_apps_empty']=not gpu.stdout.strip();proof['ready']=proof['ready'] and proof['GPU_compute_apps_empty']
    if not a.dispatch or not proof['ready']:
        proof.update(status='READY_NO_DISPATCH' if proof['ready'] else 'WAIT_FOR_SPACE_OR_GPU',NN_started=False);print(json.dumps(proof));return 0
    token=secrets.token_hex(32);attempt_id=secrets.token_hex(16)  # coordinator metadata only, never passed to NN cfg/RNG
    write(t/'dispatch.lock',dict(schema_version=1,owner=os.getpid(),at=time.time(),attempt_id=attempt_id,
        worker_token_sha256=m.digest(token.encode()),worker_script_sha256=m.sha(Path(__file__)),launch_plan_SHA256=a.expected_plan_sha256))
    argv=[str(m.PY),'-B',str(Path(__file__).resolve()),'--expected-plan-sha256',a.expected_plan_sha256,'--execute','--attempt-token',token]
    with (t/'dispatch.stdout.log').open('x') as out,(t/'dispatch.stderr.log').open('x') as err:
        child=subprocess.Popen(argv,cwd=str(m.ROOT),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    receipt=dict(actual_dispatch_exit_code=0,started=time.time(),pid_locator_only=child.pid,attempt_id=attempt_id,argv=argv,launch_plan_SHA256=a.expected_plan_sha256,
        space=proof['space'],phase_count=48,formal_updates=4000000,formal_SR_episodes=1000,backup_advisory=True,queue_completed=False,automatic_restart=False)
    write(t/'dispatch_receipt.json',receipt);print(json.dumps(receipt));return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)

"""Wait on closed Quad SR metadata, then invoke the frozen PLDM serial entry once."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
PHASES = ('01_G11_smoke', '02_G11_smoke_audit', '03_G11_full', '04_G11_full_audit',
          '05_G12_smoke', '06_G12_smoke_audit', '07_G12_full', '08_G12_full_audit')
QUAD_PLAN_SHA = 'acdb58486db6bd8499f7f34823b950232c52e2d82c553c8defd6996847a595a2'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ready(quad):
    assert sha(quad/'launch_plan.json') == QUAD_PLAN_SHA
    exits = [quad/'wrapper_exit_code', quad/'runtime_v1/outer_exit_code']
    missing = False
    for path in exits:
        if path.exists():
            assert path.read_text().strip() == '0', str(path)
        else:
            missing = True
    for name in PHASES:
        path = quad/'runtime_v1'/(name+'_exit.json')
        if path.exists():
            value = read(path)
            assert value['actual_exit_code'] == 0 and value['residual_owned_members'] == [], str(path)
        else:
            missing = True
    if missing:
        return False
    closure = read(quad/'runtime_v1/closure.json')
    assert (closure['actual_exit_code'], closure['completed_phases'], closure['SR_episodes']) == (0, 8, 500)
    for group in ('G11', 'G12'):
        for name in ('report.json', 'independent_audit.json'):
            value = read(quad/'full'/group/name)
            assert value['actual_exit_code'] == 0 and value['group_id'] == group
            assert value['episodes'] == value['unique_seeds'] == 250
            assert (value['current_dim'], value['goal_dim']) == (211, 204)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(prepared_only=True, no_model_imported=True)))
        return 0
    quad, case = ROOT/'cube4_gciql_native_sr_v1', ROOT/'cube3_pldm_gciql_v1'
    output = case/'after_quad_v1/runtime_v1'
    output.mkdir(exist_ok=False)
    def save(value):
        temporary = output/'status.json.tmp'
        temporary.write_text(json.dumps(value, indent=2)+'\n')
        temporary.replace(output/'status.json')
    code = 1
    try:
        assert sha(case/'control_v1/launch_plan.json') == args.plan_sha
        save(dict(stage='waiting_Quad_SR_actual_acceptance', actual_exit_code=None, updated=time.time(), NN_imported=False))
        while not ready(quad):
            time.sleep(30)
        assert not (case/'control_v1/runtime_v1').exists(), 'Refuse duplicate PLDM dispatch'
        assert sha(case/'control_v1/launch_plan.json') == args.plan_sha
        argv = ['bash', str(case/'control_v1/run.sh'), '--execute']
        with (output/'PLDM_launch.log').open('x') as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            (output/'actual_child_argv.json').write_text(json.dumps(dict(argv=argv, pid_only_for_location=child.pid,
                started=time.time(), plan_sha256=args.plan_sha), indent=2)+'\n')
            save(dict(stage='original_PLDM_serial_running', actual_exit_code=None, updated=time.time(), NN_imported=False))
            result = child.wait()
        (output/'actual_child_exit_code').write_text(str(result)+'\n')
        assert result == 0, 'PLDM pipeline failed; preserve exits, no automatic restart'
        assert (case/'control_v1/wrapper_exit_code').read_text().strip() == '0'
        assert (case/'control_v1/runtime_v1/outer_exit_code').read_text().strip() == '0'
        assert read(case/'control_v1/runtime_v1/closure.json')['actual_exit_code'] == 0
        code = 0
        save(dict(stage='Quad_SR_then_PLDM_pipeline_complete', actual_exit_code=0, completed=time.time(), NN_imported=False))
    except BaseException:
        save(dict(stage='failed', actual_exit_code=1, updated=time.time(), error=traceback.format_exc(), automatic_restart=False))
        raise
    finally:
        (output/'outer_exit_code').write_text(str(code)+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())

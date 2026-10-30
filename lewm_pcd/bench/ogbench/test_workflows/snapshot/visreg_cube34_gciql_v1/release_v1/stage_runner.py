"""One of four stages through the unchanged original serial/process supervisor."""
import argparse
import sys
import time
import traceback
import common as m

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=m.STAGES,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    stage=a.stage;t=m.stage_dir(stage);plan=m.read(t/'launch_plan.json')
    if not a.execute:
        m.validate_plan(plan,stage);m.fresh_outputs(stage);space=m.space_admission()
        print(m.json.dumps(dict(actual_exit_code=0,prepared_only=True,NN_started=False,stage=stage,space=space)));return 0
    for name in ('runtime','wrapper_exit_code','stage_failure.json','space_admission.json','REPORTS_ACCEPTED.json'):
        assert not (t/name).exists(),'Existing stage attempt must be inspected, not repeated: '+name
    code=1
    try:
        m.validate_plan(plan,stage);m.fresh_outputs(stage);space=m.space_admission()
        m.write_new(t/'space_admission.json',space)
        sys.path.insert(0,str(m.F/'control'));import serial
        serial.validate_plan=lambda value:m.validate_plan(value,stage)
        sys.argv=[str(__file__),'--plan-directory',str(t),'--execute']
        actual=serial.main();assert actual==0
        checked=m.stage_closed(stage,wrapper=0)
        m.write_new(t/'REPORTS_ACCEPTED.json',dict(**checked,at=time.time(),post_stage_space=m.space_admission()))
        code=0;return 0
    except BaseException:
        m.write_new(t/'stage_failure.json',dict(actual_exit_code=1,at=time.time(),error=traceback.format_exc(),automatic_restart=False))
        raise
    finally:
        with (t/'wrapper_exit_code').open('x') as f:f.write(str(code)+'\n')

if __name__=='__main__':raise SystemExit(main())

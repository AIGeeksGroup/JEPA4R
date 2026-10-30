"""Reuse the M5 per-episode auditor; add Quad membership and exact camera receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import types
import numpy as np
from quad_bridge import ROOT,POSITION,SOURCE_POSITION,QUATERNION

SOURCE=ROOT/'cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/audit.py'
SOURCE_SHA='ffccacb2f94cfd37f0480419642341e7c966e490e1a97e6056eecfe0033c5285'


def original_auditor(source=SOURCE):
    raw=Path(source).read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    text=raw.decode();before="group in ('G06', 'G09')";after="group in ('G06', 'G09', 'G12')"
    assert text.count(before)==1;effective=text.replace(before,after);assert effective.replace(after,before)==text
    module=types.ModuleType('_original_M5_Quad_audit');module.__file__=str(source)
    exec(compile(effective,str(source),'exec'),module.__dict__);return module


def verify(group,mode,output,source=SOURCE):
    output=Path(output);expected=1 if mode=='smoke' else 250;maximum=4 if mode=='smoke' else 1000
    summary=original_auditor(source).verify_report(ROOT,group,'diagnostic' if mode=='smoke' else 'full',output,expected,204,maximum)
    threads=json.loads((output/'runtime_threads.json').read_text())
    assert threads['torch_num_threads']==1 and threads['numpy_physics']['BLAS_threads']==4
    for task in range(1,2 if mode=='smoke' else 6):
        for index in range(1 if mode=='smoke' else 50):
            row=json.loads((output/'episodes'/f'task{task}'/f'episode_{index:03d}/result.json').read_text())
            if mode=='smoke':assert row['steps']==4
            camera=row['camera']
            np.testing.assert_allclose(camera['position'],POSITION,atol=1e-12,rtol=0)
            np.testing.assert_allclose(camera['source_position'],SOURCE_POSITION,atol=1e-12,rtol=0)
            np.testing.assert_allclose(camera['actual_quaternion'],QUATERNION,atol=1e-12,rtol=0)
            assert camera['pullback']==.1 and camera['crop'] is None and camera['aa_samples']==4
    return dict(actual_exit_code=0,group_id=group,mode=mode,cube_count=4,state_dim=55,
        current_dim=211,goal_dim=204,latent_dim=192,original_M5_audit_sha256=SOURCE_SHA,
        Torch1_BLAS4=True,fresh_goal_and_current_2N_required=True,**summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=('G11','G12'),required=True)
    parser.add_argument('--mode',choices=('smoke','full'),required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=verify(args.group,args.mode,args.output)
        with (args.output/'independent_audit.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    except BaseException as error:
        with (args.output/'independent_audit_failure.json').open('x') as stream:json.dump(dict(actual_exit_code=1,error=repr(error)),stream,indent=2)
        raise

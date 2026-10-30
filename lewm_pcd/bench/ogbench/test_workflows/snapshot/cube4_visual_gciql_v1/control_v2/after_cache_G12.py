"""Use the frozen original G12 dispatcher with explicit PCD recovery lineage."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import types

OLD_SHA='9e45f34b66005f28ec3f2676c85ee4a53e65c0ba48f412116373aa0b9d272a0a'
RECOVERY_SHA='e62099b13b11e5b81d3034a13187e531e7aadd708f69223f81bacc4bcf9f6e30'


def load(q,pcd,binding_path,binding_sha):
    q,pcd,binding_path=Path(q).resolve(),Path(pcd).resolve(),Path(binding_path).resolve()
    path=q/'functional_v2/recovery.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==RECOVERY_SHA
    spec=importlib.util.spec_from_file_location('_G12_recovery',path)
    recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
    records={};binding=recovery.load_binding(q,pcd,binding_path,binding_sha,records)
    source=q/'control_v1/after_cache_G12.py';raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==OLD_SHA
    edits=[
      ("control=q/'control_v1';functional=q/'functional_v1'","control=q/'control_v2';functional=q/'functional_v2';reference=q/'control_v1'"),
      ('sha(control/name)==expected','sha(reference/name)==expected'),
      ("sha(functional/'prepare_G12.py')==PREPARE_SHA","sha(functional/'recovery.py')==PREPARE_SHA"),
      ("str(functional/'prepare_G12.py'),'--q',str(q),'--pcd-case',str(pcd)","str(functional/'recovery.py'),'prepare','--q',str(q),'--pcd-case',str(pcd),*RECOVERY_ARGS"),
      ("functional/'G12_CONFIG_PREPARED_v2.json'","functional/'G12_CONFIG_PREPARED_v3.json'"),
      ("functional/'configs/G12_cube4_pcd.v2.json'","functional/'configs/G12_cube4_pcd.v3.json'"),
      ("read(functional/'configs/G11_cube4_rgb.json')","read(q/'functional_v1/configs/G11_cube4_rgb.json')"),
      ("read(control/'G11/launch_plan.json')","read(reference/'G11/launch_plan.json')"),
      ("(control/'run_G11.sh').read_text()","(reference/'run_G11.sh').read_text()"),
      ("script=text.replace(before,'RUN=\"$CASE/control_v1/G12\"')","script=text.replace(before,'RUN=\"$CASE/control_v2/G12\"')"),
      ("str(functional/'prepare_G12.py'):PREPARE_SHA","str(functional/'recovery.py'):PREPARE_SHA"),
      ("pcd/'runtime_v1/closure.json'","pcd/'recovery_receipt_v2.json'"),
      ("q/'control_v1/after_cache_G12_runtime_v1'","q/'control_v2/after_cache_G12_runtime_v2'"),
      ("q/'control_v1'/f'after_cache_G12_refused_","q/'control_v2'/f'after_cache_G12_refused_"),
    ]
    text=raw.decode()
    for before,after in edits:
        expected=3 if before=="pcd/'runtime_v1/closure.json'" else 1
        assert text.count(before)==expected,(before,text.count(before))
        text=text.replace(before,after)
    module=types.ModuleType('_original_G12_recovery_dispatch');module.__file__=str(source)
    exec(compile(text,str(source),'exec'),module.__dict__)
    module.PREPARE_SHA=RECOVERY_SHA
    module.RECOVERY_ARGS=['--binding',str(binding_path),'--binding-sha256',binding_sha]

    def cache_ready(_q,_pcd):
        exits=[pcd/'runtime_v2/outer_exit_code',q/'control_v2/PCD/wrapper_exit_code']
        for entry in exits:
            if entry.exists() and entry.read_text().strip()!='0':raise RuntimeError('New PCD recovery failed: '+str(entry))
        status=pcd/'runtime_v2/status.json'
        if status.exists() and module.read(status).get('actual_exit_code') not in (None,0):raise RuntimeError('New PCD recovery runtime failed')
        if not all(entry.exists() for entry in exits):return False
        receipt=pcd/'recovery_receipt_v2.json'
        if not receipt.exists():
            folder=q/'control_v2/after_cache_G12_runtime_v2/00_publish_recovery';folder.mkdir()
            argv=[str(q.parent/'env_runtime/bin/python'),'-B',str(path),'publish-recovery','--q',str(q),'--pcd-case',str(pcd),*module.RECOVERY_ARGS]
            module.write_new(folder/'actual_argv.json',dict(argv=argv,started=time.time()))
            with (folder/'run.log').open('x') as stream:
                result=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=False)
            module.write_new(folder/'actual_exit_code',str(result.returncode)+'\n')
            module.write_new(folder/'exit.json',dict(actual_exit_code=result.returncode,completed=time.time()))
            assert result.returncode==0,'Recovery publisher failed; no G12 launch'
        recovery.checked_receipt(q,pcd,binding,binding_path,binding_sha,
            lambda target,expected_sha=None,json_value=True:recovery.stable(target,{},expected_sha,json_value))
        return True

    module.cache_ready=cache_ready
    return module


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--q',type=Path,required=True)
    parser.add_argument('--pcd-case',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--binding-sha256',required=True);args=parser.parse_args()
    load(args.q,args.pcd_case,args.binding,args.binding_sha256).run(args.q,args.pcd_case)

"""G10 closed SR evidence; reuse the frozen state pack body, default plan only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

HERE=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
Q=ROOT/'cube4_state_g10_native_sr_v1'
OUT=Q/'closed_evidence_backup_v1/completed_v1'
ATTEMPT=Q/'closed_evidence_backup_v1/execute_v1'
ASSET='G10_final1M_Native_SR_closed_evidence'
PACK_SHA='5e6760a8f0817a79ce18f71b8a9b905b9491553bae903bd2c83dbc6a59ae62e0'
SOURCE_SHA='9588634dafec98f5f93cea5047699199bc5f0fab1cb70433728e8477ba7dffe1'
PLAN_SHA='fc697765aef33802ec0fe57f8b179bf70a79053c5bc1eb02a8645e8a129f6e4b'
QUAD_REUSE_SHA='2bf37e658ef55621ab3c327b5f7a6984352c7bc0101294af5f364bd0b4b6b877'
QUAD_COPY_V2_SHA='8b561cc989739b5603ee110b5ee50abab3f6e2583cd2120f7e847b97c6ece60a'
NAMES=('01_diagnostic_G10','02_audit_diagnostic_G10','03_full_G10','04_audit_full_G10')
PACK_EDITS=[("SR / 'release_freeze.json'","SR / 'SOURCE_FREEZE_v1.json'",2),
 ("('release_freeze.json', 'launch_plan.json', 'dispatch_receipt.json',\n                             'dispatch.log',", "('SOURCE_FREEZE_v1.json', 'launch_plan.json',",1),
 ("release['reference_pins']","plan['source_pins']",1),("release['local_files']","release['files']",1),
 ('three_final1M_SR_and_audits_complete','G10_final1M_SR_and_audits_complete',1),('== 12','== 4',2),
 ("read(refs)['all3_final_only_local_backups_closed']","read(refs)['G10_final1M_current_host_CPU_accepted']",1),
 ('formal_episodes=750','formal_episodes=250',3),
 ('G01_G04_G07_final1M_native_SR_closed.tar.gz','G10_final1M_native_SR_closed.tar.gz',1),
 ("new('receipt.json', dict(actual_exit_code=0, archive_path=","new('receipt.json', dict(actual_exit_code=0, asset_kind='G10_final1M_Native_SR_closed_evidence', archive_path=",1),
 ('gzip_CRC_full_read=True','full_gzip_CRC_passed=True',1)]


def frozen(path,digest,edits,name):
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
    original=raw.decode();effective=original
    for before,after,count in edits:
        assert effective.count(before)==count,(name,before)
        effective=effective.replace(before,after)
    restored=effective
    for before,after,count in reversed(edits):
        assert restored.count(after)==count,(name,after)
        restored=restored.replace(after,before)
    assert restored==original
    module=types.ModuleType('_G10_SR_backup_'+name);module.__file__=str(path)
    exec(compile(effective,str(path),'exec'),module.__dict__)
    module.reuse_proof=dict(source_sha256=digest,overlay_entries=len(edits),
        inverse_entire_source_exact=True,effective_sha256=hashlib.sha256(effective.encode()).hexdigest())
    return module


def verify_closed(status,exits,codes):
    assert status['stage']=='G10_final1M_SR_and_audits_complete'
    assert status['actual_exit_code']==0 and status['completed_phases']==4
    assert status['groups']==['G10'] and status['training_updates']==0
    assert len(exits)==4 and all(x['actual_exit_code']==0 and x['residual_owned_members']==[] for x in exits)
    assert codes==['0','0']


def load_pack(reference=None):
    local=HERE.parents[1]/'cube23_gciql_distance_matched_101732Test_20261018_v1/state_priority_v1/native_sr_v1/closed_raw_evidence_backup_v1/build_closed_sr.py'
    path=Path(reference) if reference else local if local.is_file() else ROOT/'cube23_distance_gciql_v1/state_priority_v1/native_sr_v1/closed_raw_evidence_backup_v1/build_closed_sr.py'
    module=frozen(path,PACK_SHA,PACK_EDITS,'pack')
    module.SR,module.OUT,module.GROUPS=Q,OUT,('G10',)
    module.RELEASE_SHA,module.PLAN_SHA=SOURCE_SHA,PLAN_SHA
    module.__file__=str(Path(__file__).resolve())
    original=module.closed_paths
    def closed_paths():
        runtime=Q/'runtime'
        verify_closed(module.read(runtime/'status.json'),[module.read(runtime/n/'exit.json') for n in NAMES],
            [(p.read_text().strip()) for p in (runtime/'outer_exit_code',Q/'wrapper_exit_code')])
        paths,external,summary=original()
        paths.update((path,HERE/'copy_backup.py',HERE/'SOURCE_FREEZE_v1.json',HERE/'HANDOFF_v1.json'))
        for name in ('ROOT_RELEASE_v1.json','dispatch_receipt.json'):
            if (Q/name).is_file():paths.add(Q/name)
        return paths,external,summary
    module.closed_paths=closed_paths
    return module


def load_copy(reference_dir=None):
    local=HERE.parents[1]/'cube4_gciql_native_sr_103032Test_20261021_v1/closed_SR_backup_v1'
    folder=Path(reference_dir) if reference_dir else local if local.is_dir() else ROOT/'cube4_gciql_native_sr_v1/closed_SR_backup_v1'
    assert hashlib.sha256((folder/'copy_backup_v2.py').read_bytes()).hexdigest()==QUAD_COPY_V2_SHA
    quad=frozen(folder/'reuse.py',QUAD_REUSE_SHA,[],'accepted_Quad_copy')
    quad.COPY_EDITS.extend([("/closed_SR_backup_v1/completed_v1'","/closed_SR_backup_v1/completed_v2'"),
        ("parser.add_argument('--asset-kind', choices=('Quad_final1M_Native_SR_closed_evidence',), default='Quad_final1M_Native_SR_closed_evidence')",
         "parser.add_argument('--asset-kind', choices=('G10_final1M_Native_SR_closed_evidence',), default='G10_final1M_Native_SR_closed_evidence')"),
        ("/cube4_gciql_native_sr_v1/closed_SR_backup_v1/completed_v2'","/cube4_state_g10_native_sr_v1/closed_evidence_backup_v1/completed_v1'"),
        ('G11_G12_Quad_final1M_Native_SR_closed_v1.tar.gz','G10_final1M_native_SR_closed.tar.gz'),
        ('formal_episodes=500, diagnostic_episodes_excluded_from_SR=2','formal_episodes=250, diagnostic_episodes_excluded_from_SR=1')])
    original_copy=HERE.parents[1]/'cube4_gciql_103032Test_20261020_v1/closed_backup_preparation_v2/copy_verified.py'
    return quad.load('copy',original_copy if original_copy.is_file() else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--build-child',action='store_true');args=parser.parse_args()
    if not args.execute and not args.build_child:
        print(json.dumps(dict(prepared_only=True,asset_kind=ASSET,output=OUT.as_posix(),formal_episodes=250,diagnostics=1,model_cache_raw_payload=False)));return 0
    if args.build_child:
        code=1
        try:load_pack().main();code=0
        finally:
            with (ATTEMPT/'builder_actual_exit_code').open('x') as f:f.write(str(code)+'\n')
        return code
    assert not OUT.exists(),'Existing archive/attempt must be preserved, never overwritten'
    ATTEMPT.mkdir(exist_ok=False)
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--build-child']
    with (ATTEMPT/'builder_actual_argv.json').open('x') as f:json.dump(dict(argv=argv,started=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f)
    with (ATTEMPT/'builder_stdout.log').open('xb') as a,(ATTEMPT/'builder_stderr.log').open('xb') as b:result=subprocess.run(argv,stdout=a,stderr=b)
    with (ATTEMPT/'builder_outer_exit_code').open('x') as f:f.write(str(result.returncode)+'\n')
    if OUT.is_dir():
        for name in ('builder_actual_exit_code','builder_outer_exit_code','builder_actual_argv.json','builder_stdout.log','builder_stderr.log'):
            assert not (OUT/name).exists();shutil.copyfile(ATTEMPT/name,OUT/name)
    return result.returncode


if __name__=='__main__':raise SystemExit(main())

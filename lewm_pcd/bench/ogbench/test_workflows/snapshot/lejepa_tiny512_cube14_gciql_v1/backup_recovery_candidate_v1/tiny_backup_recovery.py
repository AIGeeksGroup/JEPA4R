"""Versioned advisory recovery over frozen Tiny endpoint/receiver/vendor bytes; no NN imports."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
from types import SimpleNamespace

P=Path('D:/LeWM_Artifacts/coordination/lejepa_tiny512_cube14_gciql_test_20261028_v1')
CASE=PurePosixPath('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/lejepa_tiny512_cube14_gciql_v1')
OLD_ROOT=P/'backup_v1';NEW_ROOT=P/'backup_recovery_v1'
OLD_ACK_ROOT=str(CASE/'control/backup_ack/v1/acks');OLD_FAILURE_ROOT=str(CASE/'control/backup_ack/v1/failures')
NEW_ACK_ROOT=str(CASE/'control/backup_ack/recovery_v1/acks');NEW_FAILURE_ROOT=str(CASE/'control/backup_ack/recovery_v1/failures')
NEW_SCRIPT=str(CASE/'backup_recovery_candidate_v1/tiny_backup_recovery.py')
NEW_CONFIG=str(CASE/'control/backup_ack/recovery_v1/endpoint_config.json')
BASE_CONFIG_SHA='bda8bfd09bc9516b329b60abdeb25ef68dd427b7042b44a48cd0ef7496e85e95'
LEDGER_SHA='cfae072a50211e9d41ca3e4f4404abb2c97c2a54396f72985ba048a711a844d1'
COPY_TIMEOUT=1800
SOURCE_PINS={'tiny_backup_endpoint.py':'b4d7dd8c104edd84387ead894283ee4888bc3453c4a69905709beb551984fbaf','tiny_backup_receiver.py':'0d2727496610f8d25b69ddbda2d1b5290f8bbea2a2827fcb20feacd0bf594491','vendor/backup_copy.py':'0f88ebfe8a909fe68b21cb1f22a2bc4fb598110f1cff7b528a433a031babb571','vendor/backup_watch.py':'3fdcac5b385f26a4d39d8efd020580e88272ca0a1baf0b1d041cc7cb2cda4541','vendor/backup_gate.py':'8cbbc47218485b3a0206841f01ea2ea8cfd7fe9ae81dddb9cacd2b3f8f10a016','vendor/tests/test_backup.py':'fe02cf89eb5aa11b73006c50110ea34382c5ea7cb07e87565076e1644f706f97'}


def require(value,message):
    if not value:raise RuntimeError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def small_sha(path):
    path=Path(path);before=path.stat();require(before.st_size<=16<<20,'Metadata too large: '+str(path))
    value=digest(path.read_bytes());after=path.stat();require((before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_ino,before.st_dev)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino,after.st_dev),'Metadata changed: '+str(path));return value


def base_directory(local):return P/'backup_candidate_v1' if local else Path(str(CASE/'backup_candidate_v1'))


def base_config(local):
    path=P/'backup_candidate_v1/endpoint_config.json' if local else Path(str(CASE/'control/backup_ack/v1/endpoint_config.json'))
    require(small_sha(path)==BASE_CONFIG_SHA,'Frozen original endpoint config changed');return read(path)


def validate_config(cfg,base):
    overrides=dict(ack_root=NEW_ACK_ROOT,failure_root=NEW_FAILURE_ROOT,local_root=NEW_ROOT.as_posix(),remote_script=NEW_SCRIPT,remote_config=NEW_CONFIG)
    extras={'recovery_version','copy_timeout_seconds','legacy_ledger_sha256','recovery_module_sha256'}
    require(set(cfg)==set(base)|extras,'Recovery config fields differ')
    for key,value in base.items():require(cfg[key]==overrides.get(key,value),'Recovery config changed original source/target scope: '+key)
    require(cfg['recovery_version']=='tiny52_recovery_v1' and type(cfg['copy_timeout_seconds']) is int and cfg['copy_timeout_seconds']==COPY_TIMEOUT,'Wrong single-copy timeout/version')
    require(cfg['legacy_ledger_sha256']==LEDGER_SHA and cfg['recovery_module_sha256']==small_sha(__file__),'Recovery ledger/module version mismatch')


def recovery_expected_keys(cfg,base,original):
    validate_config(cfg,base)
    require(cfg['ack_root']==NEW_ACK_ROOT and cfg['failure_root']==NEW_FAILURE_ROOT,'Recovery namespace must be exact')
    view=copy.deepcopy(cfg)
    view['ack_root']=OLD_ACK_ROOT;view['failure_root']=OLD_FAILURE_ROOT
    return original(view)  # Only namespace fields map to the original strict52 checker; no directory-prefix replacement.


def load_original(local):
    base=base_directory(local)
    names=SOURCE_PINS if local else ('tiny_backup_endpoint.py','vendor/backup_watch.py')
    for name in names:require(small_sha(base/name)==SOURCE_PINS[name],'Frozen original module changed: '+name)
    sys.path[:0]=[str(base),str(base/'vendor')]
    import tiny_backup_endpoint as endpoint
    require(small_sha(endpoint.__file__)==SOURCE_PINS['tiny_backup_endpoint.py'] and small_sha(endpoint.b.__file__)==SOURCE_PINS['vendor/backup_watch.py'],'Wrong imported original endpoint/vendor')
    if not local:return endpoint,None,None
    import tiny_backup_receiver as receiver
    import backup_copy as copier
    require(small_sha(receiver.__file__)==SOURCE_PINS['tiny_backup_receiver.py'] and small_sha(copier.__file__)==SOURCE_PINS['vendor/backup_copy.py'],'Wrong imported original receiver/copier')
    return endpoint,receiver,copier


def endpoint_dispatch(cfg,action,payload=None):
    base=base_config(False);validate_config(cfg,base);endpoint,_,_=load_original(False);original=endpoint.expected_keys
    endpoint.expected_keys=lambda value:recovery_expected_keys(value,base,original)
    try:return endpoint.dispatch(cfg,action,payload)
    finally:endpoint.expected_keys=original


def ack_for(snapshot,receipt_path,receipt,b):
    ack={key:snapshot[key] for key in ('group_id','stage','counter','config_sha256','manifest_sha256')}
    ack.update(schema_version=1,actual_exit_code=0,snapshot_sha256=b.digest_json(snapshot),source_files=snapshot['files'],
        local_receipt_path=str(receipt_path),source_after_files=receipt['source_after_files'],local_receipt_sha256=small_sha(receipt_path),
        scp_exit_codes=[value['actual_exit_code'] for value in receipt['copy_records']])
    b.validate_ack(snapshot,ack);return ack


def validate_ack_response(snapshot,ack,response,b):
    b.validate_ack(snapshot,ack)
    expected=str(PurePosixPath(NEW_ACK_ROOT)/snapshot['stage']/snapshot['group_id']/f"update_{snapshot['counter']:09d}.json")
    raw=(json.dumps(ack,indent=2,sort_keys=True)+'\n').encode()
    require(response.get('accepted') is True and response.get('ack_path')==expected and response.get('ack_sha256')==digest(raw),'ACK response path/original bytes SHA mismatch')


def transport_type(c,b):
    class RecoveryTransport(c.SSHTransport):
        def command(self,argv,prefix,payload=None,timeout=240):
            if argv[0]=='scp':
                require(timeout==300,'Unexpected original SCP timeout interface')
                b.new_json(Path(prefix).with_suffix('.timeout_policy.json'),dict(original_timeout=300,single_copy_timeout=COPY_TIMEOUT,automatic_retry=False,argv=argv))
                timeout=COPY_TIMEOUT
            return super().command(argv,prefix,payload,timeout)

        def request(self,action,payload=None):
            result=super().request(action,payload)
            if action=='poststat':require(result==payload['snapshot'],'Source changed after metadata scan')
            if action=='ack':
                validate_ack_response(payload['snapshot'],payload['ack'],result,b)
                b.new_json(self.runtime/'ACK_validation'/f'{time.time_ns()}.json',dict(actual_exit_code=0,snapshot_sha256=b.digest_json(payload['snapshot']),ACK_original_bytes_sha256=result['ack_sha256'],ack_path=result['ack_path'],local_receipt_sha256=payload['ack']['local_receipt_sha256']))
            return result
    return RecoveryTransport


def legacy_identities(b):
    return {(stage,f'lejepa_tiny512_cube{n}',counter) for n in (1,2) for stage,counters in b.COUNTERS.items() for counter in counters}|{('pilot','lejepa_tiny512_cube3',1),('pilot','lejepa_tiny512_cube3',2),('formal','lejepa_tiny512_cube3',1),('formal','lejepa_tiny512_cube3',100000)}


def receipt_identity(snapshot):return snapshot['stage'],snapshot['group_id'],snapshot['counter']


def load_legacy(cfg,base,receiver,c):
    b=receiver.b;path=Path(__file__).with_name('LEGACY_ACCEPTED_LEDGER.json')
    require(small_sha(path)==cfg['legacy_ledger_sha256']==LEDGER_SHA,'Legacy ledger changed');ledger=read(path)
    require(Path(ledger['legacy_root']).resolve()==OLD_ROOT.resolve() and len(ledger['records'])==30,'Unknown legacy root/count')
    reports={}
    for record in ledger['accepted_report_pins']:
        require(small_sha(record['path'])==record['sha256'],'Accepted original report changed');value=read(record['path'])
        require(value['actual_exit_code']==0 and value['check_count']==value['passed_count'],'Original acceptance report invalid');reports[record['path']]=value
    accepted={}
    for record in ledger['records']:
        identity=record['stage'],record['group_id'],record['counter'];require(identity not in accepted and identity in legacy_identities(b),'Unknown/duplicate legacy identity')
        receipt_path=Path(record['receipt_path']);expected=OLD_ROOT/'closed_checkpoints'/identity[0]/identity[1]/f'update_{identity[2]:09d}'/'receipt.json'
        require(receipt_path.resolve()==expected.resolve() and small_sha(receipt_path)==record['receipt_sha256'],'Legacy receipt changed/outside exact accepted root')
        receipt=read(receipt_path);snapshot=receipt['snapshot'];receiver.selected_identity(base,snapshot)
        require(receipt_identity(snapshot)==identity and b.digest_json(snapshot)==record['snapshot_sha256'],'Legacy snapshot binding changed')
        c.accepted_receipt(snapshot,receipt_path)  # Original proof plus current D stats; no payload hash/download.
        proof_record=record['first_hash_proof'];require(small_sha(proof_record['path'])==proof_record['sha256'],'First local SHA proof changed');proof=read(proof_record['path'])
        require(proof['actual_exit_code']==0 and proof['receipt_sha256']==record['receipt_sha256'] and Path(proof['receipt_path']).resolve()==receipt_path.resolve(),'First SHA proof receipt differs')
        require(set(proof['files'])==set(receipt['local_files']),'First SHA file set differs')
        for name,info in proof['files'].items():
            original=receipt['local_files'][name]
            require(info['sha256']==original['sha256']==snapshot['files'][name]['sha256'] and info['bytes']==original['stat']['size'],'First SHA/size binding differs')
            require(info['stat_before']==info['stat_after']==original['stat']==b.stat(receipt_path.parent/name),'Accepted payload stat changed')
        report_record=record['accepted_report'];require(report_record in ledger['accepted_report_pins'],'Unknown acceptance report')
        matches=[x for x in reports[report_record['path']]['transactions'] if (x['stage'],x['group_id'],x['counter'])==identity]
        require(len(matches)==1 and matches[0]['receipt_sha256']==record['receipt_sha256'],'Acceptance report no longer binds receipt')
        for key in ('source_poststat','original_ACK'):require(small_sha(record[key]['path'])==record[key]['sha256'],'Original RPC proof changed')
        require(read(record['source_poststat']['path'])==snapshot,'Original source poststat differs')
        old_ack=ack_for(snapshot,receipt_path,receipt,b);result=read(record['original_ACK']['path'])
        require(result['accepted'] is True and result['ack_sha256']==record['original_ACK']['ack_sha256']==digest((json.dumps(old_ack,indent=2,sort_keys=True)+'\n').encode()),'Original ACK proof differs')
        accepted[identity]=record
    require(set(accepted)==legacy_identities(b),'Exact old30 acceptance set required')
    return ledger,accepted


def old_snapshot(snapshot):
    value=copy.deepcopy(snapshot);value['ack_root']=OLD_ACK_ROOT;value['failure_root']=OLD_FAILURE_ROOT;return value


def copy_adapter(cfg,base,records,receiver,c):
    b=receiver.b
    def accepted_receipt(snapshot,path):
        identity=receiver.selected_identity(cfg,snapshot);path=Path(path)
        if identity in records:
            record=records[identity];require(path.resolve()==Path(record['receipt_path']).resolve() and small_sha(path)==record['receipt_sha256'],'Legacy receipt lookup changed')
            legacy=old_snapshot(snapshot);require(b.digest_json(legacy)==record['snapshot_sha256'],'Current source differs from accepted legacy snapshot')
            return c.accepted_receipt(legacy,path)
        expected=NEW_ROOT.joinpath(*snapshot['target_parts'])/'receipt.json'
        require(path.resolve()==expected.resolve(),'Unknown receipt root/partial cannot be adopted')
        return c.accepted_receipt(snapshot,path)
    def copy_snapshot(snapshot,root,transport):
        identity=receiver.selected_identity(cfg,snapshot);require(Path(root).resolve()==NEW_ROOT.resolve(),'New copy root differs')
        if identity not in records:
            require(not Path(root).joinpath(*snapshot['target_parts']).exists(),'Unknown versioned destination preserved; never adopt partial/foreign receipt')
            return c.copy_snapshot(snapshot,root,transport)
        record=records[identity];receipt_path=Path(record['receipt_path']);receipt=accepted_receipt(snapshot,receipt_path)
        require(transport.request('poststat',dict(snapshot=snapshot))==snapshot,'Legacy source changed before reuse ACK')
        ack=ack_for(snapshot,receipt_path,receipt,b);result=transport.request('ack',dict(snapshot=snapshot,ack=ack));validate_ack_response(snapshot,ack,result,b)
        b.new_json(Path(root)/'legacy_reuse'/snapshot['stage']/snapshot['group_id']/f"update_{snapshot['counter']:09d}.json",dict(actual_exit_code=0,legacy_receipt_path=str(receipt_path),legacy_receipt_sha256=record['receipt_sha256'],original_first_SHA_proof=record['first_hash_proof'],new_snapshot_sha256=b.digest_json(snapshot),new_ACK_path=result['ack_path'],new_ACK_bytes_sha256=result['ack_sha256'],accepted_payload_rehashes=0,copy_calls=0))
        return dict(receipt_path=str(receipt_path),receipt_sha256=record['receipt_sha256'])
    return SimpleNamespace(copy_snapshot=copy_snapshot,accepted_receipt=accepted_receipt)


def startup_gate(cfg,base,ledger,admission,admission_sha,c,b):
    validate_config(cfg,base)
    require(small_sha(admission)==admission_sha,'Root admission record changed');proof=read(admission)
    require(proof['actual_exit_code']==0 and type(proof['checked_at']) in (int,float) and 0<=time.time()-proof['checked_at']<=600,'Root admission missing/stale')
    require(proof['ssh_G']==dict(hostname='connect.westc.seetacloud.com',user='root',port=16755) and proof['remote_hostname']==cfg['expected_remote_hostname'] and proof['case_root']==str(CASE),'Current test identity was not admitted')
    require(proof['source_pins_verified'] is True and proof['recovery_config_sha256']==small_sha(Path(__file__).with_name('recovery_config.json')) and proof['recovery_module_sha256']==cfg['recovery_module_sha256'],'Root live source/config readback not admitted')
    for key in ('old_worker_exit','old_failed_status'):
        record=ledger[key];require(small_sha(record['path'])==record['sha256'],'Old worker evidence changed')
    require(read(ledger['old_worker_exit']['path'])['actual_exit_code']==1,'Old worker has not actually exited1')
    status=read(ledger['old_failed_status']['path']);require(status['phase']=='failed' and status['exit_code']==1 and status['completed_count']==30,'Old failed status differs')
    require(not (OLD_ROOT/'backup_runtime/worker.lock').exists(),'Old worker lock still active')
    partial=ledger['preserved_partial'];directory=Path(partial['path']);require(directory==OLD_ROOT/'closed_checkpoints/formal/lejepa_tiny512_cube3/update_000200000' and not (directory/'receipt.json').exists(),'Old failed200K cannot be adopted')
    require({f.name:b.stat(f) for f in directory.iterdir() if f.is_file()}==partial['files_stat_only'],'Preserved old partial changed')
    require(not NEW_ROOT.exists(),'Recovery root already exists; preserve partial, no automatic restart')
    required=(52-30)*(128<<20)+(64<<20)
    require(c.shutil.disk_usage(NEW_ROOT.parent).free>required,'Insufficient D free space for missing22 whole-agent slots')
    return dict(actual_exit_code=0,checked_at=time.time(),old_worker_actual_exit_code=1,legacy_accepted=30,new_root=NEW_ROOT.as_posix(),required_free_D_bytes=required,Root_admission_sha256=admission_sha,copy_timeout_seconds=COPY_TIMEOUT,automatic_retry=False,advisory_only=True)


def run(cfg,config_sha,admission,admission_sha):
    require(config_sha==small_sha(Path(__file__).with_name('recovery_config.json')),'Worker config must be the exact frozen sidecar bytes')
    base=base_config(True);validate_config(cfg,base);endpoint,receiver,c=load_original(True);b=receiver.b
    ledger,records=load_legacy(cfg,base,receiver,c);proof=startup_gate(cfg,base,ledger,admission,admission_sha,c,b)
    NEW_ROOT.mkdir(parents=False,exist_ok=False);b.new_json(NEW_ROOT/'RECOVERY_STARTUP.json',proof)
    worker=transport_type(c,b)(cfg,NEW_ROOT/'backup_runtime')
    old_expected,old_c=receiver.expected_keys,receiver.c
    receiver.expected_keys=lambda value:recovery_expected_keys(value,base,endpoint.expected_keys)
    receiver.c=copy_adapter(cfg,base,records,receiver,c)
    try:return receiver.run(cfg,config_sha,transport=worker)  # Original 180s scan loop, lock/status/failure/exit semantics.
    finally:receiver.expected_keys=old_expected;receiver.c=old_c


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--action',choices=('scan','poststat','ack','failure'));p.add_argument('--run',action='store_true');p.add_argument('--admission',type=Path);p.add_argument('--admission-sha256');p.add_argument('--config-sha256');a=p.parse_args()
    cfg=read(a.config)
    if a.action:
        require(not a.run and a.admission is None,'Endpoint action cannot start a worker')
        payload=None if a.action=='scan' else json.load(sys.stdin);print(json.dumps(endpoint_dispatch(cfg,a.action,payload),sort_keys=True));return 0
    require(a.run and a.admission is not None and a.admission_sha256 and a.config_sha256==small_sha(a.config),'Explicit Root worker admission and exact config SHA required')
    return run(cfg,a.config_sha256,a.admission,a.admission_sha256)


if __name__=='__main__':raise SystemExit(main())

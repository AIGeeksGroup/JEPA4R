"""Source-only/default-prepare epoch2 preservation using the frozen original watcher."""
from pathlib import Path
import sys
import trial


def identity(root):
    v=trial.read(trial.HERE/'current_identity_v1.json');expected=v['identity'];p=Path('/proc')/str(expected['pid'])
    if str(root)!=v['root']:raise ValueError('actual current root differs')
    z=(p/'stat').read_text().rsplit(')',1)[1].split();argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1]
    if (int(z[2]),int(z[19]),argv)!=(expected['pgrp'],expected['birth'],expected['argv']):raise ValueError('actual current trainer identity differs')
    for key in ('actual_argv','protocol','continuation_freeze'):
        if trial.sha(v[key]['path'])!=v[key]['sha256']:raise ValueError('current immutable entry differs: '+key)
    if trial.read(Path(v['formal_output'])/'status.json')['active_owned_pid']!=expected['pid']:raise ValueError('current owned phase changed')
    return dict(expected,state=z[0])


def extend_parent(proof):
    v=trial.read(trial.HERE/'current_identity_v1.json');item=v['external_epoch1_proof'];p=Path(item['path'])
    if trial.sha(p)!=item['sha256']:raise ValueError('external protected epoch1 proof changed')
    first=trial.read(p)
    if first['global_step']!=45703 or first['actual_exit_code']!=0:raise ValueError('actual inherited epoch1 required')
    if proof['baseline_binding']['continuation']['parent_full_sha256']!=first['files']['full']['sha256']:raise ValueError('current lineage does not inherit real epoch1')
    original=first['files']['bare'];stat=Path(original['path']).stat()
    if (stat.st_size,stat.st_mtime_ns)!=(original['bytes'],original['mtime_ns']):raise ValueError('external epoch1 bare stat changed')
    proof.update(completed_epochs=2,external_epoch1_proof=item,boundary_source_freeze_sha256=trial.sha(trial.HERE/'source_freeze_v1.json'),
        current_formal_identity_sha256=trial.sha(trial.HERE/'current_identity_v1.json'),epoch_exports=[
        dict(epoch=1,path=first['files']['bare']['path'],sha256=first['files']['bare']['sha256'],origin='inherited_parent'),
        dict(epoch=2,path=proof['files']['bare']['path'],sha256=proof['files']['bare']['sha256'],origin='protected_parent')])
    return proof


if __name__=='__main__':
    path=trial.HERE.parent/'pldm_worker_trial_v1/boundary.py'
    original=trial.load('readtask_frozen_boundary',path,trial.boundary_tree(path.read_text()))
    original.__doc__=__doc__
    original.CURRENT_BINDING=trial.read(trial.HERE/'current_identity_v1.json')
    original.identity=identity;original._extend_parent=extend_parent
    raise SystemExit(original.main())

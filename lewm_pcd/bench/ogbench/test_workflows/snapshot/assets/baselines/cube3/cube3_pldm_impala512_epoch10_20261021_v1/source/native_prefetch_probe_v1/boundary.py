"""Protect epoch4 once at the original save boundary. No signals or trial launch."""
import ast
from pathlib import Path
import sys
import trial


def identity(root):
    value=trial.read(trial.HERE/'current_identity_v1.json');expected=value['identity'];p=Path('/proc',str(expected['pid']))
    if str(root)!=value['root']:raise ValueError('current root differs')
    stat=(p/'stat').read_text().rsplit(')',1)[1].split();argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1]
    if (int(stat[2]),int(stat[19]),argv)!=(expected['pgrp'],expected['birth'],expected['argv']):raise ValueError('current owned trainer identity differs')
    for key in ('actual_argv','protocol','continuation_freeze'):
        if trial.sha(value[key]['path'])!=value[key]['sha256']:raise ValueError('current immutable binding differs: '+key)
    if trial.read(Path(value['formal_output'])/'status.json')['active_owned_pid']!=expected['pid']:raise ValueError('current owned phase changed')
    return dict(expected,state=stat[0])


def extend_parent(proof,full):
    binding=trial.read(trial.HERE/'current_identity_v1.json');item=binding['external_parent_proof']
    if trial.sha(item['path'])!=item['sha256']:raise ValueError('actual P3 proof changed')
    parent=trial.read(item['path'])
    if (parent['stage'],parent['actual_exit_code'],parent['completed_epochs'],parent['global_step'])!=('pldm_recovery_parent_verified',0,3,137109):
        raise ValueError('actual complete P3 required')
    lineage=proof['baseline_binding']['continuation']
    if lineage['parent_full_sha256']!=parent['files']['full']['sha256']:raise ValueError('current full does not inherit actual P3')
    for row in parent['epoch_exports']:
        if not Path(row['path']).is_file():raise ValueError('inherited export missing')
    model={k[6:]:v for k,v in full['state_dict'].items() if k.startswith('model.')}
    parameter_ids=[pid for opt in full['optimizer_states'] for group in opt['param_groups'] for pid in group['params']]
    state_ids=[pid for opt in full['optimizer_states'] for pid in opt['state']]
    proof.update(stage='pldm_native_prefetch_parent_verified',completed_epochs=4,
        validation_batch_progress=full['loops']['fit_loop']['epoch_loop.val_loop.batch_progress'],
        IDM_state_present=any(k.startswith('idm.') for k in model),optimizer_all_parameters_have_state=set(parameter_ids)==set(state_ids),
        boundary_source_freeze_sha256=trial.sha(trial.HERE/'source_freeze_v1.json'),
        current_formal_identity_sha256=trial.sha(trial.HERE/'current_identity_v1.json'),
        prior_parent_references=dict(P3=dict(proof_path=item['path'],proof_sha256=item['sha256'],global_step=137109,
            files_referenced_not_copied={key:parent['files'][key] for key in ('full','bare')})),
        epoch_exports=parent['epoch_exports']+[dict(epoch=4,path=proof['files']['bare']['path'],sha256=proof['files']['bare']['sha256'],origin='protected_epoch4')])
    return trial.validate_parent(proof)


def boundary_tree(source):
    tree=trial.base().boundary_tree(source);counts=[0]*4
    class Patch(ast.NodeTransformer):
        def visit_Compare(self,node):
            text=ast.unparse(node)
            if text=="summary.get('epoch', -1) >= 2":counts[0]+=1;node.comparators=[ast.Constant(4)]
            elif text=="full['epoch'] not in (1, 2)":counts[1]+=1;node.comparators=[ast.Tuple([ast.Constant(3),ast.Constant(4)],ast.Load())]
            return self.generic_visit(node)
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='_extend_parent':counts[2]+=1;node.args.append(ast.Name('full',ast.Load()))
            return self.generic_visit(node)
        def visit_Constant(self,node):
            values={'approved_pldm_epoch2_preservation':'approved_pldm_native_prefetch_epoch4_preservation',
                'preserved_epoch2':'preserved_epoch4','last_epoch2.ckpt':'last_epoch4.ckpt','weights_epoch_2.pt':'weights_epoch_4.pt',
                'waiting_original_complete_epoch2_save':'waiting_original_complete_epoch4_save','epoch2_preservation_failed':'epoch4_preservation_failed'}
            if node.value==r'\[Epoch\s+2/10\]\s+step\s+':counts[3]+=1;return ast.copy_location(ast.Constant(r'\[Epoch\s+4/10\]\s+step\s+'),node)
            if isinstance(node.value,str) and node.value in values:return ast.copy_location(ast.Constant(values[node.value]),node)
            return node
    tree=Patch().visit(tree)
    if counts!=[1]*4:raise ValueError('frozen boundary metadata seams changed: '+str(counts))
    return ast.fix_missing_locations(tree)


def main():
    trial.source_guard(trial.HERE.parent)
    old=trial.HERE.parent/'pldm_worker_trial_v1/boundary.py'
    sys.modules['trial']=trial
    original=trial.base().load('native_prefetch_epoch4_boundary',old,boundary_tree(old.read_text()))
    original.__file__=str(trial.HERE/'boundary.py');original.__doc__=__doc__
    original.CURRENT_BINDING=trial.read(trial.HERE/'current_identity_v1.json')
    original.identity=identity;original._extend_parent=extend_parent
    return original.main()


if __name__=='__main__':raise SystemExit(main())

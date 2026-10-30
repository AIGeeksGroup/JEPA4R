"""Reuse the read-task probe from the latest verified epoch3 after interruption."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE_FREEZE = 'cdfd14b894fda7bf7f068c489bc550e819ade12ca16f36250ebf0dda2cd374d7'
MANIFEST = '98dde8e9a49bfdd3c18591939cab9208bd9d3a2af7ae095b47e020cd02ba9c29'
LEGACY_SHA = '2e656e2f2e4c0b790b6246064e82534bf5cabb4eda07cfe563dab8d9d60711d7'
PARENT_UPDATES, TOTAL_UPDATES = 137109, 457030
_API = None


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_interruption(value):
    expected = dict(stage='instance_interruption_verified', actual_exit_code=0,
        owned_original_processes_absent=True, gpu_compute_processes_empty=True,
        original_exit_codes_known=False, original_process_exit_codes=dict(trainer=None,driver=None,outer=None))
    if any(value.get(k) != v for k,v in expected.items()) or not value.get('observed_original_pid_absence') or not all(value['observed_original_pid_absence'].values()):
        raise ValueError('actual instance interruption evidence with unknown original exits required')
    return value


def verify_interruption(receipt_path, expected_sha=None, *, check_live=False):
    if expected_sha is not None and sha(receipt_path) != expected_sha:
        raise ValueError('actual interruption receipt SHA differs')
    value = require_interruption(read(receipt_path))
    if check_live:
        if any(Path('/proc', pid).exists() for pid in value['observed_original_pid_absence']):
            raise ValueError('an original process locator is present; do not assume it is safe')
        result = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if result.stdout.strip(): raise ValueError('GPU must be empty before recovery workload')
    return value


def validate_parent(value):
    expected = dict(stage='pldm_recovery_parent_verified',actual_exit_code=0,completed_epochs=3,
        global_step=PARENT_UPDATES,epoch=2,Adam_steps=[float(PARENT_UPDATES)],native_schedule_index=PARENT_UPDATES-1,
        model_bare_exact=True,model_optimizer_finite=True,IDM_state_present=True,optimizer_all_parameters_have_state=True)
    if any(value.get(k) != v for k,v in expected.items()):
        raise ValueError('actual complete protected epoch3 full state required')
    binding=value['baseline_binding']
    if (binding.get('method'),binding.get('source_freeze_sha256'),binding.get('manifest_sha256')) != ('pldm',BASE_FREEZE,MANIFEST):
        raise ValueError('actual parent method/source/data binding differs')
    progress=value['validation_batch_progress']
    if [r['epoch'] for r in value['epoch_exports']] != [1,2,3] or progress['total']['completed'] != 4571 or not progress['is_last_batch']:
        raise ValueError('three actual epoch exports and complete final validation required')
    return value


def compare_cohort(rows):
    keys=('ids','first2_inputs','RNG_before','RNG_after','start_global_step','end_global_step','native_schedule_index','parent_proof_sha256')
    for row in rows:
        if (row['start_global_step'],row['end_global_step'],row['native_schedule_index']) != (PARENT_UPDATES,PARENT_UPDATES+416,PARENT_UPDATES+415):
            raise ValueError('diagnostic update budget differs')
        if len(row['ids']) != 416 or any(len(x) != 128 for x in row['ids']) or len(row['first2_inputs']) != 2:
            raise ValueError('actual416 B128 consumption required')
        if row.get('first2_pinned_outputs') != [True,True]:
            raise ValueError('actual pinned CPU input required')
        if any(row[k] != rows[0][k] for k in keys):
            raise ValueError('cohort/input/main RNG differs')


def probe_tree(source):
    tree=base()._original_probe_tree(source)
    count=[0]
    class Patch(ast.NodeTransformer):
        def visit_Compare(self,node):
            if ast.unparse(node) == 'trainer.current_epoch != 2':
                count[0]+=1;node.comparators=[ast.Constant(3)]
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if node.value == 'Protected epoch2 plus external1 + remaining8epochs; seed0/new loader; not exact original sample/RNG continuation':
                return ast.copy_location(ast.Constant('Protected epochs1..3 + remaining7epochs; seed0/new loader; not exact original sample/RNG continuation'),node)
            return node
    tree=Patch().visit(tree)
    if count != [1]: raise ValueError('frozen probe restore epoch boundary changed')
    return ast.fix_missing_locations(tree)


def require_runtime(a):
    m=base();data_root=m.source_guard(a.root);release=read(a.release)
    output=a.output.parent if a.mode == 'case' else a.output
    expected=dict(stage='approved_readtask_restart_trial',source_freeze_sha256=sha(HERE/'source_freeze_v1.json'),
        output=str(output),parent_proof=str(a.parent_proof),parent_proof_sha256=sha(a.parent_proof))
    if any(release.get(k) != v for k,v in expected.items()):
        raise ValueError('matching Root restart trial release required')
    parent=validate_parent(read(a.parent_proof))
    for item in parent['files'].values():
        stat=Path(item['path']).stat()
        if (stat.st_size,stat.st_mtime_ns) != (item['bytes'],item['mtime_ns']):
            raise ValueError('protected epoch3 file stat differs')
    verify_interruption(release['interruption_receipt'],release['interruption_receipt_sha256'],check_live=True)
    if int(os.environ.get('SLURM_RESTART_COUNT','0')):
        raise ValueError('fresh isolated recovery run required')
    return data_root,parent


def base():
    global _API
    if _API is None:
        path=HERE.parent/'readtask_probe_v1/trial.py'
        if sha(path) != LEGACY_SHA: raise ValueError('frozen original read-task probe changed')
        spec=importlib.util.spec_from_file_location('restart_original_readtask_probe',path)
        value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value)
        value.HERE=HERE;value.PARENT_UPDATES=PARENT_UPDATES
        value._original_probe_tree=value.probe_tree
        value.probe_tree=probe_tree;value.validate_parent=validate_parent
        value.compare_cohort=compare_cohort;value.require_runtime=require_runtime
        _API=value
    return _API


def select_profile(rows_by_case): return base().select_profile(rows_by_case)
def verify_closed_trial(report_path,parent_proof_path): return base().verify_closed_trial(report_path,parent_proof_path)
NAMES,PROFILES=base().NAMES,base().PROFILES


if __name__ == '__main__':
    raise SystemExit(base().main())

"""Small stdlib contracts; no HDF5, model, worker or CUDA execution."""
import ast
import importlib.util
from pathlib import Path
import pytest

HERE=Path(__file__).parent


def api():
    path=HERE/'trial.py';assert path.exists(),'missing new thin probe'
    spec=importlib.util.spec_from_file_location('readtask_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def row(rates):
    return dict(timing_windows=[dict(updates=128,seconds=128/r,updates_per_second=r) for r in rates])


def test_profiles_and_fixed_optimizer_budget():
    m=api();assert list(m.PROFILES)==['A1_native20_pf2','B_split20_pf2','C_split32_pf2','D_split32_pf4','A2_native20_pf2']
    assert m.PROFILES['A1_native20_pf2']['read_batch_size'] is None
    assert m.PROFILES['D_split32_pf4']==dict(name='D_split32_pf4',read_batch_size=32,train_workers=32,train_prefetch=4,validation_workers=20,validation_prefetch=2,split_validation=True)
    assert (m.PARENT_UPDATES,m.TOTAL_UPDATES,m.WARM,m.WINDOW,m.TIMED)==(91406,457030,32,128,384)


def test_selection_requires_both_metrics_and_stable_controls():
    m=api();rows={name:row([6,6,6]) for name in m.PROFILES}
    rows['B_split20_pf2']=row([5.8,5.8,30]) # aggregate beats6, median does not
    rows['C_split32_pf2']=row([6.5,6.5,6.5]);rows['D_split32_pf4']=row([6.8,6.8,6.8])
    chosen=m.select_profile(rows)
    assert not chosen['candidates']['B_split20_pf2']['eligible']
    assert chosen['selected_profile']==m.PROFILES['D_split32_pf4']
    rows['A2_native20_pf2']=row([7,7,7])
    assert m.select_profile(rows)['selected_profile']==m.PROFILES['A1_native20_pf2']


def test_generic_epoch2_parent_rejects_epoch1_or_wrong_counters():
    m=api();p=dict(stage='pldm_epoch_boundary_verified',actual_exit_code=0,completed_epochs=2,global_step=91406,epoch=1,Adam_steps=[91406.0],native_schedule_index=91405,model_bare_exact=True,model_optimizer_finite=True,baseline_binding=dict(method='pldm',manifest_sha256=m.MANIFEST,source_freeze_sha256=m.BASE_FREEZE),epoch_exports=[dict(epoch=1,path='external1'),dict(epoch=2,path='protected2')])
    assert m.validate_parent(p) is p
    for key,value in [('global_step',45703),('completed_epochs',1),('native_schedule_index',91406),('model_bare_exact',False)]:
        with pytest.raises(ValueError):m.validate_parent({**p,key:value})


def test_only_probe_and_epoch2_boundary_functions_are_adapted():
    m=api();old=(HERE.parent/'pldm_worker_trial_v1/trial.py').read_text()
    changed=m.probe_tree(old);a=ast.parse(old)
    names=lambda t:{n.name:ast.dump(n) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before,after=names(a),names(changed)
    assert [name for name in before if before[name]!=after[name]]==['run_case','run_aba']
    assert 'fingerprint' in before and before['fingerprint']==after['fingerprint']
    with pytest.raises(ValueError):m.probe_tree(old.replace('train.num_workers = a.workers','pass'))
    boundary=(HERE.parent/'pldm_worker_trial_v1/boundary.py').read_text()
    patched=m.boundary_tree(boundary);before,after=names(ast.parse(boundary)),names(patched)
    assert [name for name in before if before[name]!=after[name]]==['ready','main']
    text=ast.unparse(patched)
    assert "preserved_epoch2" in text and 'completed_epochs' not in text # added by the thin proof extender
    wrapper=ast.parse((HERE/'boundary.py').read_text())
    assert len(next(n for n in wrapper.body if isinstance(n,ast.FunctionDef) and n.name=='identity').args.args)==1

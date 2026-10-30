import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('native_prefetch_trial_test',HERE/'trial.py')
trial=importlib.util.module_from_spec(spec);spec.loader.exec_module(trial)
sys.modules['trial']=trial


def test_native_prefetch_only_and_identity():
    for name in trial.NAMES:
        train=SimpleNamespace(num_workers=20,prefetch_factor=2,batch_size=128,drop_last=True,pin_memory=True,persistent_workers=False)
        val=SimpleNamespace(num_workers=20,prefetch_factor=2,batch_size=128,drop_last=False,pin_memory=True,persistent_workers=False)
        actual=trial.adapt_loaders(train,val,trial.PROFILES[name])
        assert actual==(train,val) and train.prefetch_factor==trial.PROFILES[name]['train_prefetch']
        assert val.prefetch_factor==2 and val.num_workers==20 and train.batch_size==128
        assert trial.PROFILES[name]['read_batch_size'] is None


def test_parent_boundary_and_retirement_contract():
    p=dict(stage='pldm_native_prefetch_parent_verified',actual_exit_code=0,completed_epochs=4,global_step=182812,
        epoch=3,Adam_steps=[182812.0],native_schedule_index=182811,model_bare_exact=True,model_optimizer_finite=True,
        IDM_state_present=True,optimizer_all_parameters_have_state=True,
        baseline_binding=dict(method='pldm',source_freeze_sha256=trial.BASE_FREEZE,manifest_sha256=trial.MANIFEST),
        validation_batch_progress=dict(total=dict(completed=4571),is_last_batch=True),
        epoch_exports=[dict(epoch=x) for x in range(1,5)],prior_parent_references=dict(P3=dict(global_step=137109)))
    assert trial.validate_parent(p) is p
    bad=copy.deepcopy(p);bad['global_step']=137109
    with pytest.raises(ValueError):trial.validate_parent(bad)
    retirement=dict(stage='owned_epoch4_retired',actual_exit_code=0,owned_active_after=0,gpu_empty=True,
        source_formal_output='F',parent_proof='P',parent_proof_sha256='a'*64,
        original_process_exit_codes={k:1 for k in trial.EXIT_KEYS},original_exit_paths={k:k for k in trial.EXIT_KEYS},
        original_identity=dict(pid=1,pgrp=1,birth=2,argv=['python']),original_group_drain=dict(path='D',sha256='b'*64))
    assert trial.validate_retirement(retirement) is retirement
    retirement['original_process_exit_codes']['trainer']=None
    with pytest.raises(ValueError):trial.validate_retirement(retirement)


def test_ast_epoch_budget_and_selection():
    source=(HERE.parent/'pldm_worker_trial_v1/trial.py').read_text()
    rendered=ast.unparse(trial.probe_tree(source))
    assert 'trainer.current_epoch != 4' in rendered
    assert 'trainer.current_epoch != 2' not in rendered
    api=trial.base().old_api(True,trial.PROFILES[trial.NAMES[1]])
    assert (api.PARENT_UPDATES,api.TOTAL_UPDATES,api.WARM,api.WINDOW,api.TIMED)==(182812,457030,32,128,384)
    rows={name:dict(timing_windows=[dict(updates=128,seconds=16 if i==1 else 18) for _ in range(3)]) for i,name in enumerate(trial.NAMES)}
    assert trial.select_profile(rows)['selected_profile']['train_prefetch']==4
    rows[trial.NAMES[-1]]['timing_windows']=[dict(updates=128,seconds=20) for _ in range(3)]
    assert trial.select_profile(rows)['selected_profile']['train_prefetch']==2


def test_boundary_ast_only_new_epoch_and_validation_metadata():
    spec=importlib.util.spec_from_file_location('native_boundary_test',HERE/'boundary.py')
    boundary=importlib.util.module_from_spec(spec);spec.loader.exec_module(boundary)
    source=(HERE.parent/'pldm_worker_trial_v1/boundary.py').read_text()
    rendered=ast.unparse(boundary.boundary_tree(source))
    assert "summary.get('epoch', -1) >= 4" in rendered
    assert "full['epoch'] not in (3, 4)" in rendered
    assert 'preserved_epoch4' in rendered and '_extend_parent(dict(' in rendered
    assert ', full)' in rendered
    assert 'torch.set_num_threads(1)' in rendered

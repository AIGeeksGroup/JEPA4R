"""Restart metadata controls only; no model/data/worker/GPU execution."""
import importlib.util
from pathlib import Path
import pytest


def api():
    p=Path(__file__).with_name('trial.py');assert p.exists(),'missing isolated restart bridge'
    s=importlib.util.spec_from_file_location('restart_probe_controls',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_complete_actual_parent3_and_reject_unsaved_tail():
    m=api();p=dict(stage='pldm_recovery_parent_verified',actual_exit_code=0,completed_epochs=3,global_step=137109,epoch=2,Adam_steps=[137109.0],native_schedule_index=137108,model_bare_exact=True,model_optimizer_finite=True,IDM_state_present=True,optimizer_all_parameters_have_state=True,baseline_binding=dict(method='pldm',manifest_sha256=m.MANIFEST,source_freeze_sha256=m.BASE_FREEZE),epoch_exports=[dict(epoch=i) for i in (1,2,3)],validation_batch_progress=dict(total=dict(completed=4571),is_last_batch=True))
    assert m.validate_parent(p) is p
    for key,value in [('global_step',155449),('completed_epochs',2),('native_schedule_index',137109),('model_bare_exact',False)]:
        with pytest.raises(ValueError):m.validate_parent({**p,key:value})


def test_interruption_accepts_missing_exits_only_as_unknown():
    m=api();r=dict(stage='instance_interruption_verified',actual_exit_code=0,owned_original_processes_absent=True,gpu_compute_processes_empty=True,original_exit_codes_known=False,original_process_exit_codes=dict(trainer=None,driver=None,outer=None),observed_original_pid_absence={'126717':True})
    assert m.require_interruption(r) is r
    with pytest.raises(ValueError):m.require_interruption({**r,'original_process_exit_codes':dict(trainer=0,driver=0,outer=0)})
    with pytest.raises(ValueError):m.require_interruption({**r,'original_exit_codes_known':True})
    with pytest.raises(ValueError):m.require_interruption({**r,'owned_original_processes_absent':False})


def test_original_selection_and_new_epoch_native_probe_import_are_reused():
    m=api();rows={n:dict(timing_windows=[dict(updates=128,seconds=20.0)]*3) for n in m.NAMES}
    assert m.select_profile(rows)['selected_profile']==m.PROFILES['A1_native20_pf2']
    native=m.base().old_api(True,m.PROFILES['C_split32_pf2'])
    assert native.PARENT_UPDATES==137109 and native.WARM+native.TIMED==416
    assert native.HERE==m.HERE

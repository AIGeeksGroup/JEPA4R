# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_runtime_edges.py (verbatim except: the TMPDIR test follows workflow.child_environment's adaptation)
"""Regression checks for interpreter and worker IPC setup."""
from pathlib import Path
import json,sys
from .test_workflow_controller import api

def test_posix_socket_temporary_directory_is_short(tmp_path,monkeypatch):
    module=api()
    monkeypatch.delenv('JEPA4R_TMPDIR',raising=False);monkeypatch.delenv('TMPDIR',raising=False)
    env=module.child_environment(tmp_path/'long-run-directory'/'cache',platform='posix')
    assert env['TMPDIR']=='/tmp'
    monkeypatch.setenv('JEPA4R_TMPDIR','/data/tmp')         # adapted: a short caller TMPDIR is kept
    assert module.child_environment(tmp_path/'cache',platform='posix')['TMPDIR']=='/data/tmp'
    monkeypatch.setenv('JEPA4R_TMPDIR','/'+'x'*80)          # too long for AF_UNIX names: /tmp
    assert module.child_environment(tmp_path/'cache',platform='posix')['TMPDIR']=='/tmp'
    assert Path(env['TORCHINDUCTOR_CACHE_DIR']).is_relative_to(tmp_path)

def test_venv_binary_is_not_resolved_to_its_base(tmp_path,monkeypatch,capsys):
    module=api();venv=tmp_path/'policy-env/bin/python';base=tmp_path/'base-python'
    original=Path.resolve
    monkeypatch.setattr(module.Path,'resolve',lambda p,*a,**k:base if p==venv else original(p,*a,**k))
    module.main(['--model','jepa4r','--cube-count','1','--mode','smoke','--output',str(tmp_path/'new-run'),'--wm-python',sys.executable,'--rl-python',str(venv)])
    assert json.loads(capsys.readouterr().out)['config']['python']['downstream']==str(venv)

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def workflow():
    path = ROOT / 'lewm_pcd/bench/relocate/workflow.py'
    spec = importlib.util.spec_from_file_location('workflow_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_list_works_without_simulator_dependencies():
    result = subprocess.run([sys.executable, str(ROOT / 'run.py'), '--list'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    options = json.loads(result.stdout)
    assert options['rh56-relocate'] == ['jepa4r', 'lewm']
    assert len(options['adroit-relocate']) == 6


def test_rh56_rejects_unpublished_model_before_loading_assets():
    result = subprocess.run([sys.executable, str(ROOT / 'run.py'),
                             '--env', 'rh56-relocate', '--model', 'pldm'],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'does not support' in result.stderr


def stage(name, body):
    return {'name': name, 'command': [sys.executable, '-c', body],
            'cwd': str(ROOT), 'environment': {}}


def test_actual_nonzero_exit_is_saved_and_stops_following_stage(tmp_path):
    api = workflow()
    out = tmp_path / 'run'
    later = tmp_path / 'must_not_exist'
    stages = [stage('failed', 'raise SystemExit(7)'),
              stage('later', f'open({str(later)!r}, "w").write("wrong")')]
    code = api.execute_plan(stages, out, lambda s: {'passed': True}, 'frozen')
    assert code == 7
    state = json.loads((out / 'STATUS.json').read_text(encoding='utf-8'))
    assert state['stages']['failed']['exit_code'] == 7
    assert state['failed_count'] == 1
    assert not later.exists()


def test_successful_process_with_failed_verification_is_not_accepted(tmp_path):
    api = workflow()
    out = tmp_path / 'run'
    code = api.execute_plan([stage('work', 'print("done")')], out,
                            lambda s: {'passed': False, 'error': 'bad checkpoint'},
                            'frozen')
    assert code != 0
    state = json.loads((out / 'STATUS.json').read_text(encoding='utf-8'))
    assert state['stages']['work']['exit_code'] == 0
    assert state['stages']['work']['status'] == 'verification_failed'
    assert state['status'] == 'failed'


def test_existing_output_requires_explicit_resume(tmp_path):
    api = workflow()
    out = tmp_path / 'run'
    out.mkdir()
    try:
        api.execute_plan([], out, lambda s: {'passed': True}, 'frozen')
    except FileExistsError:
        pass
    else:
        raise AssertionError('Existing output was silently reused')


def test_resume_refuses_changed_frozen_identity(tmp_path):
    api = workflow()
    out = tmp_path / 'run'
    assert api.execute_plan([], out, lambda s: {'passed': True}, 'old') == 0
    try:
        api.execute_plan([], out, lambda s: {'passed': True}, 'new', resume=True)
    except ValueError as error:
        assert 'identity' in str(error)
    else:
        raise AssertionError('Changed source/config was silently resumed')


def test_resume_rechecks_completed_artifacts_without_reexecuting(tmp_path):
    api = workflow()
    out = tmp_path / 'run'
    counter = tmp_path / 'counter'
    work = stage('work', f'open({str(counter)!r}, "a").write("1")')
    assert api.execute_plan([work], out, lambda s: {'passed': True}, 'same') == 0
    assert api.execute_plan([work], out, lambda s: {'passed': False},
                            'same', resume=True) != 0
    assert counter.read_text() == '1'


def test_duplicate_stage_names_rejected_before_output_creation(tmp_path):
    api = workflow()
    out = tmp_path / 'run'
    try:
        api.execute_plan([stage('same', ''), stage('same', '')], out,
                         lambda s: {'passed': True}, 'frozen')
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate stage names accepted')
    assert not out.exists()


def test_resume_cannot_turn_a_failed_full_plan_into_success_via_subset(tmp_path):
    api = workflow()
    out = tmp_path / 'run'
    stages = [stage('accepted', ''), stage('failed', 'raise SystemExit(7)')]
    assert api.execute_plan(stages, out, lambda s: {'passed': True}, 'same') == 7
    try:
        api.execute_plan(stages[:1], out, lambda s: {'passed': True}, 'same', resume=True)
    except ValueError:
        pass
    else:
        raise AssertionError('A subset erased the failed plan')
    state = json.loads((out / 'STATUS.json').read_text(encoding='utf-8'))
    assert state['status'] == 'failed'
    assert state['exit_code'] == 7


def test_resume_verifier_exception_persists_failure(tmp_path):
    api = workflow()
    out = tmp_path / 'run'
    work = stage('work', '')
    assert api.execute_plan([work], out, lambda s: {'passed': True}, 'same') == 0
    def broken(unused):
        raise RuntimeError('damaged receipt')
    assert api.execute_plan([work], out, broken, 'same', resume=True) == 1
    state = json.loads((out / 'STATUS.json').read_text(encoding='utf-8'))
    assert state['status'] == 'failed'
    assert state['exit_code'] == 1


def test_interrupt_during_verification_is_persisted(tmp_path):
    api = workflow()
    out = tmp_path / 'run'
    def interrupted(unused):
        raise KeyboardInterrupt()
    assert api.execute_plan([stage('work', '')], out, interrupted, 'same') == 130
    state = json.loads((out / 'STATUS.json').read_text(encoding='utf-8'))
    assert state['status'] == 'interrupted'
    assert state['exit_code'] == 130
    assert state['stages']['work']['exit_code'] == 0


def test_external_asset_fingerprint_detects_same_path_content_change(tmp_path):
    api = workflow()
    source = tmp_path / 'eai'
    source.mkdir()
    code = source / 'train_loop.py'
    code.write_text('version one', encoding='utf-8')
    assets = {'eai_vc_root': str(source)}
    before = api.asset_fingerprints(assets)
    code.write_text('version two', encoding='utf-8')
    assert api.asset_fingerprints(assets) != before


def test_resume_rejects_a_replaced_accepted_payload_and_receipt(tmp_path):
    api = workflow()
    out = tmp_path / 'run'
    payload = tmp_path / 'weights'
    work = stage('work', f'open({str(payload)!r}, "w").write("original")')
    def verify(unused):
        return {'passed': True, 'details': {'payload': payload.read_text()}}
    assert api.execute_plan([work], out, verify, 'same') == 0
    payload.write_text('replaced', encoding='utf-8')
    assert api.execute_plan([work], out, verify, 'same', resume=True) != 0
    state = json.loads((out / 'STATUS.json').read_text(encoding='utf-8'))
    assert state['status'] == 'failed'
    assert state['stages']['work']['verification']['details']['payload'] == 'original'


def test_resume_requires_the_preserved_actual_process_exit(tmp_path):
    api = workflow()
    out = tmp_path / 'run'
    work = stage('work', '')
    assert api.execute_plan([work], out, lambda s: {'passed': True}, 'same') == 0
    (out / 'logs/work.exit').write_text('7\n', encoding='ascii')
    assert api.execute_plan([work], out, lambda s: {'passed': True}, 'same', resume=True) != 0


def test_cli_simulator_override_is_explicitly_forwarded_to_backend(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    api = workflow()
    assets = tmp_path / 'assets.json'
    assets.write_text(json.dumps({'wm_python': 'wm', 'rh56_sim_python': 'json-sim',
                                  'rh56_bc_python': 'json-bc'}), encoding='utf-8')
    observed = {}
    def plan(args, unused_assets):
        observed.update(sim_python=args.sim_python,
                        explicit=getattr(args, 'sim_python_override', None))
        return []
    monkeypatch.setattr(api.importlib, 'import_module', lambda name: SimpleNamespace(plan=plan))
    assert api.main(['--env', 'rh56-relocate', '--assets', str(assets),
                     '--output', str(tmp_path / 'run'), '--sim-python', 'explicit-sim']) == 0
    assert observed == {'sim_python': 'explicit-sim', 'explicit': True}
    assert not (tmp_path / 'run').exists()


def test_source_identity_binds_the_effective_dependency_trust_rules(tmp_path, monkeypatch):
    api = workflow()
    (tmp_path / 'lewm_pcd').mkdir()
    (tmp_path / 'run.py').write_text('entry', encoding='utf-8')
    rules = tmp_path / 'DEPENDENCIES.json'
    rules.write_text('{"trusted":"first"}', encoding='utf-8')
    monkeypatch.setattr(api, 'ROOT', tmp_path)
    before = api._source_identity({})
    rules.write_text('{"trusted":"second"}', encoding='utf-8')
    assert api._source_identity({}) != before

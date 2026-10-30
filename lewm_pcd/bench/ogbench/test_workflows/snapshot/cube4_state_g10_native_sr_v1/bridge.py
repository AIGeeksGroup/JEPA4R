"""Pinned state Native functions; G10 identity and namespace only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT / 'cube23_distance_gciql_v1'
REFS = {
    'evaluate_state_xyz.py': 'f58b4d4c13523ddffb5e405ac53376af18c7a4893f0f3e7098874ec833c7cdec',
    'audit_state_xyz_sr.py': '6630b6760ebc82770cd2032e5e2655686415d4a81964749b7b3539736528e8ac',
    'run_serial_sr.py': 'f9338a85b9026942ce0f6e0bfad0949798cc86f995aec4407cd1f390d4bd40bd',
    'run_priority.py': '3b2bafcf07cb154bfbc32956375546c26fb387faf33f7d51aa2917e08ca4252a',
}
GUARD_SHA = '0331473db0b0566d3097c8cd9cf7362a820badc8c8537134ea39388ab79e6063'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def source(name, edits=(), entry=None):
    path = HERE / 'reference' / name
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFS[name], name
    original = raw.decode()
    effective = original
    for before, after in edits:
        assert effective.count(before) == 1, (name, before)
        effective = effective.replace(before, after)
    restored = effective
    for before, after in reversed(edits):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == original
    module = types.ModuleType('_G10_' + name.replace('.', '_'))
    module.__file__ = str(HERE / entry if entry else path)
    exec(compile(effective, str(path), 'exec'), module.__dict__)
    module.parameter_proof = dict(source_sha256=REFS[name], changes=len(edits),
        inverse_exact=True, effective_sha256=hashlib.sha256(effective.encode()).hexdigest())
    module.original_entry = effective[effective.rindex("if __name__ == '__main__':"):] if "if __name__ == '__main__':" in effective else None
    return module


def evaluator():
    module = source('evaluate_state_xyz.py', [
        ("sha(PRIORITY/'release_freeze.json')", "sha(config['training_source_release'])"),
    ], entry='evaluate_state_xyz.py')
    module.HERE, module.ROOT, module.CASE = HERE, ROOT, CASE
    module.PRIORITY = CASE / 'five_group_extension_v1'
    return module


def auditor():
    return source('audit_state_xyz_sr.py', [
        ("config['group_id'] in ('G01', 'G04', 'G07')", "config['group_id'] == 'G10'"),
        ("{'G01': 1, 'G04': 2, 'G07': 3}", "{'G10': 4}"),
        ("{1: 'cube-single-play-v0', 2: 'cube-double-play-v0', 3: 'cube-triple-play-v0'}", "{4: 'cube-quadruple-play-v0'}"),
        ("{1: 200, 2: 500, 3: 1000}", "{4: 1000}"),
    ], entry='audit_state_xyz_sr.py')


def verify_training(plan):
    cfg = read(plan['config'])
    training, accepted = read(cfg['training_config']), read(cfg['training_acceptance'])
    assert sha(cfg['training_config']) == cfg['training_config_sha256']
    assert sha(cfg['training_acceptance']) == cfg['training_acceptance_sha256']
    assert sha(cfg['training_source_release']) == cfg['training_source_release_sha256']
    assert (cfg['group_id'], cfg['cube_count'], cfg['current_dim'], cfg['goal_dim']) == ('G10', 4, 55, 12)
    assert (cfg['dataset_id'], cfg['max_episode_steps'], cfg['seed_base']) == ('cube-quadruple-play-v0', 1000, 970000000)
    assert training['group_id'] == 'G10' and training['modality'] == 'state'
    assert training['input_contract']['zero_padding'] is False and training['cache_binding'] is None
    assert training['features_none'] is True and training['runtime_encoder_forwards'] == 0
    assert (training['input_contract']['current_dim'], training['input_contract']['goal_dim']) == (55, 12)
    assert (training['seed'], training['batch_size'], training['total_updates']) == (0, 1024, 1000000)
    assert training['fresh'] is True and training['resume_from'] is None
    assert accepted['actual_exit_code'] == 0 and accepted['full153_state_exact'] and accepted['state']['leaves'] == 153
    assert accepted['finite_actions'] is True
    assert (accepted['Adam_count'], accepted['network_step'], accepted['CSV_validation_records']) == (1000000, 1000001, 200)
    assert (accepted['actual_current_dim'], accepted['actual_goal_dim']) == (55, 12)
    assert accepted['synthetic_zero_padding'] is False
    assert [x['counter'] for x in accepted['checkpoint_transactions']] == [1] + list(range(100000, 1000001, 100000))
    manifest = Path(cfg['final_checkpoint']) / 'manifest.json'
    assert sha(manifest) == cfg['final_manifest_sha256'] == accepted['final_manifest_sha256']
    assert read(manifest)['counter'] == 1000000
    status = read(Path(training['output']) / 'status.json')
    assert status['exit_code'] == 0 and status['completed_updates'] == status['adam_count'] == 1000000
    assert status['network_step'] == 1000001 and status['validation_count'] == 200
    assert (Path(training['output']) / 'trainer_exit_code').read_text().strip() == '0'
    for path in plan['training_phase_exit_paths']:
        value = read(path)
        assert value['actual_exit_code'] == 0 and value['residual_owned_members'] == []
    for path in plan['training_outer_paths']:
        assert Path(path).read_text().strip() == '0'


def supervisor(plan_path):
    plan = read(plan_path)
    for path, expected in plan['source_pins'].items():
        assert sha(path) == expected, path
    old = source('run_priority.py')
    guard_path = ROOT / 'prepare_runtime.py'
    assert sha(guard_path) == GUARD_SHA
    spec = importlib.util.spec_from_file_location('_G10_owned_process_guard', guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return old, guard


def serial():
    old_load = """spec = importlib.util.spec_from_file_location('_frozen_state_priority_supervisor', PRIORITY / 'run_priority.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    guard = old.release_guard()"""
    old_closed = """assert load(PRIORITY / 'runtime/status.json')['accepted_groups'] == 3
    for path in (PRIORITY / 'runtime/outer_exit_code', PRIORITY / 'wrapper_exit_code'):
        assert path.read_text().strip() == '0', path
    for group in plan['groups']:
        report = load(PRIORITY / 'acceptance' / group / 'report.json')
        assert report['actual_exit_code'] == 0 and report['Adam_count'] == 1000000"""
    module = source('run_serial_sr.py', [
        (old_load, 'old, guard = supervisor(plan_path)'),
        ("plan['groups'] == ['G01', 'G04', 'G07']", "plan['groups'] == ['G10']"),
        (old_closed, 'verify_training(plan)'),
        ('three_final1M_SR_and_audits_complete', 'G10_final1M_SR_and_audits_complete'),
    ], entry='run_serial_sr.py')
    module.HERE, module.ROOT = HERE, ROOT
    module.supervisor, module.verify_training = supervisor, verify_training
    return module

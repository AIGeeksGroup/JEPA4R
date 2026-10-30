"""Resume only the four unexecuted PCD stages using the original supervisor."""
import hashlib
import json
from pathlib import Path
import types

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
Q = ROOT / 'cube4_visual_gciql_v1'
PCD = ROOT / 'cube4_official_embeddings_v1/pcd_training_camera_v1'
HERE = Q / 'control_v2/PCD'
RUN = PCD / 'runtime_v2'
SOURCE = Q / 'control_v1/run_pcd.py'
SOURCE_SHA = '1b92756c2d15b2b99d75470d64a9f02b04f550bf2c636d85b95b238c4e1e670d'
GATE_SHA = '51da373cc18a6f352e634d48eb0b9844020f050b3536a5422ccdb8214107b5f1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prerequisites(q=Q, pcd=PCD, here=HERE):
    """Require genuine closed prior subprocesses; keep their aggregate failure."""
    assert (pcd / 'runtime_v1/outer_exit_code').read_text().strip() == '1'
    assert (q / 'control_v1/PCD/wrapper_exit_code').read_text().strip() == '1'
    assert read(pcd / 'runtime_v1/status.json')['actual_exit_code'] == 1
    for name in ('00_wait_G11_acceptance', '01_reference', '02_fresh', '03_GPU_preprocess'):
        value = read(pcd / 'runtime_v1' / (name + '_exit.json'))
        assert value['actual_exit_code'] == 0 and value['residual_owned_members'] == []
    report_path = pcd / '3d/GPU_preprocess_report.json'
    assert sha(report_path) == GATE_SHA
    report = read(report_path)
    assert report['actual_exit_code'] == 0 and report['accepted'] is False
    assert report['prepared_bitexact'] and report['latent_bitexact'] and report['model_frozen_exact']
    assert report['prepared_max_abs_diff'] == report['latent_max_abs_diff'] == 0.
    assert 0 < report['candidate_median_seconds'] < report['baseline_median_seconds']
    release = read(here / 'launch_plan.json')['performance_release']
    decision_path = Path(release['path'])
    assert sha(decision_path) == release['sha256']
    decision = read(decision_path)
    assert decision['actual_exit_code'] == 0 and decision['positive_gain_released'] is True
    assert decision['original_engineering_5percent_gate_passed'] is False
    assert decision['unchanged_numeric_tolerances'] is True
    assert decision['gate_report_sha256'] == GATE_SHA
    for path in (q / 'control_v1/G11/runtime/outer_exit_code', q / 'control_v1/G11/wrapper_exit_code'):
        assert path.read_text().strip() == '0'
    accepted = read(q / 'acceptance/G11_full_acceptance.json')
    assert accepted['actual_exit_code'] == 0 and accepted['full153_state_exact'] is True
    assert accepted['Adam_count'] == 1000000 and accepted['network_step'] == 1000001
    assert accepted['CSV_validation_records'] == 200
    assert [item['counter'] for item in accepted['checkpoint_transactions']] == [1] + list(range(100000, 1000001, 100000))
    plan = read(here / 'launch_plan.json')
    assert [item['name'] for item in plan['phases']] == ['04_profile', '05_pilot', '06_full', '07_audit']


def main():
    prerequisites()
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    preserved = types.ModuleType('_preserved_quad_pcd_serial')
    preserved.__file__ = str(SOURCE)
    exec(compile(raw, str(SOURCE), 'exec'), preserved.__dict__)
    original = preserved.original
    original.HERE, original.RUN = HERE, RUN
    owned = not RUN.exists()
    code = 1
    try:
        original.main()
        code = 0
    except BaseException as error:
        if owned and RUN.exists():
            original.save('failure.json', dict(actual_exit_code=1, error=repr(error)))
            original.save('status.json', dict(stage='failed', actual_exit_code=1, error=repr(error)))
        raise
    finally:
        if owned and RUN.exists():
            with (RUN / 'outer_exit_code').open('x') as stream:
                stream.write(str(code) + '\n')


if __name__ == '__main__':
    main()

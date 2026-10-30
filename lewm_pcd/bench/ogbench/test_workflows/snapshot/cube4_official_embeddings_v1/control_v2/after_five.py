"""Wait using stdlib only; inspect the small pilot artifacts without new inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
MAIN = ROOT / 'cube23_distance_gciql_v1/five_group_extension_v1'
PROFILE = HERE.parent / 'training_camera_v2'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main_complete():
    status = read(MAIN / 'runtime/status.json')
    assert status['actual_exit_code'] in (None, 0), 'Main training failed; do not launch encoding'
    if status['actual_exit_code'] is None:
        return False, status
    assert status['completed_phases'] == 20 and status['accepted_groups'] == status['trained_groups'] == 5
    for path in (MAIN / 'runtime/outer_exit_code', MAIN / 'wrapper_exit_code'):
        if not path.exists():
            return False, status
        assert path.read_text().strip() == '0', path
    phases = sorted((MAIN / 'runtime').glob('[0-9]*'))
    assert len(phases) == 20
    for phase in phases:
        record = read(phase / 'exit.json')
        assert record['actual_exit_code'] == 0 and not record['residual_owned_members'], phase
    for group in ('G10', 'G05', 'G06', 'G08', 'G09'):
        report = read(MAIN / 'acceptance' / (group + '_full_acceptance.json'))
        assert report['actual_exit_code'] == 0 and report['full153_state_exact']
        assert report['Adam_count'] == 1000000 and report['network_step'] == 1000001
        assert [row['counter'] for row in report['checkpoint_transactions']] == [1] + list(range(100000, 1000001, 100000))
    return True, status


def wait_for_five():
    assert sha(MAIN / 'release_freeze.json') == 'd163b157001d3e3022a8663396ff9c2ba1a23717d1b04b31e9b6e9713e1dd765'
    path = PROFILE / 'after_five_status.json'
    while True:
        done, status = main_complete()
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True)
        stage = 'five_trainings_accepted_GPU_empty' if done and not gpu.stdout.strip() else 'waiting_five_trainings_or_GPU'
        save(path, dict(stage=stage, updated=time.time(), main_group=status.get('group_id'),
            main_phase=status.get('stage'), main_accepted_groups=status.get('accepted_groups'),
            main_complete=done, GPU_pids=gpu.stdout.strip(), torch_or_model_imported=False))
        if done and not gpu.stdout.strip():
            with (PROFILE / 'after_five_receipt.json').open('x') as stream:
                json.dump(dict(actual_exit_code=0, completed=time.time(), all20_main_phases_accepted=True,
                               main_modified=False, GPU_empty=True, model_imported=False), stream, indent=2)
            return
        time.sleep(30)


def check_pilot():
    import numpy as np
    cache = PROFILE / '2d/cache'
    receipt = read(cache / 'pilot_receipt.json')
    assert receipt['actual_exit_code'] == 0 and receipt['not_full_cache'] and receipt['source_encoder_frozen']
    assert receipt['completed_episodes'] == 2 and receipt['completed_rows'] == 2002
    selection = read(cache / 'preprocess_selection.json')
    assert selection['actual_exit_code'] == 0 and selection['selected_device'] == 'cuda', 'Preserve actual CPU fallback; full auto-start held'
    assert selection['numeric_passed'] and selection['cuda_faster_by_required_gain']
    plan = read(PROFILE / 'workspace/official_state_2d_cache_v1/producer_plan.json')
    assert set(receipt['splits']) == {'train', 'val'}
    for split, rows in receipt['splits'].items():
        assert len(rows) == 1 and rows[0]['episode'] == 0
        q = Path(rows[0]['receipt_path'])
        assert sha(q) == rows[0]['receipt_sha256']
        ep = read(q)
        assert ep['actual_exit_code'] == 0 and ep['raw_row_count'] == 1001 and ep['terminal_raw_id'] == 1000
        assert ep['main_rows'] == 1000 and ep['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert ep['camera']['pullback'] == .10 and ep['camera']['crop'] is None
        np.testing.assert_allclose(ep['camera']['position'], plan['camera']['position'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(ep['camera']['actual_quaternion'], plan['camera']['actual_quaternion'], atol=1e-12, rtol=0)
        latent = ep['latent']
        path = Path(latent['path'])
        assert sha(path) == latent['sha256']
        values = np.load(path, allow_pickle=False)
        assert values.shape == (1001, 192) and values.dtype == np.float32 and np.isfinite(values).all()
    with (PROFILE / 'pilot_independent_readback.json').open('x') as stream:
        json.dump(dict(actual_exit_code=0, episodes=2, features=2002, selected_device='cuda',
                       full_cache=False, encoder_executed=False, completed=time.time()), stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('wait', 'pilot-check'))
    args = parser.parse_args()
    wait_for_five() if args.action == 'wait' else check_pilot()

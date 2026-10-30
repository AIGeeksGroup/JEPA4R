"""Stdlib-only wait for accepted G11; never initialize an encoder or trainer."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
Q = ROOT / 'cube4_visual_gciql_v1'
ROLE = Q / 'control_v1/G11'


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


while True:
    status = json.loads((ROLE / 'runtime/status.json').read_text())
    assert status['actual_exit_code'] in (None, 0), 'G11 failed; do not start PCD encoding'
    if status['actual_exit_code'] == 0:
        assert status['completed_phases'] == 4 and status['accepted_groups'] == status['trained_groups'] == 1
        exits = [ROLE / 'runtime/outer_exit_code', ROLE / 'wrapper_exit_code']
        if all(p.exists() for p in exits):
            assert all(p.read_text().strip() == '0' for p in exits)
            for folder in sorted((ROLE / 'runtime').glob('[0-9]*')):
                value = json.loads((folder / 'exit.json').read_text())
                assert value['actual_exit_code'] == 0 and not value['residual_owned_members']
            accepted = json.loads((Q / 'acceptance/G11_full_acceptance.json').read_text())
            assert accepted['actual_exit_code'] == 0 and accepted['full153_state_exact']
            assert accepted['Adam_count'] == 1000000 and accepted['network_step'] == 1000001
            assert [x['counter'] for x in accepted['checkpoint_transactions']] == [1] + list(range(100000, 1000001, 100000))
            save(Q / 'control_v1/after_G11_receipt.json', dict(actual_exit_code=0, completed=time.time(),
                 all_four_G11_phases_actual0=True, full_checkpoint_accepted=True, model_imported=False))
            break
    save(Q / 'control_v1/after_G11_wait_status.json', dict(stage='waiting_original_G11_acceptance',
         observed_G11=status, updated=time.time(), model_imported=False))
    time.sleep(30)

"""Bind the preparation source to already verified migrated source evidence."""
import hashlib
import json
from pathlib import Path
import time

CASE = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
ROOT = CASE / 'official_state_workers_v1/data_prepare'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
ready_path, migration_path = CASE / 'READY.json', CASE / 'migration_acceptance.json'
ready, migration = json.loads(ready_path.read_text()), json.loads(migration_path.read_text())
utils = CASE / 'source/ogbench/ogbench/utils.py'
assert ready['source_commit'] == 'a977253ce5b88434f4e4f531c50047cc95993a51'
assert ready['actual_exit_code'] == migration['actual_exit_code'] == 0
assert migration['members_SHA_length_verified'] == 659 and migration['gzip_read_to_EOF']
assert sha(utils) == 'f503a3a0497f91ce02958abae5770f3e08c9d6a3a740687c828c996fedb24e9b'
report = {'source_commit_from_migration': ready['source_commit'],
    'git_checkout_present': (CASE / 'source/ogbench/.git').exists(),
    'ready_receipt': {'path': str(ready_path), 'sha256': sha(ready_path)},
    'migration_receipt': {'path': str(migration_path), 'sha256': sha(migration_path), 'content': migration},
    'actual_loader_source': {'path': str(utils), 'sha256': sha(utils)},
    'actual_exit_code': 0, 'verified': time.time()}
(ROOT / 'source_attestation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

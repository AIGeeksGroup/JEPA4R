"""One fresh dispatch after frozen-file and actual CPU acceptance checks."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import time

H = Path(__file__).resolve().parent
D = H.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert not (H / 'runtime').exists() and not (H / 'dispatch.lock').exists(), 'Refuse duplicate launch'
review = json.loads((H / 'PROTOCOL_REVIEW.json').read_text())
assert review['status'] == 'source_and_frozen_entry_review_complete_no_blocker'
files = {}
for relative, expected in review['reviewed_artifacts'].items():
    path = H / relative
    actual = dict(bytes=path.stat().st_size, sha256=sha(path))
    assert actual == expected, relative
    files[str(path.relative_to(D))] = actual
for relative in ['PLAN.md', 'dispatch_formal.py']:
    path = H / relative
    files[str(path.relative_to(D))] = dict(bytes=path.stat().st_size, sha256=sha(path))
prior = json.loads((D / 'official_state_wholebatch_v1/root_preflight.json').read_text())
for relative, expected in prior['protected'].items():
    path = D / relative
    actual = dict(bytes=path.stat().st_size, sha256=sha(path))
    assert actual == expected, relative
    files[relative] = actual

# PINS deliberately refers to the earlier MAIN_SHA constant. Resolve only
# literal constants, without importing the trainer or evaluating arbitrary code.
tree = ast.parse((H / 'formal_main.py').read_text())
constants = {node.targets[0].id: ast.literal_eval(node.value)
    for node in tree.body if isinstance(node, ast.Assign)
    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    and isinstance(node.value, ast.Constant)}
pins_node = next(node.value for node in tree.body if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == 'PINS' for target in node.targets))
pins = {ast.literal_eval(key): constants[value.id] if isinstance(value, ast.Name)
    else ast.literal_eval(value) for key, value in zip(pins_node.keys, pins_node.values)}
for relative, expected in pins.items():
    path = D / 'source/ogbench' / relative
    assert sha(path) == expected, relative
    files[str(path.relative_to(D))] = dict(bytes=path.stat().st_size, sha256=expected)
for relative in ['prepare_runtime.py', 'requirements.runtime.lock']:
    path = D / relative
    files[relative] = dict(bytes=path.stat().st_size, sha256=sha(path))
assert json.loads((H / 'target_boundary_CPU_exit.json').read_text())['actual_exit_code'] == 0
frozen = json.loads((D / 'official_state_workers_v1/data_prepare/stat-freeze.json').read_text())
for name, item in frozen['datasets'].items():
    stat = Path(item['path']).stat()
    actual = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                  device=stat.st_dev, inode=stat.st_ino)
    assert actual == item['accepted_stat'], name
p = subprocess.run(['bash', '-n', str(H / 'run_formal.sh')], capture_output=True, text=True)
assert p.returncode == 0, p.stderr
gpu_processes = subprocess.check_output(['nvidia-smi',
    '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
assert not gpu_processes, 'GPU occupied; do not overlap'
cg = Path('/sys/fs/cgroup')
preflight = dict(checked_at=time.time(), GPU=subprocess.check_output(['nvidia-smi',
    '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip(),
    GPU_processes=gpu_processes, memory_current=(cg / 'memory.current').read_text().strip(),
    memory_max=(cg / 'memory.max').read_text().strip(), memory_events=(cg / 'memory.events').read_text(),
    cpu_max=(cg / 'cpu.max').read_text().strip(),
    temp_free=os.statvfs('/root/autodl-tmp').f_bavail * os.statvfs('/root/autodl-tmp').f_frsize,
    accepted_data_stat_unchanged=True, old_selected_input_unchanged=True,
    source_unmodified=True, raw_NPZ_rehashed=False, bash_syntax_actual_exit_code=0)
(H / 'formal_preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
freeze = dict(frozen_at=time.time(), files=files, review_sha256=sha(H / 'PROTOCOL_REVIEW.json'),
    real_target_CPU_actual_exit_code=0,
    real_target_CPU_receipt_sha256=sha(H / 'target_boundary_CPU_exit.json'),
    user_authorized='fresh official state seed0/B1024/1M prefetch1; first1+100K checkpoints; deferred evaluations')
(H / 'launch_freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
lock = os.open(H / 'dispatch.lock', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.write(lock, str(time.time()).encode())
os.close(lock)
argv = ['/bin/bash', str(H / 'run_formal.sh')]
with (H / 'dispatch.log').open('x') as log:
    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(H))
receipt = dict(dispatched_at=time.time(), argv=argv, supervisor_pid=child.pid,
    launch_freeze_sha256=sha(H / 'launch_freeze.json'), formal_completion_claimed=False,
    prior_prelaunch_constant_reader_failure_preserved=True)
(H / 'dispatch_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(actual_dispatch_call_exit_code=0, receipt=receipt,
                     preflight=preflight, freeze=freeze)))

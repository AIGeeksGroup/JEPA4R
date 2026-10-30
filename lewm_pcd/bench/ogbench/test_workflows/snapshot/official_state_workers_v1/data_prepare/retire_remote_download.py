"""Retire only this preparation's identified downloader after faster local source retrieval."""
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/official_state_workers_v1/data_prepare')
launches = list(ROOT.glob('launch_*.json'))
assert len(launches) == 1
launch = json.loads(launches[0].read_text())
pid = launch['pid']
proc = Path('/proc') / str(pid)
cmdline = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
assert cmdline == launch['argv']
stat = (proc / 'stat').read_text()
receipt = {'pid': pid, 'pgid': os.getpgid(pid), 'proc_stat_before': stat,
    'actual_argv_before': cmdline, 'original_launch': launch,
    'signal': 'SIGTERM only to the identified single downloader process; its thread workers are in the same PID',
    'reason': 'local official-source transport demonstrated >2.6MB/s; preserve partial remote ranges',
    'at': time.time(), 'original_status': json.loads((ROOT / 'status.json').read_text())}
os.kill(pid, signal.SIGTERM)
for _ in range(100):
    if not proc.exists() and (ROOT / 'actual_exit_code').exists():
        break
    time.sleep(0.1)
receipt['real_child_exit'] = int((ROOT / 'actual_exit_code').read_text())
assert receipt['real_child_exit'] == -15 and not proc.exists()
receipt['retired_verified'] = True
(ROOT / 'remote_download_retired.json').write_text(json.dumps(receipt, indent=2))
for source, dest in [('status.json', 'status_remote_retired.json'),
                     ('actual_argv.json', 'actual_argv_remote_retired.json'),
                     ('actual_exit_code', 'actual_exit_code_remote_retired')]:
    assert not (ROOT / dest).exists()
    (ROOT / source).rename(ROOT / dest)
(ROOT / 'transport_status.json').write_text(json.dumps({
    'stage': 'local_official_download_before_transport', 'updated': time.time(),
    'remote_download_actual_exit': -15, 'final_acceptance': 'pending'}, indent=2))
print(json.dumps(receipt, indent=2))

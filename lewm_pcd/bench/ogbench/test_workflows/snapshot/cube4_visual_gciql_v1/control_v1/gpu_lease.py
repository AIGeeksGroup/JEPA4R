"""Check GPU idle, then exec the unchanged phase in the same owned process group."""
import os
import subprocess
import sys

argv = sys.argv[1:]
assert argv and argv[0] == '--' and len(argv) > 1
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
assert not apps.strip() and max(int(line.strip()) for line in memory.splitlines()) <= 512, 'GPU occupied; preserve other jobs'
os.execv(argv[1], argv[1:])

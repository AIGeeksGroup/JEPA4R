"""Persistent launcher: records the real child exit code, preserves every launch log."""
import json
import os
from pathlib import Path
import subprocess
import time

CASE = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
ROOT = CASE / 'official_state_workers_v1/data_prepare'
stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
argv = [str(CASE / 'env_runtime/bin/python'), '-u', str(ROOT / 'prepare_official_state.py')]
log = ROOT / ('prepare_' + stamp + '.log')
started = time.time()
receipt = {'argv': argv, 'cwd': str(CASE), 'log': str(log), 'started': started, 'exit_code': None}
(ROOT / ('launch_' + stamp + '.json')).write_text(json.dumps(receipt, indent=2))
with log.open('xb') as f:
    child = subprocess.Popen(argv, cwd=CASE, env=env, stdout=f, stderr=subprocess.STDOUT)
    receipt['pid'] = child.pid
    (ROOT / ('launch_' + stamp + '.json')).write_text(json.dumps(receipt, indent=2))
    code = child.wait()
receipt.update(exit_code=code, completed=time.time())
(ROOT / ('exit_' + stamp + '.json')).write_text(json.dumps(receipt, indent=2))
(ROOT / 'actual_exit_code').write_text(str(code) + '\n')
raise SystemExit(code)

"""Freeze cache-owned sources after the real CPU contract tests."""
import json
from pathlib import Path
import subprocess
import sys
import argparse
from common import save,sha

here=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--version',default='v1');a=p.parse_args();version=a.version
result=subprocess.run([sys.executable,str(here/'test_bridges.py')],capture_output=True,text=True)
log=here/f'CPU_tests_{version}.log'
assert not log.exists()
log.write_text(result.stdout+result.stderr)
(here/f'CPU_tests_{version}.actual_exit_code').write_text(str(result.returncode)+'\n')
assert result.returncode==0
files=sorted(list(here.glob('*.py'))+list((here/'base').glob('*.py')))
save(here/f'source_freeze_{version}.json',dict(sources={str(f.relative_to(here)):sha(f) for f in files},
    baseline_reuse='counted in-memory generalization of original producer/render/audit',
    CPU_test=dict(path=str(log),sha256=sha(log),actual_exit_code=result.returncode),
    executed_models=0,executed_renderers=0,actual_exit_code=0))
print(json.dumps(dict(freeze_sha256=sha(here/f'source_freeze_{version}.json'),source_files=len(files))))

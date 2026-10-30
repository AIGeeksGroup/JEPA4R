"""Publish the frozen five-role namespace, test it on target, then start once."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = '/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1'
TARGET = ROOT + '/cube23_distance_gciql_v1/five_group_extension_v1'


def main():
    release = json.loads((HERE / 'release_freeze.json').read_text())
    review = json.loads((HERE / release['independent_review']).read_text())
    assert review['actual_exit_code'] == 0
    assets = {}
    for name, expected in release['local_files'].items():
        data = (HERE / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, name
        assets[TARGET + '/' + name] = base64.b64encode(data).decode()
    assets[TARGET + '/release_freeze.json'] = base64.b64encode((HERE / 'release_freeze.json').read_bytes()).decode()
    script = '''import base64,hashlib,json,pathlib,subprocess,os,time
root=pathlib.Path(ROOT);case=pathlib.Path(TARGET)
assert not (case/'runtime').exists() and not (case/'wrapper_exit_code').exists(),'Existing dispatch; do not repeat'
for path,encoded in assets.items():
 p=pathlib.Path(path);assert p.resolve().is_relative_to(case)
 data=base64.b64decode(encoded)
 if p.exists():assert p.read_bytes()==data,path
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(data)
 if p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
for path,expected in references.items():
 assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==expected,path
subprocess.run(['bash','-n',str(case/'run_extension.sh')],check=True)
subprocess.run([str(root/'env_runtime/bin/python'),'-B',str(case/'run_extension.py')],check=True)
handoff=json.loads((case/'FUNCTION_HANDOFF_v2.json').read_text())
environment=dict(os.environ,**handoff['CPU_test_env'],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (case/'target_CPU_tests.log').open('x') as log:
 cpu=subprocess.run(handoff['CPU_test_argv'],cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
with (case/'target_CPU_tests_exit.json').open('x') as f:json.dump(dict(actual_exit_code=cpu.returncode,completed=time.time()),f)
assert cpu.returncode==0,'Target CPU tests failed; no GPU or trainer launched'
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
assert not gpu.stdout.strip(),'Other GPU job exists; preserve it and stop before launch'
with (case/'dispatch.log').open('xb') as log:
 child=subprocess.Popen(['bash',str(case/'run_extension.sh')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(actual_exit_code=0,dispatch_only=True,pid=child.pid,started=time.time(),argv=['bash',str(case/'run_extension.sh')],source_release_sha256=hashlib.sha256((case/'release_freeze.json').read_bytes()).hexdigest(),target_CPU_tests_actual_exit_code=0,formal_training_completion_claim=False)
with (case/'dispatch_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''
    header = '\n'.join(f'{key}={value!r}' for key, value in (
        ('ROOT', ROOT), ('TARGET', TARGET), ('assets', assets), ('references', release['reference_pins'])))
    process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60', '101732Test',
        ROOT + '/env_runtime/bin/python -B -'], input=(header + '\n' + script).encode(), capture_output=True)
    (HERE / 'publish_start.stdout.log').write_bytes(process.stdout)
    (HERE / 'publish_start.stderr.log').write_bytes(process.stderr)
    with (HERE / 'publish_start_actual_exit.json').open('x') as stream:
        json.dump(dict(actual_exit_code=process.returncode, completed=time.time()), stream)
    assert process.returncode == 0, process.stderr.decode(errors='replace')
    record = json.loads(process.stdout.decode().splitlines()[-1])
    with (HERE / 'dispatch_receipt.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record))


if __name__ == '__main__':
    main()

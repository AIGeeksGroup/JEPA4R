"""Publish only the reviewed SR namespace and dispatch one serial controller."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = '/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1'
TARGET = ROOT + '/cube23_distance_gciql_v1/state_priority_v1/native_sr_v1'


def main():
    release = json.loads((HERE / 'release_freeze.json').read_text())
    assert release['groups'] == ['G01', 'G04', 'G07']
    review = json.loads((HERE / release['independent_review']).read_text())
    assert review['actual_exit_code'] == 0
    assets = {}
    for name, expected in release['local_files'].items():
        path = HERE / name
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, name
        assets[TARGET + '/' + name] = base64.b64encode(raw).decode()
    assets[TARGET + '/release_freeze.json'] = base64.b64encode((HERE / 'release_freeze.json').read_bytes()).decode()
    script = '''import base64,hashlib,json,pathlib,subprocess,os,time
root=pathlib.Path(ROOT);sr=pathlib.Path(TARGET)
assert not (sr/'runtime').exists() and not (sr/'wrapper_exit_code').exists(),'Existing SR dispatch; do not repeat'
for path,encoded in assets.items():
 p=pathlib.Path(path);assert p.resolve().is_relative_to(sr)
 raw=base64.b64decode(encoded)
 if p.exists():assert p.read_bytes()==raw,path
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
 if p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
for path,expected in source_pins.items():
 assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==expected,path
subprocess.run(['bash','-n',str(sr/'run_serial_sr.sh')],check=True)
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
subprocess.run([str(root/'env_runtime/bin/python'),'-B',str(sr/'run_serial_sr.py'),'--plan',str(sr/'launch_plan.json')],env=environment,check=True)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
assert not gpu.stdout.strip(),'Unrelated GPU job; do not overlap'
with (sr/'dispatch.log').open('xb') as log:
 child=subprocess.Popen(['bash',str(sr/'run_serial_sr.sh')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(actual_exit_code=0,dispatch_only=True,pid=child.pid,started=time.time(),argv=['bash',str(sr/'run_serial_sr.sh')],source_release_sha256=hashlib.sha256((sr/'release_freeze.json').read_bytes()).hexdigest(),training_updates=0,no_SR_completion_claim=True)
with (sr/'dispatch_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''
    header = '\n'.join(f'{name}={value!r}' for name, value in (
        ('ROOT', ROOT), ('TARGET', TARGET), ('assets', assets), ('source_pins', release['reference_pins'])))
    process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60', '101732Test',
        ROOT + '/env_runtime/bin/python -B -'], input=(header + '\n' + script).encode(), capture_output=True)
    (HERE / 'deploy_dispatch.stdout.log').write_bytes(process.stdout)
    (HERE / 'deploy_dispatch.stderr.log').write_bytes(process.stderr)
    with (HERE / 'deploy_dispatch_actual_exit.json').open('x') as stream:
        json.dump(dict(actual_exit_code=process.returncode, completed=time.time()), stream)
    assert process.returncode == 0, process.stderr.decode(errors='replace')
    receipt = json.loads(process.stdout.decode().splitlines()[-1])
    with (HERE / 'dispatch_receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

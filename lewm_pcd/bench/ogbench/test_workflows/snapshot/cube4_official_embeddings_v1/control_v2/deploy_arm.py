"""Publish only this profile's frozen control files and arm one CPU-first relay."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent
ROOT = '/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1'
CASE = ROOT + '/cube4_official_embeddings_v1'


def main():
    release = json.loads((HERE / 'release_freeze.json').read_text())
    review = json.loads((LOCAL / release['independent_review']).read_text())
    assert review['actual_exit_code'] == 0
    assets = {}
    for name, expected in release['local_files'].items():
        raw = (LOCAL / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, name
        assets[CASE + '/' + name] = base64.b64encode(raw).decode()
    assets[CASE + '/control_v2/release_freeze.json'] = base64.b64encode((HERE / 'release_freeze.json').read_bytes()).decode()
    script = '''import base64,hashlib,json,pathlib,subprocess,time
root=pathlib.Path(ROOT);case=pathlib.Path(CASE);profile=case/'training_camera_v2';control=case/'control_v2'
assert not (profile/'runtime_v2').exists() and not (control/'wrapper_exit_code').exists(),'Existing relay; do not repeat'
for path,encoded in assets.items():
 p=pathlib.Path(path);assert p.resolve().is_relative_to(case)
 raw=base64.b64decode(encoded)
 if p.exists():assert p.read_bytes()==raw,path
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
 if p.suffix=='.py':compile(p.read_bytes(),str(p),'exec')
for path,expected in references.items():
 assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==expected,path
subprocess.run(['bash','-n',str(control/'run.sh')],check=True)
profile.mkdir(parents=True,exist_ok=True)
with (control/'dispatch.log').open('xb') as log:
 child=subprocess.Popen(['bash',str(control/'run.sh')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(actual_exit_code=0,dispatch_only=True,started=time.time(),pid=child.pid,argv=['bash',str(control/'run.sh')],source_release_sha256=hashlib.sha256((control/'release_freeze.json').read_bytes()).hexdigest(),all_GPU_work_waits_existing_five_trainings_accepted=True,main_training_modified=False)
with (control/'dispatch_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''
    header = '\n'.join(f'{key}={value!r}' for key, value in (
        ('ROOT', ROOT), ('CASE', CASE), ('assets', assets), ('references', release['reference_pins'])))
    process = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60', '101732Test',
        ROOT + '/env_runtime/bin/python -B -'], input=(header + '\n' + script).encode(), capture_output=True)
    (HERE / 'deploy.stdout.log').write_bytes(process.stdout)
    (HERE / 'deploy.stderr.log').write_bytes(process.stderr)
    with (HERE / 'deploy_actual_exit.json').open('x') as stream:
        json.dump(dict(actual_exit_code=process.returncode, completed=time.time()), stream)
    assert process.returncode == 0, process.stderr.decode(errors='replace')
    record = json.loads(process.stdout.decode().splitlines()[-1])
    with (HERE / 'dispatch_receipt.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record))


if __name__ == '__main__':
    main()

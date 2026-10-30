"""Closed Cube1 metadata selector and verified small tar; no ML imports."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import traceback

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE=ROOT/'foundation_visual_gciql_v1'
GROUP='visreg_vitb768_cube1'
OUT=CASE/'closed_cache_metadata_v1'/GROUP

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    code=1
    try:
        shutil.copyfile(__file__,OUT/'builder.py')
        group=CASE/'cache/w8_groups'/GROUP;cache=group/'cache'
        planpath=group/'plan.json';plan=read(planpath)
        runtime=CASE/'control/formal_w8/runtime'
        auditpath=group/'full_audit_v1/report.json';audit=read(auditpath)
        assert audit['actual_exit_code']==0 and audit['episodes_verified']==1100
        assert audit['raw_rows_verified']==1101100 and audit['latent_dim']==768
        assert audit['plan_sha256']==sha(planpath)
        assert (auditpath.parent/'actual_exit_code').read_text().strip()=='0'
        assert audit['manifest_sha256']==sha(cache/'cache_manifest.json')
        selected={}
        def add(path):
            path=Path(path);assert path.is_file(),str(path)
            resolved=path.resolve();assert resolved.is_relative_to(ROOT),str(path)
            assert path.name not in ('status.json','resources.jsonl','train.csv')
            assert path.suffix not in ('.npy','.npz','.pt','.pkl','.h5','.hdf5','.tar','.gz')
            assert path.stat().st_size<=16*(1<<20),str(path)
            selected[str(path.relative_to(ROOT))]=path
        for name in ('plan.json','data_binding.json','camera.json','delivery.json'):add(group/name)
        for name in ('cache_manifest.json','cache_seal.json','protocol.json','batch1_alignment.json'):add(cache/name)
        receipts=sorted(cache.glob('*/episode_*/receipt.json'));assert len(receipts)==1100
        for path in receipts:add(path)
        add(auditpath);add(auditpath.parent/'actual_exit_code')
        add(Path(plan['shared_state_root'])/'source_arrays.json')
        for name in ('data_acceptance','source_stat_freeze','encoder_spec','encoder_adapter'):add(plan[name])
        for relative,digest in plan['source_pins'].items():
            path=ROOT/relative;assert sha(path)==digest;add(path)
        spec=read(plan['encoder_spec'])
        for relative,digest in spec.get('source_files_sha256',{}).items():
            path=Path(spec['source_root'])/relative;assert sha(path)==digest;add(path)
        for path in (CASE/'functional').glob('*.py'):add(path)
        config=None
        for phase in ('01_encode_full','02_cache_audit','03_deliver','04_config'):
            prefix=GROUP+'_'+phase;exitpath=runtime/(prefix+'_exit.json');record=read(exitpath)
            assert record['actual_exit_code']==0 and record['residual_owned_members']==[]
            for suffix in ('_exit.json','_argv.json','.log'):add(runtime/(prefix+suffix))
            if phase=='04_config':
                args=read(runtime/(prefix+'_argv.json'))['argv'];config=Path(args[args.index('--output')+1])
        add(config);cfg=read(config)
        assert cfg['delivery_binding']['sha256']==sha(group/'delivery.json')
        add(cfg['parent_template']['path'])
        for name in ('launch_plan.json','resource_admission.json'):add(CASE/'control/formal_w8'/name)
        # Only closed original-smoke metadata used by resource_admission; no steps,
        # checkpoint weights, active formal status/resources, or training logs.
        smoke=CASE/'control/smoke';smokeplan=read(smoke/'launch_plan.json')
        add(smoke/'launch_plan.json')
        for name in ('closure.json','outer_exit_code','actual_dispatch.json'):add(smoke/'runtime'/name)
        assert (smoke/'runtime/outer_exit_code').read_text().strip()=='0'
        assert read(smoke/'runtime/closure.json')['completed_phases']==64
        for phase in smokeplan['phases']:
            for suffix in ('_exit.json','_argv.json'):add(smoke/'runtime'/(phase['name']+suffix))
            if phase['name'].endswith('_08_native_audit'):
                args=phase['argv'];native=Path(args[args.index('--output')+1])
                for name in ('independent_audit.json','native_source_binding.json'):add(native/name)
                add(args[args.index('--config')+1])
        add(OUT/'builder.py')
        before={name:dict(bytes=path.stat().st_size,sha256=sha(path)) for name,path in sorted(selected.items())}
        save(OUT/'member_inventory.json',dict(actual_exit_code=0,scope='closed metadata only; no vectors, weights, raw arrays or active logs',
            group=GROUP,episode_receipts=1100,members=before,created=time.time()))
        archive=OUT/'closed_metadata.tar.gz'
        with tarfile.open(archive,'x:gz',compresslevel=6) as bundle:
            for name,path in sorted(selected.items()):bundle.add(path,arcname=name,recursive=False)
            bundle.add(OUT/'member_inventory.json',arcname='member_inventory.json',recursive=False)
        with gzip.open(archive,'rb') as stream:
            while stream.read(1<<20):pass
        verified=0
        with tarfile.open(archive,'r:gz') as bundle:
            members=bundle.getmembers();assert len(members)==len(before)+1
            for member in members:
                assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
                with bundle.extractfile(member) as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                if member.name=='member_inventory.json':assert digest==sha(OUT/'member_inventory.json')
                else:assert digest==before[member.name]['sha256'] and member.size==before[member.name]['bytes']
                verified+=1
        assert all(sha(path)==before[name]['sha256'] and path.stat().st_size==before[name]['bytes'] for name,path in selected.items())
        save(OUT/'remote_receipt.json',dict(actual_exit_code=0,archive=str(archive),bytes=archive.stat().st_size,
            sha256=sha(archive),member_count=verified,episode_receipts=1100,full_gzip_CRC=True,
            every_member_SHA_verified=True,source_before_after_exact=True,NN_forward=False,
            raw_or_latent_arrays_read=False,training_outputs_read=False,completed=time.time()))
        print(json.dumps(read(OUT/'remote_receipt.json')))
        code=0
    except BaseException:
        save(OUT/'failure.json',dict(actual_exit_code=1,error=traceback.format_exc(),at=time.time()));raise
    finally:
        with (OUT/'inner_exit_code').open('x') as stream:stream.write(str(code)+'\n')

if __name__=='__main__':main()

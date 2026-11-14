"""Validate direct RH56 batches and publish paired header-only VDS inputs.

The actual paired writer and numerical collection contract are unchanged. There
is no task-specific cache, relocation, scheduler or receipt-reuse mechanism.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from jepa4r.envs.twins.rh56._runtime import (RUNTIME_HELPERS_SHA256, atomic_json as save,
                                   file_sha256 as sha)

def check_registry(records):
    seeds={};initials={};counts={}
    for r in records:
        seed=int(r['seed']);initial=r['initial_state_sha256'];split=r['split']
        if seed in seeds:raise ValueError(f'duplicate seed {seed}: {seeds[seed]} / {split}')
        if initial in initials:raise ValueError(f'duplicate physical initial state: {initials[initial]} / {split}')
        if len(initial)!=64:raise ValueError('invalid physical initial hash')
        seeds[seed]=split;initials[initial]=split;counts[split]=counts.get(split,0)+1
    return dict(episodes=len(records),counts=counts,unique_seeds=len(seeds),unique_physical_initials=len(initials))

def validate_batch(attempt,plan_path=None,deep=True,report_path=None):
    """Verify original data; create a fresh receipt without rewriting inputs."""
    import h5py
    attempt=Path(attempt)
    receipt=Path(report_path) if report_path is not None else attempt/'batch_acceptance.json'
    if receipt.exists():raise FileExistsError(f'existing validation receipt: {receipt}')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    state=json.loads((attempt/'state.json').read_text())
    if state['phase']!='complete' or state['exit_code']!=0:raise ValueError('collector did not complete')
    manifest=json.loads((attempt/'manifest.json').read_text())
    if manifest.get('rejected_episodes'):raise ValueError('rejected episode cannot be published')
    mixed=manifest['episodes'];records=mixed+manifest.get('auxiliary',[])
    if plan_path is not None:
        expected=json.loads(Path(plan_path).read_text())['episodes']
        if [(r['split'],int(r['seed'])) for r in records]!=[(r['split'],int(r['seed'])) for r in expected]:
            raise ValueError('completed episodes differ from frozen batch plan')
        if [r['requested_coverage'] for r in records] != [r['coverage'] for r in expected]:
            raise ValueError('coverage differs from frozen plan')
        if manifest['provenance']['plan_sha256']!=sha(plan_path):raise ValueError('producer used another plan')
    config=json.loads((attempt/'config.json').read_text())
    config_hash=hashlib.sha256(json.dumps(config['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if config['provenance'] != manifest['provenance'] or config_hash != manifest['provenance']['config_sha256']:
        raise ValueError('producer/config identity mismatch')
    registry=check_registry(records);files=[]
    # The paired writer keeps shared numerical columns bitwise equal, with only terminal action NaN.
    if mixed:
        with h5py.File(attempt/'paired_rgb.h5','r') as a,h5py.File(attempt/'paired_pcd.h5','r') as b:
            shared=set(a)-{'pixels'}
            if shared!=set(b)-{'pcd_xyz','pcd_rgb'}:raise ValueError('paired schemas differ')
            for key in shared:
                values=a[key][:]
                if not np.array_equal(values,b[key][:],equal_nan=True):raise ValueError('paired mismatch '+key)
                if key != 'action' and values.dtype.kind == 'f' and not np.isfinite(values).all():
                    raise ValueError('nonfinite shared column ' + key)
            lengths=a['ep_len'][:];offsets=a['ep_offset'][:]
            if not np.array_equal(offsets,np.r_[0,np.cumsum(lengths)[:-1]]):raise ValueError('invalid offsets')
            if len(lengths)!=len(mixed):raise ValueError('episode count mismatch')
            for i,r in enumerate(mixed):
                off=int(offsets[i]);F=int(lengths[i]);T=F-1
                if F!=r['rows'] or T!=r['steps'] or not 1<=T<=200:raise ValueError('episode length mismatch')
                if not np.all(a['episode_seed'][off:off+F]==r['seed']):raise ValueError('stored seed mismatch')
                if not np.isnan(a['action'][off+T]).all():raise ValueError('missing terminal NaN action')
                actions=a['action'][off:off+T]
                if not np.isfinite(actions).all() or np.abs(actions).max()>1:raise ValueError('invalid executed action')
                if a['pixels'].shape[1:] != (256, 256, 3) or b['pcd_xyz'].shape[1:] != (1200, 3):
                    raise ValueError('modality shape mismatch')
                if a['pixels'].dtype != np.uint8 or b['pcd_xyz'].dtype != np.float32 or b['pcd_rgb'].dtype != np.uint8:
                    raise ValueError('modality dtype mismatch')
                if deep:
                    with np.load(attempt/r['npz'],allow_pickle=False) as z:
                        for key in ('observations','actions','proprio','qpos','qvel','trace','rewards','xyz'):
                            if not np.isfinite(z[key]).all():raise ValueError('nonfinite stored '+key)
                        for column, key, file in [
                            ('action', 'actions', a),
                            ('observation', 'observations', a),
                            ('proprio', 'proprio', a),
                            ('qpos', 'qpos', a),
                            ('qvel', 'qvel', a),
                            ('pixels', 'pixels', a),
                            ('pcd_xyz', 'xyz', b),
                            ('pcd_rgb', 'rgb', b),
                        ]:
                            selected=file[column][off:off+(T if column=='action' else F)]
                            if not np.array_equal(selected,z[key]):raise ValueError('HDF/actual episode mismatch '+key)
                        initial=json.loads(z['initial_state_json'].item())
                        fields=('integration','body_pos','body_quat','site_pos','site_quat','p','v','q_cmd')
                        initial_hash = hashlib.sha256(
                            json.dumps({k: initial[k] for k in fields}, sort_keys=True, separators=(',', ':')).encode()
                        ).hexdigest()
                        if initial_hash!=r['initial_state_sha256']:raise ValueError('physical initial hash mismatch')
                        camera=json.loads(z['camera_json'].item())
                        if (
                            camera['mode'] != 'opposite'
                            or camera['position'] != [0.0, 1.2, 1.2]
                            or camera['fovy'] != 45.0
                        ):
                            raise ValueError('wrong camera')
                        kind=r['sample_type'];branch=r['branch_t'];length=r['branch_len']
                        expected=np.array([kind==2 or kind==1 and branch<=t<branch+length for t in range(T)],np.int8)
                        if not np.array_equal(a['rand_action'][off : off + T], expected):
                            raise ValueError('random segment mask mismatch')
                        if kind == 1 and not (15 <= branch < 160 and (length == 200 or 10 <= length <= 30)):
                            raise ValueError('branch protocol mismatch')
                        if not 0 <= r['noise_std'] < 0.2 or kind == 2 and r['noise_std'] != 0:
                            raise ValueError('noise protocol mismatch')
    for r in records:
        path=attempt/r['npz']
        if sha(path)!=r['npz_sha256']:raise ValueError('NPZ hash mismatch')
        if r['producer_sha256'] != manifest['provenance']['producer_sha256'] or r['config_sha256'] != config_hash:
            raise ValueError('episode producer/config mismatch')
        if deep and r['split'] not in ('train','val','test'):
            with np.load(path,allow_pickle=False) as z:
                T=r['steps'];F=T+1
                if z['actions'].shape != (T, 12) or z['observations'].shape != (F, 56) or z['proprio'].shape != (F, 13):
                    raise ValueError('auxiliary shape mismatch')
                if z['pixels'].shape != (F, 256, 256, 3) or z['xyz'].shape != (F, 1200, 3):
                    raise ValueError('auxiliary visual shape mismatch')
                for key in ('actions','observations','proprio','qpos','qvel','integration','trace','rewards','xyz'):
                    if not np.isfinite(z[key]).all():raise ValueError('nonfinite auxiliary '+key)
                initial = json.loads(z['initial_state_json'].item())
                fields = ('integration', 'body_pos', 'body_quat', 'site_pos', 'site_quat', 'p', 'v', 'q_cmd')
                digest = hashlib.sha256(
                    json.dumps({k: initial[k] for k in fields}, sort_keys=True, separators=(',', ':')).encode()
                ).hexdigest()
                if digest != r['initial_state_sha256'] or int(z['seed']) != r['seed']:
                    raise ValueError('auxiliary initial/seed mismatch')
                if bool(z['success']) != r['success'] or bool(z['native_success']) != r['native_success']:
                    raise ValueError('auxiliary goal mismatch')
                if json.loads(z['camera_json'].item())!=r['camera']:raise ValueError('auxiliary camera mismatch')
        files.append(dict(path=str(path.resolve()),sha256=r['npz_sha256'],bytes=path.stat().st_size))
        if r['camera']['mode']!='opposite':raise ValueError('wrong task camera')
        if not r['success_consistent']:raise ValueError('unresolved success discrepancy')
    for name in ('paired_rgb.h5','paired_pcd.h5','manifest.json','state.json','config.json'):
        path=attempt/name
        if path.exists():files.append(dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size))
    result = dict(
        status='PASS_BATCH',
        validated=time.time(),
        registry=registry,
        records=records,
        validator_sha256=sha(__file__),
        runtime_helpers_sha256=RUNTIME_HELPERS_SHA256,
        files=files,
        plan_sha256=sha(plan_path) if plan_path else None,
        deep_data_check=deep,
        numeric_contract='T+1 physical states; T actual actions; outgoing HDF terminal NaN',
    )
    save(receipt,result)
    return result


def verify_receipt(attempt):
    """Verify a direct batch's immutable payloads; do not accept a shallow check."""
    attempt=Path(attempt)
    report=json.loads((attempt/'batch_acceptance.json').read_text(encoding='utf-8'))
    if report['status']!='PASS_BATCH' or not report['deep_data_check']:
        raise ValueError('batch requires its full acceptance receipt')
    for file in report['files']:
        path=Path(file['path'])
        if path.stat().st_size!=file['bytes'] or sha(path)!=file['sha256']:
            raise ValueError('accepted payload changed '+str(path))
    return report


def publish_dataset(batches,output,expected_counts):
    """Compose direct, accepted paired batches without rewriting HDF payloads."""
    import h5py
    batches=[Path(p).resolve() for p in batches]
    if not batches or len(set(batches))!=len(batches):
        raise ValueError('provide nonempty, distinct batch directories')
    if (not isinstance(expected_counts,dict) or not expected_counts or
            any(key not in ('train','val','test') or type(count) is not int or count<0
                for key,count in expected_counts.items())):
        raise ValueError('expected counts must map train/val/test to nonnegative integers')
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(f'existing dataset publication: {out}')
    records=[];receipts=[]
    for batch in batches:
        verify_receipt(batch)
        receipts.append(dict(batch=str(batch),acceptance_sha256=sha(batch/'batch_acceptance.json')))
        manifest=json.loads((batch/'manifest.json').read_text(encoding='utf-8'))
        for row in manifest['episodes']:
            records.append(dict(row,episode_index=len(records),source_dir=str(batch),
                                source_episode_index=row['episode_index'],
                                npz=str((batch/row['npz']).resolve())))
    registry=check_registry(records)
    if any(registry['counts'].get(split,0)!=count for split,count in expected_counts.items()):
        raise ValueError('global split counts do not match the requested plan')
    out.mkdir(parents=True,exist_ok=False)
    published=[]
    for route in ('rgb','pcd'):
        columns=None;shapes={};dtypes={};sources=[];lengths=[];offset=0
        for batch in batches:
            path=batch/f'paired_{route}.h5'
            with h5py.File(path,'r') as f:
                keys=sorted(k for k in f if k not in ('ep_len','ep_offset'))
                if columns is not None and columns!=keys:raise ValueError('source schema differs')
                columns=keys;ln=f['ep_len'][:];n=int(ln.sum());lengths.extend(ln.tolist())
                np.testing.assert_array_equal(f['ep_offset'][:],np.r_[0,np.cumsum(ln)[:-1]])
                for key in keys:
                    if len(f[key])!=n:raise ValueError('source row count differs')
                    if key in shapes and (shapes[key]!=f[key].shape[1:] or dtypes[key]!=f[key].dtype):
                        raise ValueError('source column shape/dtype differs')
                    shapes[key]=f[key].shape[1:];dtypes[key]=f[key].dtype
                sources.append(dict(path=os.path.relpath(path,out),rows=n,offset=offset));offset+=n
        destination=out/f'paired_{route}.h5'
        with h5py.File(destination,'w',libver='latest') as f:
            for key in columns:
                layout=h5py.VirtualLayout(shape=(offset,)+shapes[key],dtype=dtypes[key])
                for source in sources:
                    layout[source['offset']:source['offset']+source['rows']]=h5py.VirtualSource(
                        source['path'],key,shape=(source['rows'],)+shapes[key])
                f.create_virtual_dataset(key,layout)
            f['ep_len']=np.asarray(lengths,np.int32)
            f['ep_offset']=np.r_[0,np.cumsum(lengths)[:-1]].astype(np.int64)
        published.append(dict(route=route,path=str(destination),sha256=sha(destination),rows=offset,sources=sources))
    for batch,receipt in zip(batches,receipts):
        if sha(batch/'batch_acceptance.json')!=receipt['acceptance_sha256']:
            raise ValueError('batch acceptance changed during publication')
    manifest=dict(action_alignment='outgoing; last row NaN',episodes=records,registry=registry,
                  source_acceptance_checked=True,source_receipts=receipts,
                  publication='VDS only; immutable raw batch files retained',datasets=published)
    save(out/'manifest.json',manifest)
    return dict(status='PASS_VDS',episodes=len(records),registry=registry,
                manifest=str(out/'manifest.json'),datasets=published)


def _expected_counts(value):
    if value.lstrip().startswith('{'):
        return json.loads(value)
    return json.loads(Path(value).read_text(encoding='utf-8'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    validate=commands.add_parser('validate',help='deep-check one direct collection batch')
    validate.add_argument('--batch',required=True,type=Path)
    validate.add_argument('--plan',type=Path)
    validate.add_argument('--output',type=Path,help='fresh receipt JSON; default BATCH/batch_acceptance.json')
    publish=commands.add_parser('publish',help='publish paired VDS headers from accepted direct batches')
    publish.add_argument('--batches',required=True,nargs='+',type=Path)
    publish.add_argument('--output',required=True,type=Path)
    publish.add_argument('--expected-counts',required=True,type=_expected_counts,
                         help='JSON object or path to JSON split-count object')
    args=parser.parse_args(argv)
    if args.command=='validate':
        result=validate_batch(args.batch,args.plan,report_path=args.output)
        shown=dict(status=result['status'],registry=result['registry'],
                   report=str(args.output or args.batch/'batch_acceptance.json'))
    else:
        shown=publish_dataset(args.batches,args.output,args.expected_counts)
    print(json.dumps(shown,indent=2))


if __name__=='__main__':main()

"""Publish schema-preserving VDS headers from immutable committed paired batches."""
import argparse,json,os
from pathlib import Path
import h5py
import numpy as np
from paired_parent import sha,save,equal

def publish(args):
 data=args.data_root.resolve();out=args.out.resolve();assert out.is_relative_to(data),'VDS output must stay in persistent data root'
 assert not out.exists(),'fresh publisher output required';out.mkdir(parents=True)
 plan=json.loads(args.plan.read_text());state=json.loads((Path(plan['root'])/'scheduler_state.json').read_text());assert state['config_sha256']==sha(args.plan),'published plan SHA differs'
 assert state['phase']=='complete' and state['exit_code']==0
 jobs={j['id']:j for j in plan['jobs']};assert all(j['split'] in ('train','test') for j in jobs.values()),'diagnostic bank cannot be published as training data'
 records=[];registries={k:set() for k in ('seed','episode_id','id','native_initial_sha256','full_initial_sha256')};payload_paths=set()
 assert len(state['completed'])==len(set(state['completed']))==len(jobs),'incomplete or duplicate job publication'
 for name in state['completed']:
  status=json.loads((Path(plan['root'])/name/'state.json').read_text());commit=Path(status['attempt'])/'paired/commit.json';c=json.loads(commit.read_text());assert c['qc']=='PASS' and c['exit_code']==0
  attempt=commit.parent.parent
  for stage in range(3):assert (attempt/('stage_%d.exit_code'%stage)).read_text().strip()=='0'
  paired_state=json.loads((commit.parent/'state.json').read_text());assert paired_state['phase']=='committed' and paired_state['exit_code']==0 and (commit.parent/'exit_code').read_text().strip()=='0'
  assert c['manifest_sha256']==sha(attempt/'paired_manifest.json')
  expected=jobs[name]['episodes'];assert [(e['episode_id'],e['seed']) for e in c['episodes']]==[(e['episode_id'],e['seed']) for e in expected]
  lo,hi=plan['seed_ranges_half_open'][jobs[name]['split']];assert all(lo<=e['seed']<hi for e in c['episodes'])
  for ep in c['episodes']:
   for key,seen in registries.items():assert ep[key] not in seen,'duplicate episode publication '+key;seen.add(ep[key])
  for row in c['files']:
   p=commit.parent/row['name'];assert str(p.resolve()) not in payload_paths,'duplicate payload publication';payload_paths.add(str(p.resolve()));assert p.resolve().is_relative_to(data) and sha(p)==row['sha256']
  records.append(dict(job=name,split=jobs[name]['split'],commit=str(commit),commit_sha256=sha(commit),episodes=c['episodes'],files=c['files']))
 manifests=[]
 try:
  for split in ('train','test'):
   batches=sorted((r for r in records if r['split']==split),key=lambda r:r['job'])
   if not batches:continue
   for modality in ('rgb','pcd'):
    paths=[Path(r['commit']).parent/(modality+'.h5') for r in batches]
    columns=None;shapes={};dtypes={};lens=[];total=0;sources=[]
    for p in paths:
     with h5py.File(p,'r') as f:
      keys=sorted(k for k in f if k not in ('ep_len','ep_offset'));assert columns is None or columns==keys;columns=keys
      lengths=f['ep_len'][:];np.testing.assert_array_equal(f['ep_offset'][:],np.r_[0,np.cumsum(lengths)[:-1]])
      n=int(lengths.sum());lens.extend(lengths.tolist())
      for k in keys:
       ds=f[k];assert len(ds)==n
       if k in shapes:assert ds.shape[1:]==shapes[k] and ds.dtype==dtypes[k]
       shapes[k]=ds.shape[1:];dtypes[k]=ds.dtype
      sources.append(dict(path=str(p),relative_path=os.path.relpath(p,out),sha256=sha(p),rows=n,offset=total));total+=n
    destination=out/(split+'_'+modality+'.h5')
    with h5py.File(destination,'w',libver='latest') as f:
     for k in columns:
      layout=h5py.VirtualLayout(shape=(total,)+shapes[k],dtype=dtypes[k])
      for s in sources:layout[s['offset']:s['offset']+s['rows']]=h5py.VirtualSource(s['relative_path'],k,shape=(s['rows'],)+shapes[k])
      f.create_virtual_dataset(k,layout)
     lengths=np.asarray(lens,np.int32);f.create_dataset('ep_len',data=lengths);f.create_dataset('ep_offset',data=np.r_[0,np.cumsum(lengths)[:-1]].astype(np.int64))
    with h5py.File(destination,'r') as f:
     for s in sources:
      assert sha(s['path'])==s['sha256']
      with h5py.File(s['path'],'r') as raw:
       for k in columns:
        for start in range(0,s['rows'],1000):
         end=min(start+1000,s['rows']);assert equal(f[k][s['offset']+start:s['offset']+end],raw[k][start:end]),'VDS mapping differs'
    manifests.append(dict(path=str(destination),sha256=sha(destination),bytes=destination.stat().st_size,source_payloads=sources,episodes=len(lens),rows=total))
  save(out/'manifest.json',dict(vds=manifests,batches=records,sources_retained=True,qc='PASS',exit_code=0));(out/'exit_code').write_text('0\n')
 except BaseException as error:save(out/'failure.json',dict(error=repr(error),exit_code=1));(out/'exit_code').write_text('1\n');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);publish(p.parse_args())

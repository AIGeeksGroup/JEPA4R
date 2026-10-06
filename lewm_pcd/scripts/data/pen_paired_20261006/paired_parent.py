"""Modern environment transaction wrapper around the project's original paired writers."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def equal(a,b):
 return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=a.dtype.kind=='f')
SHARED_COLUMNS=('qpos','qvel','desired_orien','initial_qpos','initial_qvel','initial_desired_orien','native_observation','initial_native_observation','proprio','state_sha256','initial_state_sha256','legacy_initial_state_sha256','episode_seed','native_step_idx','executed_action','executed_action_valid','sample_type','noise_std','branch_t','branch_len','rand_action','goal_achieved','task_success','observation','reward','terminated','truncated','action','step_idx','id')
def episode_check(ep,meta,image_size=256):
 assert set(ep)==set(SHARED_COLUMNS)|{'pixels','render_time','pcd_xyz','pcd_rgb'},'World schema differs'
 n=len(ep['observation']);assert 2<=n<=101
 assert all(len(v)==n for v in ep.values()),'nonuniform episode columns'
 assert ep['render_time'].shape==(n,) and np.isfinite(ep['render_time']).all() and np.all(ep['render_time']>=0)
 assert ep['pixels'].shape==(n,image_size,image_size,3) and ep['pixels'].dtype==np.uint8
 assert ep['pcd_xyz'].shape==ep['pcd_rgb'].shape==(n,1200,3)
 assert ep['pcd_xyz'].dtype==np.float32 and ep['pcd_rgb'].dtype==np.uint8
 assert ep['observation'].shape==(n,45) and ep['observation'].dtype.kind=='f'
 np.testing.assert_array_equal(ep['observation'],ep['native_observation'])
 np.testing.assert_array_equal(ep['native_step_idx'],ep['step_idx'])
 for k in ('qpos','qvel','initial_qpos','initial_qvel'):assert ep[k].shape==(n,30) and ep[k].dtype==np.float64
 for k in ('desired_orien','initial_desired_orien'):assert ep[k].shape==(n,4) and ep[k].dtype==np.float64
 for k in ('state_sha256','initial_state_sha256'):assert ep[k].shape==(n,32) and ep[k].dtype==np.uint8
 assert ep['legacy_initial_state_sha256'].shape==(n,64) and ep['legacy_initial_state_sha256'].dtype==np.uint8
 assert ep['initial_native_observation'].shape==(n,45)
 np.testing.assert_array_equal(ep['proprio'],ep['qpos'][:,:24])
 assert ep['qpos'].shape==ep['qvel'].shape==(n,30)
 assert ep['proprio'].shape==(n,24)
 assert ep['action'].dtype==np.float32
 assert ep['action'].shape==(n,24) and np.isnan(ep['action'][-1]).all()
 assert np.isfinite(ep['action'][:-1]).all() and np.max(np.abs(ep['action'][:-1]))<=1
 assert np.isnan(ep['reward'][0]).all() and np.isfinite(ep['reward'][1:]).all()
 for k,v in ep.items():
  if v.dtype.kind in 'fiu' and k not in ('action','reward'):assert np.isfinite(v).all(),k
 assert ep['executed_action'].shape==(n,24)
 np.testing.assert_array_equal(ep['executed_action'][1:],ep['action'][:-1])
 assert int(ep['executed_action_valid'][0])==0 and np.all(ep['executed_action_valid'][1:]==1)
 assert np.all(ep['episode_seed']==meta['seed']) and int(ep['sample_type'][0]) in (0,1,2)
 assert ep['sample_type'].dtype==np.int64 and ep['rand_action'].dtype==np.int64
 assert np.isin(ep['rand_action'],[0,1]).all(),'random action flag must be exact0/1'
 assert ep['executed_action_valid'].dtype==np.int64
 assert ep['terminated'].dtype==ep['truncated'].dtype==np.bool_
 assert np.all(ep['sample_type']==ep['sample_type'][0])
 assert ep['id'].dtype==np.int64 and np.all(ep['id']==ep['id'][0])
 for key in ('noise_std','branch_t','branch_len'):
  assert np.all(ep[key]==ep[key][0]),'episode plan changed: '+key
 kind=int(ep['sample_type'][0]);noise=float(ep['noise_std'][0]);b0=int(ep['branch_t'][0]);bl=int(ep['branch_len'][0])
 assert 0<=noise<=.2
 if kind==1:assert 15<=b0<60 and (10<=bl<=30 or bl==100)
 else:assert b0==-1 and bl==0
 if kind==2:assert noise==0
 expected_random=(np.arange(n)>=b0)&(np.arange(n)<b0+bl) if kind==1 else np.full(n,kind==2)
 np.testing.assert_array_equal(ep['rand_action'].astype(bool),expected_random)
 assert np.array_equal(ep['step_idx'],np.arange(n))
 assert bool(ep['terminated'][-1]) or bool(ep['truncated'][-1])
 assert not (bool(ep['terminated'][-1]) and bool(ep['truncated'][-1]))
 assert not np.any(ep['terminated'][:-1]) and not np.any(ep['truncated'][:-1])
 if bool(ep['truncated'][-1]):assert n==101,'unexpected short horizon truncation'
 assert meta['strict_replay_gate']=='PASS' and meta['render_gate']=='PASS'
 return n

def run(args):
 import h5py
 sys.path.insert(0,str(args.code_root/'lewm_pcd'));sys.path.insert(0,str(args.code_root/'lewm_pcd/scripts/data'))
 from collect_cube_pix import GzipHDF5Writer
 from collect_cube_pcd import PairedWriter
 spec=json.loads(args.manifest.read_text());out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
 commit=out/'commit.json'
 if commit.exists():
  old=json.loads(commit.read_text());prior_state=json.loads((out/'state.json').read_text());assert prior_state['phase']=='committed' and prior_state['exit_code']==0 and (out/'exit_code').read_text().strip()=='0'
  assert old['qc']=='PASS' and old['exit_code']==0
  assert old['manifest_sha256']==sha(args.manifest)
  for row in old['files']:assert sha(out/row['name'])==row['sha256']
  return 0
 assert not (out/'rgb.h5').exists() and not (out/'pcd.h5').exists(),'uncommitted staging exists: retain and use new attempt directory'
 records=[];seen_seeds=set();seen_initial=set();seen_native=set();seen_ids=set();seen_episode_ids=set();state={'phase':'writing','started_at':time.time(),'updated_at':time.time(),'completed':0,'failed':0,'exit_code':None};save(out/'state.json',state)
 try:
  with PairedWriter(GzipHDF5Writer(out/'rgb.h5',mode='error'),GzipHDF5Writer(out/'pcd.h5',mode='error',gzip_cols=())) as writer:
   for item in spec['episodes']:
    meta=json.loads(Path(item['metadata']).read_text())
    assert item['episode_id']==meta['episode_id'],'planned episode ID differs'
    assert meta['config_sha256']==spec['config_sha256'] and meta['source_hashes']==spec['expected_source_hashes'],'worker/source configuration seal differs'
    assert meta['image_size']==int(spec.get('image_size',args.image_size))
    assert sha(item['pcd'])==meta['payload_sha256'] and sha(item['rgb'])==meta['rgb_payload_sha256'],'worker payload SHA mismatch'
    audit_path=Path(item['pcd']).with_name(Path(item['pcd']).stem+'_audit.npz')
    assert sha(audit_path)==meta['audit_sha256'],'PCD audit mutated'
    assert sha(meta['rgb_metadata_path'])==meta['rgb_metadata_sha256'],'RGB metadata mutated'
    assert meta['seed'] not in seen_seeds and meta['full_initial_sha256'] not in seen_initial and meta['native_initial_sha256'] not in seen_native
    seen_seeds.add(meta['seed']);seen_initial.add(meta['full_initial_sha256']);seen_native.add(meta['native_initial_sha256'])
    files=[Path(item['rgb']),Path(item['pcd']),Path(item['metadata'])];hashes=[sha(p) for p in files]
    with np.load(files[0],allow_pickle=False) as r,np.load(files[1],allow_pickle=False) as p:
     for k in set(r.files)&set(p.files):assert equal(r[k],p[k]),'paired column differs: '+k
     ep={k:r[k].copy() for k in r.files};ep.update({k:p[k].copy() for k in p.files})
    n=episode_check(ep,meta,int(spec.get('image_size',args.image_size)))
    assert int(ep['id'][0]) not in seen_ids and item['episode_id'] not in seen_episode_ids,'duplicate episode ID'
    seen_ids.add(int(ep['id'][0]));seen_episode_ids.add(item['episode_id'])
    assert bool(ep['terminated'][-1])==bool(meta['terminated']) and bool(ep['truncated'][-1])==bool(meta['truncated'])
    assert meta['terminal_reason']==('native_done' if ep['terminated'][-1] else 'horizon')
    assert int(np.sum(ep['goal_achieved'][1:]))==int(meta['goal_rows'])
    for key in ('sample_type','noise_std','branch_t','branch_len'):assert np.asarray(ep[key][0]).item()==meta[key],'drawn plan metadata differs: '+key
    native_digest=np.frombuffer(bytes.fromhex(meta['native_initial_sha256']),np.uint8)
    assert np.all(ep['initial_state_sha256']==native_digest[None,:]),'native initial digest rows differ'
    writer.write_episodes([ep])
    assert [sha(p) for p in files]==hashes,'input changed'
    records.append(dict(episode_id=item['episode_id'],id=int(ep['id'][0]),seed=meta['seed'],full_initial_sha256=meta['full_initial_sha256'],native_initial_sha256=meta['native_initial_sha256'],rows=n,sample_type=int(ep['sample_type'][0]),source_files=[dict(path=str(p),sha256=h) for p,h in zip(files,hashes)]))
    state['completed']=len(records);state['updated_at']=time.time();save(out/'state.json',state)
  with h5py.File(out/'rgb.h5','r') as r,h5py.File(out/'pcd.h5','r') as p:
   common=set(r)&set(p)
   for k in common:assert equal(r[k][:],p[k][:]),'persisted shared column differs: '+k
   lengths=np.array([v['rows'] for v in records]);np.testing.assert_array_equal(r['ep_len'][:],lengths);np.testing.assert_array_equal(r['ep_offset'][:],np.r_[0,np.cumsum(lengths)[:-1]])
   for index,row in enumerate(records):
    start=int(r['ep_offset'][index]);end=start+row['rows']
    combined={k:r[k][start:end] for k in r if k not in ('ep_len','ep_offset')};combined.update({k:p[k][start:end] for k in p if k not in ('ep_len','ep_offset')})
    episode_check(combined,dict(seed=row['seed'],strict_replay_gate='PASS',render_gate='PASS'),int(spec.get('image_size',args.image_size)))
  rng_path=Path(spec['policy_rng_end']);rng_seal=dict(path=str(rng_path),sha256=sha(rng_path))
  last_rgb_meta=json.loads(Path(records[-1]['source_files'][0]['path']).with_suffix('.json').read_text())
  assert json.loads(rng_path.read_text())==last_rgb_meta['policy_rng_after'],'policy RNG ending state differs from final RGB episode'
  result={'policy_rng_end':rng_seal,'manifest_sha256':sha(args.manifest),'episodes':records,'files':[dict(name=name,sha256=sha(out/name),bytes=(out/name).stat().st_size) for name in ('rgb.h5','pcd.h5')],'qc':'PASS','exit_code':0}
  save(commit,result);state.update(phase='committed',exit_code=0);return 0
 except BaseException as error:
  state.update(phase='failed',failed=1,updated_at=time.time(),error=repr(error),exit_code=1);raise
 finally:save(out/'state.json',state);(out/'exit_code').write_text(str(state['exit_code'])+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--code-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--image-size',type=int,choices=(224,256),default=256);a=p.parse_args();raise SystemExit(run(a))

import hashlib,json,sys
from pathlib import Path
import numpy as np,pytest,torch
sys.path.insert(0,str(Path(__file__).parent))
from cache_dataset import CachedCubeStateLatentDataset

@pytest.fixture
def fixture(tmp_path):
 root=tmp_path/'cache';numeric=tmp_path/'numeric';root.mkdir();(numeric/'train').mkdir(parents=True)
 action=np.arange(2002*5,dtype=np.float32).reshape(2002,5)/np.float32(100000)
 obs=np.arange(2002*28,dtype=np.float32).reshape(2002,28)
 np.save(numeric/'train/action.npy',action);np.save(numeric/'train/observation.npy',obs)
 entries=[];stats={}
 for ep in range(2):
  p=root/'train'/f'episode_{ep:04d}'/'latent.npy';p.parent.mkdir(parents=True)
  np.save(p,np.repeat((np.arange(1000,dtype=np.float32)+ep*10000)[:,None],192,axis=1))
  s=p.stat();stats[str(p)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
  entries.append(dict(episode=ep,path=str(p),shape=[1000,192],dtype='float32'))
 index=tmp_path/'index.json';index.write_text(json.dumps(dict(status='passed',exit_code=0,train=[dict(episode=0),dict(episode=1)],val=[])))
 manifest=root/'cache_manifest.json';manifest.write_text(json.dumps(dict(schema_version=1,status='passed',exit_code=0,source_rgb_index=str(index),source_rgb_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),splits=dict(train=entries,val=[]),numeric_sources=dict(train=dict(action_path=str(numeric/'train/action.npy'),observation_path=str(numeric/'train/observation.npy'))))))
 seal=root/'seal.json';seal.write_text(json.dumps(dict(status='passed',exit_code=0,manifest_path=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),cache_stats=stats)))
 return root,numeric,index,seal,action,obs

def test_raw_compact_boundary_and_original_state_wrapper(fixture):
 root,numeric,index,seal,action,obs=fixture
 ds=CachedCubeStateLatentDataset(root,'train',numeric,index,seal)
 assert len(ds)==2000 and ds.rows==1001 and ds.transitions==1000 and ds.readers=={}
 for cur,goal,raw,graw in [(0,1,0,1),(999,999,999,999),(1000,1001,1001,1002),(1999,1999,2000,2000)]:
  row=ds[(cur,goal)];assert int(row['source_row'])==raw and int(row['goal_source_row'])==graw
  assert row['current'].shape==row['goal'].shape==(192,) and row['current'].dtype==torch.float32
  assert torch.equal(row['action'],torch.from_numpy(action[raw]))
  assert 'current_state' not in row
 sys.path.insert(0,str(Path(__file__).parents[2]))
 from dataset_state_fusion import CubeStateVisualDataset
 wrapped=CubeStateVisualDataset(ds,numeric/'train/observation.npy');row=wrapped[(1000,1001)]
 assert torch.equal(row['current_state'],torch.from_numpy(obs[1001])) and torch.equal(row['goal_state'],torch.from_numpy(obs[1002]))

def test_invalid_goal_and_changed_cache_witness(fixture):
 root,numeric,index,seal,_,_=fixture
 ds=CachedCubeStateLatentDataset(root,'train',numeric,index,seal)
 for pair in [(999,1000),(1000,1000),(-1,0),(2000,2000)]:
  with pytest.raises(ValueError):ds[pair]
 p=root/'train/episode_0000/latent.npy'
 with p.open('ab') as out:out.write(b'changed')
 with pytest.raises(ValueError):ds[(0,1)]

def test_exact_val_tail_pair_override_not_goal_cache(fixture):
 root,numeric,index,seal,_,_=fixture
 (numeric/'val').mkdir();np.save(numeric/'val/action.npy',np.zeros((2002,5),np.float32))
 idx=json.loads(index.read_text());idx['val']=idx['train'];index.write_text(json.dumps(idx))
 mf=root/'cache_manifest.json';m=json.loads(mf.read_text());m['splits']['val']=m['splits']['train'];m['numeric_sources']['val']=dict(action_path=str(numeric/'val/action.npy'));m['source_rgb_index_sha256']=hashlib.sha256(index.read_bytes()).hexdigest()
 p=root/'val_tail32.npz';cur=np.arange(32,dtype=np.int64);goal=cur+100
 np.savez(p,current_compact=cur,goal_compact=goal,current_latents=np.ones((32,192),np.float32)*-7,goal_latents=np.ones((32,192),np.float32)*-8)
 s=p.stat();m['val_tail32']=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),stat=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev));mf.write_text(json.dumps(m));z=json.loads(seal.read_text());z['manifest_sha256']=hashlib.sha256(mf.read_bytes()).hexdigest();seal.write_text(json.dumps(z))
 ds=CachedCubeStateLatentDataset(root,'val',numeric,index,seal)
 assert torch.equal(ds[(0,100)]['current'],torch.full((192,),-7.)) and torch.equal(ds[(0,100)]['goal'],torch.full((192,),-8.))
 with pytest.raises(ValueError):ds[(0,101)]
 # Another current with the same goal must retain its main64 row.
 assert torch.equal(ds[(50,100)]['goal'],torch.full((192,),100.))

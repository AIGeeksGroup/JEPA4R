"""Resumable per-episode frozen RGB latent producer; no training dispatch."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def stat(path):
 s=Path(path).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
def save(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,path)

class EpisodeFrames(torch.utils.data.Dataset):
 def __init__(self,index,split,preprocessor,episodes):
  from dataset import CubeOfficialDataset
  self.data=CubeOfficialDataset(index,split,preprocessor,open_files=2);self.episodes=episodes
 def __len__(self):return len(self.episodes)
 def __getitem__(self,i):
  ep=self.episodes[i];reader=self.data._reader(ep);frames=[]
  for step in range(1000):
   row=reader._load_slice(0,step,step+1)
   if int(row['source_row'][0])!=ep*1001+step or int(row['source_episode'][0])!=ep or row['source_terminal'].any():raise ValueError('Original valid raw source row differs')
   frames.append(row['pixels'][0])
  return dict(episode=ep,frames=torch.stack(frames))

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--limit-episodes',type=int);p.add_argument('--output',type=Path);a=p.parse_args();plan=read(a.plan);plan_sha=sha(a.plan)
 if not a.execute:
  print(json.dumps(dict(status='prepared_only_no_encoder_or_RGB_reads',plan=str(a.plan),workers=16,prefetch_factor=2,batch64=True,tail40_pad24_discard=True)));return 0
 root=(a.output or Path(plan['cache_root'])).resolve();root.mkdir(parents=True,exist_ok=True)
 if (root/'cache_seal.json').exists():raise ValueError('Immutable sealed cache already exists')
 save(root/'status.json',dict(stage='starting',status='running',started_at=time.time(),completed_valid_frames=0,exit_code=None))
 for path,h in plan['source_pins'].items():
  if sha(path)!=h:raise ValueError('Cache source input changed:'+path)
 code=Path(plan['code_root']);repo=Path(plan['repo_root'])/'lewm_pcd';sys.path[:0]=[str(code),plan['smoke_utils_root'],str(repo),str(repo/'third_party/stable-worldmodel')]
 import hydra
 from omegaconf import OmegaConf
 import trainer as original
 from lewm_gcbc_adapter import LeWMFrozenEncoder
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 if not torch.cuda.is_available():raise ValueError('Root-approved CUDA producer runtime required')
 cfg=OmegaConf.load(Path(plan['source_checkpoint']).parent/'config.yaml');preprocessor=hydra.utils.instantiate(cfg.data.preprocessor)
 model=LeWMFrozenEncoder(source_checkpoint=plan['source_checkpoint'],latent_dim=192).cuda().eval()
 if any(x.training for x in model.modules()) or any(x.requires_grad for x in model.parameters()):raise ValueError('Frozen encoder/projector eval required')
 if any(isinstance(x,torch.nn.BatchNorm1d) and (not x.track_running_stats or x.running_mean is None) for x in model.modules()):raise ValueError('Fixed BN running stats required')
 frozen=original.module_hash(model)
 if frozen!=plan['encoder_hashes']['trainer_numpy_dtype']:raise ValueError('Actual source encoder hash differs')
 index=read(plan['source_rgb_index']);records={};total=0;source_stats={};producer_sha=sha(__file__)
 for split in ['train','val']:
  selected=list(range(len(index[split])));selected=selected[:a.limit_episodes] if a.limit_episodes else selected;pending=[];records[split]=[]
  for ep in selected:
   epdir=root/split/f'episode_{ep:04d}';receipt=epdir/'receipt.json';src=index[split][ep]['path'];source_stats[src]=stat(src)
   if receipt.exists():
    record=read(receipt);latent=Path(record['path'])
    if record['cache_plan_sha256']!=plan_sha or record['producer_source_sha256']!=producer_sha or record['source_rgb_stat_before']!=stat(src) or record['sha256']!=sha(latent):raise ValueError('Committed cache resume identity differs')
    records[split].append(record);total+=1000
   else:pending.append(ep)
  dataset=EpisodeFrames(plan['source_rgb_index'],split,preprocessor,pending)
  loader=original.make_loader(dataset,[[i] for i in range(len(pending))],16,pin_memory=True);iterator=iter(loader)
  try:
   for batch in iterator:
    ep=int(batch['episode'][0]);frames=batch['frames'][0];src=index[split][ep]['path'];before=source_stats[src];values=[]
    for start in range(0,1000,64):
     image=frames[start:start+64];valid=len(image)
     if valid<64:image=torch.cat((image,image[:64-valid]),0)
     with torch.inference_mode():latent=model(image.cuda(non_blocking=True)).logits.float().cpu().numpy()[:valid].copy()
     if latent.shape!=(valid,192) or not np.isfinite(latent).all():raise ValueError('Invalid real latent')
     values.append(latent)
    value=np.concatenate(values);epdir=root/split/f'episode_{ep:04d}';epdir.mkdir(parents=True,exist_ok=True);path=epdir/'latent.npy';tmp=epdir/'latent.partial.npy'
    if path.exists():os.replace(path,epdir/('latent.uncommitted_'+str(time.time_ns())+'.npy'))
    np.save(tmp,value,allow_pickle=False);os.replace(tmp,path)
    if stat(src)!=before:raise ValueError('RGB source changed during encoding')
    record=dict(episode=ep,path=str(path),shape=[1000,192],dtype='float32',sha256=sha(path),source_rgb_path=src,source_rgb_stat_before=before,source_rgb_stat_after=stat(src),source_row_start=ep*1001,source_row_end_exclusive=ep*1001+1000,source_valid_frames=1000,producer_encoded_frames=1024,producer_padding_frames=24,producer_source_sha256=producer_sha,cache_plan_sha256=plan_sha)
    save(epdir/'receipt.json',record);records[split].append(record);total+=1000;save(root/'status.json',dict(stage='encoding',status='running',completed_valid_frames=total,split=split,episode=ep,exit_code=None))
  finally:original.cleanup_loader(iterator)
  records[split].sort(key=lambda x:x['episode'])
 # Reproduce the original complete validation sampler traversal, then encode its real B32 tail.
 from dataset import CubeOfficialDataset,LogicalSampler
 sampler=LogicalSampler(100000,micro_batch=64,accumulation=1,seed=0)
 for pairs in sampler:tail=pairs
 if len(tail)!=32:raise ValueError('Original validation tail is not B32')
 data=CubeOfficialDataset(plan['source_rgb_index'],'val',preprocessor,open_files=16)
 for ep in set(x//1000 for pair in tail for x in pair):
  src=index['val'][ep]['path'];source_stats.setdefault(src,stat(src))
 pairrows=[data[pair] for pair in tail];tailvalues={}
 with torch.inference_mode():
  for key in ['current','goal']:tailvalues[key+'_latents']=model(torch.stack([x[key] for x in pairrows]).cuda()).logits.float().cpu().numpy().copy()
 for reader in data.readers.values():
  if reader.h5_file is not None:reader.h5_file.close()
 current=np.array([x[0] for x in tail],dtype=np.int64);goal=np.array([x[1] for x in tail],dtype=np.int64);tailpath=root/'val_tail32.npz'
 if tailpath.exists():os.replace(tailpath,root/('val_tail32.uncommitted_'+str(time.time_ns())+'.npz'))
 np.savez(tailpath,current_compact=current,goal_compact=goal,source_rows=current//1000*1001+current%1000,goal_source_rows=goal//1000*1001+goal%1000,**tailvalues)
 if any(x.shape!=(32,192) or x.dtype!=np.float32 or not np.isfinite(x).all() for x in tailvalues.values()):raise ValueError('Real validation tail latent invalid')
 tailrecord=dict(path=str(tailpath),sha256=sha(tailpath),stat=stat(tailpath),pairs=32,source_batch=32,sampler_source_sha256=sha(code/'dataset.py'),sampler=dict(size=100000,micro_batch=64,accumulation=1,seed=0,full_traversal=True),torch_version=torch.__version__,numpy_version=np.__version__,sampler_state_digest=hashlib.sha256(sampler.generator.get_state().numpy().tobytes()+sampler.goal_sampler.rng.get_state()[1].tobytes()).hexdigest(),precision=plan['precision'],source_rows=(current//1000*1001+current%1000).tolist(),goal_source_rows=(goal//1000*1001+goal%1000).tolist())
 save(root/'val_tail32_receipt.json',tailrecord)
 if original.module_hash(model)!=frozen:raise ValueError('Frozen source encoder changed')
 if any(stat(p)!=s for p,s in source_stats.items()):raise ValueError('RGB sources changed')
 if a.limit_episodes:
  save(root/'pilot_receipt.json',dict(status='passed_diagnostic_cache_only',exit_code=0,splits=records,frames=total,source_encoder_frozen=frozen,not_full_cache=True,val_tail32=tailrecord));return 0
 if len(records['train'])!=1000 or len(records['val'])!=100:raise ValueError('Full official split cache required')
 numeric=plan['numeric_sources']
 for item in numeric.values():
  for kind in ['action','observation']:
   if sha(item[kind+'_path'])!=item[kind+'_sha256']:raise ValueError('Original numeric source changed')
 manifest=dict(plan,schema_version=1,status='passed',exit_code=0,splits=records,producer_source_sha256=producer_sha,encoder_hashes=plan['encoder_hashes'],val_tail32=tailrecord)
 save(root/'cache_manifest.json',manifest)
 stats={item['path']:stat(item['path']) for rows in records.values() for item in rows}
 for item in [x for rows in records.values() for x in rows]:
  value=np.load(item['path'],mmap_mode='r',allow_pickle=False)
  if value.shape!=(1000,192) or value.dtype!=np.float32 or not np.isfinite(value).all() or sha(item['path'])!=item['sha256']:raise ValueError('Final full cache audit differs')
 save(root/'cache_seal.json',dict(status='passed',exit_code=0,manifest_path=str(root/'cache_manifest.json'),manifest_sha256=sha(root/'cache_manifest.json'),cache_stats=stats,train_valid_frames=1000000,val_valid_frames=100000,source_RGB_stats_exact=True,encoder_frozen=True,feature_precision_equivalence='Separate root numerical gate required; no bitexact claim'))
 save(root/'status.json',dict(stage='sealed_full_cache',status='passed',completed_valid_frames=total,exit_code=0));return 0

if __name__=='__main__':
 try:code=main()
 except BaseException as error:
  import traceback;traceback.print_exc();code=1
  try:
   argv=sys.argv;plan=read(argv[argv.index('--plan')+1]);root=Path(argv[argv.index('--output')+1]) if '--output' in argv else Path(plan['cache_root']);root.mkdir(parents=True,exist_ok=True)
   if not (root/'cache_seal.json').exists():save(root/'status.json',dict(stage='producer_failed',status='failed',updated_at=time.time(),error=repr(error),exit_code=1))
  except Exception:pass
 if '--execute' in sys.argv:
  plan=read(sys.argv[sys.argv.index('--plan')+1]);root=Path(sys.argv[sys.argv.index('--output')+1]) if '--output' in sys.argv else Path(plan['cache_root'])
  if code==0 or not (root/'cache_seal.json').exists():(root/'producer_exit_code').write_text(str(code)+'\n')
 raise SystemExit(code)

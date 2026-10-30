"""Read frozen compact latents/action rows; original wrapper owns raw states."""
from collections import OrderedDict
import hashlib,json
from pathlib import Path
import numpy as np
import torch

def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stat(path):
 s=Path(path).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)

class CachedCubeStateLatentDataset(torch.utils.data.Dataset):
 def __init__(self,cache_root,split,numeric_root,original_index,cache_seal):
  self.root=Path(cache_root).resolve();self.split=split;self.rows=1001;self.transitions=1000;self.readers={}
  if split not in ('train','val'):raise ValueError('Explicit independent split required')
  index=read(original_index);seal=read(cache_seal);manifest_path=Path(seal['manifest_path']);manifest=read(manifest_path)
  if any(x.get('status')!='passed' or x.get('exit_code')!=0 for x in [index,seal,manifest]):raise ValueError('Complete accepted index/cache required')
  if digest(manifest_path)!=seal['manifest_sha256'] or digest(original_index)!=manifest['source_rgb_index_sha256']:raise ValueError('Cache/source manifest SHA changed')
  self.tail={}
  if split=='val' and 'val_tail32' in manifest:
   t=manifest['val_tail32'];p=Path(t['path']).resolve()
   if not p.is_relative_to(self.root) or stat(p)!=t['stat'] or digest(p)!=t['sha256']:raise ValueError('Validation B32 pair cache changed')
   with np.load(p,allow_pickle=False) as z:
    if z['current_latents'].shape!=(32,192) or z['goal_latents'].shape!=(32,192) or z['current_latents'].dtype!=np.float32 or z['goal_latents'].dtype!=np.float32 or len(set(z['current_compact'].tolist()))!=32:raise ValueError('Actual validation tail32 schema differs')
    self.tail={int(c):(int(g),np.array(x,copy=True),np.array(y,copy=True)) for c,g,x,y in zip(z['current_compact'],z['goal_compact'],z['current_latents'],z['goal_latents'])}
  self.entries=index[split];self.cached=manifest['splits'][split];self.stats=seal['cache_stats'];self._maps=OrderedDict();self._action=None
  if [x['episode'] for x in self.entries]!=list(range(len(self.entries))) or [x['episode'] for x in self.cached]!=list(range(len(self.entries))):raise ValueError('Cache compact episode order differs')
  for entry in self.cached:
   p=Path(entry['path']).resolve()
   if not p.is_relative_to(self.root) or entry['shape']!=[1000,192] or entry['dtype']!='float32':raise ValueError('Cache path/float32 global192 differs')
  self.action_path=(Path(numeric_root)/split/'action.npy').resolve()
  if self.action_path!=Path(manifest['numeric_sources'][split]['action_path']).resolve():raise ValueError('Original numeric action owner differs')
 def __len__(self):return len(self.entries)*1000
 def __getstate__(self):
  state=self.__dict__.copy();state['_maps']=OrderedDict();state['_action']=None;return state
 def _latent(self,ep,step):
  if ep not in self._maps:
   p=Path(self.cached[ep]['path']).resolve()
   if stat(p)!=self.stats[str(p)]:raise ValueError('Sealed latent read-time stat changed')
   value=np.load(p,mmap_mode='r',allow_pickle=False)
   if value.shape!=(1000,192) or value.dtype!=np.float32:raise ValueError('Actual latent array differs')
   self._maps[ep]=value
   if len(self._maps)>16:self._maps.popitem(last=False)[1]._mmap.close()
  self._maps.move_to_end(ep);return torch.from_numpy(np.array(self._maps[ep][step],copy=True))
 def __getitem__(self,pair):
  current,goal=map(int,pair)
  if not 0<=current<len(self) or not 0<=goal<len(self):raise ValueError('Compact transition outside split')
  ep,step=divmod(current,1000);gep,gstep=divmod(goal,1000)
  if ep!=gep or gstep<min(step+1,999):raise ValueError('Trajectory goal crosses episode or is past')
  if current in self.tail and goal!=self.tail[current][0]:raise ValueError('Validation B32 current matched but goal differs')
  raw=ep*1001+step;graw=gep*1001+gstep
  if self._action is None:
   self._action=np.load(self.action_path,mmap_mode='r',allow_pickle=False)
   if self._action.shape!=(len(self.entries)*1001,5) or self._action.dtype!=np.float32:raise ValueError('Original raw1001 action source differs')
  result=dict(current=self._latent(ep,step),goal=self._latent(gep,gstep),action=torch.from_numpy(np.array(self._action[raw],copy=True)),source_row=torch.tensor(raw),goal_source_row=torch.tensor(graw),source_episode=torch.tensor(ep),goal_source_episode=torch.tensor(gep))
  if current in self.tail:
   result['current']=torch.from_numpy(self.tail[current][1].copy());result['goal']=torch.from_numpy(self.tail[current][2].copy())
  if not all(torch.isfinite(v).all() for v in result.values()) or (result['action'].abs()>1).any():raise ValueError('Nonfinite cached latent/original normalized Box action')
  return result

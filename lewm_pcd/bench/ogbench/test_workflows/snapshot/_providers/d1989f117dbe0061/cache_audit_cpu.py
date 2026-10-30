"""One CPU integrity/mapping audit of the completed cache; no RGB/encoder/GPU reads."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np,torch
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parents[2])]
from cache_producer import sha,stat,read,save
from cache_dataset import CachedCubeStateLatentDataset
from dataset_state_fusion import CubeStateVisualDataset
from dataset import LogicalSampler

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);m=read(a.manifest);seal=read(a.manifest.parent/'cache_seal.json');idx=read(m['source_rgb_index']);before={str(a.manifest):stat(a.manifest),str(a.manifest.parent/'cache_seal.json'):stat(a.manifest.parent/'cache_seal.json')};frames={};pairs_checked=0
 if m['status']!='passed' or m['exit_code']!=0 or seal['status']!='passed' or seal['exit_code']!=0 or seal['manifest_sha256']!=sha(a.manifest) or m['source_rgb_index_sha256']!=sha(m['source_rgb_index']):raise ValueError('Actual complete sealed source binding required')
 for split,count in [('train',1000),('val',100)]:
  rows=m['splits'][split]
  if len(rows)!=count or len(idx[split])!=count or [x['episode'] for x in rows]!=list(range(count)):raise ValueError('Full original episode count/order differs')
  for ep,x in enumerate(rows):
   p=Path(x['path']);before[str(p)]=stat(p);z=np.load(p,mmap_mode='r',allow_pickle=False)
   if z.shape!=(1000,192) or z.dtype!=np.float32 or not np.isfinite(z).all() or sha(p)!=x['sha256'] or before[str(p)]!=seal['cache_stats'][str(p)]:raise ValueError('Full real latent finite/SHA/stat differs')
   if x['source_rgb_path']!=idx[split][ep]['path'] or x['source_row_start']!=ep*1001 or x['source_row_end_exclusive']!=ep*1001+1000 or stat(x['source_rgb_path'])!=x['source_rgb_stat_before'] or x['source_rgb_stat_before']!=x['source_rgb_stat_after']:raise ValueError('Original RGB/source raw row ownership differs')
  numeric_root=Path(m['numeric_sources'][split]['action_path']).parents[1];ds=CachedCubeStateLatentDataset(a.manifest.parent,split,numeric_root,m['source_rgb_index'],a.manifest.parent/'cache_seal.json');wrapped=CubeStateVisualDataset(ds,m['numeric_sources'][split]['observation_path']);actions=np.load(m['numeric_sources'][split]['action_path'],mmap_mode='r');states=np.load(m['numeric_sources'][split]['observation_path'],mmap_mode='r')
  if actions.shape!=(count*1001,5) or states.shape!=(count*1001,28) or actions.dtype!=np.float32 or states.dtype!=np.float32:raise ValueError('Original raw numeric schema differs')
  for current,goal in [(0,1),(999,999),(1000,1001),(count*1000-2,count*1000-1)]:
   if current in ds.tail:goal=ds.tail[current][0]
   row=wrapped[(current,goal)];raw=current//1000*1001+current%1000;grow=goal//1000*1001+goal%1000
   if int(row['source_row'])!=raw or int(row['goal_source_row'])!=grow or not np.array_equal(row['action'].numpy(),actions[raw]) or not np.array_equal(row['current_state'].numpy(),states[raw]) or not np.array_equal(row['goal_state'].numpy(),states[grow]):raise ValueError('Cache/action/full-state current and goal raw pairing differs')
   pairs_checked+=1
  frames[split]=count*1000
 sampler=LogicalSampler(100000,micro_batch=64,accumulation=1,seed=0)
 for pairs in sampler:tail=pairs
 t=m['val_tail32'];tp=Path(t['path']);before[str(tp)]=stat(tp)
 if len(tail)!=32 or stat(tp)!=t['stat'] or sha(tp)!=t['sha256']:raise ValueError('Real validation B32 cache binding differs')
 with np.load(tp,allow_pickle=False) as z:
  if list(zip(z['current_compact'].tolist(),z['goal_compact'].tolist()))!=tail or not np.isfinite(z['current_latents']).all() or not np.isfinite(z['goal_latents']).all():raise ValueError('True fixed sampler full-traversal tail differs')
  for i,pair in enumerate(tail):
   row=ds[pair]
   if not np.array_equal(row['current'].numpy(),z['current_latents'][i]) or not np.array_equal(row['goal'].numpy(),z['goal_latents'][i]):raise ValueError('Validation exact pair B32 override differs')
   pairs_checked+=1
 if any(stat(p)!=s for p,s in before.items()):raise ValueError('Cache changed during CPU independent scan')
 save(a.output/'report.json',dict(status='passed_cache_integrity_source_mapping_and_tail32_CPU_only',exit_code=0,files=1100,latent_frames=frames,finite_all=True,files_SHA_all=True,raw_source_rows_and_terminal_exclusion=True,original_action_state_sample_pairs=pairs_checked,manifest_sha256=sha(a.manifest),tail32_sha256=t['sha256'],source_before_after_exact=True,RGB_reads=0,encoder_forwards=0,GPU_used=False,audit_source_sha256=sha(__file__),scope_limit='Raw-vs-cache numerical equivalence is separate actual GPU gate, not certified by this CPU scan'));return 0

if __name__=='__main__':
 try:code=main()
 except SystemExit:raise
 except BaseException:
  import traceback;traceback.print_exc();code=1
 argv=sys.argv
 if '--output' in argv:
  out=Path(argv[argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True);(out/'exit_code').write_text(str(code)+'\n')
 raise SystemExit(code)

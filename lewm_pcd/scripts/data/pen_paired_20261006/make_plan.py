"""Build an explicit pending-launch 16-shard plan; no processes are started."""
import argparse,json
from pathlib import Path
from paired_parent import sha,save

def main(a):
 assert a.train_count%a.shards==0 and (a.train_count//a.shards)%a.batch==0
 cfg=Path(a.worker_config).resolve();out=Path(a.plan).resolve();assert not out.exists()
 jobs=[];episode_id=0
 for split,total,parts,start in [('train',a.train_count,a.shards,a.train_seed),('test',a.test_count,1,a.test_seed)]:
  if total==0:continue
  per=total//parts
  for shard in range(parts):
   predecessor=None
   for offset in range(0,per,a.batch):
    n=min(a.batch,per-offset);jobid=f'{split}_shard{shard:02d}_batch{offset//a.batch:03d}'
    eps=[{'episode_id':episode_id+i,'seed':start+shard*per+offset+i} for i in range(n)];episode_id+=n
    command0=[a.legacy_python,a.worker,'--phase','rgb','--config',str(cfg),'--episodes','{episodes}','--out','{attempt}/rgb']
    command1=[a.legacy_python,a.worker,'--phase','pcd','--config',str(cfg),'--episodes','{episodes}','--out','{attempt}/pcd','--rgb-root','{attempt}/rgb']
    command2=[a.modern_python,str(Path(__file__).with_name('paired_parent.py').resolve()),'--code-root',a.code_root,'--manifest','{manifest}','--out','{attempt}/paired']
    jobs.append(dict(id=jobid,split=split,shard=shard,policy_seed=start+shard*per,episodes=eps,policy_parent=predecessor,depends_on=[] if predecessor is None else [predecessor],commands=[command0,command1,command2]));predecessor=jobid
 ranges={'train':[a.train_seed,a.train_seed+a.train_count],'test':[a.test_seed,a.test_seed+a.test_count],
         'gate':[a.gate_seed,a.gate_seed+a.gate_count],'benchmark':[a.benchmark_seed,a.benchmark_seed+a.benchmark_count]}
 for name,(lo,hi) in ranges.items():assert 0<=lo<hi<=2**32,(name,'legacy seed overflow')
 names=list(ranges)
 for i,name in enumerate(names):
  for other in names[i+1:]:
   x,y=ranges[name],ranges[other];assert x[1]<=y[0] or y[1]<=x[0],('seed ranges overlap',name,other)
 seed_list=[ep['seed'] for job in jobs for ep in job['episodes']];ids=[ep['episode_id'] for job in jobs for ep in job['episodes']]
 assert len(set(seed_list))==len(seed_list) and len(set(ids))==len(ids)
 files=[a.worker,str(cfg),str(Path(__file__).with_name('paired_parent.py')),str(Path(__file__).with_name('paired_scheduler.py'))]+a.source
 image_size=int(json.loads(cfg.read_text()).get('image_size',256));assert image_size in (224,256)
 manifest=[dict(path=str(Path(f).resolve()),sha256=sha(f)) for f in files]
 worker_settings=json.loads(cfg.read_text());expected_source_hashes=worker_settings['source_hashes']
 save(out,dict(worker_config_sha256=sha(cfg),expected_source_hashes=expected_source_hashes,image_size=image_size,seed_ranges_half_open=ranges,expected_committed_episodes=a.train_count+a.test_count,root=a.root,concurrency=a.concurrency,train_count=a.train_count,test_count=a.test_count,shards=a.shards,batch=a.batch,source_manifest=manifest,jobs=jobs,environment={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','LP_NUM_THREADS':'1','MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl'}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('worker-config','plan','root','worker','legacy-python','modern-python','code-root'):p.add_argument('--'+n,required=True)
 p.add_argument('--train-count',type=int,default=10000);p.add_argument('--test-count',type=int,default=150);p.add_argument('--shards',type=int,default=16);p.add_argument('--batch',type=int,default=25);p.add_argument('--train-seed',type=int,default=1706000000);p.add_argument('--test-seed',type=int,default=1806000000);p.add_argument('--gate-seed',type=int,default=1906000000);p.add_argument('--gate-count',type=int,default=1024);p.add_argument('--benchmark-seed',type=int,default=2006000000);p.add_argument('--benchmark-count',type=int,default=4096);p.add_argument('--concurrency',type=int,choices=(1,2,4,8,16),default=4);p.add_argument('--source',action='append',default=[]);main(p.parse_args())

# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/collection/__main__.py (verbatim)
"""Current paper Cube collector: plans by default."""
import argparse,json
from . import build_plan,run

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cube-count',type=int,choices=(1,2,3,4),required=True)
 p.add_argument('--output',required=True)
 p.add_argument('--mode',choices=('formal','smoke'),default='formal')
 p.add_argument('--collection-seed',type=int,default=0)
 p.add_argument('--python')
 p.add_argument('--execute',action='store_true')
 p.add_argument('--resume',action='store_true',help='validate completed shards and run only missing shards')
 p.add_argument('--part-id',type=int,action='append',dest='part_ids',help='explicit canonical shard ID; does not change seeds')
 a=p.parse_args();plan=build_plan(a.cube_count,a.output,a.mode,a.python,collection_seed=a.collection_seed)
 print(json.dumps(plan,indent=2),flush=True)
 if a.execute:run(plan,resume=a.resume,part_ids=a.part_ids)
 return 0
if __name__=='__main__':raise SystemExit(main())

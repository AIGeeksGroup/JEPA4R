"""PCD provider/schema adaptation of the immutable RGB-cache producer; no copied loop."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
CODE=HERE.parents[1]
BASE=CODE/'goal_xyz_mask_100k_b256_20261011_v1/rgb_latent_cache_20261012_v1'
REPO=Path('/root/autodl-tmp/workspace/users/lewm/JEPA4R-ogbench-native-sr-df0496d/lewm_pcd')
sys.path[:0]=[str(CODE),str(BASE),'/root/autodl-tmp/workspace/users/lewm/lewm-gcbc-smoke-20261006-0002',str(REPO),str(REPO/'third_party/stable-worldmodel')]
if hashlib.sha256((BASE/'cache_producer.py').read_bytes()).hexdigest()!='f40dea29f18346e538edd15db2d035967245d6841c02a7afba781ed2b1cab745':raise ValueError('Pinned generic cache loop changed')
import dataset,dataset_pcd,cache_producer as common
# Existing PCD view exposes real BN6 through its original generic pixels key.
dataset.CubeOfficialDataset=dataset_pcd.CubeOfficialPCDDataset
raw_read,raw_save=common.read,common.save
KEYS={'source_rgb':'source_pcd','source_RGB':'source_PCD'}
def schema(value,out=False):
    if isinstance(value,list):return [schema(v,out) for v in value]
    if not isinstance(value,dict):return value
    pairs=KEYS.items() if out else [(b,a) for a,b in KEYS.items()]
    result={}
    for key,v in value.items():
        for a,b in pairs:
            if key.startswith(a):key=b+key[len(a):];break
        result[key]=schema(v,out)
    return result
common.read=lambda path:schema(raw_read(path),False)
def save(path,value):
    value=schema(value,True)
    if isinstance(value,dict):value.update(source_modality='pcd',modality_adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reused_cache_loop_source_sha256=hashlib.sha256((BASE/'cache_producer.py').read_bytes()).hexdigest())
    return raw_save(path,value)
common.save=save
if __name__=='__main__':
    code=1
    try:code=common.main()
    except BaseException as error:
        import traceback;traceback.print_exc()
        try:
            plan=common.read(sys.argv[sys.argv.index('--plan')+1]);root=Path(sys.argv[sys.argv.index('--output')+1]) if '--output' in sys.argv else Path(plan['cache_root']);root.mkdir(parents=True,exist_ok=True)
            if not (root/'cache_seal.json').exists():save(root/'status.json',dict(stage='producer_failed',status='failed',error=repr(error),exit_code=1))
        except Exception:pass
    if '--execute' in sys.argv:
        plan=common.read(sys.argv[sys.argv.index('--plan')+1]);root=Path(sys.argv[sys.argv.index('--output')+1]) if '--output' in sys.argv else Path(plan['cache_root'])
        if code==0 or not (root/'cache_seal.json').exists():(root/'producer_exit_code').write_text(str(code)+'\n')
    raise SystemExit(code)

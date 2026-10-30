"""Only Quad schema/identity edits over accepted Double 2D cache sources.

Preparation imports no Torch, renderer, encoder or model. Existing workspaces,
checkpoints, datasets and cache files are never written by this adapter.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

MODULE='official_state_2d_cache_v1'
EXPECTED={
    MODULE+'/produce.py':'fa6ff3dcb2d8fb709de069ea3aebfa5970cf3686a4a8aec20def899bd9848cd8',
    MODULE+'/render_adapter.py':'7607611907523a384d567f093c7c76d2438c4de9968e7ce9d3ef6c883300b144',
    MODULE+'/audit_cache.py':'75eb5137c30cac3628f0e781c7d2eb4a96b29eb7689a7fcae6ce1509dc303adc',
}
CHECKPOINT_SHA='7d8340c28b23910602f1328e6ce3ed56668ac04015f80ee1725c02131325bc6d'
EPISODES={'train':5000,'val':500}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    blob=value if isinstance(value,bytes) else (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as stream:stream.write(blob)


def build_sources(original):
    sources,ledger={},[]
    def edit(name,changes):
        before=(original/name).read_bytes()
        assert hashlib.sha256(before).hexdigest()==EXPECTED[name],name
        text=before.decode()
        for old,new,count in changes:
            assert text.count(old)==count,(name,old,count,text.count(old))
            text=text.replace(old,new)
        ast.parse(text);sources[name]=text.encode()
        ledger.append(dict(path=name,original_sha256=EXPECTED[name],
            generated_sha256=hashlib.sha256(sources[name]).hexdigest(),replacements=changes,
            diff=''.join(difflib.unified_diff(before.decode().splitlines(True),text.splitlines(True),
                fromfile='accepted_cube2/'+name,tofile='new_cube4/'+name))))
    edit(MODULE+'/produce.py',[
        ("{'train': 1000, 'val': 100}","{'train': 5000, 'val': 500}",1),
        ("('qpos', 28), ('qvel', 26)","('qpos', 42), ('qvel', 38)",1),
        ('qpos_width=28, qvel_width=26','qpos_width=42, qvel_width=38',1),
        ('shape[1] == 37','shape[1] == 55',1),('37 * 4','55 * 4',2),
        ('range(2)','range(4)',1),
        ('completed == 1100 and total_rows == 1101100','completed == 5500 and total_rows == 5505500',1),
        ('train_rows=1001000, val_rows=100100','train_rows=5005000, val_rows=500500',1)])
    edit(MODULE+'/render_adapter.py',[
        ('cube-double-v0','cube-quadruple-v0',1),('(28, 26, 7)','(42, 38, 7)',1),
        ('range(2)','range(4)',2),('% 2','% 4',1),('nq=28, nv=26','nq=42, nv=38',1)])
    edit(MODULE+'/audit_cache.py',[
        ("{'train': 1000, 'val': 100}","{'train': 5000, 'val': 500}",1),
        ("accepted['data_id'] == 'cube-double-play-v0' and accepted['cube_count'] == 2",
         "accepted['data_id'] == 'cube-quadruple-play-v0' and accepted['cube_count'] == 4",1),
        ('qpos_width=28, qvel_width=26','qpos_width=42, qvel_width=38',1),
        ("('qpos', 28), ('qvel', 26)","('qpos', 42), ('qvel', 38)",1),
        ('nq=28, nv=26','nq=42, nv=38',1),
        ('Require full official1000/100 Double cache','Require full official5000/500 Quad cache',1)])
    return sources,ledger


def make_plan(original,case,base,binding,sources):
    shadow=case/'workspace'
    plan=copy.deepcopy(json.loads((original/MODULE/'producer_plan.json').read_text()))
    assert binding['cube_count']==4 and binding['data_id']=='cube-quadruple-play-v0'
    checkpoint=Path(binding['checkpoint']['path'])
    assert checkpoint==base/'checkpoints/cube4/2d_epoch10/weights_epoch_10.pt'
    assert checkpoint.stat().st_size==binding['checkpoint']['bytes']==72292862
    assert sha(checkpoint)==binding['checkpoint']['sha256']==CHECKPOINT_SHA
    for key in ('data_acceptance','source_stat_freeze'):
        assert sha(binding[key]['path'])==binding[key]['sha256'],key
    accepted=json.loads(Path(binding['data_acceptance']['path']).read_text())
    frozen=json.loads(Path(binding['source_stat_freeze']['path']).read_text())
    assert accepted['actual_exit_code']==frozen['actual_exit_code']==0
    assert accepted['cube_count']==4 and accepted['data_id']=='cube-quadruple-play-v0'
    assert frozen['acceptance_sha256']==binding['data_acceptance']['sha256']
    for split,count in EPISODES.items():
        item=frozen['datasets']['validation' if split=='val' else 'train']
        expected='cube-quadruple-play-v0'+('-val' if split=='val' else '')+'.npz'
        assert Path(item['path']).name==expected
    assert binding['camera']==plan['camera'],'Declared encoding camera must be explicit, unchanged front+5 RGB profile'
    plan.update(workspace=str(shadow),shared_state_root=str(case/'source_arrays'),
        checkpoint=str(shadow/'checkpoints/cube4/2d_epoch10/weights_epoch_10.pt'),
        checkpoint_sha256=CHECKPOINT_SHA,split_episodes=EPISODES,cache_dir=str(case/'2d/cache'),
        data_acceptance=binding['data_acceptance']['path'],data_acceptance_sha256=binding['data_acceptance']['sha256'],
        source_stat_freeze=binding['source_stat_freeze']['path'],source_stat_freeze_sha256=binding['source_stat_freeze']['sha256'])
    pins={}
    for relative in plan['source_pins']:
        relative=relative.replace('checkpoints/cube2/2d_epoch10/','checkpoints/cube4/2d_epoch10/')
        if relative in sources:pins[relative]=hashlib.sha256(sources[relative]).hexdigest()
        else:
            target=base/relative
            assert target.is_file(),str(target)
            pins[relative]=sha(target)
    plan['source_pins']=pins
    return plan


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-root',required=True,type=Path)
    parser.add_argument('--original-workspace',required=True,type=Path)
    parser.add_argument('--case',required=True,type=Path)
    parser.add_argument('--binding',required=True,type=Path)
    parser.add_argument('--phase',choices=('materialize','prepare-arrays'),default='materialize')
    args=parser.parse_args()
    base,original,case=args.base_root.resolve(),args.original_workspace.resolve(),args.case.resolve()
    assert case!=base and case.is_relative_to(base) and original.is_relative_to(base)
    assert not original.is_relative_to(case),'Original code must remain outside new case'
    shadow=case/'workspace'
    if args.phase=='materialize':
        assert not shadow.exists(),'Never overwrite a previous materialization'
        sources,ledger=build_sources(original)
        binding=json.loads(args.binding.read_text());plan=make_plan(original,case,base,binding,sources)
        # Resolve every old/new pin before creating a shadow tree; no missing-pin partial plan.
        for relative in plan['source_pins']:
            assert relative in sources or (base/relative).is_file(),relative
        links=('source','external','reference','env_runtime','checkpoint_encoder.py','checkpoints')
        assert all((base/name).exists() for name in links)
        shadow.mkdir(parents=True)
        for name in links:(shadow/name).symlink_to(base/name,target_is_directory=(base/name).is_dir())
        for name,value in sources.items():write(shadow/name,value)
        write(shadow/MODULE/'producer_plan.json',plan)
        write(case/'materialization_receipt.json',dict(actual_exit_code=0,cube_count=4,
            binding_sha256=sha(args.binding),producer_plan_sha256=sha(shadow/MODULE/'producer_plan.json'),
            modules=ledger,original_sources_modified=False,model_renderer_GPU_executed=False))
    else:
        sys.path.insert(0,str(shadow/MODULE))
        spec=importlib.util.spec_from_file_location('_quad_original_state_array_extractor',shadow/MODULE/'produce.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        plan=json.loads((shadow/MODULE/'producer_plan.json').read_text())
        file_sha,stat,save=module.writer_helpers(shadow)
        contract=dict(data_acceptance_sha256=plan['data_acceptance_sha256'],
            source_stat_freeze_sha256=plan['source_stat_freeze_sha256'],split_episodes=EPISODES,
            raw_rows_per_episode=1001,qpos_width=42,qvel_width=38,
            extractor_source_sha256=file_sha(shadow/MODULE/'produce.py'))
        module.source_arrays(plan,module.object_sha(contract),file_sha,stat,save,contract)
    print(json.dumps(dict(actual_exit_code=0,phase=args.phase,model_renderer_GPU_executed=False)))


if __name__=='__main__':main()

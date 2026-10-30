"""Tiny1-4 advisory committed-checkpoint metadata endpoint; no NN/payload reads."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import socket
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import backup_watch as b

CASE=PurePosixPath('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/lejepa_tiny512_cube14_gciql_v1')
STORE=PurePosixPath('/root/autodl-tmp/lejepa_tiny512_cube14_20261029_v1_outputs')
FAMILY='lejepa_tiny512'
_BRIDGES={}


def wire(path):return str(PurePosixPath(str(path).replace('\\','/')))
def hexsha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def expected_keys(cfg):
    case=PurePosixPath(wire(cfg['case_root']))
    b.require(case==CASE and PurePosixPath(wire(cfg['store_root']))==STORE,'wrong dedicated CASE/STORE')
    b.require(cfg['advisory_only'] is True and cfg['expected_checkpoint_count']==52 and cfg['expected_file_count']==208,'wrong physical-only52/208 scope')
    b.require(cfg['future_config_SHA_prefilled'] is False and 120<=cfg['poll_seconds']<=300,'no future config SHA or fast polling')
    control=case/'control/backup_ack/v1'
    b.require(wire(cfg['ack_root'])==str(control/'acks') and wire(cfg['failure_root'])==str(control/'failures'),'wrong independent backup control roots')
    want={(stage,FAMILY+f'_cube{n}') for n in range(1,5) for stage in b.COUNTERS};seen=set()
    for entry in cfg['config_sources']:
        b.require(set(entry)=={'stage','group_id','config_path'},'future config SHA must not be prefilled')
        identity=entry['stage'],entry['group_id'];b.require(identity in want and identity not in seen,'wrong or duplicate source identity');seen.add(identity)
        b.require(wire(entry['config_path'])==str(case/'configs'/entry['stage']/(entry['group_id']+'.json')),'source cfg path outside own case')
    b.require(seen==want and len(cfg['config_sources'])==8 and set(cfg['groups'])=={'1','2','3','4'},'exact eight future cfgs required')
    components={str(case/'release_v1'/name) for name in ['tiny_bridge.py','tiny_encoder.py','ENCODER_COMPONENT_FREEZE.json','ADAPTER_SOURCE_FREEZE.json']}
    b.require(set(cfg['component_sources'])==components and wire(cfg['bridge_path'])==str(case/'release_v1/tiny_bridge.py'),'wrong frozen metadata component scope')
    for source in cfg['component_sources'].values():b.require(hexsha(source['sha256']) and type(source['bytes']) is int and 0<source['bytes']<=b.MAX_METADATA,'invalid component binding')
    for n in range(1,5):
        g=cfg['groups'][str(n)];group=FAMILY+f'_cube{n}'
        b.require(g['group_id']==group and wire(g['encoder_spec_path'])==str(case/'cache/groups'/group/'encoder_spec.json') and hexsha(g['encoder_spec_sha256']) and hexsha(g['checkpoint_sha256']),'wrong per-Cube spec/weight binding')
    return {(stage,group,counter) for stage,group in want for counter in b.COUNTERS[stage]}


def metadata(path):
    evidence=b.small_file(path);raw=Path(path).read_bytes()
    b.require(hashlib.sha256(raw).hexdigest()==evidence['sha256'] and b.stat(path)==evidence['stat'],'small metadata changed while reading')
    evidence['path']=wire(path)
    return json.loads(raw),evidence


def load_bridge(cfg):
    path=Path(cfg['bridge_path']);key=(wire(path),cfg['component_sources'][wire(path)]['sha256'])
    b.require(b.sha(path)==key[1],'metadata bridge source changed')
    if key not in _BRIDGES:
        spec=importlib.util.spec_from_file_location('_advisory_original_tiny_metadata',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        _BRIDGES[key]=module  # Pinned Tiny module imports only its stdlib-only encoder metadata module.
    return _BRIDGES[key]


def validate_config(cfg,value,n,stage):
    group=FAMILY+f'_cube{n}';g=cfg['groups'][str(n)];case=PurePosixPath(wire(cfg['case_root']));enc=value['encoder_binding']
    b.require(value['group_id']==value['role']==group and value['cube_count']==n and value['purpose']==stage and value['modality']=='rgb','wrong Tiny cfg group/purpose')
    b.require('durability' not in value and value['fresh'] is True and value['resume_from'] is None,'NN cfg changed for durability or resume')
    b.require(value['input_contract']==dict(current_dim=531,goal_dim=512+3*n,latent_dim=512,zero_padding=False),'wrong Tiny input widths')
    b.require(value['native_max_episode_steps']=={1:200,2:500,3:1000,4:1000}[n],'wrong Native cap')
    b.require(enc['family']==FAMILY and enc['latent_dim']==512 and enc['checkpoint_sha256']==g['checkpoint_sha256'] and wire(enc['spec_path'])==wire(g['encoder_spec_path']) and enc['spec_sha256']==g['encoder_spec_sha256'],'wrong own encoder provenance')
    adapter=str(case/'release_v1/tiny_encoder.py');b.require(wire(enc['adapter']['path'])==adapter and enc['adapter']['sha256']==cfg['component_sources'][adapter]['sha256'],'wrong Tiny adapter')
    b.require(wire(value['execution_root'])==str(case/'outputs'/stage) and wire(value['output'])==str(case/'outputs'/stage/'runs'/group),'wrong output role paths')
    b.require(wire(value['config_seal_path'])==str(case/'configs/seals'/(group+'.json')),'wrong local cfg seal path')
    b.require((value['seed'],value['batch_size'],value['total_updates'],value['feature_lookup_workers'])==(0,1024,1000000,4) and value['save_step1'] is True and value['checkpoint_steps']==b.COUNTERS['formal'],'changed frozen GCIQL budget/input contract')
    a=value['agent_config'];b.require((a['lr'],a['alpha'],a['discount'],a['expectile'],a['tau'])==(3e-4,1.0,.99,.9,.005) and a['layer_norm'] and a['actor_hidden_dims']==a['value_hidden_dims']==[512,512,512],'changed optimizer/network science')


def group_binding(cfg,n):
    group=FAMILY+f'_cube{n}';case=PurePosixPath(wire(cfg['case_root']));seal_path=case/'configs/seals'/(group+'.json');hook_path=case/'control/full/runtime'/(group+'_04_hooks_exit.json')
    if not Path(str(seal_path)).is_file() or not Path(str(hook_path)).is_file():return None
    hook,hook_file=metadata(str(hook_path))
    b.require(type(hook['actual_exit_code']) is int and hook['actual_exit_code']==0 and hook['residual_owned_members']==[],'local cfg hooks not actually0/drained')
    seal,seal_file=metadata(str(seal_path));b.require(seal['actual_exit_code']==0 and seal['group_id']==group and seal['NN_performed'] is False,'wrong cfg seal')
    values={};evidence={wire(seal_path):seal_file,wire(hook_path):hook_file}
    for stage in b.COUNTERS:
        path=case/'configs'/stage/(group+'.json');value,info=metadata(str(path));validate_config(cfg,value,n,stage)
        b.require(seal['configs'][stage]==dict(path=str(path),sha256=info['sha256']),'unsealed cfg bytes changed')
        values[stage]=value;evidence[str(path)]=info
    for key in ['encoder_binding','data_binding','camera','cache_plan_binding','delivery_binding','cache_control_binding','parent_template']:
        b.require(values['pilot'][key]==values['formal'][key],'pilot/formal source lineage differs')
    for path,binding in cfg['component_sources'].items():
        info=b.small_file(path);info['path']=wire(path)
        b.require(info['sha256']==binding['sha256'] and info['stat']['size']==binding['bytes'],'published component changed');evidence[path]=info
    group_source=cfg['groups'][str(n)];_,spec_file=metadata(group_source['encoder_spec_path']);b.require(spec_file['sha256']==group_source['encoder_spec_sha256'],'published Tiny spec changed');evidence[wire(group_source['encoder_spec_path'])]=spec_file
    bridge=load_bridge(cfg)
    pins=bridge.cache_pins(values['formal'],require_seal=True)  # Original verified pure-metadata function; no install/bootstrap/model.
    bridge.check_config(values['pilot'])
    for path,info in evidence.items():b.require(b.stat(path)==info['stat'],'source changed through cfg/cache metadata validation')
    return dict(configs=values,files=evidence,binding=dict(config_seal=dict(path=str(seal_path),sha256=seal_file['sha256']),
        sealed_configs=seal['configs'],hook_exit=dict(path=str(hook_path),sha256=hook_file['sha256'],actual_exit_code=0,residual_owned_members=[]),
        files=evidence,cache_pins_sha256=b.digest_json(pins)))


def checkpoint_directory(cfg,entry,counter):
    case=PurePosixPath(wire(cfg['case_root']));return case/'outputs'/entry['stage']/('gates' if entry['stage']=='pilot' else 'runs')/entry['group_id']/'checkpoints'/f'update_{counter:09d}'


def checkpoint_snapshot(cfg,entry,counter,context=None):
    expected_keys(cfg);b.require(type(counter) is int and counter in b.COUNTERS[entry['stage']],'wrong committed counter')
    directory=checkpoint_directory(cfg,entry,counter);manifest_path=directory/'manifest.json'
    if not Path(str(manifest_path)).is_file():return None
    context=group_binding(cfg,int(entry['group_id'][-1])) if context is None else context
    if context is None:return None
    config_file=context['files'][wire(entry['config_path'])]
    manifest,manifest_file=metadata(str(manifest_path));b.require(type(manifest['counter']) is int and manifest['counter']==counter,'manifest counter changed')
    files={'manifest.json':manifest_file,'training_config.json':config_file}
    store=Path(cfg['store_root']).resolve(strict=True)
    b.require(Path(str(manifest_path)).resolve(strict=True).is_relative_to(store),'checkpoint manifest outside own STORE')
    for name,key in [(f'params_{counter}.pkl','agent_sha256'),('run_state.pkl','sidecar_sha256')]:
        b.require(hexsha(manifest[key]),'invalid committed payload SHA');path=Path(str(directory/name))
        b.require(path.is_file() and path.resolve(strict=True).is_relative_to(store),'committed payload missing or outside own STORE')
        files[name]=dict(path=wire(path),sha256=manifest[key],stat=b.stat(path))  # No payload open/hash.
    b.require(b.stat(str(manifest_path))==manifest_file['stat'],'manifest changed during snapshot')
    return dict(schema_version=1,kind='checkpoint',group_id=entry['group_id'],stage=entry['stage'],counter=counter,
        config_path=wire(entry['config_path']),config_sha256=config_file['sha256'],manifest_sha256=manifest_file['sha256'],
        files=files,ack_root=cfg['ack_root'],failure_root=cfg['failure_root'],
        target_parts=['closed_checkpoints',entry['stage'],entry['group_id'],f'update_{counter:09d}'],metadata_binding=context['binding'])


def scan(cfg):
    expected_keys(cfg);b.require(not (Path(cfg['failure_root'])/'global.json').exists(),'sticky advisory failure exists')
    checkpoints=[];pending=[]
    for n in range(1,5):
        entries=[x for x in cfg['config_sources'] if x['group_id']==FAMILY+f'_cube{n}']
        present=[(e,c) for e in entries for c in b.COUNTERS[e['stage']] if Path(str(checkpoint_directory(cfg,e,c)/'manifest.json')).is_file()]
        if not present:pending.append(FAMILY+f'_cube{n}');continue
        context=group_binding(cfg,n)  # Once per group per scan, never once per52 counters.
        if context is None:pending.append(FAMILY+f'_cube{n}');continue
        for entry,counter in present:
            value=checkpoint_snapshot(cfg,entry,counter,context)
            if value is not None:checkpoints.append(value)
    return dict(checkpoints=checkpoints,logs=[],pending_groups=pending)


def refresh(cfg,snapshot):
    b.require(snapshot.get('kind')=='checkpoint','only checkpoint snapshots supported')
    entries=[e for e in cfg['config_sources'] if wire(e['config_path'])==snapshot.get('config_path') and e['stage']==snapshot.get('stage') and e['group_id']==snapshot.get('group_id')]
    b.require(len(entries)==1,'unconfigured snapshot identity');return checkpoint_snapshot(cfg,entries[0],snapshot['counter'])


def dispatch(cfg,action,payload=None):
    expected_keys(cfg);b.require(socket.gethostname()==cfg['expected_remote_hostname'],'wrong remote endpoint hostname')
    if action=='scan':return scan(cfg)
    if action=='failure':
        path=Path(cfg['failure_root'])/'global.json'
        if not path.exists():b.new_json(path,dict(payload,schema_version=1,actual_exit_code=1,at=time.time(),advisory_only=True,automatic_restart=False))
        return dict(recorded=True,advisory_only=True)
    b.require(action in ('poststat','ack'),'wrong endpoint action');snapshot=payload['snapshot'];actual=refresh(cfg,snapshot)
    b.require(actual==snapshot,'source changed since metadata snapshot')
    if action=='poststat':return actual
    ack=payload['ack'];b.validate_ack(actual,ack);path=b.ack_path(actual)
    if path.exists():b.require(json.loads(path.read_text(encoding='utf-8'))==ack,'existing ACK differs; refuse overwrite')
    else:b.new_json(path,ack)
    return dict(accepted=True,ack_path=wire(path),ack_sha256=b.sha(path),advisory_only=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--action',choices=['scan','poststat','ack','failure'],required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text(encoding='utf-8'));payload=None if a.action=='scan' else json.load(sys.stdin)
    print(json.dumps(dispatch(cfg,a.action,payload),sort_keys=True))


if __name__=='__main__':main()

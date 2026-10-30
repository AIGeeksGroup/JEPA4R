"""Generate fresh fixed-protocol configs from an accepted cache delivery."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT/'foundation_visual_gciql_v1'
COUNTERS = [1]+list(range(100000,1000001,100000))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_formal_encoder(binding):
    path=Path(binding['spec_path'])
    assert sha(path)==binding['spec_sha256'],'Encoder spec bytes changed'
    assert json.loads(path.read_text()).get('diagnostic_only',False) is False,'Diagnostic encoder cannot enter formal training'


def make_config(template, delivery):
    cfg={'agent_config':copy.deepcopy(template['agent_config'])}
    n,d,g=delivery['cube_count'],delivery['latent_dim'],delivery['group_id']
    assert n in (1,2,3,4) and d in (768,1024) and '/' not in g
    purpose=delivery.get('purpose','formal');assert purpose in ('pilot','formal')
    execution_root=CASE/'pilot' if purpose=='pilot' else CASE
    cfg.update(schema_version=1, executable=True, group_id=g,role=g,cube_count=n,modality='rgb',
        seed=0,batch_size=1024,total_updates=1000000,resume_from=None,fresh=True,
        validation_interval=5000,save_interval=100000,save_step1=True,checkpoint_steps=COUNTERS,
        heartbeat_interval=100,feature_lookup_workers=4,evaluation_episodes_during_training=0,
        runtime_encoder_forwards=0,WM_dynamics_calls=0,output=str(execution_root/'runs'/g),
        purpose=purpose,execution_root=str(execution_root),
        native_max_episode_steps={1:200,2:500,3:1000,4:1000}[n],
        input_contract=dict(current_dim=d+19,goal_dim=d+3*n,latent_dim=d,zero_padding=False),
        reward=dict(rule='ALL(each cube L2<=0.4 scaled=4cm)',values=[-1,0],mask='1-success'))
    for key in ('data_binding','feature_loader_arguments','encoder_binding','camera','cache_audit'):
        cfg[key]=copy.deepcopy(delivery[key])
    if cfg['data_binding']:
        cfg['dataset_id']=cfg['data_binding']['data_id']
    return cfg


def commands(config_path,cfg):
    base=str(CASE/'functional/runtime.py');g=cfg['group_id']
    home=Path(cfg['execution_root'])
    gate=str(home/'gates'/g);run=cfg['output'];accept=str(home/'acceptance'/f'{g}.json')
    result=dict(gate=[base,'gate','--config',str(config_path),'--output',gate],
        reload=[base,'reload','--config',str(config_path),'--output',gate,'--report',str(home/'gates'/f'{g}_reload.json')],
        train=[base,'train','--config',str(config_path)],
        accept=[base,'accept','--config',str(config_path),'--output',run,'--report',accept],
        native=[str(CASE/'functional/native_sr.py'),'--config',str(config_path),'--mode','full','--output',str(home/'native_sr'/g)])
    if cfg['purpose']=='pilot':
        del result['train'],result['accept']
        result['native']=[str(CASE/'functional/native_sr.py'),'--config',str(config_path),'--mode','tiny','--gate-smoke','--output',str(home/'native_sr'/g)]
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--delivery',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();delivery=json.loads(a.delivery.read_text())
    audit=delivery['cache_audit'];assert sha(audit['path'])==audit['sha256']
    assert json.loads(Path(audit['path']).read_text())['actual_exit_code']==0
    if delivery.get('purpose','formal')=='formal':
        require_formal_encoder(delivery['encoder_binding'])
        expected={1:(1000,100),2:(1000,100),3:(3000,300),4:(5000,500)}[delivery['cube_count']]
        assert tuple(delivery['data_binding']['splits'][s]['episodes'] for s in ('train','val'))==expected
    template_path=ROOT/'cube23_distance_gciql_v1/five_group_extension_v1/configs/G08_cube3_rgb.json'
    cfg=make_config(json.loads(template_path.read_text()),delivery)
    cfg['delivery_binding']=dict(path=str(a.delivery),sha256=sha(a.delivery))
    cfg['parent_template']=dict(path=str(template_path),sha256=sha(template_path))
    cfg['commands']=commands(a.output,cfg)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(cfg,f,indent=2)


if __name__=='__main__':main()

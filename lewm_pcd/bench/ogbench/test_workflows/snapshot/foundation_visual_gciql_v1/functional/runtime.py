"""Original async100 train/save/CPU acceptance with actual foundation widths."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from inputs import load_inputs
from prepare import ROOT, CASE, COUNTERS, sha, require_formal_encoder

LEGACY=ROOT/'cube23_distance_gciql_v1'


def check_config(cfg):
    n,d=cfg['cube_count'],cfg['input_contract']['latent_dim']
    assert n in (1,2,3,4) and d in (768,1024) and cfg['modality']=='rgb'
    assert cfg['input_contract']==dict(current_dim=d+19,goal_dim=d+3*n,latent_dim=d,zero_padding=False)
    assert (cfg['seed'],cfg['batch_size'],cfg['total_updates'],cfg['resume_from'])==(0,1024,1000000,None)
    assert cfg['save_step1'] and cfg['checkpoint_steps']==COUNTERS and cfg['feature_lookup_workers']==4
    agent=cfg['agent_config']
    assert (agent['lr'],agent['alpha'],agent['discount'],agent['expectile'],agent['tau'])==(3e-4,1.,.99,.9,.005)
    assert agent['layer_norm'] and agent['actor_hidden_dims']==agent['value_hidden_dims']==[512,512,512]
    assert cfg['encoder_binding']['latent_dim']==d


def install(cfg):
    check_config(cfg)
    sys.path[:0]=[str(LEGACY/'state_priority_v1'),str(LEGACY),str(ROOT)]
    assert sha(LEGACY/'state_priority_v1/state_runtime.py')=='623fff33b0c836c6268257689216cf78de8694fbab1107dddbd4bdbfc3ee1956'
    preserved=importlib.import_module('state_runtime')
    runtime=preserved.install('baseline','async100')
    inputs=load_inputs(LEGACY/'distance_inputs.py',cfg['input_contract']['latent_dim'])
    runtime.DistanceInputs=inputs.VisualInputs
    runtime.load_cached_features=inputs.load_cached_features
    def prepare(config,shared_datasets=None):
        check_config(config)
        samplers,resources,binding=preserved.ORIGINAL_PREPARE(config,shared_datasets)
        resources.actual_update_count=resources.train_batches=0
        resources.barrier=lambda:None
        binding['runtime'].update(input_mode='samebatch_roles4',host_metrics='async100',
            formal_metric_finite_scope='each100updates_and5000validation',visual_future_sampling=False,
            width_proof=inputs.width_proof,encoder_binding=config['encoder_binding'],runtime_encoder_calls=0)
        binding['code'].update({str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('inputs.py'))})
        return samplers,resources,binding
    runtime.prepare=prepare
    boundary=importlib.import_module('gciql_boundary')
    boundary.prepare,boundary.get_batch=runtime.prepare,runtime.get_batch
    return runtime,boundary


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('verify-hooks','gate','reload','train','accept'))
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--report',type=Path)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.action=='train':
        assert cfg['purpose']=='formal','Pilot cannot enter formal training'
        require_formal_encoder(cfg['encoder_binding'])
    runtime,boundary=install(cfg)
    if cfg['purpose']=='pilot':assert a.action in ('verify-hooks','gate','reload')
    if a.action=='verify-hooks':
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and runtime.jax.default_backend()=='cpu'
        result=dict(actual_exit_code=0,boundary_prepare_identity=boundary.prepare is runtime.prepare,
            boundary_get_batch_identity=boundary.get_batch is runtime.get_batch,agent_or_encoder_created=False)
        print(json.dumps(result));return
    if a.action=='train':return runtime.run(str(a.config))
    assert a.output is not None
    if a.report:a.report.parent.mkdir(parents=True,exist_ok=True)
    if a.action=='gate':boundary.gate(cfg,a.output)
    elif a.action=='reload':boundary.reload_gate(cfg,a.output,a.report)
    else:boundary.accept(cfg,a.output,a.report)


if __name__=='__main__':raise SystemExit(main())

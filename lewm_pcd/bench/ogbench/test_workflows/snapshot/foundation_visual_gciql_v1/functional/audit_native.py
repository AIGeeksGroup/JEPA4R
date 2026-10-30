"""Reuse the original episode auditor, independently verify bytes and step arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import types
import numpy as np
from prepare import ROOT

AUDIT_SHA='ffccacb2f94cfd37f0480419642341e7c966e490e1a97e6056eecfe0033c5285'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def source_binding(cfg,config_path):
    encoder=cfg['encoder_binding'];specpath=Path(encoder['spec_path'])
    assert sha(specpath)==encoder['spec_sha256']
    spec=json.loads(specpath.read_text())
    assert spec['checkpoint_sha256']==encoder['checkpoint_sha256'] and spec['output_dim']==cfg['input_contract']['latent_dim']
    assert spec['family']==encoder['family']
    pins={str(specpath):encoder['spec_sha256'],str(config_path):sha(config_path),
        encoder['adapter']['path']:encoder['adapter']['sha256'],spec['checkpoint']:spec['checkpoint_sha256']}
    for name,digest in spec.get('source_files_sha256',{}).items():pins[str(Path(spec['source_root'])/name)]=digest
    for path in Path(__file__).parent.glob('*.py'):pins[str(path)]=sha(path)
    pins[str(ROOT/'cube23_distance_gciql_v1/native_sr_visual_v1/visual_sr.py')]='e40a2ff52bd0915b7953d9011c65e02c75f9e0b2d0e454b613d18c275f6c13f0'
    home=Path(cfg['execution_root']);group=cfg['group_id']
    if cfg['purpose']=='pilot':
        run=home/'gates'/group;counter=2
        reports=[run/'report.json',home/'gates'/f'{group}_reload.json']
    else:
        run=Path(cfg['output']);counter=1000000
        reports=[home/'acceptance'/f'{group}.json']
    checkpoint=run/'checkpoints'/f'update_{counter:09d}'
    manifest_path=checkpoint/'manifest.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['counter']==counter
    pins[str(manifest_path)]=sha(manifest_path)
    pins[str(checkpoint/f'params_{counter}.pkl')]=manifest['agent_sha256']
    pins[str(checkpoint/'run_state.pkl')]=manifest['sidecar_sha256']
    for report_path in reports:
        report=json.loads(report_path.read_text());assert report['actual_exit_code']==0
        pins[str(report_path)]=sha(report_path)
    if cfg['purpose']=='formal':
        assert report['final_checkpoint']==str(checkpoint) and report['final_manifest_sha256']==pins[str(manifest_path)]
    for path,digest in pins.items():assert sha(path)==digest,path
    return dict(actual_exit_code=0,group_id=cfg['group_id'],config_sha256=sha(config_path),
        encoder_family=encoder['family'],latent_dim=cfg['input_contract']['latent_dim'],
        source_pins=pins,camera=cfg['camera'])


def auditor(current_dim):
    path=ROOT/'cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/audit.py'
    assert sha(path)==AUDIT_SHA
    source=path.read_text();assert source.count('211')==3
    module=types.ModuleType('_foundation_original_result_auditor')
    exec(compile(source.replace('211',str(current_dim)),str(path),'exec'),module.__dict__)
    return module


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--mode',choices=('tiny','full'),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());binding=source_binding(cfg,a.config)
    actual=json.loads((a.output/'native_source_binding.json').read_text())
    for key,value in binding.items():assert actual[key]==value,key
    assert actual['mode']==a.mode
    if a.mode=='full':assert cfg['purpose']=='formal' and actual['gate_smoke'] is False
    expected,cap=(1,4) if a.mode=='tiny' else (250,cfg['native_max_episode_steps'])
    result=auditor(cfg['input_contract']['current_dim']).verify_report(ROOT,cfg['group_id'],
        'diagnostic' if a.mode=='tiny' else 'full',a.output,expected,cfg['input_contract']['goal_dim'],cap)
    episodes=list((a.output/'episodes').glob('task*/episode_*'))
    assert len(episodes)==expected
    for episode in episodes:
        item=json.loads((episode/'result.json').read_text());steps=item['steps']
        assert item['camera']==dict(cfg['camera'],encoder_spec_sha256=cfg['encoder_binding']['spec_sha256'])
        with np.load(episode/'steps.npz',allow_pickle=False) as data:
            actions,rewards,successes=(data[k] for k in ('actions','rewards','successes'))
            assert actions.shape==(steps,5) and rewards.shape==successes.shape==(steps,)
            assert np.isfinite(actions).all() and np.isfinite(rewards).all() and np.abs(actions).max()<=1.000001
            # The preserved evaluator aggregates the final step's info.
            assert int(successes[-1])==item['success']
    result.update(actual_exit_code=0,mode=a.mode,group_id=cfg['group_id'],config_sha256=binding['config_sha256'],
        source_binding_sha256=sha(a.output/'native_source_binding.json'),original_auditor_sha256=AUDIT_SHA,
        actual_step_arrays_checked=True,diagnostic_excluded_from_formal_SR=a.mode=='tiny',no_NN_or_update_performed=True)
    with (a.output/'independent_audit.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()

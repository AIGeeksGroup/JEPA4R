"""Seal an original serial250 plan only after actual tiny + independent acceptance."""
import argparse,hashlib,json
from pathlib import Path
N=Path(__file__).resolve().parents[1]
B=Path('/root/autodl-tmp/workspace/users/lewm/ogbench-official-single-play-20261006')
DRIVER=B/'official_lewm_gcbc_20261006/native_sr_v1/run_full_sequence.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
    config_path=N/f'configs/{a.kind}_frozen.json';config=json.loads(config_path.read_text());out=Path(config['native_output'])
    freeze=N/'aux/freeze_v1/manifest.json';f=json.loads(freeze.read_text())
    assert sha(freeze)=='d8f4adfa41aaeb002198dc1f134503e27d2033f0250c847baee2ac9cb91283dc'
    for name,h in f['files'].items():assert sha(N/name)==h
    assert sha(DRIVER)=='bbbb96caf3ea7ac013a2841cede2c0a83c622ba2e7cb65b86fbd0574233552f1'
    assert sha(a.tiny_review)==a.tiny_review_sha
    review=json.loads(a.tiny_review.read_text());assert review['exit_code']==0 and review['status'].startswith('passed')
    for p in (a.tiny_outer_exit,a.tiny_stage_exit):assert p.read_text().strip()=='0'
    tiny=json.loads((out/'gate_receipt.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    assert tiny['status']=='passed' and tiny['exit_code']==0 and tiny['diagnostic_only'] and 0<tiny['steps']<=4
    assert (tiny['task'],tiny['index'],tiny['seed'])==(1,0,970100000)
    assert hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()==tiny['protocol_sha256']
    assert protocol['checkpoint_sha256']==config['agent_sha256'] and protocol['policy_kind']=='cached-visual-state-xyz'
    attempt=out/'gate'/tiny['attempt'];assert (attempt/'exit_code').read_text().strip()=='0'
    case=json.loads((attempt/'input_case.json').read_text());assert case['model_calls']==case['policy_steps']==tiny['steps']
    assert config['selected_updates']==1000000 and config['goal_dims']==195
    assert case['policy_input_shapes']==[[1,config['current_dims']],[1,195]]
    if config['visual_mode']=='zero192':
        assert case['goal_encoder_calls']==case['current_encoder_calls']==0 and case['zero_visual_channels'] and case['total_encoder_calls']==0
    else:
        assert config['visual_mode']=='cache' and config.get('state_profile')=='robot_proprio'
        assert case['goal_encoder_calls']==case['current_encoder_calls']==case['model_calls'] and case['total_encoder_calls']==2*case['model_calls']
        assert case['state_profile']=='robot_proprio' and case['network_state_indices']==list(range(19)) and case['excluded_object_state_indices']==list(range(19,28))
    sealed=dict(config['source_sha256'])
    for p in [freeze,config_path,DRIVER,out/'gate_receipt.json',out/'protocol.json',out/'binding.json',a.tiny_review,a.tiny_outer_exit,a.tiny_stage_exit,attempt/'exit_code']:
        sealed[str(p)]=sha(p)
    driver_output=B/f'native_rl3_{a.kind.lower()}_1m_full_sequence_20261013_v1'
    assert not driver_output.exists()
    plan=dict(driver_output=str(driver_output),cwd=str(N),environment={},sealed_inputs=sealed,tiny_gate=str(out/'gate_receipt.json'),
        aggregate=str(out/'aggregate.json'),stages=[{'name':f'task{i}','argv':['bash',str(N/'run_stage.sh'),str(config_path),'task',str(i)]} for i in range(1,6)]+[{'name':'aggregate','argv':['bash',str(N/'run_stage.sh'),str(config_path),'aggregate']}])
    destination=N/f'aux/sequence_plans/{a.kind}.json';destination.parent.mkdir(exist_ok=True)
    with destination.open('x') as f:json.dump(plan,f,indent=2)
    print(json.dumps({'plan':str(destination),'sha256':sha(destination),'driver_output':str(driver_output)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['M0','M6','M5'],required=True)
    for name in ['tiny-review','tiny-outer-exit','tiny-stage-exit']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tiny-review-sha',required=True);main(p.parse_args())

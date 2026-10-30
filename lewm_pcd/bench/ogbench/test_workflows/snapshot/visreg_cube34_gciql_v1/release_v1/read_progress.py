"""Read-only state/exit/checkpoint inventory for monitoring and advisory D backups."""
import json
import time
import common as m

def maybe(p):
    if not p.exists():return None
    try:return m.read(p)
    except (ValueError,UnicodeDecodeError):return p.read_text(errors='replace')[-5000:]

def main():
    out=dict(at=time.time(),read_only=True,NN_started_by_reader=False,stages={},groups={},backup_advisory=True)
    for name in ('status.json','closure.json','failure.json','outer_exit_code','wrapper_exit_code'):
        out['chain_'+name]=maybe(m.CASE/'control/chain/runtime'/name)
    for stage in m.STAGES:
        t=m.stage_dir(stage);r=t/'runtime'
        out['stages'][stage]={name:maybe(r/name) for name in ('status.json','closure.json','failure.json','outer_exit_code')}
        out['stages'][stage].update(wrapper_exit=maybe(t/'wrapper_exit_code'),stage_failure=maybe(t/'stage_failure.json'),reports=maybe(t/'REPORTS_ACCEPTED.json'),phase_exits={p.name:maybe(p) for p in r.glob('*_exit.json')})
    for n in (3,4):
        g=f'visreg_vitb768_cube{n}';out['groups'][g]={}
        for purpose in ('pilot','formal'):
            home=m.home(purpose);run=home/('gates' if purpose=='pilot' else 'runs')/g
            item={name:maybe(run/name) for name in ('status.json','trainer_exit_code','report.json')};item['checkpoint_inventory']=[]
            for p in sorted((run/'checkpoints').glob('update_*')):
                if not (p/'manifest.json').exists():continue
                manifest=m.read(p/'manifest.json');counter=manifest['counter']
                files=[p/'manifest.json',p/f'params_{counter}.pkl',p/'run_state.pkl']
                if not all(f.is_file() for f in files):continue
                item['checkpoint_inventory'].append(dict(path=str(p),manifest_sha256=m.sha(files[0]),manifest=manifest,
                    files=[dict(path=str(f),stat=m.stat(f)) for f in files],tensor_payloads_rehashed=False))
            item['native_status']=maybe(home/'native_sr'/g/'status.json');item['native_report']=maybe(home/'native_sr'/g/'report.json');item['native_audit']=maybe(home/'native_sr'/g/'independent_audit.json')
            out['groups'][g][purpose]=item
    print(json.dumps(out,separators=(',',':')))

if __name__=='__main__':main()

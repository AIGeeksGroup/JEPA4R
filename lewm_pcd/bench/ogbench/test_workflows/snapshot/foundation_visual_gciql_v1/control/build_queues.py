"""Bind existing commands by family; every active smoke precedes training."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT/'foundation_visual_gciql_v1'
CONTROL = CASE/'control'
PY = str(ROOT/'env_runtime/bin/python')
LEASE = str(ROOT/'cube4_visual_gciql_v1/control_v1/gpu_lease.py')
FAMILIES = {'visreg_vitb768':768, 'vjepa21_vitl1024':1024}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        json.dump(value, out, indent=2)


def make_plans(source_pins, active_families, group_prefix='', smoke_directory='smoke'):
    assert active_families and len(set(active_families))==len(active_families)
    assert all(family in FAMILIES for family in active_families)
    assert group_prefix in ('','random_init_v1_')
    phases = {'smoke': [], 'formal': []}
    def add(queue, name, script, args=(), *, gpu=False, checks=()):
        argv = [PY, '-B', str(script), *map(str, args)]
        if gpu:
            argv = [PY, '-B', LEASE, '--', *argv]
        env = dict(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
            XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_THREEFRY_PARTITIONABLE='false',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
            MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
            LD_LIBRARY_PATH=str(ROOT/'runtime/lib')+':/usr/lib/x86_64-linux-gnu',CUDA_VISIBLE_DEVICES='0' if gpu else '',
            JAX_PLATFORMS='cuda' if gpu and script.name == 'runtime.py' else 'cpu')
        phases[queue].append(dict(name=name, argv=argv, cwd=str(ROOT), environment=env, checks=list(checks)))
    for family,n in ((family,n) for family in active_families for n in range(1,5)):
        group = f'{group_prefix}{family}_cube{n}'
        home = CASE/'cache/groups'/group
        plan = home/'plan.json'
        pilot_config = CASE/'pilot/configs'/f'{group}.json'
        formal_config = CASE/'configs'/f'{group}.json'
        sm = group
        add('smoke', sm+'_01_encode_pilot', CASE/'cache/producer_bridge.py', ['--plan',plan,'--execute','--pilot'], gpu=True)
        add('smoke', sm+'_02_deliver_pilot', CASE/'cache/deliver.py', ['--plan',plan,'--pilot',
            '--producer-exit-file', CONTROL/smoke_directory/'runtime'/f'{sm}_01_encode_pilot_exit.json'])
        add('smoke', sm+'_03_config', CASE/'functional/prepare.py', ['--delivery',home/'pilot_fixture/delivery.json','--output',pilot_config])
        add('smoke', sm+'_04_hooks', CASE/'functional/runtime.py', ['verify-hooks','--config',pilot_config])
        gate = CASE/'pilot/gates'/group
        add('smoke', sm+'_05_gate', CASE/'functional/runtime.py', ['gate','--config',pilot_config,'--output',gate], gpu=True)
        add('smoke', sm+'_06_reload', CASE/'functional/runtime.py', ['reload','--config',pilot_config,'--output',gate,
            '--report',CASE/'pilot/gates'/f'{group}_reload.json'])
        native = CASE/'pilot/native_sr'/group
        add('smoke', sm+'_07_native', CASE/'functional/native_sr.py', ['--config',pilot_config,'--mode','tiny','--gate-smoke','--output',native], gpu=True)
        add('smoke', sm+'_08_native_audit', CASE/'functional/audit_native.py', ['--config',pilot_config,'--mode','tiny','--output',native])

        add('formal', sm+'_01_encode_full', CASE/'cache/producer_bridge.py', ['--plan',plan,'--execute'], gpu=True)
        audit = home/'full_audit_v1'
        add('formal', sm+'_02_cache_audit', CASE/'cache/audit_bridge.py', ['--plan',plan,'--cache-dir',home/'cache','--output-dir',audit])
        add('formal', sm+'_03_deliver', CASE/'cache/deliver.py', ['--plan',plan,'--audit-report',audit/'report.json',
            '--producer-exit-file',CONTROL/'formal/runtime'/f'{sm}_01_encode_full_exit.json'])
        add('formal', sm+'_04_config', CASE/'functional/prepare.py', ['--delivery',home/'delivery.json','--output',formal_config])
        add('formal', sm+'_05_train', CASE/'functional/runtime.py', ['train','--config',formal_config], gpu=True)
        add('formal', sm+'_06_accept', CASE/'functional/runtime.py', ['accept','--config',formal_config,
            '--output',CASE/'runs'/group,'--report',CASE/'acceptance'/f'{group}.json'])
        native = CASE/'native_sr'/group
        add('formal', sm+'_07_native', CASE/'functional/native_sr.py', ['--config',formal_config,'--mode','full','--output',native], gpu=True)
        add('formal', sm+'_08_native_audit', CASE/'functional/audit_native.py', ['--config',formal_config,'--mode','full','--output',native])
    count=len(active_families)
    return {name: dict(schema_version=1, queue=name, families=active_families, cube_counts=[1,2,3,4],group_prefix=group_prefix,
        diagnostic_random=bool(group_prefix),
        groups=[dict(group_id=f'{group_prefix}{family}_cube{n}',family=family,cube_count=n,latent_dim=FAMILIES[family]) for family in active_families for n in range(1,5)],
        expected_features=(11011000 if name=='formal' else 8008)*count,
        training_updates=4000000*count if name=='formal' else 0, SR_episodes=1000*count if name=='formal' else 0,
        diagnostic_optimizer_updates=0 if name=='formal' else 8*count, diagnostic_SR_episodes=0 if name=='formal' else 4*count,
        process_guard=str(ROOT/'prepare_runtime.py'),own_RSS_limit_bytes=60*(1<<30),
        source_pins=source_pins,phases=value) for name,value in phases.items() if not group_prefix or name=='smoke'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--diagnostic-random',action='store_true');args=parser.parse_args()
    release=json.loads(args.release.read_text())
    pins=release['source_pins'].copy()
    for path,digest in pins.items():
        assert sha(path)==digest,path
    active=release['active_families']
    prefix='random_init_v1_' if args.diagnostic_random else ''
    smoke_directory='random_smoke' if args.diagnostic_random else 'smoke'
    for family,n in ((family,n) for family in active for n in range(1,5)):
        path=CASE/'cache/groups'/f'{prefix}{family}_cube{n}'/'plan.json'
        plan=json.loads(path.read_text())
        assert (plan['cube_count'],plan['latent_dim'],plan['encoder_family'])==(n,FAMILIES[family],family)
        assert plan['cache_dir']==str(path.parent/'cache')
        assert plan['source_pins'][str((CASE/'encoder/encoder_adapter.py').relative_to(ROOT))]==pins[str(CASE/'encoder/encoder_adapter.py')]
        pins[str(path)]=sha(path)
    for family in active:
        spec=CASE/'encoder/specs'/f'{"random_diagnostic_" if args.diagnostic_random else ""}{family}.json'
        assert bool(json.loads(spec.read_text()).get('diagnostic_only',False))==args.diagnostic_random
        pins[str(spec)]=sha(spec)
    plans=make_plans(pins,active,prefix,smoke_directory)
    for name,plan in plans.items():
        save(CONTROL/(smoke_directory if name=='smoke' else name)/'launch_plan.json',plan)
    save(CONTROL/('random_queues_ready.json' if args.diagnostic_random else 'queues_ready.json'),dict(actual_exit_code=0,smoke_phases=len(plans['smoke']['phases']),
        formal_phases=len(plans.get('formal',{}).get('phases',[])),all_smokes_before_formal=True,formal_groups=0 if args.diagnostic_random else 4*len(active),
        release_sha256=sha(args.release),plans={name:sha(CONTROL/(smoke_directory if name=='smoke' else name)/'launch_plan.json') for name in plans}))


if __name__=='__main__':main()

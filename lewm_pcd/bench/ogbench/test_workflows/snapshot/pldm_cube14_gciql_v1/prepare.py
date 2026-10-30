"""Create configs/phase lists only after both immutable PLDM caches are accepted."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
EMBED = ROOT / 'pldm_cube14_embeddings_v1'
CASE = ROOT / 'pldm_cube14_gciql_v1'
STORE = Path('/autodl-fs/data/lewm/pldm_cube14_embeddings_20261025_v1/downstream_gciql_v1')
PY = ROOT / 'env_runtime/bin/python'
BUDGET = 2400000000
MARGIN = 512 << 20


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def cache_admission():
    runtime = EMBED / 'control/full/runtime'
    assert (runtime / 'outer_exit_code').read_text().strip() == '0'
    assert (runtime.parent / 'wrapper_exit_code').read_text().strip() == '0'
    closure = read(runtime / 'closure.json')
    assert closure['actual_exit_code'] == 0 and closure['completed_phases'] == 6
    accepted = {}
    for n, episodes in [(1, 1100), (4, 5500)]:
        group = f'pldm_cube{n}'
        home = EMBED / 'groups' / group
        plan = read(home / 'plan.json')
        cache = Path(plan['cache_dir'])
        audit_path = home / 'full_audit_v1/report.json'
        audit, delivery = read(audit_path), read(home / 'delivery.json')
        assert (audit_path.parent / 'actual_exit_code').read_text().strip() == '0'
        assert audit['actual_exit_code'] == 0 and audit['episodes_verified'] == episodes
        assert audit['raw_rows_verified'] == episodes * 1001 and audit['main_rows_verified'] == episodes * 1000
        assert audit['latent_dim'] == 512 and audit['dtype'] == 'float32'
        assert all(audit[k] for k in ('all_new_latent_SHA_shape_dtype_finite', 'all_raw_ID_and_true_terminal_next_contracts', 'tracked_stats_stable'))
        assert audit['plan_sha256'] == sha(home / 'plan.json')
        assert audit['manifest_sha256'] == sha(cache / 'cache_manifest.json')
        assert audit['producer_seal_sha256'] == sha(cache / 'cache_seal.json')
        assert delivery['cache_audit']['sha256'] == sha(audit_path)
        assert delivery['cube_count'] == n and delivery['latent_dim'] == 512 and delivery['encoder_binding']['family'] == 'pldm'
        for suffix in ['01_encode_full', '02_cache_audit', '03_deliver']:
            phase = read(runtime / (group + '_' + suffix + '_exit.json'))
            assert phase['actual_exit_code'] == 0 and phase['residual_owned_members'] == []
        accepted[n] = dict(delivery_path=str(home / 'delivery.json'), delivery_sha256=sha(home / 'delivery.json'),
                           cache_plan_path=str(home / 'plan.json'), cache_plan_sha256=sha(home / 'plan.json'))
    return accepted


def prepare():
    accepted = cache_admission()
    assert not STORE.exists() and not (CASE / 'configs').exists(), 'Fresh downstream namespace required'
    free = shutil.disk_usage(STORE.parent).free
    assert free >= BUDGET + MARGIN, (free, BUDGET + MARGIN)
    STORE.mkdir()
    for name in ['runs', 'gates', 'acceptance', 'native_sr', 'pilot', 'control']:
        (STORE / name).mkdir()
        (CASE / name).symlink_to(STORE / name, target_is_directory=True)
    from functional import bootstrap
    modules = bootstrap(pldm=True)
    factory = modules['prepare']
    template_path = ROOT / 'cube23_distance_gciql_v1/five_group_extension_v1/configs/G08_cube3_rgb.json'
    template = read(template_path)
    configs = {}
    for n, item in accepted.items():
        delivery = read(item['delivery_path'])
        for purpose in ['pilot', 'formal']:
            view = copy.deepcopy(delivery)
            view['purpose'] = purpose
            cfg = factory.make_config(template, view)
            cfg['delivery_binding'] = dict(path=item['delivery_path'], sha256=item['delivery_sha256'])
            cfg['cache_plan_binding'] = dict(path=item['cache_plan_path'], sha256=item['cache_plan_sha256'])
            cfg['parent_template'] = dict(path=str(template_path), sha256=sha(template_path))
            path = CASE / 'configs' / purpose / f'pldm_cube{n}.json'
            cfg['commands'] = factory.commands(path, cfg)
            save(path, cfg)
            configs[(n, purpose)] = (path, cfg)
    environments = read(CASE / 'baseline_environments.json')['phases']
    cpu = dict(next(x['environment'] for x in environments if x['name'].endswith('_06_accept')))
    gpu = dict(next(x['environment'] for x in environments if x['name'].endswith('_05_train')))
    native = dict(next(x['environment'] for x in environments if x['name'].endswith('_07_native')))
    pins = {}
    for directory in [CASE, ROOT / 'foundation_visual_gciql_v1/functional']:
        for p in directory.glob('*.py'):
            pins[str(p)] = sha(p)
    for p in CASE.glob('*.sh'):
        pins[str(p)] = sha(p)
    for p in (CASE / 'configs').glob('*/*.json'):
        pins[str(p)] = sha(p)
    for n in [1, 4]:
        plan_path = Path(accepted[n]['cache_plan_path'])
        pins[str(plan_path)] = accepted[n]['cache_plan_sha256']
        for rel, digest in read(plan_path)['source_pins'].items():
            pins[str(ROOT / rel)] = digest
    for rel in ['cube2_official_embeddings_v1/serial_relay.py', 'prepare_runtime.py',
                'foundation_visual_gciql_v1/control/serial.py', 'foundation_visual_gciql_v1/control/barrier.py',
                'cube4_visual_gciql_v1/control_v1/gpu_lease.py', 'cube23_distance_gciql_v1/distance_inputs.py',
                'cube23_distance_gciql_v1/trainer_runtime.py', 'cube23_distance_gciql_v1/gciql_boundary.py',
                'cube23_distance_gciql_v1/state_priority_v1/state_runtime.py',
                'cube23_distance_gciql_v1/native_sr_visual_v1/visual_sr.py']:
        pins[str(ROOT / rel)] = sha(ROOT / rel)
    for queue in ['smoke', 'full']:
        phases = []
        for n in [1, 4]:
            path, cfg = configs[(n, 'pilot' if queue == 'smoke' else 'formal')]
            commands = cfg['commands']
            steps = [('verify_hooks', [str(CASE / 'functional.py'), 'runtime', 'verify-hooks', '--config', str(path)], cpu)]
            if queue == 'smoke':
                steps += [('gate', commands['gate'], gpu), ('reload', commands['reload'], cpu),
                          ('native_tiny', commands['native'], native), ('tiny_audit', commands['native_audit'], cpu)]
            else:
                steps = [('train', commands['train'], gpu), ('accept', commands['accept'], cpu),
                         ('native_full', commands['native'], native), ('native_audit', commands['native_audit'], cpu)]
            for i, (label, command, env) in enumerate(steps, 1):
                argv = [str(PY), '-B'] + command
                if env['CUDA_VISIBLE_DEVICES'] == '0':
                    argv = [str(PY), '-B', str(ROOT / 'cube4_visual_gciql_v1/control_v1/gpu_lease.py'), '--'] + argv
                phases.append(dict(name=f'pldm_cube{n}_{i:02d}_{label}', argv=argv, cwd=str(ROOT), environment=env))
        save(CASE / 'control' / queue / 'launch_plan.json', dict(schema_version=1, queue='pldm_downstream', mode=queue,
             families=['pldm'], cube_counts=[1, 4], expected_features=6606600,
             training_updates=0 if queue == 'smoke' else 2000000, SR_episodes=0 if queue == 'smoke' else 500,
             diagnostic_optimizer_updates=4 if queue == 'smoke' else 0,
             process_guard=str(ROOT / 'prepare_runtime.py'), own_RSS_limit_bytes=60 << 30, phases=phases, source_pins=pins))
    save(CASE / 'PREPARED.json', dict(actual_exit_code=0, at=time.time(), accepted_caches=accepted,
         fresh_seed=0, batch=1024, updates_per_group=1000000, formal_SR_per_group=250,
         storage_free_before=free, artifact_budget_bytes=BUDGET, final_margin_bytes=MARGIN,
         original_sources_modified=False, training_started=False, SR_started=False))
    print(json.dumps(dict(actual_exit_code=0, configs=4, smoke_phases=10, full_phases=8)))


if __name__ == '__main__':
    prepare()

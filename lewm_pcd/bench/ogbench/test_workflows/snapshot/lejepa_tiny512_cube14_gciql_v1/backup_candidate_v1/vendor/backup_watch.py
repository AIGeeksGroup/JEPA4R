"""Small Linux metadata endpoint for committed checkpoints and drained phase logs.

Only this endpoint writes the new backup control acknowledgments/failures. It
never opens checkpoint payloads, signals processes, or changes experiment code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

COUNTERS = {'pilot': [1, 2], 'formal': [1] + list(range(100000, 1000001, 100000))}
MAX_METADATA = 16 << 20


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                inode=value.st_ino, device=value.st_dev)


def new_json(path, value):
    """Publish one immutable JSON file atomically, never replacing an old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()  # Only the temporary file created by this invocation.


def small_file(path):
    path = Path(path)
    before = stat(path)
    require(before['size'] <= MAX_METADATA, f'metadata exceeds 16MiB: {path}')
    digest = sha(path)
    require(stat(path) == before, f'source changed while hashing metadata: {path}')
    return dict(path=str(path), sha256=digest, stat=before)


def checkpoint_snapshot(config_path, stage, counter):
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    durability = cfg['durability']
    require(set(durability) == {'group_id', 'pilot_output', 'formal_output', 'ack_root', 'failure_root'},
            'durability must contain exactly the agreed five fields')
    group = durability['group_id']
    require(re.fullmatch(r'[A-Za-z0-9_-]+', group) and cfg['group_id'] == group, 'group binding invalid')
    require(cfg['purpose'] in ('pilot', 'formal') and stage == cfg['purpose'], 'purpose binding invalid')
    execution_root = Path(cfg['execution_root'])
    pilot_root = execution_root if stage == 'pilot' else execution_root/'pilot'
    require(Path(durability['pilot_output']) == pilot_root/'gates'/group, 'pilot output binding invalid')
    if stage == 'formal':
        require(Path(durability['formal_output']) == Path(cfg['output']), 'formal output binding invalid')
    require(counter in COUNTERS[stage], 'unexpected checkpoint counter')
    output = Path(durability['pilot_output' if stage == 'pilot' else 'formal_output'])
    directory = output/'checkpoints'/f'update_{counter:09d}'
    manifest_path = directory/'manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['counter'] == counter, 'manifest counter binding invalid')
    files = {'manifest.json': small_file(manifest_path), 'training_config.json': small_file(config_path)}
    for name, key in ((f'params_{counter}.pkl', 'agent_sha256'), ('run_state.pkl', 'sidecar_sha256')):
        require(re.fullmatch('[0-9a-f]{64}', manifest[key]), 'manifest SHA256 invalid')
        files[name] = dict(path=str(directory/name), sha256=manifest[key], stat=stat(directory/name))
    return dict(schema_version=1, kind='checkpoint', group_id=group, stage=stage, counter=counter,
        config_path=str(config_path), config_sha256=files['training_config.json']['sha256'],
        manifest_sha256=files['manifest.json']['sha256'], files=files,
        ack_root=durability['ack_root'], failure_root=durability['failure_root'],
        target_parts=['closed_checkpoints', stage, group, f'update_{counter:09d}'])


def phase_snapshot(cfg, queue, phase_name):
    require(queue in ('smoke', 'full') and re.fullmatch('[A-Za-z0-9_-]+', phase_name), 'invalid phase')
    control = Path(cfg['case_root'])/'control'/queue
    plan_path = control/'launch_plan.json'
    if not plan_path.exists():
        return None
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    phase = next((item for item in plan['phases'] if item['name'] == phase_name), None)
    require(phase is not None, 'phase absent from frozen launch plan')
    runtime = control/'runtime'
    exit_path = runtime/f'{phase_name}_exit.json'
    if not exit_path.exists():
        return None
    exit_record = json.loads(exit_path.read_text(encoding='utf-8'))
    if exit_record['residual_owned_members']:
        return None
    files = {'launch_plan.json': small_file(plan_path)}
    for name in (f'{phase_name}_argv.json', f'{phase_name}.log', f'{phase_name}_exit.json'):
        files[name] = small_file(runtime/name)
    argv = phase['argv']
    def option(flag):
        return Path(argv[argv.index(flag) + 1]) if flag in argv else None
    if exit_record['actual_exit_code'] == 0:
        report = option('--report')
        if report:
            files['phase_report_' + report.name] = small_file(report)
        output = option('--output')
        if 'train' in argv:
            output = Path(json.loads(option('--config').read_text(encoding='utf-8'))['output'])
        if output and output.is_dir():
            # The audit adds its own report after Native; keep the Native snapshot stable.
            is_audit = any(Path(item).name == 'audit_native.py' for item in argv)
            names = ['independent_audit.json'] if is_audit else [
                'report.json', 'status.json', 'resolved.json', 'train.csv',
                'final_state_fingerprint.json', 'trainer_exit_code',
                'native_source_binding.json', 'summary.json', 'source_binding.json']
            for name in names:
                if (output/name).is_file():
                    files['artifact_' + name] = small_file(output/name)
    return dict(schema_version=1, kind='logs', queue=queue, phase_name=phase_name,
        phase_exit_code=exit_record['actual_exit_code'], files=files,
        target_parts=['closed_logs', queue, phase_name])


def scan(cfg):
    checkpoints, logs = [], []
    for config_path in cfg['config_paths']:
        path = Path(config_path)
        if not path.exists():
            continue
        config = json.loads(path.read_text(encoding='utf-8'))
        stage = 'pilot' if config['purpose'] == 'pilot' else 'formal'
        for counter in COUNTERS[stage]:
            snapshot = checkpoint_snapshot(path, stage, counter)
            if snapshot:
                checkpoints.append(snapshot)
    for queue in ('smoke', 'full'):
        plan = Path(cfg['case_root'])/'control'/queue/'launch_plan.json'
        if plan.exists():
            for phase in json.loads(plan.read_text(encoding='utf-8'))['phases']:
                snapshot = phase_snapshot(cfg, queue, phase['name'])
                if snapshot:
                    logs.append(snapshot)
    return dict(checkpoints=checkpoints, logs=logs)


def refresh(cfg, snapshot):
    if snapshot['kind'] == 'checkpoint':
        require(snapshot['config_path'] in cfg['config_paths'], 'unconfigured checkpoint source')
        return checkpoint_snapshot(snapshot['config_path'], snapshot['stage'], snapshot['counter'])
    require(snapshot['kind'] == 'logs', 'unknown backup kind')
    return phase_snapshot(cfg, snapshot['queue'], snapshot['phase_name'])


def ack_path(snapshot):
    return Path(snapshot['ack_root'])/snapshot['stage']/snapshot['group_id']/f"update_{snapshot['counter']:09d}.json"


def validate_ack(snapshot, ack):
    for key in ('group_id', 'stage', 'counter', 'config_sha256', 'manifest_sha256'):
        require(ack.get(key) == snapshot[key], f'ack {key} binding mismatch')
    require(ack.get('snapshot_sha256') == digest_json(snapshot), 'ack source snapshot binding mismatch')
    require(ack.get('source_files') == snapshot['files'], 'ack source stat/hash binding mismatch')
    require(ack.get('source_after_files') == snapshot['files'], 'ack source after-copy stat binding mismatch')
    require(ack.get('actual_exit_code') == 0, 'ack copy failed')
    require(re.fullmatch('[0-9a-f]{64}', ack.get('local_receipt_sha256', '')), 'invalid D receipt SHA256')
    require(ack.get('scp_exit_codes') == [0]*len(snapshot['files']), 'ack SCP exits invalid')


def write_failure(cfg, failure):
    paths = set()
    for config_path in cfg['config_paths']:
        if Path(config_path).exists():
            item = json.loads(Path(config_path).read_text(encoding='utf-8'))
            paths.add(str(Path(item['durability']['failure_root'])/'global.json'))
    require(paths, 'no configured backup failure root')
    for path in paths:
        if not Path(path).exists():
            new_json(path, dict(failure, schema_version=1, actual_exit_code=1, at=time.time()))
    return dict(recorded=True)


def dispatch(cfg, action, payload=None):
    if action == 'scan':
        return scan(cfg)
    if action == 'failure':
        return write_failure(cfg, payload)
    snapshot = payload['snapshot']
    actual = refresh(cfg, snapshot)
    require(actual == snapshot, 'source changed since metadata scan')
    if action == 'poststat':
        return actual
    require(action == 'ack' and snapshot['kind'] == 'checkpoint', 'invalid action')
    ack = payload['ack']
    validate_ack(snapshot, ack)
    path = ack_path(snapshot)
    if path.exists():
        require(json.loads(path.read_text(encoding='utf-8')) == ack, 'existing ack differs; refusing overwrite')
    else:
        new_json(path, ack)
    return dict(accepted=True, ack_path=str(path), ack_sha256=sha(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--action', choices=('scan', 'poststat', 'ack', 'failure'), required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    payload = None if args.action == 'scan' else json.load(sys.stdin)
    print(json.dumps(dispatch(cfg, args.action, payload), sort_keys=True))


if __name__ == '__main__':
    main()

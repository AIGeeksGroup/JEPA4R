"""VISReg3/4 checkpoint metadata bridge; no model imports or payload reads."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent/'vendor'))
import backup_watch as b

FAMILY = 'visreg_vitb768'
ENCODER_SHA = '23e6aa5fe93cc61b3d9ab13e35cb4966580612d39bae39b1991f35c19c8087b8'
SPEC_SHA = '16cee3c1a0cd26f38b09691b7d71ade5dd40085fbfba15f639cf269917e31354'
ADAPTER_SHA = '3f5cb292728c809737ff0dbe851095cec8825883b0f009d913d315677f462bfe'


def expected_keys(cfg):
    """Validate this independent metadata configuration, never training configs."""
    case = Path(cfg['case_root'])
    b.require(cfg['advisory_only'] is True and cfg['expected_checkpoint_count'] == 26
              and cfg['expected_file_count'] == 104, 'invalid advisory scope')
    b.require(120 <= cfg['poll_seconds'] <= 300, 'poll interval must be 120..300 seconds')
    control = case/'control/backup_ack/v1'
    b.require(Path(cfg['ack_root']) == control/'acks' and
              Path(cfg['failure_root']) == control/'failures', 'invalid independent control roots')
    actual = set()
    for entry in cfg['config_sources']:
        stage, group = entry['stage'], entry['group_id']
        b.require(stage in b.COUNTERS and group in (FAMILY+'_cube3', FAMILY+'_cube4'), 'invalid source scope')
        b.require(Path(entry['config_path']) == case/'configs'/stage/(group+'.json'), 'config source root binding invalid')
        b.require(re.fullmatch('[0-9a-f]{64}', entry['config_sha256']), 'config SHA invalid')
        b.require((stage, group) not in actual, 'duplicate source identity')
        actual.add((stage, group))
    expected = {(stage, FAMILY+f'_cube{n}') for stage in b.COUNTERS for n in (3, 4)}
    b.require(actual == expected, 'exactly four VISReg3/4 source configs required')
    return {(stage, group, counter) for stage, group in expected for counter in b.COUNTERS[stage]}


def metadata(path):
    evidence = b.small_file(path)
    raw = Path(path).read_bytes()
    b.require(hashlib.sha256(raw).hexdigest() == evidence['sha256'] and
              b.stat(path) == evidence['stat'], 'metadata changed during read')
    return json.loads(raw), evidence


def checkpoint_snapshot(cfg, entry, counter):
    expected_keys(cfg)
    stage, group = entry['stage'], entry['group_id']
    b.require(counter in b.COUNTERS[stage], 'unexpected checkpoint counter')
    config_path = Path(entry['config_path'])
    config, config_file = metadata(config_path)
    b.require(config_file['sha256'] == entry['config_sha256'], 'published config SHA binding changed')
    n = int(group[-1])
    b.require(config['group_id'] == config['role'] == group and config['cube_count'] == n
              and config['purpose'] == stage, 'training config group/stage binding invalid')
    b.require(config['input_contract'] == dict(current_dim=787, goal_dim=768+3*n,
              latent_dim=768, zero_padding=False), 'training config input dimensions changed')
    encoder = config['encoder_binding']
    b.require(encoder['family'] == FAMILY and encoder['latent_dim'] == 768 and
              encoder['checkpoint_sha256'] == ENCODER_SHA and encoder['spec_sha256'] == SPEC_SHA
              and encoder['adapter']['sha256'] == ADAPTER_SHA, 'encoder provenance changed')
    home = Path(cfg['case_root'])/'outputs'/stage
    b.require(Path(config['execution_root']) == home and
              Path(config['output']) == home/'runs'/group, 'training config output binding changed')
    output = home/'gates'/group if stage == 'pilot' else Path(config['output'])
    directory = output/'checkpoints'/f'update_{counter:09d}'
    manifest_path = directory/'manifest.json'
    if not manifest_path.exists():
        return None
    manifest, manifest_file = metadata(manifest_path)
    b.require(type(manifest['counter']) is int and manifest['counter'] == counter, 'manifest counter binding invalid')
    files = {'manifest.json': manifest_file, 'training_config.json': config_file}
    for name, key in ((f'params_{counter}.pkl', 'agent_sha256'), ('run_state.pkl', 'sidecar_sha256')):
        b.require(re.fullmatch('[0-9a-f]{64}', manifest[key]), 'manifest payload SHA invalid')
        path = directory/name
        b.require(path.is_file(), 'committed checkpoint payload missing')
        files[name] = dict(path=str(path), sha256=manifest[key], stat=b.stat(path))
    b.require(b.stat(manifest_path) == manifest_file['stat'] and b.stat(config_path) == config_file['stat'],
              'source metadata changed during snapshot')
    return dict(schema_version=1, kind='checkpoint', group_id=group, stage=stage, counter=counter,
        config_path=str(config_path), config_sha256=config_file['sha256'],
        manifest_sha256=manifest_file['sha256'], files=files, ack_root=cfg['ack_root'],
        failure_root=cfg['failure_root'],
        target_parts=['closed_checkpoints', stage, group, f'update_{counter:09d}'])


def scan(cfg):
    expected_keys(cfg)
    b.require(not (Path(cfg['failure_root'])/'global.json').exists(), 'sticky advisory backup failure exists')
    checkpoints = []
    for entry in cfg['config_sources']:
        for counter in b.COUNTERS[entry['stage']]:
            snapshot = checkpoint_snapshot(cfg, entry, counter)
            if snapshot is not None:
                checkpoints.append(snapshot)
    return dict(checkpoints=checkpoints, logs=[])


def refresh(cfg, snapshot):
    b.require(snapshot.get('kind') == 'checkpoint', 'only checkpoint snapshots supported')
    entries = [e for e in cfg['config_sources'] if e['config_path'] == snapshot.get('config_path')
               and e['stage'] == snapshot.get('stage') and e['group_id'] == snapshot.get('group_id')]
    b.require(len(entries) == 1, 'unconfigured checkpoint identity')
    return checkpoint_snapshot(cfg, entries[0], snapshot['counter'])


def dispatch(cfg, action, payload=None):
    expected_keys(cfg)
    if action == 'scan':
        return scan(cfg)
    if action == 'failure':
        path = Path(cfg['failure_root'])/'global.json'
        if not path.exists():
            b.new_json(path, dict(payload, schema_version=1, actual_exit_code=1,
                at=time.time(), advisory_only=True, automatic_restart=False))
        return dict(recorded=True, advisory_only=True)
    b.require(action in ('poststat', 'ack'), 'invalid endpoint action')
    snapshot = payload['snapshot']
    actual = refresh(cfg, snapshot)
    b.require(actual == snapshot, 'source changed since metadata snapshot')
    if action == 'poststat':
        return actual
    ack = payload['ack']
    b.validate_ack(actual, ack)
    path = b.ack_path(actual)
    if path.exists():
        b.require(json.loads(path.read_text(encoding='utf-8')) == ack, 'existing ack differs; refusing overwrite')
    else:
        b.new_json(path, ack)
    return dict(accepted=True, ack_path=str(path), ack_sha256=b.sha(path), advisory_only=True)


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

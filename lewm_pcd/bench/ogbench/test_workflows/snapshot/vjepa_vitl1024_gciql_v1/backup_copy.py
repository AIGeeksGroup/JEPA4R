"""Persistent Windows D copier for the dedicated V-JEPA run; stdlib plus ssh/scp.

Copy only committed transactions and drained phase metadata. A partial or
changed destination is preserved and fails closed; no automatic experiment
restart, payload unpickling, deletion, or source mutation is performed.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import traceback
import uuid

from backup_watch import COUNTERS, digest_json, new_json, require, sha, stat


class SSHTransport:
    def __init__(self, cfg, runtime):
        self.cfg, self.runtime = cfg, Path(runtime)

    def command(self, argv, prefix, payload=None, timeout=240):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        record = dict(argv=argv, started=time.time())
        with prefix.with_suffix('.stdout.log').open('xb') as stdout, prefix.with_suffix('.stderr.log').open('xb') as stderr:
            try:
                child = subprocess.run(argv, input=payload, stdout=stdout, stderr=stderr, timeout=timeout)
                record['actual_exit_code'] = child.returncode
            except BaseException:
                record.update(actual_exit_code=None, error=traceback.format_exc())
                new_json(prefix.with_suffix('.exit.json'), record)
                raise
        record['completed'] = time.time()
        new_json(prefix.with_suffix('.exit.json'), record)
        return record

    def request(self, action, payload=None):
        cfg = self.cfg
        remote = shlex.join([cfg['remote_python'], '-B', cfg['remote_script'],
                             '--config', cfg['remote_config'], '--action', action])
        argv = ['ssh', '-p', str(cfg['ssh_port']), '-o', 'BatchMode=yes',
                '-o', 'ConnectTimeout=90', '-o', 'ServerAliveInterval=30',
                '-o', 'ServerAliveCountMax=3', cfg['ssh_host'], remote]
        prefix = self.runtime/'commands'/f'{time.time_ns()}_{action}_{uuid.uuid4().hex[:8]}'
        record = self.command(argv, prefix, None if payload is None else json.dumps(payload).encode())
        require(record['actual_exit_code'] == 0, f'SSH {action} failed: {prefix}')
        return json.loads(prefix.with_suffix('.stdout.log').read_text(encoding='utf-8'))

    def copy(self, source, target, log_prefix):
        cfg = self.cfg
        argv = ['scp', '-P', str(cfg['ssh_port']), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=90',
                '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
                f"{cfg['ssh_host']}:{source}", str(target)]
        return self.command(argv, log_prefix, timeout=300)


def accepted_receipt(snapshot, receipt_path):
    receipt_path = Path(receipt_path)
    require(receipt_path.is_file(), f'partial or unaccepted D directory preserved: {receipt_path.parent}')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    require(receipt['actual_exit_code'] == 0 and receipt['snapshot'] == snapshot,
            'accepted receipt source binding changed; refusing overwrite')
    require(receipt['source_after_files'] == snapshot['files'], 'accepted source after-copy binding changed')
    require(set(receipt['local_files']) == set(snapshot['files']), 'accepted local file set incomplete')
    require(receipt.get('source_stable') is True, 'accepted source stability proof missing')
    require(len(receipt['copy_records']) == len(snapshot['files']) and
            all(item['actual_exit_code'] == 0 for item in receipt['copy_records']),
            'accepted SCP copy record count/exits invalid')
    for name, evidence in receipt['local_files'].items():
        require(evidence['sha256'] == snapshot['files'][name]['sha256'] and
                evidence['stat']['size'] == snapshot['files'][name]['stat']['size'],
                f'accepted local file SHA/size binding changed: {name}')
        require(stat(receipt_path.parent/name) == evidence['stat'], f'accepted local file stat changed: {name}')
    return receipt


def copy_snapshot(snapshot, local_root, transport):
    root = Path(local_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    parts = snapshot['target_parts']
    require(all(part not in ('.', '..') and '/' not in part and '\\' not in part for part in parts),
            'unsafe local target path')
    directory = root.joinpath(*parts)
    receipt_path = directory/'receipt.json'
    try:
        if directory.exists():
            receipt = accepted_receipt(snapshot, receipt_path)
        else:
            required_bytes = sum(value['stat']['size'] for value in snapshot['files'].values())
            require(shutil.disk_usage(root).free > required_bytes + (64 << 20), 'insufficient D copy free space')
            directory.mkdir(parents=True, exist_ok=False)
            local_files, copy_records = {}, []
            for index, (name, source) in enumerate(snapshot['files'].items()):
                require(Path(name).name == name, 'invalid copy filename')
                target = directory/name
                record = transport.copy(source['path'], target, directory/f'scp_{index:02d}')
                copy_records.append(record)
                require(record['actual_exit_code'] == 0, f'SCP copy failed: {name}')
                require(target.stat().st_size == source['stat']['size'], f'copy size mismatch: {name}')
                require(sha(target) == source['sha256'], f'copy SHA256 hash mismatch: {name}')
                with target.open('r+b') as handle:
                    os.fsync(handle.fileno())
                local_files[name] = dict(sha256=source['sha256'], stat=stat(target))
            after = transport.request('poststat', dict(snapshot=snapshot))
            require(after == snapshot, 'source changed after SCP copy')
            receipt = dict(schema_version=1, actual_exit_code=0, snapshot=snapshot,
                local_files=local_files, copy_records=copy_records, completed=time.time(),
                source_after_files=after['files'], source_stable=True,
                no_pickle_deserialization=True)
            new_json(receipt_path, receipt)
        receipt_sha = sha(receipt_path)
        if snapshot['kind'] == 'checkpoint':
            ack = {key: snapshot[key] for key in ('group_id', 'stage', 'counter', 'config_sha256', 'manifest_sha256')}
            ack.update(schema_version=1, actual_exit_code=0, snapshot_sha256=digest_json(snapshot),
                source_files=snapshot['files'], local_receipt_path=str(receipt_path),
                source_after_files=receipt['source_after_files'],
                local_receipt_sha256=receipt_sha,
                scp_exit_codes=[record['actual_exit_code'] for record in receipt['copy_records']])
            transport.request('ack', dict(snapshot=snapshot, ack=ack))
        return dict(receipt_path=str(receipt_path), receipt_sha256=receipt_sha)
    except BaseException:
        failure = dict(error=traceback.format_exc(), target=str(directory), snapshot=snapshot,
                       at=time.time(), actual_exit_code=1, automatic_restart=False)
        if directory.exists() and not (directory/'failure.json').exists():
            new_json(directory/'failure.json', failure)
        try:
            transport.request('failure', failure)
        except BaseException:
            # Preserve the original error; the local daemon also persists SSH errors.
            pass
        raise


def completion_requested(runtime, accepted_checkpoints):
    stop_path = Path(runtime)/'STOP.json'
    if not stop_path.exists():
        return False
    stop = json.loads(stop_path.read_text(encoding='utf-8'))
    require(stop.get('reason') == 'task_complete' and stop.get('expected_checkpoint_count') == 52,
            'STOP must request task_complete with expected_checkpoint_count 52')
    expected = {(stage, f'vjepa21_vitl1024_cube{cube}', counter)
        for stage, counters in COUNTERS.items() for cube in range(1, 5) for counter in counters}
    require(accepted_checkpoints == expected, 'STOP received with incomplete accepted checkpoint set; expected exactly 52')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    root = Path(cfg['local_root'])
    root.mkdir(parents=True, exist_ok=True)
    runtime = root/'backup_runtime'
    runtime.mkdir(exist_ok=True)
    lock_path = runtime/'worker.lock'
    lock = lock_path.open('x', encoding='utf-8')
    lock.write(json.dumps(dict(pid=os.getpid(), started=time.time(), config_sha256=sha(args.config))))
    lock.flush()
    worker = SSHTransport(cfg, runtime)
    accepted = {}
    accepted_checkpoints = set()
    started = time.time()
    exit_code = 1
    state_path = runtime/'status.json'
    def state(stage, **extra):
        value = dict(schema_version=1, phase=stage, started=started, updated=time.time(),
            config_sha256=sha(args.config), completed_count=len(accepted), failed_count=int(stage == 'failed'),
            automatic_restart=False, **extra)
        temporary = runtime/f'status_{uuid.uuid4().hex}.tmp'
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(temporary, state_path)  # Only this worker's mutable progress file.
    try:
        state('scanning', exit_code=None)
        while True:
            result = worker.request('scan')
            for snapshot in result['checkpoints'] + result['logs']:
                key = digest_json(snapshot)
                if key in accepted:
                    evidence = accepted[key]
                    require(sha(evidence['receipt_path']) == evidence['receipt_sha256'], 'accepted D receipt changed')
                    accepted_receipt(snapshot, evidence['receipt_path'])
                    continue
                state('copying', current_target=snapshot['target_parts'], exit_code=None)
                accepted[key] = copy_snapshot(snapshot, root, worker)
                if snapshot['kind'] == 'checkpoint':
                    accepted_checkpoints.add((snapshot['stage'], snapshot['group_id'], snapshot['counter']))
            if completion_requested(runtime, accepted_checkpoints):
                exit_code = 0
                state('completed', exit_code=0, checkpoint_count=len(accepted_checkpoints), stop_reason='task_complete')
                return 0
            state('waiting', exit_code=None)
            if args.once:
                exit_code = 0
                state('completed_once', exit_code=0)
                return 0
            time.sleep(60)
    except BaseException:
        exit_code = 1
        failure = dict(error=traceback.format_exc(), at=time.time(), actual_exit_code=1)
        new_json(runtime/f'failure_{time.time_ns()}.json', failure)
        state('failed', exit_code=1, error=failure['error'])
        try:
            worker.request('failure', failure)
        except BaseException:
            pass
        raise
    finally:
        lock.close()
        lock_path.unlink()  # The exclusive lock owned by this worker.
        new_json(runtime/f'worker_exit_{time.time_ns()}.json', dict(actual_exit_code=exit_code, completed=time.time()))


if __name__ == '__main__':
    raise SystemExit(main())

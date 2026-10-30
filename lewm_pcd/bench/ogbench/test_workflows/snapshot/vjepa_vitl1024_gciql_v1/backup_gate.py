"""CPU-only durability barrier; no imports from the model or training runtime."""
import argparse
import json
from pathlib import Path
import time
import traceback

from backup_watch import COUNTERS, ack_path, checkpoint_snapshot, new_json, require, sha, validate_ack


def check_once(config_path, stage, progress=None):
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    durability = cfg['durability']
    failure_root = Path(durability['failure_root'])
    require(not any(failure_root.rglob('*.json')) if failure_root.exists() else True,
            'backup failure exists; no automatic continuation')
    backup_stage = 'pilot' if stage == 'gate' else 'formal'
    records = []
    missing_manifest, missing_ack = [], []
    for counter in COUNTERS[backup_stage]:
        snapshot = checkpoint_snapshot(config_path, backup_stage, counter)
        if snapshot is None:
            missing_manifest.append(counter)
            continue
        path = ack_path(snapshot)
        if not path.exists():
            missing_ack.append(counter)
            continue
        ack = json.loads(path.read_text(encoding='utf-8'))
        validate_ack(snapshot, ack)
        records.append(dict(counter=counter, manifest_sha256=snapshot['manifest_sha256'],
            ack_path=str(path), ack_sha256=sha(path), local_receipt_sha256=ack['local_receipt_sha256']))
    if progress is not None:
        progress.update(expected_checkpoint_count=len(COUNTERS[backup_stage]),
            accepted_ack_counters=[item['counter'] for item in records],
            missing_manifest_counters=missing_manifest, missing_ack_counters=missing_ack,
            checked_at=time.time())
    if missing_manifest or missing_ack:
        return None
    return dict(schema_version=1, actual_exit_code=0, stage=stage, backup_stage=backup_stage,
        group_id=cfg['group_id'], config_path=str(config_path), config_sha256=sha(config_path),
        checkpoint_count=len(records), checkpoints=records, completed=time.time(), timeout_seconds=600)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('gate', 'train'), required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    require(0 < args.timeout <= 600, 'backup timeout must be in (0,600] seconds')
    started = time.monotonic()
    progress = {}
    try:
        while True:
            result = check_once(args.config, args.stage, progress)
            if result:
                result.update(timeout_seconds=args.timeout, waited_seconds=time.monotonic()-started,
                              progress=progress)
                new_json(args.report, result)
                print(json.dumps(result, sort_keys=True))
                return 0
            remaining = args.timeout-(time.monotonic()-started)
            require(remaining > 0, f'checkpoint D backup acknowledgment timeout ({args.timeout:g}s)')
            time.sleep(min(2, remaining))
    except BaseException:
        if not args.report.exists():
            new_json(args.report, dict(schema_version=1, actual_exit_code=1, stage=args.stage,
                config_path=str(args.config), at=time.time(), error=traceback.format_exc(), automatic_restart=False,
                timeout_seconds=args.timeout, waited_seconds=time.monotonic()-started, progress=progress))
        raise


if __name__ == '__main__':
    raise SystemExit(main())

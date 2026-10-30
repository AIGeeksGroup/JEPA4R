"""Closed last3 evidence only. Root separately records builder and SCP exits.

Prepared artifact: do not execute until Root's explicit closed signal.
Stdlib only; no imports of model code, pickle loads, raw data reads or deletion.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tarfile
import time


D = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
H = D / 'official_state_last3_v1'
R = H / 'runtime_v2'
A = H / 'attempt_v2'
L = D / 'official_state_formal_v1/runtime'
STEPS = (800000, 900000, 1000000)
SMALL_LIMIT = 32 * 1024**2
MODEL_LIMIT = 64 * 1024**2
EVAL_SHA = '4d9a2732bc9ef89ce99fbc0f07a76d3f4d035783af3a658251709140ff9de3d5'
R_NAMES = (
    'actual_argv.json', 'entry_binding.json', 'entry_status.json', 'entry_events.jsonl',
    'launch_plan.json', 'launch_freeze.json', 'status.json', 'runtime_check.log',
    'runtime_check_exit.json', 'train.log', 'resources.jsonl', 'trainer_exit.json',
    'trainer_actual_exit_code', 'eval.csv', 'subset_AST_proof.json',
    'original_eval_body_binding.json', 'old_eval_csv_protected.json',
    'restore_template_1000000.json',
) + tuple(f'{stem}_{step}.json' for step in STEPS
          for stem in ('restore_evaluation', 'deferred_eval'))
A_NAMES = ('launch_plan.json', 'launch_freeze.json', 'dispatch_receipt.json',
           'dispatch.log', 'supervisor_actual_exit_code')
H_NAMES = (
    'build_last3_evidence.py', 'PACKAGE_PLAN.md', 'evaluate_last3.py',
    'evaluate_last3_v2.py', 'supervise_last3.py', 'supervise_last3_v2.py',
    'run_last3.sh', 'run_last3_v2.sh', 'test_evaluate_last3.py',
    'test_evaluate_last3_v2.py', 'check_last3_v2_flags.py', 'README.md',
    'source_freeze_v1.json', 'source_freeze_v2.json', 'launch_plan.json',
    'REVIEW_v2.json', 'runtime/trainer_exit.json',
    'runtime/trainer_actual_exit_code', 'runtime/supervisor_failure.json',
    'runtime/train.log', 'supervisor_actual_exit_code',
)
# Fixed small metadata that may exist only on Root's local side. No globbing.
H_OPTIONAL = ('ROOT_START_RECEIPT.json', 'root_status.json', 'startup_snapshot_1.json',
              'Root_prelaunch_acceptance.json', 'dispatch_receipt.json',
              'Root_dispatch_exit.json', 'Root_dispatch_stdout.log',
              'Root_dispatch_stderr.log', 'REVIEW.md', 'REVIEW.json',
              'CPU_exit.json', 'CPU_stdout.log', 'CPU_stderr.log', 'CPU_v2_exit.json',
              'CPU_v2_stdout.log', 'CPU_v2_stderr.log', 'real_flag_regression.json',
              'real_flag_regression_actual_exit_code', 'real_flag_regression_stdout.log',
              'real_flag_regression_stderr.log', 'runtime/entry_status.json',
              'LAST3_COMPLETION_READBACK.json', 'LAST3_COMPLETION_READBACK.md')
DEPENDENCIES = (
    'official_state_formal_v1/formal_main.py', 'official_state_formal_v1/supervise.py',
    'official_state_formal_v1/runtime_check.py', 'prepare_runtime.py',
    'requirements.runtime.lock', 'official_state_workers_v1/ordered_input.py',
    'official_state_workers_v1/measure.py', 'official_state_workers_v1/selected_input.json',
    'official_state_formal_v1/entry_freeze_v1.json',
    'source/ogbench/impls/main.py', 'source/ogbench/impls/agents/gciql.py',
    'source/ogbench/impls/utils/datasets.py', 'source/ogbench/impls/utils/env_utils.py',
    'source/ogbench/impls/utils/evaluation.py', 'source/ogbench/impls/utils/log_utils.py',
    'source/ogbench/impls/utils/flax_utils.py', 'source/ogbench/impls/hyperparameters.sh',
    'source/ogbench/ogbench/utils.py',
)
# Original migration manifest.json entries, read locally during preparation.
MIGRATION_PINS = {
    'source/ogbench/impls/utils/encoders.py': dict(bytes=4644,
        sha256='6bd7fdd82838b754d26af01f9e84d70fa63fca1e09efee38a09f325884073ecc'),
    'source/ogbench/impls/utils/networks.py': dict(bytes=16753,
        sha256='e7e4c4a439b691911e5ae6c2653f253c5703e6b5d190d813e28ebc32e7306094'),
}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_stat(path):
    need(path.is_absolute() and path.is_relative_to(D), 'Outside fixed case: ' + str(path))
    for node in (path, *path.parents):
        if node == D.parent:
            break
        need(not node.is_symlink(), 'Refuse symlink: ' + str(node))
    value = path.stat()
    need(stat.S_ISREG(value.st_mode), 'Not a regular file: ' + str(path))
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns,
                ctime_ns=value.st_ctime_ns, inode=value.st_ino, device=value.st_dev)


def read_bytes(path, limit=SMALL_LIMIT):
    before = source_stat(path)
    need(before['size'] <= limit, 'Oversize source: ' + str(path))
    content = path.read_bytes()
    need(source_stat(path) == before and len(content) == before['size'],
         'Source changed while reading: ' + str(path))
    return content


def read_json(path):
    return json.loads(read_bytes(path))


def sha(content):
    return hashlib.sha256(content).hexdigest()


def digest_file(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def code_zero(path):
    need(read_bytes(path).decode().strip() == '0', 'Nonzero exit: ' + str(path))


def closure():
    trainer = read_json(R / 'trainer_exit.json')
    need(trainer['actual_exit_code'] == 0 and trainer['residual_members'] == [],
         'Trainer not successfully closed')
    code_zero(R / 'trainer_actual_exit_code')
    code_zero(A / 'supervisor_actual_exit_code')
    status = read_json(R / 'entry_status.json')
    need(status['phase'] == 'last3_evaluation_returned' and status['actual_exit_code'] == 0,
         'Entry did not return successfully')
    need(read_json(R / 'status.json')['stage'] == 'last3_evaluation_completed',
         'Supervisor completion status missing')
    need(read_json(R / 'runtime_check_exit.json')['actual_exit_code'] == 0,
         'Runtime check did not pass')
    binding = read_json(R / 'entry_binding.json')
    need(binding['checkpoint_steps'] == list(STEPS) and
         binding['training_updates_in_this_process'] == 0 and
         binding['source_training_updates'] == 1000000 and
         binding['dataset_loaded'] is False and binding['agent_update_called'] is False,
         'Unexpected evaluation-only contract')
    need(binding['evaluation'] == dict(tasks=5, episodes_per_task=50,
         video_episodes_per_task=1, max_episode_steps=1000, temperature=0., Gaussian=None, CPU=True),
         'Unexpected evaluation protocol')
    model = Path(binding['model_dir'])
    need(model.is_relative_to(L / 'model_outputs'), 'Model pointer outside original model root')
    plan = read_json(A / 'launch_plan.json')
    need(plan == read_json(R / 'launch_plan.json') and
         read_json(A / 'launch_freeze.json') == read_json(R / 'launch_freeze.json') and
         plan['output'] == str(R) and read_json(R / 'actual_argv.json')['argv'] == plan['argv'],
         'Actual invocation does not match launch plan')
    for flag, expected in (('--model-dir', model), ('--ledger-dir', L), ('--output-dir', R)):
        need(plan['argv'][plan['argv'].index(flag) + 1] == str(expected), 'Wrong ' + flag)
    reports, restores = {}, {}
    for step in STEPS:
        report = read_json(R / f'deferred_eval_{step}.json')
        need(report['actual_exit_code'] == 0 and report['checkpoint_epoch'] == step and
             report['RNG_ledger_match'] is True and report['original_evaluation_body_used'] is True,
             'Incomplete checkpoint report: ' + str(step))
        metrics = report['metrics']
        need(metrics.get('step') == step, 'Wrong checkpoint step in metrics')
        success_metrics = {key: value for key, value in metrics.items() if key != 'step'}
        values = [v for k, v in success_metrics.items() if k != 'evaluation/overall_success']
        need(len(success_metrics) == 6 and len(values) == 5 and 'evaluation/overall_success' in success_metrics and
             all(k.startswith('evaluation/') and k.endswith('_success') for k in success_metrics) and
             all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
                 for v in success_metrics.values()), 'Invalid success metrics')
        need(all(abs(v * 50 - round(v * 50)) < 1e-6 for v in values) and
             abs(sum(values) / 5 - metrics['evaluation/overall_success']) < 1e-7,
             'Success metrics disagree with 50 measured episodes/task')
        reports[str(step)] = report
        for purpose in (('evaluation', 'template') if step == 1000000 else ('evaluation',)):
            name = f'restore_{purpose}_{step}.json'
            receipt = read_json(R / name)
            need(receipt['actual_exit_code'] == 0 and receipt['checkpoint_epoch'] == step and
                 receipt['state_leaves'] == 153 and receipt['all_state_finite'] is True and
                 receipt['network_step'] == step + 1 and receipt['adam_count'] == step and
                 receipt['training_updates_in_this_process'] == 0 and
                 receipt['checkpoint_path'] == str(model / f'params_{step}.pkl'),
                 'Invalid restore receipt: ' + name)
            restores[name] = receipt
    events = [json.loads(line) for line in read_bytes(R / 'entry_events.jsonl').splitlines()]
    returned = [e for e in events if e['phase'] == 'evaluation_task_returned']
    need(len(returned) == 15 and [(e['evaluation_checkpoint'], e['task_id']) for e in returned]
         == [(step, task) for step in STEPS for task in range(1, 6)], 'Expected exactly 15 ordered task returns')
    checkpoint_returns = [e['checkpoint_epoch'] for e in events
                          if e['phase'] == 'deferred_checkpoint_evaluation_returned']
    need(checkpoint_returns == list(STEPS) and events[-1] == status,
         'Missing checkpoint returns or final entry event')
    rows = list(csv.DictReader(io.StringIO(read_bytes(R / 'eval.csv').decode())))
    need(len(rows) == 3 and [int(row['step']) for row in rows] == list(STEPS), 'Wrong CSV checkpoint rows')
    for row, step in zip(rows, STEPS):
        need(all(abs(float(row[k]) - v) < 1e-7 for k, v in reports[str(step)]['metrics'].items()),
             'CSV/report metrics mismatch')
    protected = read_json(R / 'old_eval_csv_protected.json')
    need(protected['actual_exit_code'] == 0 and
         protected['before_sha256'] == protected['after_sha256'] == binding['old_eval_csv_before_sha256'],
         'Old evaluation CSV protection did not pass')
    return dict(trainer=trainer, supervisor_actual_exit_code=0, entry_status=status,
                binding=binding, reports=reports, restore_receipts=restores,
                task_return_events=15, checkpoint_return_events=checkpoint_returns,
                old_eval_csv_protected=protected)


def collect(gates):
    paths = {}

    def add(path):
        source_stat(path)
        paths[path.relative_to(D).as_posix()] = path

    for root, names in ((R, R_NAMES), (A, A_NAMES), (H, H_NAMES)):
        for name in names:
            add(root / name)
    for name in H_OPTIONAL:
        if (H / name).exists():
            add(H / name)
    for name in (*DEPENDENCIES, *MIGRATION_PINS):
        add(D / name)
    freeze = read_json(A / 'launch_freeze.json')['files']
    historical = {}
    for name, path in paths.items():
        if name in freeze:
            content = read_bytes(path)
            need(len(content) == freeze[name]['bytes'] and sha(content) == freeze[name]['sha256'],
                 'Historical launch freeze mismatch: ' + name)
            historical[name] = freeze[name]
    need(all(name in historical for name in DEPENDENCIES), 'Missing fixed dependency launch pins')
    for name, expected in MIGRATION_PINS.items():
        content = read_bytes(D / name)
        need(len(content) == expected['bytes'] and sha(content) == expected['sha256'],
             'Original migration source pin mismatch: ' + name)
        historical[name] = expected
    need(sha(read_bytes(H / 'evaluate_last3_v2.py')) == gates['binding']['entry_sha256'],
         'Entry source differs from actual binding')
    need(sha(read_bytes(D / 'source/ogbench/impls/utils/evaluation.py')) == EVAL_SHA,
         'Original evaluation.py changed')
    model = Path(gates['binding']['model_dir'])
    add(model / 'flags.json')
    add(L / 'entry_binding.json')
    need(read_json(model / 'flags.json')['agent'] == gates['binding']['official_flags']['agent'] ==
         read_json(L / 'entry_binding.json')['official_flags']['agent'], 'Original agent config mismatch')
    pointers = {}
    for step in STEPS:
        sidecar_path = model / f'checkpoint_{step}_sidecar.json'
        sidecar = read_json(sidecar_path)
        path = model / f'params_{step}.pkl'
        current_stat = source_stat(path)  # 800K/900K pickle content is never opened.
        need(sidecar['epoch'] == step and sidecar['checkpoint_path'] == str(path) and
             current_stat['size'] == sidecar['checkpoint_bytes'] and
             sidecar['network_step'] == step + 1 and sidecar['adam_count'] == step,
             'Original sidecar/stat mismatch: ' + str(step))
        for receipt in gates['restore_receipts'].values():
            if receipt['checkpoint_epoch'] == step:
                need(receipt['checkpoint_sha256'] == sidecar['checkpoint_sha256'], 'Restore SHA pointer mismatch')
        pointers[str(step)] = dict(path=str(path), bytes=current_stat['size'],
            sha256=sidecar['checkpoint_sha256'], sha256_source='original_checkpoint_sidecar',
            source_stat=current_stat, content_read_by_builder=step == 1000000,
            archived=step == 1000000, sidecar_path=str(sidecar_path))
        add(sidecar_path)
        rng = model / f'numpy_rng_{step}.npz'
        need(sidecar['host_numpy_rng_sidecar'] == str(rng) and
             sha(read_bytes(rng)) == sidecar['host_numpy_rng_sidecar_sha256'], 'Checkpoint RNG sidecar mismatch')
        add(rng)
        ledger_path = L / f'eval_node_{step}_seed_ledger.json'
        ledger_content = read_bytes(ledger_path)
        ledger = json.loads(ledger_content)
        bound = gates['binding']['ledger_pointers'][str(step)]
        need(bound['path'] == str(ledger_path) and bound['sha256'] == sha(ledger_content) and
             ledger['step'] == step and ledger['task_seeds'] == bound['task_seeds'] and
             [s['task_id'] for s in ledger['task_seeds']] == [1, 2, 3, 4, 5], 'Ledger binding mismatch')
        add(ledger_path)
        for phase in ('before', 'after'):
            rng = L / f'eval_node_{step}_{phase}_numpy.npz'
            need(ledger[f'{phase}_numpy_path'] == str(rng) and
                 sha(read_bytes(rng)) == ledger[f'{phase}_numpy_sha256'], 'Ledger RNG mismatch')
            add(rng)
    add(model / 'params_1000000.pkl')
    return dict(sorted(paths.items())), pointers, historical


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed-signal', required=True)
    args = parser.parse_args()
    need(args.closed_signal.strip(), 'A nonempty explicit Root closed signal is required')
    need(Path(__file__).resolve() == H / 'build_last3_evidence.py', 'Wrong deployment path')
    os.nice(10)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    out = H / 'closed_evidence_v1'
    need(not out.exists(), 'Refuse existing output directory; no overwrite or deletion')
    gates = closure()
    paths, pointers, historical = collect(gates)
    before = {name: source_stat(path) for name, path in paths.items()}
    out.mkdir(exist_ok=False)
    write_new(out / 'closure_checks.json', dict(closed_signal=args.closed_signal, checked=time.time(), **gates))
    write_new(out / 'checkpoint_pointers.json', pointers)
    write_new(out / 'package_policy.json', dict(
        checkpoints_copied=[1000000], checkpoints_pointer_only=[800000, 900000],
        raw_datasets_accessed=False, raw_datasets_rehashed=False, model_deserialized=False,
        original_evaluation_source_sha256=EVAL_SHA, measured_episodes_per_task=50,
        video_episodes_per_task=1, video_episodes_excluded_from_SR=True,
        original_initial_scenes_reproduced=False, original_actor_seed_ledgers_reused=True,
        historical_source_pins_verified=historical,
        extra_source_pin_origin='original local coordination manifest.json migration entries',
        extra_source_migration_pins=MIGRATION_PINS,
        archive_sources=list(paths), excluded='raw datasets, caches, environments, tmp, WandB, media, other checkpoints, old archives'))
    for name in ('closure_checks.json', 'checkpoint_pointers.json', 'package_policy.json'):
        paths['_backup/' + name] = out / name
        before['_backup/' + name] = source_stat(out / name)
    manifest = {}
    target = out / 'closed_last3_SR_evidence_v1.tar.gz'
    # Exclusive create and a fresh output directory: failed partials are retained.
    with target.open('xb') as raw, tarfile.open(fileobj=raw, mode='w:gz') as archive:
        for name, path in paths.items():
            need(source_stat(path) == before[name], 'Source changed before archive: ' + name)
            final_model = path.name == 'params_1000000.pkl'
            content = read_bytes(path, MODEL_LIMIT if final_model else SMALL_LIMIT)
            value = dict(bytes=len(content), sha256=sha(content), source_path=str(path),
                         source_stat=before[name])
            if name in historical:
                need(value['sha256'] == historical[name]['sha256'] and
                     value['bytes'] == historical[name]['bytes'], 'Archived historical pin mismatch: ' + name)
            if final_model:
                need(value['sha256'] == pointers['1000000']['sha256'], 'Final model SHA differs from original sidecar')
            manifest[name] = value
            archive.addfile(archive.gettarinfo(str(path), arcname=name), io.BytesIO(content))
            need(source_stat(path) == before[name], 'Source changed during archive: ' + name)
        # The manifest describes all preceding members; its own bytes are verified
        # by the external receipt SHA, avoiding a circular self-hash definition.
        manifest_content = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
        info = tarfile.TarInfo('_backup/evidence_manifest.json')
        info.size, info.mode = len(manifest_content), 0o644
        archive.addfile(info, io.BytesIO(manifest_content))
    need({name: source_stat(path) for name, path in paths.items()} == before,
         'Archived sources changed after archive')
    need(closure() == gates, 'Closed-run gate changed while packaging')
    _, after_pointers, _ = collect(gates)
    need(after_pointers == pointers, 'Selected checkpoint stat/sidecar pointers changed')
    # Read the complete gzip stream to verify EOF and trailer CRC/length.
    with gzip.open(target, 'rb') as stream:
        for _ in iter(lambda: stream.read(1024**2), b''):
            pass
    seen = set()
    expected_members = dict(manifest)
    expected_members['_backup/evidence_manifest.json'] = dict(
        bytes=len(manifest_content), sha256=sha(manifest_content))
    with tarfile.open(target, 'r:gz') as archive:
        for member in archive:
            need(member.isfile() and member.name in expected_members and member.name not in seen,
                 'Unexpected/duplicate/non-file tar member')
            content = archive.extractfile(member).read()
            item = expected_members[member.name]
            need(member.size == len(content) == item['bytes'] and sha(content) == item['sha256'],
                 'Tar member SHA/length mismatch: ' + member.name)
            seen.add(member.name)
    need(seen == set(expected_members), 'Tar member set mismatch')
    need({name: source_stat(path) for name, path in paths.items()} == before,
         'Archived source stats changed during verification')
    for pointer in pointers.values():
        need(source_stat(Path(pointer['path'])) == pointer['source_stat'],
             'Selected checkpoint stat changed during verification')
    write_new(out / 'evidence_manifest.json', manifest)
    archive_stat = source_stat(target)
    archive_sha = digest_file(target)
    need(source_stat(target) == archive_stat, 'Archive changed while computing its SHA')
    receipt = dict(actual_exit_code=0, completed=time.time(), archive=target.name,
        bytes=archive_stat['size'], sha256=archive_sha, members=len(expected_members),
        gzip_EOF_and_CRC_verified=True, all_member_SHA_and_length_verified=True,
        source_before_after_stat_exact=True, final_1M_SHA_matches_original_sidecar=True,
        checkpoints_copied=[1000000], checkpoints_pointer_only=[800000, 900000],
        checkpoint_800K_900K_content_read=False, raw_NPZ_read_or_rehashed=False,
        model_deserialized=False, reports=3, measured_episodes=750, video_episodes=15,
        task_return_events=15, restore_receipts=4,
        manifest_sha256=sha(manifest_content), manifest_member='_backup/evidence_manifest.json',
        manifest_excludes_only_its_own_member=True, nice_increment=10, affinity_CPUs_max=2,
        closure_checks_sha256=digest_file(out / 'closure_checks.json'),
        builder_process_exit_requires_external_capture=True, SCP_exit_requires_external_capture=True)
    write_new(out / 'evidence_receipt.json', receipt)
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()

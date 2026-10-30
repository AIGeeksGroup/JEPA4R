"""Explicit v1->v2 config-only migration; v1 actual bytes remain archived."""
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import zipfile

from resume_entry import adapt_main, restore_main

case = Path(__file__).resolve().parent
v1 = json.loads((case / 'freeze_v1.json').read_text())
old_sha = hashlib.sha256((case / 'freeze_v1.json').read_bytes()).hexdigest()
assert old_sha == '37e382ec7875f67c225ce65c4db9e8d73fcc698263e30993ca030a4b15e7071f'
for name, expected in v1['files'].items():
    if hashlib.sha256((case / 'failed_source_v1' / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Preserved v1 mismatch: {name}')
for name in ['resume_helpers.py', 'launch.py', 'strict_reload_export.py', 'test_cpu_recovery.py']:
    if hashlib.sha256((case / name).read_bytes()).hexdigest() != v1['files'][name]:
        raise ValueError(f'Unrelated runtime/math/test file changed: {name}')
before = (case / 'failed_source_v1/resume_entry.py').read_text()
after = (case / 'resume_entry.py').read_text()
expected = before.replace("ap.add_argument('--resume-model', type=Path", "ap.add_argument('--resume-model', type=str")
expected = expected.replace("ap.add_argument('--verified-export-root', type=Path", "ap.add_argument('--verified-export-root', type=str")
if after != expected:
    raise ValueError('V2 runtime change exceeded the two declared argparse types')
reference = case.parent / 'double_cube_lejepa_match_3cube_20261026_v1'
adapted = adapt_main((reference / 'source/train_lejepa_mw.py').read_text())
artifacts = {'adapted_native_main.py': adapted + '\n',
             'MAIN_ALLOWED_DIFF.patch': ''.join(difflib.unified_diff(
                 restore_main(adapted).splitlines(keepends=True), adapted.splitlines(keepends=True),
                 fromfile='frozen-native-main', tofile='weight-only-recovery-main-v2')) + '\n',
             'V2_CONFIG_FIX.patch': ''.join(difflib.unified_diff(
                 before.splitlines(keepends=True), after.splitlines(keepends=True),
                 fromfile='failed-v1-resume-entry', tofile='config-json-v2-resume-entry')) + '\n'}
for name, content in artifacts.items():
    (case / name).write_text(content)
migration = {'schema': 'explicit_config_serialization_v2_migration',
             'at_utc': datetime.now(timezone.utc).isoformat(), 'old_freeze_sha256': old_sha,
             'old_actual_files': 'failed_source_v1/; freeze_v1.json and plan_v1.json; source_bundle.zip remains v1',
             'failed_runtime': 'root_smoke_v1 native actual_exit_code1; config Path JSON failure before data/model',
             'runtime_delta': 'only two added argparse path values type=Path -> type=str',
             'unchanged_runtime_files': {n: v1['files'][n] for n in ['resume_helpers.py', 'launch.py', 'strict_reload_export.py']},
             'affected_verification': 'real native parser -> original native config JSON source-only regression',
             'reused_evidence': 'unchanged v1 CPU recovery/LR/save eight tests; no rerun or claimed GPU pass',
             'v1_failure_preserved': True, 'v2_gpu_smoke_required': 'Root exclusive root_smoke_v2',
             'storage_authority': 'user accepts expanded instance disk as long-term; external backup not required',
             'freeze_alias_migration': 'freeze.json will become exact v2 bytes; previous exact bytes stay freeze_v1.json'}
with (case / 'V2_MIGRATION.json').open('x') as stream:
    json.dump(migration, stream, indent=2)
    stream.write('\n')
names = list(v1['files']) + ['plan_v2.json', 'run_source_checks_v2.py', 'freeze_source_v2.py',
                            'V2_MIGRATION.json', 'V2_CONFIG_FIX.patch', 'freeze_v1.json', 'plan_v1.json']
files = {}
for name in names:
    data = (case / name).read_bytes()
    if name.endswith('.py'):
        ast.parse(data)
        compile(data, name, 'exec')
    files[name] = hashlib.sha256(data).hexdigest()
freeze = {'schema': 'weight_only_recovery_source_freeze_v2',
          'at_utc': datetime.now(timezone.utc).isoformat(),
          'stage': 'config_json_v2_source_frozen_affected_regression_pending', 'files': files,
          'supersedes_failed_v1_sha256': old_sha, 'migration_record': 'V2_MIGRATION.json',
          'runtime_delta': 'two added argparse path types only',
          'reference_freeze_sha256': v1['reference_freeze_sha256'],
          'exact_resume': False, 'formal_training_launched': False,
          'reused_cpu_checks': 'cpu_checks_frozen_v1; unchanged helper SHA',
          'v2_gpu_gate': 'pending Root root_smoke_v2; v1 real smoke failure preserved'}
data = json.dumps(freeze, indent=2) + '\n'
with (case / 'freeze_v2.json').open('x') as stream:
    stream.write(data)
(case / 'freeze.json').write_text(data)
with zipfile.ZipFile(case / 'source_bundle_v2.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in names + ['freeze_v2.json', 'freeze.json']:
        archive.write(case / name, name)
print(json.dumps({'files': len(files), 'freeze_v2_sha256': hashlib.sha256((case / 'freeze_v2.json').read_bytes()).hexdigest(),
                  'bundle_v2_sha256': hashlib.sha256((case / 'source_bundle_v2.zip').read_bytes()).hexdigest(),
                  'bundle_v2_bytes': (case / 'source_bundle_v2.zip').stat().st_size}))

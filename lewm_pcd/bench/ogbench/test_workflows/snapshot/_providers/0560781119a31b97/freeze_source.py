"""Freeze this isolated recovery source; archive only the declared new files."""
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import zipfile

from resume_entry import adapt_main, restore_main

case = Path(__file__).resolve().parent
reference = case.parent / 'double_cube_lejepa_match_3cube_20261026_v1'
source = (reference / 'source/train_lejepa_mw.py').read_text()
adapted = adapt_main(source)
for name, content in [('adapted_native_main.py', adapted + '\n'),
                      ('MAIN_ALLOWED_DIFF.patch', ''.join(difflib.unified_diff(
                          restore_main(adapted).splitlines(keepends=True),
                          adapted.splitlines(keepends=True), fromfile='frozen-native-main',
                          tofile='weight-only-recovery-main')) + '\n')]:
    with (case / name).open('x') as stream:
        stream.write(content)

names = ['resume_entry.py', 'resume_helpers.py', 'launch.py', 'plan.json', 'prepare_plan.py',
         'run_train.sh', 'strict_reload_export.py', 'test_source_contract.py', 'test_cpu_recovery.py',
         'run_cpu_checks.py', 'CPU_TEST_HISTORY.md', 'HANDOFF.md', 'freeze_source.py',
         'adapted_native_main.py', 'MAIN_ALLOWED_DIFF.patch']
files = {}
for name in names:
    data = (case / name).read_bytes()
    if name.endswith('.py'):
        ast.parse(data)
        compile(data, name, 'exec')
    files[name] = hashlib.sha256(data).hexdigest()
freeze = {'schema': 'weight_only_recovery_source_freeze_v1',
          'at_utc': datetime.now(timezone.utc).isoformat(),
          'stage': 'source_frozen_cpu_checked_root_gpu_and_launch_gates_pending',
          'files': files,
          'reference_freeze_sha256': hashlib.sha256((reference / 'freeze.json').read_bytes()).hexdigest(),
          'adapted_main_sha256_without_final_newline': hashlib.sha256(adapted.encode()).hexdigest(),
          'allowed_main_replacement_count': 7, 'reference_model_loss_augment_copied': False,
          'exact_resume': False, 'formal_training_launched': False,
          'environment_installed_or_modified': False}
with (case / 'freeze.json').open('x') as stream:
    json.dump(freeze, stream, indent=2)
    stream.write('\n')
with zipfile.ZipFile(case / 'source_bundle.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in names + ['freeze.json']:
        archive.write(case / name, name)
print(json.dumps({'files': len(files), 'freeze_sha256': hashlib.sha256((case / 'freeze.json').read_bytes()).hexdigest(),
                  'bundle_sha256': hashlib.sha256((case / 'source_bundle.zip').read_bytes()).hexdigest(),
                  'bundle_bytes': (case / 'source_bundle.zip').stat().st_size}))

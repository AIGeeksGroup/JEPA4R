"""Use the frozen Range download on Windows, retaining original official provenance."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ['LOCALAPPDATA']) / 'Temp/codex_ogbench_101732Test_20261017_v1'
DATA = ROOT / 'official_state'
DATA.mkdir(parents=True, exist_ok=True)
source = HERE / 'prepare_official_state.py'
namespace = {'__name__': 'frozen_download_functions', '__file__': str(source)}
exec(compile(source.read_text(encoding='utf-8').replace('import fcntl\n', ''), str(source), 'exec'), namespace)
namespace['ROOT'], namespace['DATA'] = ROOT, DATA
namespace['resources'] = lambda: {'disk_free': shutil.disk_usage(DATA).free}
started = time.time()
namespace['save'](ROOT / 'actual_local_argv.json', {
    'argv': [sys.executable] + sys.argv, 'started': started, 'pid': os.getpid(),
    'frozen_download_source': str(source),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'purpose': 'download official bytes locally before transport to target; no data transform'})
try:
    receipts = []
    for name, size in namespace['SPECS']:
        receipts.append(namespace['download'](name, size))
    namespace['save'](ROOT / 'local_download_exit.json', {
        'actual_exit_code': 0, 'completed': time.time(), 'receipts': receipts})
    namespace['status'](stage='local_raw_download_complete', actual_exit_code=0)
except BaseException as e:
    namespace['save'](ROOT / 'local_download_exit.json', {
        'actual_exit_code': 1, 'error': repr(e), 'traceback': traceback.format_exc(), 'completed': time.time()})
    raise

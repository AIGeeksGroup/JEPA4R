"""Resume the same verified Range parts with curl deadlines and disk-streamed responses."""
import email
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ['LOCALAPPDATA']) / 'Temp/codex_ogbench_101732Test_20261017_v1'
DATA = ROOT / 'official_state'
DATA.mkdir(parents=True, exist_ok=True)
TEMP = ROOT / 'curl_responses'
TEMP.mkdir(exist_ok=True)
source = HERE / 'prepare_official_state.py'
namespace = {'__name__': 'frozen_download_functions', '__file__': str(source)}
exec(compile(source.read_text(encoding='utf-8').replace('import fcntl\n', ''), str(source), 'exec'), namespace)
namespace['ROOT'], namespace['DATA'] = ROOT, DATA
namespace['resources'] = lambda: {'disk_free': shutil.disk_usage(DATA).free}

class CurlResponse:
    def __init__(self, request, timeout):
        token = uuid.uuid4().hex
        self.body, self.header = TEMP / (token + '.body'), TEMP / (token + '.headers')
        argv = ['curl.exe', '--silent', '--show-error', '--fail', '--location', '--max-time', str(timeout),
                '--dump-header', str(self.header), '--output', str(self.body), '--write-out', '%{url_effective}']
        if request.get_method() == 'HEAD':
            argv += ['--head']
        for key, value in request.header_items():
            argv += ['--header', key + ': ' + value]
        argv.append(request.full_url)
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout + 10)
        if result.returncode:
            self.body.unlink(missing_ok=True)
            self.header.unlink(missing_ok=True)
            raise RuntimeError('curl exit ' + str(result.returncode) + ': ' + result.stderr.decode(errors='replace'))
        self.url = result.stdout.decode().strip()
        blocks = self.header.read_text().replace('\r\n', '\n').strip().split('\n\n')
        block = [b for b in blocks if b.startswith('HTTP/')][-1]
        first, rest = block.split('\n', 1)
        self.status = int(first.split()[1])
        self.headers = email.message_from_string(rest)
        self.file = None
    def geturl(self):
        return self.url
    def __enter__(self):
        self.file = self.body.open('rb')
        return self
    def read(self, n):
        return self.file.read(n)
    def __exit__(self, *args):
        self.file.close()
        self.body.unlink()
        self.header.unlink()

namespace['urllib'].request.urlopen = lambda request, timeout: CurlResponse(request, timeout)
namespace['save'](ROOT / 'actual_local_v2_argv.json', {
    'argv': [sys.executable] + sys.argv, 'started': time.time(), 'pid': os.getpid(),
    'frozen_download_source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'http_transport': 'curl.exe with 120s whole-request deadline, file-backed response, same 8 Range workers',
    'purpose': 'resume only missing official parts; unchanged frozen integrity logic'})
try:
    receipts = [namespace['download'](name, size) for name, size in namespace['SPECS']]
    namespace['save'](ROOT / 'local_download_exit.json', {
        'actual_exit_code': 0, 'completed': time.time(), 'receipts': receipts,
        'actual_local_v2_argv': str(ROOT / 'actual_local_v2_argv.json')})
    namespace['status'](stage='local_raw_download_complete', actual_exit_code=0)
except BaseException as e:
    namespace['save'](ROOT / 'local_download_exit.json', {
        'actual_exit_code': 1, 'error': repr(e), 'traceback': traceback.format_exc(), 'completed': time.time()})
    raise

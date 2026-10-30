"""Reverify transported official bytes on target before publishing, then run frozen acceptance."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import prepare_official_state as frozen
SOURCE_ROOT = ROOT / 'transport_from_local'
started = time.time()
frozen.save(ROOT / 'actual_transport_finalize_argv.json', {
    'argv': [sys.executable] + sys.argv, 'started': started, 'pid': os.getpid(),
    'producer_sha256': frozen.sha(__file__), 'frozen_prepare_sha256': frozen.sha(frozen.__file__)})
for name, size in frozen.SPECS:
    receipt_path = SOURCE_ROOT / 'receipts' / (name + '.download.json')
    local_receipt = json.loads(receipt_path.read_text())
    incoming = frozen.DATA / (name + '.npz.from_local.incoming')
    target = frozen.DATA / (name + '.npz')
    assert incoming.stat().st_size == size == local_receipt['bytes']
    assert local_receipt['url'] == frozen.URL + name + '.npz'
    digest = frozen.sha(incoming)
    assert digest == local_receipt['sha256']
    with zipfile.ZipFile(incoming) as z:
        assert z.testzip() is None
        members = [{'name': x.filename, 'bytes': x.file_size, 'compressed_bytes': x.compress_size,
                    'crc32': x.CRC} for x in z.infolist()]
    assert not target.exists(), 'Refuse overwrite of published dataset'
    os.link(incoming, target)
    incoming.unlink()
    receipt = dict(local_receipt, path=str(target), zip_members=members,
        transport={'original_local_receipt': str(receipt_path), 'receipt_sha256': frozen.sha(receipt_path),
                   'original_local_path': local_receipt['path'], 'target_SHA_verified': True,
                   'target_ZIP_CRC_verified': True, 'target_verified': time.time()}, actual_exit_code=0)
    frozen.save(ROOT / 'receipts' / (name + '.download.json'), receipt)
frozen.main()
frozen.save(ROOT / 'transport_status.json', {
    'stage': 'complete', 'actual_exit_code': 0, 'updated': time.time(),
    'acceptance_path': str(ROOT / 'ACCEPTANCE.json'), 'acceptance_sha256': frozen.sha(ROOT / 'ACCEPTANCE.json')})

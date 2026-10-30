"""Adopt already verified Double NPZ; inspect headers and terminals only."""
from pathlib import Path
import hashlib
import json
import sys
import time
import zipfile

import numpy as np

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT / 'cube2_official_embeddings_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stat(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns,
                ctime_ns=s.st_ctime_ns, device=s.st_dev, inode=s.st_ino)


def write_new(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n')


def main():
    CASE.mkdir(exist_ok=True)
    assert not (CASE / 'data_acceptance.json').exists()
    assert not (CASE / 'stat-freeze.json').exists()
    before = time.time()
    datasets, frozen = [], {}
    download_root = ROOT / 'official_state_download_all_v1'
    outer_path = download_root / 'outer_exit.json'
    assert json.loads(outer_path.read_text())['actual_exit_code'] == 0
    for split, suffix, eps in [('train', '', 1000), ('validation', '-val', 100)]:
        name = f'cube-double-play-v0{suffix}.npz'
        path = ROOT / 'data/official_state' / name
        receipt_path = download_root / 'receipts' / f'{name}.json'
        receipt = json.loads(receipt_path.read_text())
        oldstat = stat(path)
        assert receipt['actual_exit_code'] == 0 and receipt['pinned_SHA_exact']
        assert receipt['archive_CRC_all_members'] and receipt['validated_before_atomic_publish']
        assert str(path) == receipt['path']
        assert (oldstat['size'], oldstat['mtime_ns'], oldstat['inode']) == (
            receipt['bytes'], receipt['mtime_ns'], receipt['inode'])
        rows = eps * 1001
        expected = {'observations': (rows, 37), 'actions': (rows, 5),
                    'terminals': (rows,), 'qpos': (rows, 28), 'qvel': (rows, 26)}
        fields = {}
        with zipfile.ZipFile(path) as z:
            assert set(z.namelist()) == {k + '.npy' for k in expected}
            for key, shape in expected.items():
                with z.open(key + '.npy') as f:
                    version = np.lib.format.read_magic(f)
                    got, fort, dtype = np.lib.format._read_array_header(f, version)
                assert got == shape and not fort
                assert dtype == np.dtype('bool' if key == 'terminals' else 'float32')
                fields[key] = dict(shape=shape, dtype=str(dtype))
            with z.open('terminals.npy') as f:
                terminals = np.lib.format.read_array(f, allow_pickle=False)
        ends = np.flatnonzero(terminals)
        assert np.array_equal(ends, np.arange(1000, rows, 1001))
        assert len(ends) == eps and bool(terminals[-1])
        assert stat(path) == oldstat
        schema = dict(actual_exit_code=0, path=str(path), rows=rows,
                      episodes=eps, episode_lengths=dict(min=1001, max=1001, unique=[1001]),
                      fields=fields, terminal_count=len(ends), raw_last_terminal=True,
                      terminals_exact_every_1001=True, raw_sha256=receipt['sha256'],
                      field_values_scope='only terminal values inspected; other raw values not read',
                      numpy_version=np.__version__)
        datasets.append(dict(split=split, download=receipt, schema_loader=schema,
                             original_download_receipt_path=str(receipt_path),
                             original_download_receipt_sha256=sha(receipt_path),
                             accepted_stat=oldstat, current_stat_exact=True))
        frozen[split] = dict(path=str(path), raw_sha256=receipt['sha256'],
                             accepted_stat=oldstat, download_receipt_path=str(receipt_path),
                             download_receipt_sha256=sha(receipt_path))
    acceptance = dict(actual_exit_code=0, started=before, completed=time.time(),
                      data_id='cube-double-play-v0', cube_count=2, datasets=datasets,
                      producer_sha256=sha(__file__), argv=sys.argv,
                      source_download_outer_exit_path=str(outer_path),
                      source_download_outer_exit_sha256=sha(outer_path),
                      scope='reuse accepted raw SHA/CRC; fresh headers/terminals/stat only',
                      raw_content_hash_repeated=False, qpos_qvel_values_read=False,
                      observations_values_read=False, GPU_or_encoder_executed=False)
    accpath = CASE / 'data_acceptance.json'
    write_new(accpath, acceptance)
    write_new(CASE / 'stat-freeze.json', dict(actual_exit_code=0, frozen=time.time(),
              acceptance_path=str(accpath), acceptance_sha256=sha(accpath), datasets=frozen,
              producer_sha256=sha(__file__)))
    print(json.dumps(dict(actual_exit_code=0, acceptance=str(accpath),
                         acceptance_sha256=sha(accpath), freeze=str(CASE / 'stat-freeze.json'),
                         freeze_sha256=sha(CASE / 'stat-freeze.json'))))


if __name__ == '__main__':
    main()

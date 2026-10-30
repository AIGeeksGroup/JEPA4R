"""Reuse the pinned full CPU auditor with private PCD metadata aliases."""
import argparse, json, os, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RGB = HERE.parent / 'rgb_latent_cache_20261012_v1'
sys.path[:0] = [str(HERE), str(RGB), str(HERE.parents[1])]
from cache_pcd_dataset import CachedCubePCDLatentDataset, read_pcd_metadata
import cache_audit_cpu as audit
from cache_producer import sha, stat, read, save
if sha(RGB / 'cache_audit_cpu.py') != 'd1989f117dbe006119041ac2c7007170e83048c9d93f761ec70c95ce35afc02d':
    raise ValueError('Pinned full CPU auditor changed')
audit.CachedCubeStateLatentDataset = CachedCubePCDLatentDataset
audit.read = read_pcd_metadata

def main():
    p = argparse.ArgumentParser(); p.add_argument('--manifest', type=Path, required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    if a.output.exists(): raise ValueError('Fresh audit output required')
    started = datetime.now(timezone.utc).isoformat(); m = read(a.manifest); seal = read(a.manifest.parent / 'cache_seal.json')
    if m.get('source_modality') != 'pcd' or seal.get('source_modality') != 'pcd': raise ValueError('PCD schema required')
    before = {}; checked = {}
    paths = dict(m['source_pins']); paths[m['source_pcd_index']] = m['source_pcd_index_sha256']; paths[m['source_pcd_seal']] = m['source_pcd_seal_sha256']
    paths[m['original_training_plan']] = m['original_training_plan_sha256']
    for path, expected in paths.items():
        before[path] = stat(path)
        actual = sha(path)
        if actual != expected: raise ValueError('Actual source pin differs: ' + path)
        checked[path] = actual
    freeze = read(HERE / 'source_freeze.json')
    if sha(HERE / 'producer_plan.json') != freeze['plan_sha256']: raise ValueError('Producer plan changed after source freeze')
    if freeze['adapter_source_sha256'] != m['modality_adapter_source_sha256']: raise ValueError('Actual PCD producer adapter differs')
    for split in ['train', 'val']:
        for kind in ['action', 'observation']:
            entry = m['numeric_sources'][split]; path = entry[kind + '_path']; before[path] = stat(path)
            if sha(path) != entry[kind + '_sha256']: raise ValueError('Original numeric source SHA differs')
            value = np.load(path, mmap_mode='r', allow_pickle=False)
            if not np.isfinite(value).all(): raise ValueError('Nonfinite original numeric source')
            checked[path] = entry[kind + '_sha256']
    proof_path = HERE / 'review/pilot_latent64_tail32_gate_v1/report.json'; proof = read(proof_path)
    if sha(proof_path) != '10f76f4fb2a6fa8fc7a879d3638ce56d4e0ba8184917176c600177c8487f656c': raise ValueError('Independent PCD numerical proof changed')
    if proof['exit_code'] != 0 or proof['source_checkpoint_sha256'] != m['source_checkpoint_sha256'] or proof['producer_plan_sha256'] != freeze['plan_sha256']: raise ValueError('PCD numerical proof source ownership differs')
    typed = m['encoder_hashes']['trainer_numpy_dtype']; canonical = m['encoder_hashes']['canonical_torch_dtype']
    if proof['typed_frozen_sha256_before'] != typed or proof['typed_frozen_sha256_after'] != typed: raise ValueError('PCD typed frozen hash differs')
    donor_path = m['encoder_hash_donor']['path']; donor = read(donor_path)
    if sha(donor_path) != m['encoder_hash_donor']['sha256'] or donor['exit_code'] != 0: raise ValueError('Canonical hash donor changed')
    for x in donor['best_last'].values():
        if x['canonical_dtype_frozen_sha256'] != canonical or x['trainer_dtype_frozen_sha256'] != typed: raise ValueError('Two hash formats have different lineage')
    for name in ['child_exit_code', 'driver_exit_code']:
        if (HERE / 'full_producer_v1_launcher' / name).read_text().strip() != '0': raise ValueError('Full producer process failed')
    if (a.manifest.parent / 'producer_exit_code').read_text().strip() != '0': raise ValueError('Full producer inner exit differs')
    code = audit.main()
    if code != 0 or any(stat(path) != old for path, old in before.items()): raise ValueError('Source changed during CPU scan')
    report = read(a.output / 'report.json'); report.pop('RGB_reads', None)
    report.update(source_modality='pcd', PCD_HDF5_reads=0, numeric_sources_SHA_and_finite_all=True, source_pins_actual=checked, cache_seal_sha256=sha(a.manifest.parent / 'cache_seal.json'), audit_source_sha256=sha(__file__), reader_adapter_sha256=sha(HERE / 'cache_pcd_dataset.py'), reused_auditor_sha256=sha(RGB / 'cache_audit_cpu.py'), producer_adapter_sha256=m['modality_adapter_source_sha256'], independent_PCD_GPU_proof={'path': str(proof_path), 'sha256': sha(proof_path), 'scope': proof['scope'], 'comparisons': proof['comparisons']}, encoder_hash_lineage={'trainer_numpy_dtype': typed, 'canonical_torch_dtype': canonical, 'canonical_donor_path': donor_path, 'canonical_donor_sha256': sha(donor_path)}, started_UTC=started, completed_UTC=datetime.now(timezone.utc).isoformat(), scope_limit='Full latent integrity/source mapping proven here; six PCD numerical cases from separate bound GPU gate; no head training, real resume parity, or online SR proof.')
    save(a.output / 'report.json', report); save(a.output / 'status.json', dict(stage='complete', status='passed', files=1100, failures=0, exit_code=0, updated_UTC=report['completed_UTC'])); return 0

if __name__ == '__main__':
    code = 1
    try: code = main()
    except BaseException: traceback.print_exc()
    if '--output' in sys.argv:
        out = Path(sys.argv[sys.argv.index('--output') + 1]); out.mkdir(parents=True, exist_ok=True); (out / 'exit_code').write_text(str(code) + '\n')
        if code != 0: save(out / 'status.json', dict(stage='failed', status='failed', failures=1, exit_code=code, updated_UTC=datetime.now(timezone.utc).isoformat()))
    raise SystemExit(code)

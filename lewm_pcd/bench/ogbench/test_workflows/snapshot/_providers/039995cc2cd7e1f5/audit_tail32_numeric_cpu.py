"""Check exact action/full-state ownership at every true validation tail pair."""
import json, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from cache_pcd_dataset import CachedCubePCDLatentDataset
from cache_producer import sha, stat, read, save
from dataset_state_fusion import CubeStateVisualDataset

def main():
    out = HERE / 'review/tail32_numeric_cpu_v1'; out.mkdir(exist_ok=False)
    manifest = HERE / 'features_v1/cache_manifest.json'; m = read(manifest); numeric = m['numeric_sources']['val']; tail = m['val_tail32']
    binding = HERE / 'review/full_cache_cpu_audit_v1/report.json'
    if sha(binding) != '1caedff64a52ea7f0588111978c50dc562d9a3d660ed93bd09acd3b1076d6d1b': raise ValueError('Bound full audit changed')
    paths = [manifest, Path(tail['path']), Path(numeric['action_path']), Path(numeric['observation_path'])]; before = {str(p): stat(p) for p in paths}
    ds = CachedCubePCDLatentDataset(manifest.parent, 'val', Path(numeric['action_path']).parents[1], m['source_pcd_index'], manifest.parent / 'cache_seal.json'); wrapped = CubeStateVisualDataset(ds, numeric['observation_path'])
    actions = np.load(numeric['action_path'], mmap_mode='r'); states = np.load(numeric['observation_path'], mmap_mode='r'); cases = []
    with np.load(tail['path'], allow_pickle=False) as z:
        for i, pair in enumerate(zip(z['current_compact'], z['goal_compact'])):
            current, goal = map(int, pair); raw = current // 1000 * 1001 + current % 1000; grow = goal // 1000 * 1001 + goal % 1000; row = wrapped[(current, goal)]
            if raw != tail['source_rows'][i] or grow != tail['goal_source_rows'][i] or int(row['source_row']) != raw or int(row['goal_source_row']) != grow: raise ValueError('Tail raw source mapping differs')
            if not np.array_equal(row['action'].numpy(), actions[raw]) or not np.array_equal(row['current_state'].numpy(), states[raw]) or not np.array_equal(row['goal_state'].numpy(), states[grow]): raise ValueError('Tail action/full-state ownership differs')
            if not np.array_equal(row['current'].numpy(), z['current_latents'][i]) or not np.array_equal(row['goal'].numpy(), z['goal_latents'][i]): raise ValueError('Tail B32 latent override differs')
            cases.append(dict(current=current, goal=goal, source_row=raw, goal_source_row=grow, action_current_state_goal_state_exact=True, tail_B32_latent_override_exact=True))
    if len(cases) != 32 or any(stat(p) != s for p, s in before.items()): raise ValueError('Tail source changed or pair count differs')
    save(out / 'report.json', dict(status='passed_tail32_numeric_and_latent_pairing_CPU_only', exit_code=0, cases=cases, tail32_numeric_pairs_checked=32, full_audit_numeric_sample_pairs=8, combined_numeric_pairs_checked=40, full_audit_path=str(binding), full_audit_sha256=sha(binding), manifest_sha256=sha(manifest), audit_source_sha256=sha(__file__), reader_source_sha256=sha(HERE / 'cache_pcd_dataset.py'), GPU_used=False, encoder_forwards=0, PCD_HDF5_reads=0, completed_UTC=datetime.now(timezone.utc).isoformat()))
    (out / 'exit_code').write_text('0\n'); return 0

if __name__ == '__main__':
    code = 1
    try: code = main()
    except BaseException: traceback.print_exc()
    if code != 0:
        out = HERE / 'review/tail32_numeric_cpu_v1'; out.mkdir(parents=True, exist_ok=True); (out / 'exit_code').write_text(str(code) + '\n')
    raise SystemExit(code)

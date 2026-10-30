"""One B32 CPU-vs-GPU original-preprocessor diagnostic, with no rendering/writes to cache."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

from gpu_preprocess import install


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('prepared_only_no_model_or_CUDA')
        return
    import numpy as np
    import torch
    plan = json.loads(args.plan.read_text())
    root = Path(plan['workspace'])
    sys.path.insert(0, str(args.plan.parent))
    from common import load_source, validate_plan
    validate_plan(plan, sha)
    encoder = load_source('_original_pcd_encoder_preproc_gate', root / 'checkpoint_encoder.py')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35)
    model = encoder.load_frozen(plan['checkpoint'], device='cuda')
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    samples = np.load(args.samples, allow_pickle=False)
    xyz, rgb = samples['pcd_xyz'], samples['pcd_rgb']
    assert xyz.dtype == np.float32 and rgb.dtype == np.uint8 and xyz.shape == rgb.shape
    indices = np.arange(32) % len(xyz)
    batch = dict(pcd_xyz=torch.from_numpy(xyz[indices].copy()), pcd_rgb=torch.from_numpy(rgb[indices].copy()))
    original = install(encoder)
    def baseline():
        return original(batch, '3d')[model.obs_key].to('cuda')
    def candidate():
        return encoder.preprocess(batch, '3d')[model.obs_key]
    before = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    expected, actual = baseline(), candidate()
    z_expected, z_actual = encoder.encode(model, expected), encoder.encode(model, actual)
    prepared_same = torch.equal(expected, actual)
    latent_same = torch.equal(z_expected, z_actual)
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    values = {'baseline': [], 'candidate': []}
    # Paired alternating timings exclude rendering; every timed branch includes H2D+prep+encode.
    for step in range(23):
        for key, prepare in [('baseline', baseline), ('candidate', candidate)]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            encoder.encode(model, prepare())
            torch.cuda.synchronize()
            if step >= 3:
                values[key].append(time.perf_counter()-started)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    base_med, gpu_med = (statistics.median(values[key]) for key in ('baseline', 'candidate'))
    report = dict(actual_exit_code=0, accepted=bool(prepared_same and latent_same and gpu_med <= base_med*.95),
        prepared_bitexact=prepared_same, latent_bitexact=latent_same,
        prepared_max_abs_diff=float((expected-actual).abs().max()), latent_max_abs_diff=float((z_expected-z_actual).abs().max()),
        baseline_median_seconds=base_med, candidate_median_seconds=gpu_med,
        timing_scope='H2D+original_preprocessor+encode_only_no_render_no_cache_writes',
        paired_alternating_trials=20, warmup_pairs=3, batch_size=32,
        source_samples_count=int(len(xyz)), real_raw_ids=samples['raw_ids'].tolist(),
        deterministic_tiling_to_B32=True, XYZ_input_dtype='float32', RGB_input_dtype='uint8',
        plan_path=str(args.plan), plan_sha256=sha(args.plan), checkpoint_sha256=plan['checkpoint_sha256'],
        samples_path=str(args.samples), samples_sha256=sha(args.samples),
        candidate_source_sha256=sha(Path(__file__).with_name('gpu_preprocess.py')),
        benchmark_source_sha256=sha(__file__), model_frozen_exact=True, training_updates=0, render_frames=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()

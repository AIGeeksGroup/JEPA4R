"""One real B32+all256 uint8 color domains; no model load, encode or rendering."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('prepared_only_no_model_no_CUDA')
        return
    import numpy as np
    import torch
    torch.set_num_threads(1)
    plan = json.loads(args.plan.read_text())
    root = Path(plan['workspace'])
    encoder = load('_original_encoder_prepared_only', root / 'checkpoint_encoder.py')
    assert sha(root/'checkpoint_encoder.py') == plan['source_pins']['checkpoint_encoder.py']
    encoder.source_paths()
    assert sha(root/'source/lewm_pcd/utils.py') == plan['source_pins']['source/lewm_pcd/utils.py']
    original = encoder.preprocess
    prior = Path(__file__).with_name('versions')/'gpu_preprocess_v2.py'
    assert sha(prior) == '7c278d64dc64fd70cc8ff2e9c08d003a99fcc4796a8296a4ee63c99453cd232d'
    load('_failed_rounding_v2', prior).install(encoder)
    old_gpu = encoder.preprocess
    candidate_file = Path(__file__).with_name('gpu_preprocess.py')
    load('_LUT_candidate_v3', candidate_file).install(encoder)
    new_gpu = encoder.preprocess
    samples = np.load(args.samples, allow_pickle=False)
    indices = np.arange(32) % len(samples['pcd_xyz'])
    batch = dict(pcd_xyz=torch.from_numpy(samples['pcd_xyz'][indices].copy()),
                 pcd_rgb=torch.from_numpy(samples['pcd_rgb'][indices].copy()))
    def check(data):
        base, old, new = (fn(data, '3d')['pcd'].cpu() for fn in (original, old_gpu, new_gpu))
        return dict(old_per_channel_max_abs_diff=(base-old).abs().reshape(-1,6).amax(0).tolist(),
            new_per_channel_max_abs_diff=(base-new).abs().reshape(-1,6).amax(0).tolist(),
            old_XYZ_bitexact=torch.equal(base[..., :3],old[..., :3]),
            old_RGB_bitexact=torch.equal(base[..., 3:],old[..., 3:]),
            new_XYZ_bitexact=torch.equal(base[..., :3],new[..., :3]),
            new_RGB_bitexact=torch.equal(base[..., 3:],new[..., 3:]))
    real = check(batch)
    domain = check(dict(pcd_xyz=torch.zeros((256,1,3),dtype=torch.float32),
        pcd_rgb=torch.arange(256,dtype=torch.uint8)[:,None,None].expand(-1,1,3)))
    accepted = all(data[key] for data in (real, domain)
                   for key in ('old_XYZ_bitexact','new_XYZ_bitexact','new_RGB_bitexact'))
    report = dict(actual_exit_code=0 if accepted else 1, candidate_prepared_bitexact=accepted,
        real_B32=real, all_256_uint8_colors=domain, samples_raw_ids=samples['raw_ids'].tolist(),
        deterministic_tiling_to_B32=True, samples_sha256=sha(args.samples),
        plan_sha256=sha(args.plan), original_checkpoint_encoder_sha256=sha(root/'checkpoint_encoder.py'),
        original_preprocessor_sha256=sha(root/'source/lewm_pcd/utils.py'),
        prior_failed_adapter_sha256=sha(prior), candidate_source_sha256=sha(candidate_file),
        diagnostic_source_sha256=sha(__file__), model_loads=0, encoder_calls=0, render_frames=0,
        XYZ_input_dtype='float32', RGB_input_dtype='uint8', LUT_source='original CPU preprocessor all256 uint8 values',
        training_updates=0, tolerance_relaxed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))
    assert accepted, 'XYZ differs or original CPU color LUT failed strict alignment'


if __name__ == '__main__':
    main()

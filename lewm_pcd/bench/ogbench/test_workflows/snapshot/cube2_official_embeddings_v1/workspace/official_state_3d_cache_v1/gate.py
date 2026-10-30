"""One small real PCD batch and independent-process strict reload; no updates or SR."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common import legacy, load_arrays, load_source, object_sha, validate_plan


def tensor_digest(state):
    value = hashlib.sha256()
    for key, tensor in state.items():
        array = tensor.detach().cpu().numpy()
        value.update(key.encode() + str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--stage', choices=('reference', 'fresh'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    old = legacy(plan['workspace'])
    sha, stat, save = old.writer_helpers(plan['workspace'])
    validate_plan(plan, sha)
    arrays_path, arrays = load_arrays(plan, old, sha, stat)
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / f'{args.stage}_report.json').exists(), 'Never overwrite gate evidence'
    fixture = args.output / 'real_PCD32.npz'
    if args.stage == 'reference':
        assert not fixture.exists() and not (args.output / 'reference_report.json').exists()
        from pcd_renderer import init_worker, render_rows, missing_state_witness
        init_worker(plan['workspace'], str(arrays_path), plan['camera'], plan['pcd'])
        ids = np.r_[np.arange(29, dtype=np.int64), 999, 1000, 1001]
        rendered = render_rows(('train', ids))
        witness = missing_state_witness(('train', 999))
        with fixture.open('xb') as stream:
            np.savez(stream, raw_ids=ids, pcd_xyz=rendered['pcd_xyz'], pcd_rgb=rendered['pcd_rgb'],
                     pixels=rendered['pixels'])
        save(args.output / 'missing_state_witness.json', witness)
        save(args.output / 'capture_witnesses.json', rendered['capture_witnesses'])
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35)
    encoder = load_source('_original_pcd_encoder_gate', Path(plan['workspace']) / 'checkpoint_encoder.py')
    started = time.time()
    model = encoder.load_frozen(plan['checkpoint'], device='cuda')
    saved = torch.load(plan['checkpoint'], map_location='cpu', weights_only=True)
    assert list(model.state_dict()) == list(saved)
    assert all(torch.equal(value.cpu(), saved[key]) for key, value in model.state_dict().items())
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters()) and not torch.is_autocast_enabled()
    frozen = tensor_digest(model.state_dict())
    data = np.load(fixture, allow_pickle=False)
    batch = {key: torch.from_numpy(data[key]) for key in ('pcd_xyz', 'pcd_rgb')}
    assert batch['pcd_xyz'].dtype == torch.float32 and batch['pcd_rgb'].dtype == torch.uint8
    start = time.perf_counter()
    prepared = encoder.preprocess(batch, '3d')[model.obs_key]
    assert prepared.shape == (32, 1200, 6) and prepared.dtype == torch.float32
    actual = encoder.encode(model, prepared.to('cuda')).cpu().numpy()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert tensor_digest(model.state_dict()) == frozen
    report = dict(stage=args.stage, actual_exit_code=0, elapsed_seconds=time.time()-started,
        checkpoint_sha256=plan['checkpoint_sha256'], fixture_sha256=sha(fixture),
        plan_sha256=sha(args.plan), protocol_sha256=object_sha(plan), strict_all_tensors_equal=True,
        frozen_model_state_sha256=frozen, encoder_projector_only=True, updates=0, SR_episodes=0,
        raw_ids=data['raw_ids'].tolist(), input_shape=list(prepared.shape), output_shape=list(actual.shape),
        dtype='float32', AMP=False, finite=bool(np.isfinite(actual).all()),
        batch_preprocess_transfer_encode_seconds=elapsed, rendering_in_timing=False,
        runtime=dict(torch=torch.__version__, CUDA=torch.version.cuda, device=torch.cuda.get_device_name(),
            matmul_precision=torch.get_float32_matmul_precision(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
    if args.stage == 'reference':
        with (args.output / 'reference_latent.npy').open('xb') as stream:
            np.save(stream, actual, allow_pickle=False)
        report['reference_latent_sha256'] = sha(args.output / 'reference_latent.npy')
    else:
        reference = np.load(args.output / 'reference_latent.npy', allow_pickle=False)
        assert np.array_equal(actual, reference), 'Fresh identical batch/model output differs; do not relax tolerance'
        original = json.loads((args.output / 'reference_report.json').read_text())
        assert sha(fixture) == original['fixture_sha256'] and sha(args.plan) == original['plan_sha256']
        assert sha(args.output / 'reference_latent.npy') == original['reference_latent_sha256']
        assert frozen == original['frozen_model_state_sha256']
        report.update(fresh_reload_bitexact=True, max_abs_difference=float(np.max(np.abs(actual-reference))))
    save(args.output / f'{args.stage}_report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

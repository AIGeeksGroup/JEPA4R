"""Bounded FP32 CPU parity of original versus wrapped LeWM, without updates.

Run from lewm_pcd with its existing world-model dependencies:
  python bench/ogbench/tests/verify_lewm_cpu.py --output NEW --pcd-points 32
Uses the accepted full model widths and RGB224. The PCD point count is explicit
smoke scope, independent of the production 1200-point dataset. This does not
test CUDA/BF16, production data identity, a complete training epoch or final SR.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import types

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True


def verify(output, pcd_points=32):
    import torch
    import hydra
    from omegaconf import OmegaConf
    from bench.ogbench import lewm
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    original_train = CODE / 'train.py'
    original_sha = hashlib.sha256(original_train.read_bytes()).hexdigest()
    if original_sha != '733564c824ce4023abdcdcf0a65ad72f5b71efe9d21d7958a466f87c7cdfe6e5':
        raise ValueError('original canonical LeWM source SHA differs')
    if pcd_points < 1:
        raise ValueError('pcd_points must be positive')
    results = {}
    with lewm.pinned_native() as native:
        original = lewm._load('lewm_cpu_original_forward', original_train)
        try:
            from stable_worldmodel.wm.utils import load_pretrained
            for mode in ('2d', '3d'):
                torch.manual_seed(0)
                reference_cfg = OmegaConf.load(lewm.NATIVE_DIRECTORY / ('accepted_' + mode + '_source.yaml'))
                cfg = OmegaConf.create(lewm.build_config('double500', mode, output / 'unused_data'))
                # Compare resolved accepted model/loss/preprocessor contracts before constructing models.
                for key in ('model', 'optimizer', 'loss'):
                    if OmegaConf.to_container(reference_cfg[key], resolve=True) != OmegaConf.to_container(cfg[key], resolve=True):
                        raise ValueError(mode + ' accepted ' + key + ' contract differs')
                if OmegaConf.to_container(reference_cfg.data.preprocessor, resolve=True) != OmegaConf.to_container(cfg.data.preprocessor, resolve=True):
                    raise ValueError('accepted preprocessing config differs')
                stats = torch.randn(64, 5)
                stats[-1] = float('nan')  # Original normalizer excludes incomplete boundary actions.
                stats_dataset = types.SimpleNamespace(get_col_data=lambda key: stats.numpy())
                reference_transform = original.spt.data.transforms.Compose(
                    hydra.utils.instantiate(reference_cfg.data.preprocessor),
                    original.get_column_normalizer(stats_dataset, 'action', 'action'))
                current_transform = native.spt.data.transforms.Compose(
                    hydra.utils.instantiate(cfg.data.preprocessor),
                    native.get_column_normalizer(stats_dataset, 'action', 'action'))
                raw = []
                for _ in range(2):
                    sample = {'action': torch.randn(20, 5)}
                    if mode == '2d':
                        sample['pixels'] = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.uint8)
                    else:
                        sample['pcd_xyz'] = torch.randn(4, pcd_points, 3)
                        sample['pcd_rgb'] = torch.randint(0, 256, (4, pcd_points, 3), dtype=torch.uint8)
                    raw.append(sample)
                def batch(transform):
                    samples = [transform({k: v.clone() for k, v in row.items()}) for row in raw]
                    for row in samples:
                        row['action'] = row['action'].reshape(4, 25)
                    return torch.utils.data.default_collate(samples)
                reference_batch, current_batch = batch(reference_transform), batch(current_transform)
                def equal(a, b):
                    if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all():
                        raise ValueError('shape/dtype/finite contract differs')
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                for key in reference_batch:
                    equal(reference_batch[key], current_batch[key])
                reference_model = hydra.utils.instantiate(reference_cfg.model).cpu().eval()
                current_model = hydra.utils.instantiate(cfg.model).cpu().eval()
                current_model.load_state_dict(reference_model.state_dict(), strict=True)
                def clone_batch(value):
                    return {k: v.clone() for k, v in value.items()}
                def encode_predict(model, value):
                    encoded = model.encode(clone_batch(value))
                    predicted = model.predict(encoded['emb'][:, :3], encoded['act_emb'][:, :3])
                    return encoded['emb'], encoded['act_emb'], predicted
                with torch.no_grad():
                    old_encoded = encode_predict(reference_model, reference_batch)
                    new_encoded = encode_predict(current_model, current_batch)
                for a, b in zip(old_encoded, new_encoded):
                    equal(a, b)
                reference_model.train()
                current_model.train()
                initial_parameters = {k: v.detach().clone() for k, v in current_model.named_parameters()}
                def forward(entry, model, value, active_cfg):
                    owner = types.SimpleNamespace(model=model, sigreg=entry.SIGReg(**active_cfg.loss.sigreg.kwargs),
                                                  log_dict=lambda *args, **kwargs: None)
                    torch.manual_seed(314159)  # Identical dropout and SIGReg random projections.
                    return entry.lejepa_forward(owner, clone_batch(value), 'train', active_cfg)
                old_result = forward(original, reference_model, reference_batch, reference_cfg)
                new_result = forward(native, current_model, current_batch, cfg)
                for key in ('emb', 'act_emb', 'pred_loss', 'sigreg_loss', 'loss'):
                    equal(old_result[key], new_result[key])
                old_result['loss'].backward()
                new_result['loss'].backward()
                old_parameters = dict(reference_model.named_parameters())
                gradients_checked = 0
                for key, parameter in current_model.named_parameters():
                    left = old_parameters[key]
                    if (left.grad is None) != (parameter.grad is None):
                        raise ValueError('gradient ownership differs: ' + key)
                    if parameter.grad is not None:
                        equal(left.grad, parameter.grad)
                        gradients_checked += 1
                    equal(initial_parameters[key], parameter.detach())
                mode_root = output / mode
                mode_root.mkdir()
                weights = mode_root / 'weights.pt'
                torch.save(current_model.state_dict(), weights)
                model_config = OmegaConf.to_container(cfg.model, resolve=True)
                (mode_root / 'config.json').write_text(json.dumps(model_config, indent=2) + '\n', encoding='utf8')
                restored = load_pretrained(str(weights.resolve()), cache_dir=str(output / 'factory_cache')).cpu().eval()
                current_model.eval()
                for key, value in current_model.state_dict().items():
                    equal(value, restored.state_dict()[key])
                with torch.no_grad():
                    before_reload = encode_predict(current_model, current_batch)
                    after_reload = encode_predict(restored, current_batch)
                    current_loss = forward(native, current_model, current_batch, cfg)
                    restored_loss = forward(native, restored, current_batch, cfg)
                for a, b in zip(before_reload, after_reload):
                    equal(a, b)
                for key in ('pred_loss', 'sigreg_loss', 'loss'):
                    equal(current_loss[key], restored_loss[key])
                results[mode] = dict(preprocess_exact=True, encode_exact=True, predict_exact=True,
                    loss_exact=True, gradients_exact=True, gradient_tensors_checked=gradients_checked,
                    strict_original_factory_reload_exact=True, parameter_updates=0,
                    obs_shape=list(current_batch[cfg.model.obs_key].shape),
                    action_shape=list(current_batch['action'].shape), latent_shape=list(new_encoded[0].shape),
                    weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                    loss=float(new_result['loss'].detach()), rtol=0, atol=0)
                del reference_model, current_model, restored, old_result, new_result, initial_parameters, old_parameters
        finally:
            sys.modules.pop('lewm_cpu_original_forward', None)
    return dict(device='cpu', dtype='float32', threads=1, torch_version=torch.__version__,
                original_train_sha256=original_sha, native_train_sha256=lewm.NATIVE_SHA256,
                modes=results, optimizer_updates=0, complete_training_epochs=0,
                pcd_smoke_points=pcd_points, production_pcd_points=1200,
                cuda_bf16_verified=False, production_data_verified=False, formal_acceptance_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pcd-points', type=int, default=32)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    state = dict(stage='lewm_cpu_parity_running', started_at=started, updated_at=started,
                 argv=sys.argv[:], pcd_points=args.pcd_points, actual_exit_code=None)
    def save():
        (output / 'status.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf8')
    save()
    with (output / 'stdout_stderr.log').open('x', encoding='utf8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                result = verify(output, args.pcd_points)
                (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
                state.update(stage='lewm_cpu_parity_verified', actual_exit_code=0)
            except BaseException:
                state.update(stage='lewm_cpu_parity_failed', actual_exit_code=1, error=traceback.format_exc())
                traceback.print_exc()
    state['finished_at'] = state['updated_at'] = datetime.now(timezone.utc).isoformat()
    save()
    (output / 'actual_exit_code').write_text(str(state['actual_exit_code']) + '\n', encoding='utf8')
    return state['actual_exit_code']


if __name__ == '__main__':
    raise SystemExit(main())

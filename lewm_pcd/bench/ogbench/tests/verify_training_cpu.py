"""Bounded CPU parity against the historical Quad PLDM source, with no update.

From lewm_pcd, run with CUDA_VISIBLE_DEVICES='' and PYTHONPATH=.: 
python bench/ogbench/tests/verify_training_cpu.py --reference-root OLD/source --output NEW
The output must be new. No training loop, optimizer step or CUDA SIGReg runs.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True  # Historical reference files remain untouched.


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(reference_root, output):
    import torch
    from bench.ogbench.pldm import method, shared_data
    from bench.ogbench import tiny
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    root = Path(shared_data.__file__).resolve().parent
    expected = '0400e518a220d7843f62b70427dd645300a46f3ddb951ce58e15ab52625b7712'
    original_method = reference_root / 'methods/pldm/method.py'
    for path in (original_method, root / 'method.py'):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('reference/common method SHA differs')
    manifest = json.loads((root / 'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if name.startswith('vendor/source/'):
            original = reference_root / 'vendor/pldm/source' / name[len('vendor/source/'):]
            if hashlib.sha256(original.read_bytes()).hexdigest() != digest:
                raise ValueError('author source parity differs: ' + name)
    old_method = load_file('parity_original_method', original_method)
    old_shared = load_file('parity_original_data', reference_root / 'shared_data.py')
    pixels = torch.randint(0, 256, (2, 16, 3, 224, 224), dtype=torch.uint8)
    actions = torch.randn(2, 16, 5)
    stats = dict(mean=actions.flatten(0, 1).mean(0).tolist(),
                 std=actions.flatten(0, 1).std(0, correction=1).tolist())

    class MemoryReader:
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return dict(pixels=pixels[index].clone(), action=actions[index].clone())

    original_ds = old_shared.VisualDataset(None, 'pldm', reader=MemoryReader(), action_stats=stats)
    current_ds = shared_data.VisualDataset(MemoryReader(), stats)

    def batch(dataset):
        samples = [dataset[i] for i in range(2)]
        value = {key: torch.stack([item[key] for item in samples]) for key in ('pixels', 'actions')}
        return dataset.gpu_transform(value)

    original_batch = batch(original_ds)
    current_batch = batch(current_ds)
    for key in ('pixels', 'actions'):
        torch.testing.assert_close(original_batch[key], current_batch[key], rtol=0, atol=0)
    official_source = root / 'vendor/source'
    original_model = old_method.build_model(official_source, pretrained=False)
    current_model = method.build_model(official_source, pretrained=False)
    current_model.load_state_dict(original_model.state_dict(), strict=True)
    original_out = original_model(original_batch['pixels'], original_batch['actions'])
    current_out = current_model(current_batch['pixels'], current_batch['actions'])
    for key in original_out:
        torch.testing.assert_close(original_out[key], current_out[key], rtol=0, atol=0)
    original_out['loss'].backward()
    current_out['loss'].backward()
    for (left_name, left), (right_name, right) in zip(original_model.named_parameters(), current_model.named_parameters()):
        if left_name != right_name or (left.grad is None) != (right.grad is None):
            raise ValueError('parameter/gradient ownership differs')
        if left.grad is not None:
            if not torch.isfinite(left.grad).all():
                raise ValueError('nonfinite reference gradient')
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    pldm_path = output / 'pldm_cpu_state.pt'
    torch.save(current_model.state_dict(), pldm_path)
    restored = method.build_model(official_source, pretrained=False)
    restored.load_state_dict(torch.load(pldm_path, map_location='cpu', weights_only=True), strict=True)
    for key, tensor in current_model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)
    with torch.no_grad():
        restored_out = restored(current_batch['pixels'], current_batch['actions'])
    for key in current_out:
        torch.testing.assert_close(current_out[key], restored_out[key], rtol=0, atol=0)
    del original_model, current_model, restored, original_out, current_out, restored_out

    native = tiny._load_canonical()
    tiny_model = native.Encoder(proj_dim=16).eval()
    image = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        embedding = tiny_model.embed(image)
    if tuple(embedding.shape) != (2, 512) or not torch.isfinite(embedding).all():
        raise ValueError('Tiny canonical head512 embedding differs')
    tiny_path = output / 'tiny_cpu_state.pt'
    torch.save(dict(backbone=tiny_model.backbone.state_dict(), head=tiny_model.head.state_dict(),
                    proj=tiny_model.proj.state_dict()), tiny_path)
    saved = torch.load(tiny_path, map_location='cpu', weights_only=True)
    restored_tiny = native.Encoder(proj_dim=16).eval()
    for key in ('backbone', 'head', 'proj'):
        getattr(restored_tiny, key).load_state_dict(saved[key], strict=True)
    with torch.no_grad():
        restored_embedding = restored_tiny.embed(image)
    torch.testing.assert_close(embedding, restored_embedding, rtol=0, atol=0)
    return dict(torch_version=torch.__version__, device='cpu', threads=1,
                pldm_preprocess_exact=True, pldm_forward_loss_exact=True,
                pldm_gradients_exact=True, pldm_strict_reload_exact=True,
                tiny_embedding_shape=[2, 512], tiny_strict_reload_exact=True,
                optimizer_updates=0, cuda_sigreg_calls=0, reference_root=str(reference_root),
                checkpoint_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (pldm_path, tiny_path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.reference_root = args.reference_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(stage='cpu_parity_running', started_at=datetime.now(timezone.utc).isoformat(),
                 argv=sys.argv[:], actual_exit_code=None)
    (args.output / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
    with (args.output / 'stdout.log').open('x', encoding='utf8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                result = verify(args.reference_root, args.output)
                (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                state.update(stage='cpu_parity_verified', actual_exit_code=0)
            except Exception:
                state.update(stage='cpu_parity_failed', actual_exit_code=1, error=traceback.format_exc())
                traceback.print_exc()
    state['finished_at'] = datetime.now(timezone.utc).isoformat()
    (args.output / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
    (args.output / 'actual_exit_code').write_text(str(state['actual_exit_code']) + '\n')
    return state['actual_exit_code']


if __name__ == '__main__':
    raise SystemExit(main())

"""Eight real paired rows, frozen checkpoint loading and bounded GPU encoding."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from checkpoint_encoder import load_frozen, preprocess, encode

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.20)
    raw_path = ROOT/'incoming/smoke_paired_rows_v1.npz'
    receipt = json.loads((ROOT/'incoming/smoke_input_receipt.json').read_text())
    assert digest(raw_path) == receipt['sha256']
    raw = dict(np.load(raw_path, allow_pickle=False))
    migration = json.loads((ROOT/'migration_manifest.json').read_text())
    result = dict(actual_exit_code=None, device=torch.cuda.get_device_name(), torch=torch.__version__,
        source_input_receipt=receipt, encoder_batches={}, training_updates=0,
        world_model_dynamics_calls=0, SR_episodes=0)
    features = dict(observations=raw['observation'], actions=raw['action'], source_rows=raw['source_rows'])
    for role, directory, filename in [('2d','2d_epoch10','weights_epoch_10.pt'),
                                      ('3d','3d_epoch6','weights_epoch_6.pt')]:
        checkpoint = ROOT/'checkpoints'/directory/filename
        expected = migration['files'][checkpoint.relative_to(ROOT).as_posix()]
        assert digest(checkpoint) == expected['sha256']
        model = load_frozen(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert model.state_dict().keys() == saved.keys()
        assert all(torch.equal(value.cpu(), saved[key]) for key, value in model.state_dict().items())
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        if role == '2d':
            batch = {'pixels': torch.from_numpy(raw['pixels'])}
        else:
            batch = {k: torch.from_numpy(raw[k]) for k in ['pcd_xyz','pcd_rgb']}
        prepared = preprocess(batch, role)[model.obs_key]
        chunks = []
        for start in range(0, len(prepared), 2):
            x = prepared[start:start+2].to('cuda')
            z = encode(model, x)
            chunks.append(z.cpu().numpy())
        features['features_' + role] = np.concatenate(chunks)
        assert all(torch.equal(value.cpu(), saved[key]) for key, value in model.state_dict().items())
        result['encoder_batches'][role] = dict(checkpoint=str(checkpoint), sha256=expected['sha256'],
            rows=len(prepared), microbatch=2, feature_shape=list(features['features_'+role].shape),
            full_strict_factory=True, full_state_unchanged=True, encoder_projector_only=True,
            accepted_final_checkpoint=(role=='2d'))
        del model, saved, chunks, prepared, batch, x, z
        torch.cuda.empty_cache()
    np.savez(ROOT/'smoke/features.npz', **features)
    result.update(actual_exit_code=0, completed=time.time(),
                  peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                  peak_cuda_reserved=torch.cuda.max_memory_reserved())
    (ROOT/'smoke/encoders_report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

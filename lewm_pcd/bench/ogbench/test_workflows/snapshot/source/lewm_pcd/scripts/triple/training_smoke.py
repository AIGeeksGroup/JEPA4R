"""One isolated real-model optimizer step, then strict reload in a new process.

This is a smoke receipt, never a formal model or completed training result.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_freeze(path, package_root):
    frozen = json.loads(Path(path).read_text(encoding='utf-8'))
    for name, expected in frozen['files'].items():
        source = Path(name)
        if not source.is_absolute():
            source = package_root / source
        if sha(source) != expected:
            raise ValueError(f'Frozen runtime source differs: {source}')
    return sha(path)


def forward(entry, model, sigreg, batch, cfg, stage):
    wrapper = type('ForwardOnly', (), dict(model=model, sigreg=sigreg, log_dict=lambda *a, **k: None))()
    return entry.lejepa_forward(wrapper, {key: value.clone() for key, value in batch.items()}, stage, cfg)


def run(args, record):
    import hydra
    import hdf5plugin  # noqa: F401: register codecs before any compressed HDF5 read
    import torch
    from omegaconf import OmegaConf
    import train as entry
    import training_hdf5  # noqa: F401 registers only the explicit private reader
    from triple_training_contract import validate_contract

    cfg = OmegaConf.load(args.profile)
    if args.views is not None:
        modality = '2d' if cfg.model.obs_key == 'pixels' else '3d'
        cfg.data.dataset.name = str(args.views / f'{modality}_train_action_rows1000.h5')
        cfg.data.validation_dataset.name = str(args.views / f'{modality}_validation_action_rows1000.h5')
    validate_contract(cfg.triple_contract)
    entry.pl.seed_everything(int(cfg.seed), workers=True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('An available CUDA device is required for CUDA smoke')
    record.update(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), profile_sha256=sha(args.profile),
                  source_origins=dict(train=entry.__file__, stable_worldmodel=entry.swm.__file__),
                  fixture=bool(args.allow_fixture), formal_initialization='fresh seed only; never smoke weights')
    readers = []
    try:
        for key in ('dataset', 'validation_dataset'):
            options = OmegaConf.to_container(cfg.data[key], resolve=True)
            name = options.pop('name')
            options['allow_fixture'] = bool(args.allow_fixture)
            readers.append(entry.swm.data.load_dataset(name, transform=None, **options))
        training, validation = readers
        if validation.clip_indices[980] != (0, 980):
            raise ValueError('Triple boundary clip980 must remain in episode0')
        transforms = [hydra.utils.instantiate(cfg.data.preprocessor),
                      entry.get_column_normalizer(training, 'action', 'action')]
        validation.transform = entry.spt.data.transforms.Compose(*transforms)
        batch = torch.utils.data.default_collate([validation[0], validation[980]])
    finally:
        for reader in readers:
            reader.close()
    if set(batch) != {cfg.model.obs_key, 'action'} or batch['action'].shape != (2, 4, 25):
        raise ValueError('Smoke must use only original visual/action inputs and action chunks25')
    if not all(torch.isfinite(value).all() for value in batch.values()):
        raise ValueError('Nonfinite real normalized model input')
    cpu_batch = {key: value.cpu() for key, value in batch.items()}
    batch = {key: value.to(args.device) for key, value in batch.items()}
    model = hydra.utils.instantiate(cfg.model).to(args.device)
    sigreg = entry.SIGReg(**cfg.loss.sigreg.kwargs).to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    with torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
        output = forward(entry, model, sigreg, batch, cfg, 'train')
    output['loss'].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    encoder = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    if (not torch.isfinite(output['loss']) or not gradients or
            not all(torch.isfinite(value).all() for value in gradients) or
            not any(torch.count_nonzero(value) > 0 for value in encoder)):
        raise ValueError('Real loss/gradient/nonzero-encoder gate failed')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clip_val)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise ValueError('Nonfinite or zero gradient norm')
    probe = next(p for p in model.encoder.parameters()
                 if p.grad is not None and torch.count_nonzero(p.grad) > 0)
    before = probe.detach().clone()
    optimizer.step()
    if torch.equal(before, probe.detach()):
        raise ValueError('Actual optimizer step did not change the nonzero-gradient encoder parameter')
    record.update(optimizer_steps=1, loss=float(output['loss']), gradient_norm=float(norm),
                  real_encoder_backward=True, encoder_parameter_changed=True,
                  model_input_shapes={k: list(v.shape) for k, v in batch.items()})
    model.eval()
    with torch.no_grad():
        torch.manual_seed(int(cfg.seed))
        reference = forward(entry, model, sigreg, batch, cfg, 'val')
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    (args.output / 'config.json').write_text(json.dumps(model_config, indent=2), encoding='utf-8')
    OmegaConf.save(cfg, args.output / 'config.yaml')
    torch.save(model.state_dict(), args.output / 'smoke_weights.pt')
    torch.save(cpu_batch, args.output / 'smoke_batch.pt')
    torch.save(reference['emb'].cpu(), args.output / 'reference_emb.pt')
    record['smoke_checkpoint_sha256'] = sha(args.output / 'smoke_weights.pt')
    del model, optimizer, batch, output, reference, sigreg, gradients, encoder, probe, before
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    command = [sys.executable, str(Path(__file__).resolve()), '--stage', 'reload', '--device', args.device,
               '--profile', str(args.profile), '--output', str(args.output),
               '--source-freeze', str(args.source_freeze), '--package-root', str(args.package_root)]
    if args.allow_fixture:
        command.append('--allow-fixture')
    (args.output / 'reload_argv.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    with (args.output / 'reload.log').open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    (args.output / 'reload.exit_code').write_text(str(result.returncode) + '\n')
    if result.returncode != 0:
        raise ValueError('Fresh-process strict reload failed; see reload.log')
    record['fresh_process_reload'] = json.loads((args.output / 'reload.json').read_text())


def reload(args):
    import hydra
    import torch
    from omegaconf import OmegaConf
    import train as entry
    cfg = OmegaConf.load(args.output / 'config.yaml')
    model = hydra.utils.instantiate(json.loads((args.output / 'config.json').read_text())).to(args.device)
    state = torch.load(args.output / 'smoke_weights.pt', map_location='cpu', weights_only=True)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Nonfinite smoke checkpoint')
    model.load_state_dict(state, strict=True)
    model.eval()
    batch = {key: value.to(args.device) for key, value in
             torch.load(args.output / 'smoke_batch.pt', weights_only=True).items()}
    sigreg = entry.SIGReg(**cfg.loss.sigreg.kwargs).to(args.device)
    with torch.no_grad():
        torch.manual_seed(int(cfg.seed))
        output = forward(entry, model, sigreg, batch, cfg, 'val')
    expected = torch.load(args.output / 'reference_emb.pt', weights_only=True).to(args.device)
    if (not torch.isfinite(output['loss']) or not torch.isfinite(output['emb']).all() or
            not torch.allclose(expected, output['emb'], rtol=1e-5, atol=1e-5)):
        raise ValueError('Fresh-process strict reload or real-input feature parity failed')
    (args.output / 'reload.json').write_text(json.dumps(dict(exit_code=0, strict=True,
        separate_process=True, finite_real_forward=True, embedding_parity=True,
        loss=float(output['loss']), tensors=len(state)), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('run', 'reload'), default='run')
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--source-freeze', type=Path, required=True)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--allow-fixture', action='store_true')
    parser.add_argument('--views', type=Path, help='smoke-only view leaf; formal profiles remain unchanged')
    args = parser.parse_args()
    if args.stage == 'reload':
        check_freeze(args.source_freeze, args.package_root)
        reload(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(stage='running', scope='one-step-smoke-only', optimizer_steps=0,
                  started_at=datetime.now(timezone.utc).isoformat(), argv=sys.argv)
    try:
        record['source_freeze_sha256'] = check_freeze(args.source_freeze, args.package_root)
        run(args, record)
        record.update(stage='passed', exit_code=0)
    except Exception:
        record.update(stage='failed', exit_code=1, error=traceback.format_exc())
    record['finished_at'] = datetime.now(timezone.utc).isoformat()
    (args.output / 'status.json').write_text(json.dumps(record, indent=2) + '\n')
    (args.output / 'exit_code').write_text(str(record['exit_code']) + '\n')
    print(json.dumps(record, indent=2))
    return record['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())

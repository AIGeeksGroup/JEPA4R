"""RH56 data-boundary hooks around the unchanged CortexBench LeWM entrypoint.

Task data boundaries, run budgets and checkpoint evidence are adapted here.
The original reader, transforms, model, forward/loss, optimizer and scheduler
are delegated without editing baseline source.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import time
import traceback

import numpy as np

from jepa4r.envs.twins.rh56._runtime import (RUNTIME_HELPERS_SHA256, atomic_json, file_sha256 as sha,
                                   temporary_bindings, validate_output_location)


class EpisodePartition:
    def __init__(self, dataset, manifest):
        from jepa4r.envs.twins.rh56.input_identity import validate_episode_producers
        self.producer_binding = validate_episode_producers(dataset, manifest)
        records = manifest['episodes']
        n = len(dataset.lengths)
        if len(records) != n or sorted(r['episode_index'] for r in records) != list(range(n)):
            raise ValueError('manifest episode_index must cover exactly this HDF5 pool')
        records = sorted(records, key=lambda r: r['episode_index'])
        seeds = [int(r['seed']) for r in records]
        hashes = [r['initial_state_sha256'] for r in records]
        if len(set(seeds)) != n:
            raise ValueError('duplicate episode seed in manifest')
        if len(set(hashes)) != n or any(len(h) != 64 for h in hashes):
            raise ValueError('duplicate or invalid physical initial state hash')
        if any(r['split'] not in ('train', 'val', 'test') for r in records):
            raise ValueError('unsupported episode split')
        actual_seeds = np.asarray(dataset.get_col_data('episode_seed')).reshape(-1)
        for ep, (length, offset) in enumerate(zip(dataset.lengths, dataset.offsets)):
            values = actual_seeds[int(offset):int(offset) + int(length)]
            if len(values) != int(length) or not np.all(values == seeds[ep]):
                raise ValueError(f'episode_seed mismatch in HDF5 episode {ep}')
        self.episode_ids = {s: [r['episode_index'] for r in records if r['split'] == s]
                            for s in ('train', 'val', 'test')}
        episode_sets = {s: set(ids) for s, ids in self.episode_ids.items()}
        self.clip_indices = {s: [i for i, (ep, _) in enumerate(dataset.clip_indices)
                                 if ep in episode_sets[s]] for s in self.episode_ids}
        for s in ('train', 'val'):
            if not self.clip_indices[s]:
                raise ValueError(f'{s} split has no eligible clip')
        self.rows = {s: [(int(dataset.offsets[ep]), int(dataset.offsets[ep] + dataset.lengths[ep]))
                         for ep in ids] for s, ids in self.episode_ids.items()}
        self.seeds = {s: [seeds[ep] for ep in ids] for s, ids in self.episode_ids.items()}

    def receipt(self):
        return dict(episode_ids=self.episode_ids, seeds=self.seeds,
                    clips={k: len(v) for k, v in self.clip_indices.items()},
                    rows={k: sum(hi - lo for lo, hi in v) for k, v in self.rows.items()},
                    producer_binding=self.producer_binding,
                    policy='episode-disjoint; test excluded; train-only normalization')


class TrainColumnView:
    """Feed only train rows to the original, unchanged normalizer formula."""
    def __init__(self, dataset, partition):
        self.dataset, self.partition = dataset, partition
        self._cache = {}

    def get_col_data(self, col):
        if col not in self._cache:
            data = np.asarray(self.dataset.get_col_data(col))
            self._cache[col] = np.concatenate([data[lo:hi] for lo, hi in self.partition.rows['train']])
        return self._cache[col]


@contextmanager
def install_episode_split(train_module, dataset_path, manifest, subset_factory=None, generator_state=None,
                          gpu_image_preprocess=False):
    """Process-local, path-scoped hooks. Always restore both original bindings."""
    expected = Path(dataset_path).resolve()
    normalizer = train_module.get_column_normalizer
    splitter = train_module.spt.data.random_split
    audit = type('SplitAudit', (), {})()
    audit.partition = None
    audit.normalizers = {}
    audit.train_loader_generator_restored = False
    audit.parent_hdf5_handle_closed = False
    audit.gpu_image_preprocessor = None
    bound = None

    def selected(dataset):
        path = getattr(dataset, 'h5_path', None)
        return path is not None and Path(path).resolve() == expected

    def bind(dataset):
        nonlocal bound
        if bound is None:
            bound = dataset
            audit.partition = EpisodePartition(dataset, manifest)
        if bound is not dataset:
            raise ValueError('multiple selected dataset instances in one training run')

    def fit(dataset, source, target):
        if not selected(dataset):
            return normalizer(dataset, source, target)
        bind(dataset)
        view = TrainColumnView(dataset, audit.partition)
        values = view.get_col_data(source)
        audit.normalizers[source] = dict(
            fit_episode_ids=audit.partition.episode_ids['train'], rows=len(values),
            finite_rows=int(np.isfinite(values).all(axis=1).sum()),
            input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
            formula='delegated original get_column_normalizer; excludes NaN rows; sample std')
        return normalizer(view, source, target)

    def split(dataset, lengths, generator=None):
        if not selected(dataset):
            return splitter(dataset, lengths, generator=generator)
        bind(dataset)
        if generator_state is not None:
            if generator is None:
                raise ValueError('resume requires the original train DataLoader generator')
            generator.set_state(generator_state)
            audit.train_loader_generator_restored = True
        # Seed auditing opens the original reader in the parent. Linux fork
        # does not call its __getstate__; let every worker lazy-open its own
        # handle instead of inheriting this live HDF5/VDS file object.
        parent_handle = getattr(dataset, 'h5_file', None)
        if parent_handle is not None:
            parent_handle.close()
            dataset.h5_file = None
            audit.parent_hdf5_handle_closed = True
        if gpu_image_preprocess:
            from jepa4r.envs.twins.rh56.uint8_gpu import offload_image_preprocessor
            audit.gpu_image_preprocessor = offload_image_preprocessor(dataset)
        factory = subset_factory
        if factory is None:
            from torch.utils.data import Subset
            factory = Subset
        return [factory(dataset, audit.partition.clip_indices[s]) for s in ('train', 'val')]

    with temporary_bindings((train_module, 'get_column_normalizer', fit),
                            (train_module.spt.data, 'random_split', split)):
        yield audit


def resolve_budget(mode, batch_size=None, workers=None, prefetch_factor=2):
    if mode not in ('smoke', 'profile', 'formal'):
        raise ValueError('unsupported mode')
    batch = int(batch_size if batch_size is not None else (2 if mode == 'smoke' else 128))
    readers = int(workers if workers is not None else (0 if mode == 'smoke' else 8))
    if batch < 2 or readers < 0:
        raise ValueError('batch_size must be >=2 and workers >=0')
    prefetch = int(prefetch_factor)
    if not 1 <= prefetch <= 3:
        raise ValueError('prefetch_factor must be between 1 and 3')
    bounded = mode != 'formal'
    return dict(mode=mode, batch_size=batch, workers=readers, prefetch_factor=prefetch,
                scheduler_epochs=10, completed_epochs=1 if bounded else 10,
                train_limit=2 if bounded else None, val_limit=1 if bounded else None)


def expected_optimizer_steps(budget, train_clip_count):
    batches = int(train_clip_count) // budget['batch_size']
    if budget['mode'] != 'formal':
        if batches < 2:
            raise ValueError('bounded WM requires at least two complete training batches')
        return 2
    if batches < 1:
        raise ValueError('formal WM has no complete training batch')
    return batches * 10


def maybe_stop_after_bounded_epoch(trainer, budget):
    if budget['mode'] == 'formal':
        return
    if int(trainer.current_epoch) != 0 or int(trainer.global_step) != 2:
        raise RuntimeError('one-epoch/two-optimizer-step budget violated')
    trainer.should_stop = True


def stop_after_smoke_epoch(trainer):
    """Backward-compatible helper for the already-verified smoke contract."""
    return maybe_stop_after_bounded_epoch(trainer, resolve_budget('smoke'))


def validate_training_budget(budget, evidence, train_clip_count, start_step=0, val_clip_count=None):
    expected = expected_optimizer_steps(budget, train_clip_count)
    epochs = list(range(1, budget['completed_epochs'] + 1))
    if evidence['epochs'] != epochs or evidence['optimizer_steps'] != expected:
        raise RuntimeError('epoch/optimizer budget mismatch')
    if evidence['train_batches'] != expected or evidence['val_batches'] < budget['completed_epochs']:
        raise RuntimeError('train/validation batch budget mismatch')
    if val_clip_count is not None:
        val_per_epoch = ((int(val_clip_count) + budget['batch_size'] - 1) // budget['batch_size']
                         if budget['mode'] == 'formal' else 1)
        if evidence['val_batches'] != val_per_epoch * budget['completed_epochs']:
            raise RuntimeError('full validation batch budget mismatch')
    if start_step < 0 or start_step > expected:
        raise RuntimeError('resume step outside budget')


def require_same_identity(expected, actual):
    if expected != actual:
        raise ValueError('frozen run identity changed; refuse resume')


def checked_resume_checkpoint(output, receipt):
    output = Path(output).resolve()
    candidate = Path(receipt['checkpoint']).resolve()
    if not candidate.is_relative_to(output):
        raise ValueError('resume checkpoint is outside this task output')
    if not candidate.is_file() or sha(candidate) != receipt['sha256']:
        raise ValueError('resume checkpoint hash mismatch')
    return candidate


BASELINE_FINGERPRINT_SCOPE = (
    'models/*.py; data/clips.py; models/encoders/**/*.py; ../configs/method/relocate/train YAML; '
    'policy/bc/lewm_cortex.py; '
    'excludes datasets, outputs, caches, external dependencies and adapter files'
)


def tree_sha(root):
    """Fingerprint learning sources, never mutable run outputs or cache files.

    This deliberately differs from the original standalone experiment's whole
    baseline-directory hash. Adapter identity prevents reusing its old receipts.
    """
    root = Path(root)
    if not (root / 'models/world_model.py').is_file():
        raise ValueError('baseline root must contain the original train.py')
    paths = set((root / 'models').glob('*.py')) | {root / 'data/clips.py'}
    paths.update((root / 'models/encoders').rglob('*.py'))
    for suffix in ('*.yaml', '*.yml'):
        paths.update((root.parent / 'configs/method/relocate/train').rglob(suffix))
    loader = root / 'policy/bc/lewm_cortex.py'
    if loader.is_file():
        paths.add(loader)
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file() and '__pycache__' not in path.parts:
            name = path.relative_to(root) if path.is_relative_to(root) else Path('..') / path.relative_to(root.parent)
            digest.update(str(name).replace('\\', '/').encode())
            digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def capture_training_batch(batch):
    """Clone two examples; reload requires model.obs_key (pixels/pcd) and action.

    Forward adds scalar losses in place; reload recomputes them from these inputs.
    """
    import torch
    return {k: v[:2].detach().cpu().clone() for k, v in batch.items()
            if torch.is_tensor(v) and v.ndim > 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', default=str(Path(__file__).resolve().parents[3]),
                        help='Existing LeWM source root; defaults to this checkout')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--route', required=True, choices=('rgb', 'pcd'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--mode', choices=('smoke', 'profile', 'formal'), default='smoke')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--uint8-gpu-preprocess', action='store_true',
                        help='Experimental RGB CUDA only, default off; prior full-model CPU/GPU numerical gate failed')
    parser.add_argument('--seed', type=int, default=3072)
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Experimental: only this output sealed checkpoint; bit-exact continuation unverified; not old standalone '
             'receipts',
    )
    parser.add_argument(
        '--recovery', action='store_true', help='Explicit bounded workers0 complete-state continuation; default off'
    )
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--pause-after-batches', type=int)
    args = parser.parse_args(argv)
    if args.uint8_gpu_preprocess and (args.route != 'rgb' or args.device != 'cuda'):
        parser.error('--uint8-gpu-preprocess requires --route rgb --device cuda')
    budget = resolve_budget(args.mode, args.batch_size, args.workers, args.prefetch_factor)
    if (args.resume_checkpoint is not None or args.pause_after_batches is not None) and not args.recovery:
        parser.error('bounded checkpoint/pause requires --recovery')
    if args.recovery and (
        args.mode == 'formal' or args.resume or budget['workers'] != 0 or args.seed != 3072 or args.uint8_gpu_preprocess
    ):
        parser.error(
            'bounded recovery requires smoke/profile, workers0, guide3072, original preprocessing and a fresh phase'
        )
    if args.pause_after_batches is not None and args.pause_after_batches != 1:
        parser.error('two-step bounded recovery pauses after exactly one real batch')
    if args.resume and args.mode != 'formal':
        raise ValueError('resume is only supported for formal mode')
    out = Path(args.output).resolve()
    validate_output_location(out, args.baseline_root)
    out.mkdir(parents=True, exist_ok=args.resume)
    state_path = out / 'state.json'
    if args.resume:
        if not state_path.is_file():
            raise ValueError('resume requires existing task state')
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state.get('phase') == 'complete':
            raise ValueError('completed formal output cannot be retrained or overwritten')
        state['resume_started'] = time.time()
    else:
        state = dict(phase='initializing', started=time.time(), completed=0, failures=0,
                     exit_code=None, route=args.route, mode=args.mode)
    def write_state():
        state['updated'] = time.time()
        atomic_json(state_path, state)
    write_state()
    metrics = None
    try:
        import torch
        import hydra
        import stable_pretraining as spt
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf, open_dict
        from types import SimpleNamespace
        torch.set_num_threads(1)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if args.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable')
            torch.cuda.manual_seed_all(args.seed)
        baseline = Path(args.baseline_root).resolve()
        sys.path.insert(0, str(baseline.parent))
        train = importlib.import_module('jepa4r.models.world_model')
        import stable_pretraining
        train.spt = stable_pretraining  # the moved train.py imports it lazily
        if Path(train.__file__).resolve() != baseline / 'models/world_model.py':
            raise RuntimeError('wrong train.py imported')
        manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        if not str(manifest.get('action_alignment', '')).startswith('outgoing'):
            raise ValueError('expected original World.collect outgoing action convention')
        if not OmegaConf.has_resolver('eval'):
            OmegaConf.register_new_resolver('eval', eval)
        overrides = ['data=adroit_relocate', 'model=lewm'] if args.route == 'rgb' else [
            'data=adroit_relocate_pcd', 'model=lewm_pcd_dexart']
        train_configs = str(baseline.parent / 'configs/method/relocate/train')
        with initialize_config_dir(version_base=None, config_dir=train_configs):
            cfg = compose(config_name='lewm', overrides=overrides)
        run_name = f'rh56_relocate_{args.route}_{args.mode}_v1'
        with open_dict(cfg):
            cfg.seed = args.seed
            cfg.data.dataset.name = str(Path(args.dataset).resolve())
            cfg.data.dataset.keys_to_load = (
                ['pixels', 'action'] if args.route == 'rgb' else ['pcd_xyz', 'pcd_rgb', 'action']
            )
            cfg.data.dataset.keys_to_cache = ['action']
            cfg.output_model_name = run_name; cfg.subdir = run_name
            cfg.num_workers = budget['workers']
            cfg.loader.batch_size = budget['batch_size']; cfg.loader.num_workers = budget['workers']
            cfg.loader.persistent_workers = budget['workers'] > 0
            cfg.loader.pin_memory = args.device == 'cuda'
            if budget['workers']:
                cfg.loader.prefetch_factor = budget['prefetch_factor']
            else:
                cfg.loader.pop('prefetch_factor', None)
            cfg.wandb = {'enabled': False}
            cfg.trainer.max_epochs = 10
            # Formal mode never inherits the bounded smoke limits.
            cfg.trainer.pop('limit_train_batches', None); cfg.trainer.pop('limit_val_batches', None)
            if args.mode != 'formal':
                cfg.trainer.limit_train_batches = budget['train_limit']
                cfg.trainer.limit_val_batches = budget['val_limit']
            cfg.trainer.accelerator = 'gpu' if args.device == 'cuda' else 'cpu'
            cfg.trainer.devices = 1
            cfg.trainer.precision = 'bf16-mixed' if args.device == 'cuda' else '32-true'
            cfg.trainer.enable_progress_bar = False
        identity = dict(mode=args.mode, route=args.route, seed=args.seed, device=args.device,
                        seed_scope='explicit cfg/data/construction/Manager guide seed',
                        budget=budget, dataset=str(Path(args.dataset).resolve()),
                        dataset_sha256=sha(args.dataset), manifest_sha256=sha(args.manifest),
                        adapter_sha256=sha(__file__), runtime_helpers_sha256=RUNTIME_HELPERS_SHA256,
                        baseline_tree_sha256=tree_sha(baseline),
                        baseline_fingerprint_scope=BASELINE_FINGERPRINT_SCOPE,
                        config=OmegaConf.to_container(cfg, resolve=False),
                        torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        spt_version=getattr(spt, '__version__', None))
        identity['image_preprocessing'] = 'uint8_gpu' if args.uint8_gpu_preprocess else 'original_cpu'
        if args.recovery:
            from jepa4r.envs.twins.rh56 import wm_recovery
            identity['bounded_recovery'] = dict(enabled=True,helper_sha256=sha(wm_recovery.__file__))
        if args.uint8_gpu_preprocess:
            identity['image_preprocessing_sha256'] = sha(Path(__file__).with_name('uint8_gpu.py'))
        resume_receipt = None; resume_checkpoint = None; start_step = 0
        if args.resume:
            require_same_identity(identity, json.loads((out / 'run_identity.json').read_text(encoding='utf-8')))
            resume_receipt = json.loads((out / 'resume.json').read_text(encoding='utf-8'))
            resume_checkpoint = checked_resume_checkpoint(out, resume_receipt)
            start_step = int(resume_receipt['global_step'])
            restored_checkpoint = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
            require_same_identity(identity, restored_checkpoint.get('rh56_run_identity'))
            if int(restored_checkpoint['global_step']) != start_step:
                raise ValueError('resume receipt step disagrees with actual checkpoint')
            restored_rng = restored_checkpoint.get('rh56_rng_state')
            if restored_rng is None or restored_rng.get('train_loader_generator') is None:
                raise ValueError('resume checkpoint lacks recorded RNG and DataLoader generator state')
        else:
            atomic_json(out / 'run_identity.json', identity)
            restored_rng = None
        os.environ['STABLEWM_HOME'] = str(out / 'worldmodel_cache')
        spt.set(cache_dir=str(out / 'spt_cache'), cleanup={
            'checkpoints': False, 'logs': False, 'callback_artifacts': False})
        original_callback = train.SaveCkptCallback
        original_manager = train.spt.Manager
        evidence = (resume_receipt['evidence'] if resume_receipt else dict(
            gradient_checks=0, train_batches=0, val_batches=0, pre_step_lrs=[],
            train_loss=None, epochs=[], epoch_loss_means=[], epoch_validation_losses=[]))
        capture = {}; last_progress = [0.]
        recovery_evidence = None
        if args.recovery:
            recovery_evidence = wm_recovery.EvidenceState(evidence,capture,out)
        gradient_interval = 100 if args.mode == 'formal' else 1
        metrics = (out / 'metrics.jsonl').open('a' if args.resume else 'x', encoding='utf-8')
        def log(row):
            metrics.write(json.dumps(dict(time=time.time(), **row)) + '\n'); metrics.flush()
        def rng_state(trainer):
            generator = getattr(trainer.train_dataloader, 'generator', None)
            return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        train_loader_generator=generator.get_state() if generator is not None else None)
        def restore_rng(saved, trainer):
            random.setstate(saved['python']); np.random.set_state(saved['numpy']); torch.set_rng_state(saved['torch'])
            if saved['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(saved['cuda'])
        class Evidence(original_callback):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                capture['evidence_callback'] = self
            def on_load_checkpoint(self, trainer, pl_module, checkpoint):
                require_same_identity(identity, checkpoint.get('rh56_run_identity'))
                self.loaded_rng = checkpoint.get('rh56_rng_state')
                if self.loaded_rng is None:
                    raise ValueError('resume checkpoint has no task RNG state')
            def on_save_checkpoint(self, trainer, pl_module, checkpoint):
                checkpoint['rh56_run_identity'] = identity
                checkpoint['rh56_rng_state'] = rng_state(trainer)
            def on_fit_start(self, trainer, pl_module):
                expected_optimizer_steps(budget, len(audit.partition.clip_indices['train']))
                evidence['seed_scopes'] = dict(configured_seed=int(cfg.seed),
                    data_generator_seed=int(cfg.seed),shim_startup_torch_seed=args.seed,
                    actual_torch_initial_seed_at_fit_start=int(torch.initial_seed()),
                    explicit_manager_seed=int(args.seed))
                if int(torch.initial_seed())!=int(args.seed):raise RuntimeError('guide WM Manager seed was not applied')
                capture['initial'] = {n: p.detach().cpu().clone() for n, p in pl_module.model.named_parameters()}
                capture['fit_started'] = time.perf_counter()
                atomic_json(out/'data_boundary.json',dict(split=audit.partition.receipt(),normalizers=audit.normalizers,
                    train_loader_generator_restored=audit.train_loader_generator_restored,
                    parent_hdf5_handle_closed=audit.parent_hdf5_handle_closed))
            def on_train_start(self, trainer, pl_module):
                if getattr(self, 'loaded_rng', None) is not None:
                    # The split hook restores the dedicated loader generator
                    # before DataLoader construction; do not reset a prefetched
                    # iterator here. Restore model RNG after the sanity pass.
                    restore_rng(self.loaded_rng, trainer)
                    log(dict(event='resume_rng_restored', global_step=int(trainer.global_step),
                             loader_restored_before_construction=audit.train_loader_generator_restored))
            def on_train_epoch_start(self, trainer, pl_module):
                capture['epoch_index'] = int(trainer.current_epoch)
                restored = capture.pop('restored_epoch_progress',None) if args.recovery else None
                if restored is None or not wm_recovery.restore_epoch_progress(self,restored,int(trainer.current_epoch)):
                    self.epoch_sum = 0.; self.epoch_batches = 0
                    self.val_sum = 0.; self.val_batches = 0
                log(
                    dict(
                        event='epoch_started',
                        epoch=int(trainer.current_epoch) + 1,
                        global_step=int(trainer.global_step),
                    )
                )
            def on_before_optimizer_step(self, trainer, pl_module, optimizer):
                step = int(trainer.global_step)
                if step >= 2 and step % gradient_interval and args.mode == 'formal':return
                grads = [p.grad for p in pl_module.model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise RuntimeError('missing or nonfinite model gradients')
                if not any(torch.count_nonzero(g) for g in grads):raise RuntimeError('all model gradients are zero')
                evidence['pre_step_lrs'].append(dict(step=step, lrs=[float(g['lr']) for g in optimizer.param_groups]))
                evidence['gradient_checks'] += 1
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
                if not torch.isfinite(loss).all():raise RuntimeError('nonfinite training loss')
                value = float(loss.detach().cpu()); evidence['train_batches'] += 1
                evidence['train_loss'] = value; self.epoch_sum += value; self.epoch_batches += 1
                if batch_idx == 0 and (not args.recovery or 'batch' not in capture):
                    # A small real input probe avoids replaying a formal 128x4 batch on one CPU thread.
                    capture['batch'] = capture_training_batch(batch)
                    torch.save(capture['batch'], out / 'real_training_batch.pt')
                log(dict(event='train_batch', epoch=int(trainer.current_epoch)+1,
                         global_step=int(trainer.global_step), loss=value))
                if time.time() - last_progress[0] > 10:
                    state.update(
                        phase='training',
                        epoch=int(trainer.current_epoch) + 1,
                        global_step=int(trainer.global_step),
                        train_batches=evidence['train_batches'],
                        loss=value,
                    )
                    write_state(); last_progress[0] = time.time()
            def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                if trainer.sanity_checking:return
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
                if not torch.isfinite(loss).all():raise RuntimeError('nonfinite validation loss')
                value = float(loss.detach().cpu())
                evidence['val_batches'] += 1; self.val_sum += value; self.val_batches += 1
                log(dict(event='validation_batch', epoch=int(trainer.current_epoch)+1, loss=value))
            def on_train_epoch_end(self, trainer, pl_module):
                epoch = int(trainer.current_epoch)+1
                if epoch in evidence['epochs']:raise RuntimeError('resume would overwrite an accepted epoch checkpoint')
                if not self.epoch_batches or not self.val_batches:
                    raise RuntimeError('epoch missing train or validation work')
                super().on_train_epoch_end(trainer, pl_module)
                evidence['epochs'].append(epoch)
                evidence['epoch_loss_means'].append(self.epoch_sum/self.epoch_batches)
                evidence['epoch_validation_losses'].append(self.val_sum/self.val_batches)
                evidence['optimizer_steps'] = int(trainer.global_step)
                evidence['changed_parameter_tensors'] = sum(not torch.equal(p.detach().cpu(),capture['initial'][n])
                    for n,p in pl_module.model.named_parameters())
                resume_path = out / f'resume_epoch_{epoch:02d}_step_{int(trainer.global_step)}.ckpt'
                if resume_path.exists():raise FileExistsError(resume_path)
                trainer.save_checkpoint(str(resume_path))
                atomic_json(out/'resume.json',dict(checkpoint=str(resume_path),sha256=sha(resume_path),
                    epoch=epoch,global_step=int(trainer.global_step),evidence=evidence))
                state.update(phase='training',completed_epochs=evidence['epochs'],global_step=int(trainer.global_step))
                write_state(); log(dict(event='epoch_completed',epoch=epoch,global_step=int(trainer.global_step)))
                maybe_stop_after_bounded_epoch(trainer,budget)
            def on_fit_end(self, trainer, pl_module):
                evidence['optimizer_steps'] = int(trainer.global_step)
                evidence['fit_wall_seconds'] = time.perf_counter()-capture['fit_started']
                trainer.save_checkpoint(str(out/'trainer_final.ckpt'))
        def manager(*manager_args, **manager_kwargs):
            manager_kwargs['seed'] = int(args.seed)
            if resume_checkpoint is not None:manager_kwargs['ckpt_path'] = str(resume_checkpoint)
            return original_manager(*manager_args, **manager_kwargs)
        state['phase']='training';write_state()
        verification_only = bool(resume_receipt and int(resume_receipt['epoch'])==10)
        if not verification_only:
            with temporary_bindings((train, 'SaveCkptCallback', Evidence),
                                    (train.spt, 'Manager', manager)):
                from contextlib import nullcontext
                recovery_context = (wm_recovery.training_context(train,
                    lambda:wm_recovery.recovery_identity(args,cfg,identity),out/'recovery',seed=args.seed,
                    pause_after_batches=args.pause_after_batches,resume_checkpoint=args.resume_checkpoint,
                    evidence_provider=recovery_evidence.state_dict,evidence_restorer=recovery_evidence.load_state_dict)
                    if args.recovery else nullcontext())
                with recovery_context,install_episode_split(train,args.dataset,manifest,
                        generator_state=restored_rng['train_loader_generator'] if restored_rng else None,
                        gpu_image_preprocess=args.uint8_gpu_preprocess) as audit:
                    train.run_random_split(cfg, None, original_layout=True)
                    partition = audit.partition; normalizers = audit.normalizers
                    hook = audit.gpu_image_preprocessor
                    image_io = dict(mode=identity['image_preprocessing'],
                        processed_batches=hook.processed_batches if hook else 0,
                        schema=hook.schema if hook else None)
                    if args.uint8_gpu_preprocess and not image_io['processed_batches']:
                        raise RuntimeError('SPT uint8 GPU image preprocessing hook was never executed')
            atomic_json(out/'data_boundary.json',dict(split=partition.receipt(),normalizers=normalizers,
                train_loader_generator_restored=audit.train_loader_generator_restored,
                parent_hdf5_handle_closed=audit.parent_hdf5_handle_closed,
                image_preprocessing=image_io))
        else:
            # Completed training can be re-validated after a QA interruption without another optimizer step.
            boundary=json.loads((out/'data_boundary.json').read_text(encoding='utf-8'))
            partition=None;normalizers=boundary['normalizers']
        boundary=json.loads((out/'data_boundary.json').read_text(encoding='utf-8'))
        validate_training_budget(budget,evidence,boundary['split']['clips']['train'],start_step,
                                 val_clip_count=boundary['split']['clips']['val'])
        if not evidence['pre_step_lrs'] or not any(lr>0 for row in evidence['pre_step_lrs'] for lr in row['lrs']):
            raise RuntimeError('no checked optimizer step used a nonzero learning rate')
        if evidence['changed_parameter_tensors']==0:raise RuntimeError('optimizer did not change model parameters')
        state['phase']='validating_checkpoints';write_state()
        ckdir=out/'worldmodel_cache/checkpoints'/run_name
        completed=budget['completed_epochs'];weight_path=ckdir/f'weights_epoch_{completed}.pt'
        checkpoint_records=[];weight_schema=None
        for epoch in range(1,completed+1):
            path=ckdir/f'weights_epoch_{epoch}.pt'
            weights=torch.load(path,map_location='cpu',weights_only=True)
            schema={key:(list(value.shape),str(value.dtype)) for key,value in weights.items()}
            if weight_schema is None:weight_schema=schema
            if schema!=weight_schema:raise RuntimeError(f'epoch {epoch} checkpoint schema changed')
            if not all(torch.isfinite(v).all() for v in weights.values() if v.is_floating_point()):
                raise RuntimeError(f'nonfinite epoch {epoch} checkpoint')
            checkpoint_records.append(dict(epoch=epoch,path=str(path),sha256=sha(path)))
        weights=torch.load(weight_path,map_location='cpu',weights_only=True)
        final_trainer=(resume_checkpoint if verification_only else out/'trainer_final.ckpt')
        trainer_ck=torch.load(final_trainer,map_location='cpu',weights_only=False)
        require_same_identity(identity,trainer_ck.get('rh56_run_identity'))
        final_weights={k[len('model.'):]:v for k,v in trainer_ck['state_dict'].items() if k.startswith('model.')}
        if set(weights)!=set(final_weights) or not all(torch.equal(v,final_weights[k]) for k,v in weights.items()):
            raise RuntimeError('last epoch weights differ from the actual trained Lightning model')
        from jepa4r.models.world_model import _retarget   # config.json keeps LeWM's names; the classes moved
        model_cfg=OmegaConf.create(_retarget(json.loads((ckdir/'config.json').read_text(encoding='utf-8'))))
        first,second=hydra.utils.instantiate(model_cfg).eval(),hydra.utils.instantiate(model_cfg).eval()
        first.load_state_dict(weights,strict=True);second.load_state_dict(weights,strict=True)
        batch=torch.load(out/'real_training_batch.pt',map_location='cpu',weights_only=True)
        sigreg_state={k[len('sigreg.'):]:v for k,v in trainer_ck['state_dict'].items() if k.startswith('sigreg.')}
        forward=hydra.utils.get_method(cfg.forward) if cfg.get('forward') else train.lejepa_forward
        def probe(model):
            sigreg=train.SIGReg(**OmegaConf.to_container(cfg.loss.sigreg.kwargs,resolve=True)).eval()
            sigreg.load_state_dict(sigreg_state,strict=True)
            holder=SimpleNamespace(model=model,sigreg=sigreg,log_dict=lambda *args,**kwargs:None)
            with torch.random.fork_rng(devices=[]),torch.no_grad():
                torch.manual_seed(args.seed)
                values=forward(holder,{k:v.clone() for k,v in batch.items()},'reload',cfg)
                prediction=model.predict(values['emb'][:,:cfg.history_size],values['act_emb'][:,:cfg.history_size])
            return {k:values[k] for k in ('emb','act_emb','pred_loss','sigreg_loss','loss')}|{'pred':prediction}
        left,right=probe(first),probe(second)
        differences={k:float((left[k]-right[k]).abs().max()) for k in left}
        if not all(torch.isfinite(v).all() for v in left.values()) or any(differences.values()):
            raise RuntimeError('strict CPU reload encode/predict/loss mismatch')
        steps=[int(v['step']) for o in trainer_ck['optimizer_states'] for v in o['state'].values() if 'step' in v]
        expected=expected_optimizer_steps(budget,boundary['split']['clips']['train'])
        if not steps or set(steps)!={expected}:raise RuntimeError('actual Adam state step does not match full budget')
        if tree_sha(baseline)!=identity['baseline_tree_sha256']:raise RuntimeError('protected baseline changed')
        result = dict(
            status='passed',
            mode=args.mode,
            scope=(
                'formal ten-epoch world-model training'
                if args.mode == 'formal'
                else 'bounded interface/capacity profile; not formal training'
            ),
            route=args.route,
            budget=budget,
            evidence=evidence,
            split=boundary['split'],
            normalizers=normalizers,
            data_io_boundary=dict(
                parent_hdf5_handle_closed=boundary['parent_hdf5_handle_closed'],
                image_preprocessing=boundary.get('image_preprocessing'),
            ),
            strict_reload_maxabs=differences,
            reload_probe_samples=len(next(iter(batch.values()))),
            actual_adam_step_values=sorted(set(steps)),
            action_encoder_input_dim=int(model_cfg.action_encoder.input_dim),
            checkpoint=str(weight_path),
            checkpoint_epoch=completed,
            checkpoints=checkpoint_records,
            checkpoint_sha256=sha(weight_path),
            manifest_sha256=identity['manifest_sha256'],
            dataset_sha256=identity['dataset_sha256'],
            script_sha256=identity['adapter_sha256'],
            runtime_helpers_sha256=RUNTIME_HELPERS_SHA256,
            baseline_tree_sha256=identity['baseline_tree_sha256'],
            baseline_fingerprint_scope=BASELINE_FINGERPRINT_SCOPE,
            bounded_recovery=dict(
                enabled=args.recovery,
                resume_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint else None,
                final_checkpoint=str(out / 'recovery/final.ckpt') if args.recovery else None,
            ),
            train_py_sha256=sha(baseline / 'models/world_model.py'),
            utils_py_sha256=sha(baseline / 'data/clips.py'),
            gradient_check_interval=gradient_interval,
            resume_used=args.resume,
            verification_only=verification_only,
        )
        (out/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True),encoding='utf-8')
        atomic_json(out/'validation.json',result)
        state.update(phase='complete',completed=1,completed_epochs=completed,exit_code=0);write_state()
        print(json.dumps(result,indent=2),flush=True)
        return result
    except BaseException as error:
        if args.recovery:
            from jepa4r.models.relocate.recovery import RecoveryPause
            if isinstance(error,RecoveryPause):
                state.update(
                    phase='intentional_sealed_pause',
                    completed=0,
                    exit_code=75,
                    checkpoint=str(error),
                    scope='original RH56 update sealed; independent continuation pending',
                )
                write_state()
                raise SystemExit(75) from error
        state.update(phase='failed', failures=state.get('failures', 0) + 1, exit_code=1, error=traceback.format_exc())
        write_state()
        raise
    finally:
        if metrics is not None:metrics.close()


if __name__=='__main__':main()

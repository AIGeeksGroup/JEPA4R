"""Sealed Cube data seam around the byte-identical CortexBench Tiny entry.

The native reader, encoder, augmentations, loss, optimizer, schedule and loop
are called unchanged. Only sealed-input guards and optional prefetch are added.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

from .contracts import get_profile, validate_manifest, validate_view
from .training import build_environment

CANONICAL_SHA256 = '64caecbd3f2282d38c159018cb3d6f204079bc60ccecfa41b231d00332937b51'


def _load_canonical():
    path = Path(__file__).resolve().parents[2] / 'scripts/baselines/train_lejepa_mw.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != CANONICAL_SHA256:
        raise ValueError('canonical Tiny source differs')
    spec = importlib.util.spec_from_file_location('cube_native_lejepa_tiny', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def native_module(profile_name, data_root, prefetch=None):
    profile = get_profile(profile_name)
    validate_manifest(data_root, profile_name)
    view = Path(data_root) / profile['view_directory'] / profile['train_view']
    validate_view(view, profile_name, 'train')
    module = _load_canonical()
    if prefetch is not None:
        if prefetch < 1:
            raise ValueError('prefetch must be positive')
        original = module.DataLoader

        class PrefetchLoader(original):
            def __init__(self, *args, **kwargs):
                kwargs['prefetch_factor'] = prefetch
                super().__init__(*args, **kwargs)

        module.DataLoader = PrefetchLoader
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--prefetch', type=int)
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error('native Tiny requires positive workers')
    if args.prefetch is not None and args.prefetch < 1:
        parser.error('prefetch must be positive')
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(build_environment(args.data_root, args.output))
    state = dict(stage='native_tiny_running', started_at=datetime.now(timezone.utc).isoformat(),
                 actual_exit_code=None, argv=sys.argv[:], profile=get_profile(args.profile),
                 canonical_sha256=CANONICAL_SHA256)
    state_path = args.output / 'status.json'
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    original_argv = sys.argv[:]
    with (args.output / 'stdout.log').open('x', encoding='utf8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                module = native_module(args.profile, args.data_root, args.prefetch)
                sys.argv = [module.__file__, '--task', args.profile,
                            '--dataset', str(args.data_root / state['profile']['view_directory'] /
                                             state['profile']['train_view']),
                            '--name', 'lejepa_' + args.profile]
                if args.workers is not None:
                    sys.argv += ['--workers', str(args.workers)]
                state['native_argv'] = sys.argv[:]
                state_path.write_text(json.dumps(state, indent=2) + '\n')
                module.main()
                state.update(stage='native_tiny_training_returned', actual_exit_code=0)
            except Exception:
                state.update(stage='native_tiny_failed', actual_exit_code=1,
                             error=traceback.format_exc())
                traceback.print_exc()
            finally:
                sys.argv = original_argv
    state['finished_at'] = datetime.now(timezone.utc).isoformat()
    state['updated_at'] = state['finished_at']
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    (args.output / 'actual_exit_code').write_text(str(state['actual_exit_code']) + '\n')
    return state['actual_exit_code']


if __name__ == '__main__':
    raise SystemExit(main())

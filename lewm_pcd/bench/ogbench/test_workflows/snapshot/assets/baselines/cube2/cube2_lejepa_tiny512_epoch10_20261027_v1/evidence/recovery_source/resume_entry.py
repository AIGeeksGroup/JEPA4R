"""Opt-in weight-only recovery of the frozen native main; no copied model/loss."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

CORE_SHA = '64caecbd3f2282d38c159018cb3d6f204079bc60ccecfa41b231d00332937b51'
IO_SHA = '320bff92d774c8499e9a6f69a622a1d6b64a5e43bd038d7124246f4f50cec8b9'

# Only main is compiled. Imports, model, SIGReg, augmentation and PF2 loader are
# the old module's exact definitions. Every replacement must match exactly once.
EDITS = [
    ('    args = ap.parse_args()',
     "    ap.add_argument('--resume-model', type=str, default=None)\n"
     "    ap.add_argument('--start-epoch', type=int, default=0)\n"
     "    ap.add_argument('--verified-export-root', type=str, default=None)\n"
     '    args = ap.parse_args()\n    _validate_resume_args(args)'),
    ("    sigreg = SIGReg().to('cuda')",
     "    sigreg = SIGReg().to('cuda')\n    _parent_receipt = _load_resume_model(net, args, out, _RESUME_BINDING)"),
    ('    scaler = GradScaler(enabled=True)',
     '    scaler = GradScaler(enabled=True)\n'
     '    _first_update = _prepare_resume(scheduler, opt, net, args, len(train), out, _parent_receipt, _RESUME_BINDING)'),
    ('    for epoch in range(args.epochs):', '    for epoch in range(args.start_epoch, args.epochs):'),
    ('            scheduler.step()',
     '            scheduler.step()\n            _record_first_update(_first_update, scheduler, out)'),
    ('                return\n',
     '                _smoke_export(net, out, args, step + 1)\n                return\n'),
    ("        torch.save({'backbone': net.backbone.state_dict(), 'head': net.head.state_dict(), 'proj': net.proj.state_dict()},\n"
     "                   out / f'weights_epoch_{epoch + 1}.pt')",
     '        _save_epoch(net, out, epoch + 1, args.verified_export_root)'),
]


def adapt_main(source):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    main_source = ast.get_source_segment(source, node)
    for before, after in EDITS:
        if main_source.count(before) != 1:
            raise ValueError(f'Native main anchor changed: {before!r}')
        main_source = main_source.replace(before, after, 1)
    compile(main_source, '<weight-only-native-main>', 'exec')
    return main_source


def restore_main(adapted):
    for before, after in reversed(EDITS):
        if adapted.count(after) != 1:
            raise ValueError('Adapted main no longer matches declared edits')
        adapted = adapted.replace(after, before, 1)
    return adapted


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('native_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    case = args.case.resolve()
    plan = json.loads((case / 'plan.json').read_text())
    reference = Path(plan['reference_case'])
    for name, expected in [('source/train_lejepa_mw.py', CORE_SHA), ('worker_prefetch_v1/native_io.py', IO_SHA)]:
        if hashlib.sha256((reference / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen reference mismatch: {name}')
    io = load_file('frozen_native_io_resume7', reference / 'worker_prefetch_v1/native_io.py')
    native = io.module(reference, plan['recipe']['prefetch_factor'])
    helpers = load_file('resume7_verified_helpers', case / 'resume_helpers.py')
    for name in ['validate_resume_args', 'load_resume_model', 'prepare_resume', 'record_first_update', 'smoke_export', 'save_epoch']:
        native.__dict__['_' + name] = getattr(helpers, name)
    native.__dict__['_RESUME_BINDING'] = plan['recovery']
    adapted = adapt_main((reference / 'source/train_lejepa_mw.py').read_text())
    exec(compile(adapted, str(case / 'adapted_native_main.py'), 'exec'), native.__dict__)
    tail = args.native_args[1:] if args.native_args[:1] == ['--'] else args.native_args
    sys.argv = [str(case / 'resume_entry.py')] + tail
    native.main()


if __name__ == '__main__':
    main()

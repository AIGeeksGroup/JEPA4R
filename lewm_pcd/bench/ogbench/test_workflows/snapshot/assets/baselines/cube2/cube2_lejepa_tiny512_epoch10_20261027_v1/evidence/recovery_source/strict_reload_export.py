"""Root-owned execution gate: fresh CPU native Encoder + three-module strict load."""
import argparse
import importlib.util
import json
from pathlib import Path

from resume_entry import load_file
from resume_helpers import audit_checkpoint, tensor_bytes_equal, write_json_new, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.case / 'plan.json').read_text())
    reference = Path(plan['reference_case'])
    io = load_file('strict_reload_reference_io', reference / 'worker_prefetch_v1/native_io.py')
    module = io.module(reference, plan['recipe']['prefetch_factor'])
    metadata, payload = audit_checkpoint(args.model)
    net = module.Encoder(proj_dim=plan['recipe']['projector'][-1]).cpu()
    for name in ['backbone', 'head', 'proj']:
        actual = getattr(net, name)
        actual.load_state_dict(payload[name], strict=True)
        loaded = actual.state_dict()
        if set(loaded) != set(payload[name]):
            raise ValueError(f'Strict reload keys changed: {name}')
        for key, tensor in loaded.items():
            if not tensor_bytes_equal(tensor, payload[name][key]):
                raise ValueError(f'Strict reload bytes mismatch: {name}.{key}')
    write_json_new(args.receipt, {'schema': 'fresh_cpu_native_strict_reload_v1', 'at_utc': utc_now(),
                                 'model': metadata, 'strict_modules': ['backbone', 'head', 'proj'],
                                 'all_tensor_bytes_equal': True, 'cuda_used': False,
                                 'forward_executed': False})


if __name__ == '__main__':
    main()

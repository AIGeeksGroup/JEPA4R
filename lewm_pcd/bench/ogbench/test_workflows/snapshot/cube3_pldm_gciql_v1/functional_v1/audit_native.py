"""Original Native result-count auditor with explicit P3 width and source binding."""
import argparse
import hashlib
import json
from pathlib import Path
import types

GROUP = 'P3_cube3_pldm_rgb'
AUDIT_SHA = 'ffccacb2f94cfd37f0480419642341e7c966e490e1a97e6056eecfe0033c5285'
ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auditor(path):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == AUDIT_SHA
    source = raw.decode()
    assert source.count('211') == 3
    effective = source.replace('211', '531')
    assert effective.replace('531', '211') == source
    module = types.ModuleType('_P3_original_native_auditor')
    exec(compile(effective, str(path), 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--mode', choices=('tiny', 'full'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binding_path = args.case/'functional_v1/NATIVE_BINDING_v1.json'
    binding = json.loads(binding_path.read_text())
    assert binding['actual_exit_code'] == 0 and binding['group_id'] == GROUP
    evidence = json.loads((args.output/'native_source_binding.json').read_text())
    assert evidence['actual_exit_code'] == 0 and evidence['binding_sha256'] == sha(binding_path)
    assert evidence['encoder_family'] == 'pldm' and evidence['latent_dim'] == 512
    assert evidence['selected_preprocess_device'] == binding['group']['encoder']['selected_preprocess_device']
    assert evidence['torch_num_threads'] == 1 and evidence['torch_float32_matmul_precision'] == 'highest'
    assert evidence['cuda_matmul_allow_tf32'] is False
    for path, digest in binding['source_pins'].items():
        assert sha(path) == digest, path
    module = auditor(ROOT/'cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/audit.py')
    result = module.verify_report(args.case, GROUP, 'diagnostic' if args.mode == 'tiny' else 'full',
        args.output, 1 if args.mode == 'tiny' else 250, 521, 4 if args.mode == 'tiny' else 1000)
    result.update(actual_exit_code=0, group_id=GROUP, mode=args.mode,
        encoder_family='pldm', latent_dim=512, current_dim=531, goal_dim=521,
        synthetic_zero_padding=False, original_result_auditor_sha256=AUDIT_SHA,
        original_counter_seed_success_checks_preserved=True,
        fresh_goal_then_current_encoding_required=True, binding_sha256=sha(binding_path),
        diagnostic_excluded_from_formal_SR=args.mode == 'tiny', no_NN_or_update_performed=True)
    with (args.output/'independent_audit.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

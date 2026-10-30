"""Register one PLDM512 role; use frozen async100 training and boundary functions."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from pldm_inputs import CHECKPOINT_SHA, DIM, FAMILY, load_features, load_inputs

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT / 'cube2_pldm_gciql_v1'
LEGACY = ROOT / 'cube23_distance_gciql_v1'
GROUP = 'P2_cube2_pldm_rgb'
EXTENSION_SHA = 'a9735b9b4f3fec3b01f189e910e12e985cbba3a8ba13e1fafeac878f1222c9c1'


def check_config(config):
    assert config['executable'] is True and config['group_id'] == GROUP
    assert (config['cube_count'], config['modality']) == (2, 'rgb')
    assert (config['input_contract']['current_dim'], config['input_contract']['goal_dim']) == (531, 518)
    assert config['input_contract']['latent_dim'] == DIM and config['input_contract']['zero_padding'] is False
    encoder = config['encoder_binding']
    assert encoder['family'] == FAMILY and encoder['latent_dim'] == DIM and encoder['checkpoint_sha256'] == CHECKPOINT_SHA


def bind_boundary(runtime, boundary):
    # These two names were imported by value in the preserved boundary module.
    boundary.prepare, boundary.get_batch = runtime.prepare, runtime.get_batch
    assert boundary.prepare is runtime.prepare and boundary.get_batch is runtime.get_batch


def install():
    sys.path[:0] = [str(LEGACY/'five_group_extension_v1'), str(LEGACY/'state_priority_v1'), str(LEGACY), str(ROOT)]
    path = LEGACY/'five_group_extension_v1/extension_runtime.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXTENSION_SHA
    spec = importlib.util.spec_from_file_location('_P3_preserved_runtime', path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    extension.GROUPS = {GROUP: (2, 'rgb', 531, 518)}
    runtime = extension.install('baseline', 'async100')
    inputs = load_inputs(LEGACY/'distance_inputs.py')
    original_prepare = runtime.prepare
    runtime.DistanceInputs = inputs.VisualInputs
    runtime.load_cached_features = lambda **kw: load_features(inputs, **kw)
    def prepare(config, shared_datasets=None):
        check_config(config)
        samplers, resources, binding = original_prepare(config, shared_datasets)
        binding['runtime'].update(encoder_family=FAMILY, latent_dim=DIM,
            native_input_width_proof=inputs.width_proof, runtime_encoder_calls=0)
        binding['code'].update({str(p): runtime.utility.sha(p) for p in
            (Path(__file__), Path(__file__).with_name('pldm_inputs.py'))})
        return samplers, resources, binding
    runtime.prepare = prepare
    bind_boundary(runtime, importlib.import_module('gciql_boundary'))
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify-hooks', 'gate', 'reload', 'train', 'accept'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.action == 'verify-hooks':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('JAX_PLATFORMS') == 'cpu'
    else:
        assert args.config is not None
        config = json.loads(args.config.read_text())
        check_config(config)
    runtime = install()
    if args.action == 'verify-hooks':
        boundary = importlib.import_module('gciql_boundary')
        assert boundary.prepare is runtime.prepare and boundary.get_batch is runtime.get_batch
        assert runtime.jax.default_backend() == 'cpu'
        result = dict(actual_exit_code=0, boundary_prepare_identity=True, boundary_get_batch_identity=True,
            backend='cpu', sampler_called=False, agent_or_encoder_created=False,
            original_boundary_SHA='a81363583a4826194b90974be2ad62c5895ab6b963e15ba042c9f893a46ba5e6')
        if args.report:
            with args.report.open('x') as stream:
                json.dump(result, stream, indent=2)
        print(json.dumps(result))
        return 0
    if args.action == 'train':
        return runtime.run(str(args.config))
    # Import after installing hooks: boundary's imported functions must be the new inputs.
    boundary = importlib.import_module('gciql_boundary')
    assert boundary.prepare is runtime.prepare and boundary.get_batch is runtime.get_batch
    assert args.output is not None
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.action == 'gate':
        boundary.gate(config, args.output)
    elif args.action == 'reload':
        boundary.reload_gate(config, args.output, args.report)
    else:
        boundary.accept(config, args.output, args.report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

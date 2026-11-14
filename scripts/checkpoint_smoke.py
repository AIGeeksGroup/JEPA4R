"""Verify a pinned downloaded bundle, load one component and run two fixed forwards.

No environment, dataset, training loop or optimizer update is started. Use separate
original world-model / policy interpreters. --help needs only the standard library.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / 'released_checkpoints.json'
TASKS = {'cube_single': 1, 'cube_double': 2, 'cube_triple': 3, 'cube_quadruple': 4,
         'adroit_relocate': 0, 'rh56_relocate': 0}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_bundle(bundle, registry_path=None):
    registry = read(registry_path or REGISTRY)
    if registry.get('schema') != 'jepa4r_released_checkpoints_v1':
        raise ValueError('unsupported checkpoint registry schema')
    matches = [r for r in registry['releases'] if (r['task'], r['model']) ==
               (bundle.get('task'), bundle.get('model'))]
    if len(matches) != 1:
        raise ValueError('bundle task/model is not one pinned release')
    release = matches[0]
    if release['task'] not in TASKS or release['model'] != 'jepa4r':
        raise ValueError('load smoke supports the published JEPA4R tasks only')
    if not re.fullmatch('[0-9a-f]{40}', release['revision']):
        raise ValueError('an immutable revision is required')
    if any(bundle.get(k) != v for k, v in release.items() if k != 'files'):
        raise ValueError('bundle differs from pinned registry release')
    expected, actual = release['files'], bundle.get('files', [])
    if [{k: v for k, v in row.items() if k != 'local_path'} for row in actual] != expected:
        raise ValueError('bundle file pins differ from registry')
    root, directory = Path(bundle['root']), Path(bundle['directory'])
    if not root.is_absolute() or not directory.is_absolute() or directory.resolve() != (root / release['task'] / release['model']).resolve() or not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError('bundle directory path differs from root/task/model')
    files = {}
    for row in actual:
        name = PurePosixPath(row['path'])
        if not name.parts or name.is_absolute() or '..' in name.parts or '\\' in row['path'] or ':' in row['path']:
            raise ValueError('unsafe release file path')
        path = Path(row['local_path'])
        if not path.is_absolute() or path.resolve() != directory.joinpath(*name.parts).resolve() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('resolved file path differs from bundle directory')
        if row['path'] in files or type(row['bytes']) is not int or row['bytes'] <= 0 or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('invalid or duplicate release file pin')
        if path.stat().st_size != row['bytes']:
            raise ValueError('file bytes differ: ' + row['path'])
        if sha(path) != row['sha256']:
            raise ValueError('file SHA256 differs: ' + row['path'])
        files[row['path']] = path
    return release, files


def outputs(first, second):
    import numpy as np
    a, b = np.asarray(first), np.asarray(second)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.array_equal(a, b):
        raise ValueError('fixed forwards are nonfinite or not exactly repeatable')
    return dict(shape=list(a.shape), dtype=str(a.dtype), finite=True,
                repeat_max_abs=float(np.max(np.abs(a - b))), sha256=hashlib.sha256(a.tobytes()).hexdigest())


def load_wm(release, files, device):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from stable_worldmodel.wm.utils import load_pretrained
    spec = release['world_model']
    weight, config = files[spec['weights']], files[spec['config']]
    if config.name != 'config.json' or weight.parent != config.parent:
        raise ValueError('original WM loader requires sibling config.json')
    pre = OmegaConf.load(files[spec['training_config']]).data.preprocessor
    if str(pre['_target_']) != 'utils.get_pcd_preprocessor' or not pre.get('use_rgb'):
        raise ValueError('expected the original XYZRGB point-cloud preprocessor')
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='jepa4r-load-') as cache:
        model = load_pretrained(str(weight), cache_dir=cache).to(device).eval().requires_grad_(False)
    transform = instantiate(pre)
    xyz = torch.linspace(-0.03, 0.03, 7200).reshape(2, 1200, 3) + torch.tensor(list(pre.center))
    raw = {str(pre.source_xyz): xyz, str(pre.source_rgb): torch.arange(7200).reshape(2, 1200, 3).remainder(256).byte()}
    cloud = transform(raw)[str(pre.target)].unsqueeze(0).to(device)
    actions = torch.zeros((1, 2, read(config)['action_encoder']['input_dim']), device=device)
    def forward():
        encoded = model.encode({model.obs_key: cloud, 'action': actions})
        return torch.cat((encoded['emb'], model.predict(encoded['emb'], encoded['act_emb'])), dim=1).cpu().numpy().copy()
    with torch.inference_mode():
        first, second = forward(), forward()
    return dict(loader='stable_worldmodel.wm.utils.load_pretrained', strict_state_dict=True,
                input_shape=list(cloud.shape), preprocessing=OmegaConf.to_container(pre), output=outputs(first, second))


def policy_config(metadata, sidecar, cubes):
    if sidecar['network_step'] != 1000001 or sidecar['adam_count'] != 1000000:
        raise ValueError('original policy counter differs')
    binding = sidecar['binding']
    if metadata.get('binding', binding) != binding or ('binding' not in metadata and binding['config'] != metadata):
        raise ValueError('original policy metadata/state binding differs')
    config = binding['config']
    agent = metadata['binding']['agent_config'] if 'binding' in metadata else metadata['agent_config']
    contract = config.get('input_contract', {'current_dim': 211, 'goal_dim': 195, 'zero_padding': False})
    if (config.get('cube_count', 1), config.get('modality', config.get('kind')), config['seed'], config['total_updates'],
        contract['current_dim'], contract['goal_dim'], contract['zero_padding']) != (cubes, 'pcd', 0, 1000000, 211, 192 + 3 * cubes, False):
        raise ValueError('original policy task/input contract differs')
    if binding['agent_config'] != agent:
        raise ValueError('original agent configuration differs')
    return dict(agent)


def load_cube_policy(release, files, spec, device):
    # run_state contains Torch RNG tensors. Read it in a short child before importing
    # JAX here, keeping the two large runtimes out of the same resident process.
    code = "import json,pickle,sys; s=pickle.load(open(sys.argv[1],'rb')); print(json.dumps({k:s[k] for k in ('network_step','adam_count','binding')}))"
    sidecar = json.loads(subprocess.check_output([sys.executable, '-c', code, str(files[spec['run_state']])], text=True))
    metadata = read(files[spec['agent_config']])
    manifest = read(files[spec['manifest']])
    counter = 1000000
    if manifest['counter'] != counter or manifest['agent_sha256'] != sha(files[spec['path']]) or manifest['sidecar_sha256'] != sha(files[spec['run_state']]):
        raise ValueError('original policy checkpoint transaction differs')
    cubes = TASKS[release['task']]
    config = policy_config(metadata, sidecar, cubes)
    os.environ['JAX_PLATFORMS'] = 'cpu' if device == 'cpu' else 'cuda'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['JAX_THREEFRY_PARTITIONABLE'] = 'false'
    from bench.ogbench.downstream._source import install_impls
    install_impls()
    import jax
    import numpy as np
    from bench.ogbench.downstream.sampler import create_agent
    from bench.ogbench.downstream.vendor.gciql import GCIQLAgent
    from bench.ogbench.gate_checks import _matrix_restore_policy
    from utils.flax_utils import restore_agent
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        config[key] = tuple(config[key])
    batch = dict(observations=np.linspace(-0.1, 0.1, 211, dtype=np.float32)[None],
                 actor_goals=np.linspace(-0.1, 0.1, 192 + 3 * cubes, dtype=np.float32)[None],
                 actions=np.zeros((1, 5), dtype=np.float32))
    template = create_agent(GCIQLAgent, 0, batch, config)
    checkpoint = dict(path=str(files[spec['path']].parent), counter=counter, sha256=manifest['agent_sha256'])
    agent, _, proof = _matrix_restore_policy(template, checkpoint, restore_agent)
    def forward():
        return np.asarray(agent.sample_actions(batch['observations'], batch['actor_goals'], seed=jax.random.PRNGKey(0), temperature=0))
    return dict(loader='original create_agent / restore_agent / _matrix_restore_policy', strict_state_dict=True,
                state_leaves=proof['full_numeric_leaves'], checkpoint_counter=counter, output=outputs(forward(), forward()))


def load_policy(release, files, seed, device):
    candidates = [p for p in release['policies'] if p['seed'] == seed]
    if len(candidates) != 1:
        raise ValueError('choose one published policy --seed')
    spec = candidates[0]
    if spec['kind'] == 'cube_gciql' and TASKS[release['task']]:
        return load_cube_policy(release, files, spec, device)
    expected = {'adroit_relocate': ('native_adroit_bc', 606, 30), 'rh56_relocate': ('rh56_bc', 589, 12)}
    if release['task'] not in expected or spec['kind'] != expected[release['task']][0]:
        raise ValueError('unsupported policy kind for this task')
    import pickle
    import numpy as np
    import torch
    from bench.relocate.policy_evidence import policy_actions
    with files[spec['path']].open('rb') as stream:
        policy = pickle.load(stream)  # hash-pinned, original mjrl policy and network classes
    _, width, action_dim = expected[release['task']]
    if (policy.n, policy.m) != (width, action_dim):
        raise ValueError('original BC policy dimensions differ')
    torch.set_num_threads(1)
    if device != 'cpu':
        raise ValueError('original pickled BC head smoke currently supports CPU only')
    policy.model.cpu().eval()
    feature = np.linspace(-0.1, 0.1, width, dtype=np.float32)
    with torch.inference_mode():
        first, second = policy_actions(policy, feature[None])[0], policy_actions(policy, feature[None])[0]
    if np.asarray(first).shape != (action_dim,):
        raise ValueError('BC action shape differs')
    return dict(loader=type(policy).__module__ + '.' + type(policy).__name__, serialization='original pickle',
                input_shape=[width], output=outputs(first, second))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--component', choices=('wm', 'policy'), required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True, help='new JSON receipt; existing files are refused')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(schema='jepa4r_checkpoint_load_smoke_v1', started_at=datetime.now(timezone.utc).isoformat(),
                  component=args.component, seed=args.seed, device=args.device, environment_steps=0, optimizer_updates=0,
                  script_sha256=sha(__file__), python=sys.version, executable=sys.executable, status='failed', actual_exit_code=1)
    try:
        release, files = verify_bundle(read(args.bundle))
        report.update(task=release['task'], model=release['model'], revision=release['revision'], verified_files=release['files'],
                      bundle_sha256=sha(args.bundle), registry_sha256=sha(REGISTRY))
        sys.path[:0] = [str(REPO / 'lewm_pcd'), str(REPO / 'lewm_pcd/third_party/stable-worldmodel')]
        report['result'] = load_wm(release, files, args.device) if args.component == 'wm' else load_policy(release, files, args.seed, args.device)
        report.update(status='passed', actual_exit_code=0)
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    report['versions'] = {}
    for package in ('torch', 'numpy', 'jax', 'jaxlib', 'flax', 'optax', 'hydra-core', 'stable-pretraining'):
        try:
            report['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, allow_nan=False))
    return report['actual_exit_code']


if __name__ == '__main__':
    raise SystemExit(main())

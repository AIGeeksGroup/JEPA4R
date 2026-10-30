"""PLDM1/4 bindings over the frozen shared GCIQL and Native implementations."""
import argparse
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import types

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
MODULES = ('inputs', 'prepare', 'runtime', 'audit_native', 'native_sr')
SOURCE_SHA = {
    'inputs': 'ff66c68f50a6c73bd7928e4281a30adf4686538a2829d31d6981a4fd86b5f2ec',
    'prepare': '17141ea3dffd485455f9a54dbcb52a3b569c583b2f1326d7074d2d96b4a78464',
    'runtime': '03986a4c4dd03598b9ae06e57955ce164ad3a73fcc655dcc73acf12df0632326',
    'audit_native': 'c2840c183d9279623be16b5f1e882a4b121a856039fd74e8f4584fca82f8501a',
    'native_sr': 'ae16abe24a609719630357f69c2c472d4df2f58a63f044edd9fbe028f4f10e41',
}
EDITS = {
    'inputs': (('dim in (768, 1024)', 'dim in (512, 768, 1024)'),),
    'prepare': (('d in (768,1024)', 'd in (512,768,1024)'),),
    'runtime': (('d in (768,1024)', 'd in (512,768,1024)'),),
    'native_sr': ((
        '                out=self.adapter.encode_uint8(self.model,torch.from_numpy(np.ascontiguousarray(value[None])))',
        "                if encoder['family']=='pldm':\n"
        '                    out=self.model.encode_uint8(torch.from_numpy(np.ascontiguousarray(value[None])))\n'
        '                else:\n'
        '                    out=self.adapter.encode_uint8(self.model,torch.from_numpy(np.ascontiguousarray(value[None])))'),),
}
_MISSING = object()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def effective_source(name, *, pldm=True):
    path = ROOT/'foundation_visual_gciql_v1/functional'/f'{name}.py'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA[name], str(path)
    source = raw.decode()
    if pldm:
        for before, after in EDITS.get(name, ()):
            assert source.count(before) == 1, (name, before)
            source = source.replace(before, after, 1)
    return source


def recipe():
    return dict(schema_version=1, functional_path=str(Path(__file__).resolve()), functional_SHA256=sha(__file__),
        sources={name:dict(original_SHA256=SOURCE_SHA[name],
            effective_SHA256=hashlib.sha256(effective_source(name).encode()).hexdigest(),
            counted_edits=[list(pair) for pair in EDITS.get(name, ())]) for name in MODULES},
        wrappers='PLDM config/path dispatch, restored module aliases, original runtime.prepare trace and shared Native source binding; original loops retained')


@contextmanager
def activated(modules):
    """Keep delayed imports on this module set, then restore unrelated aliases."""
    previous = {name:sys.modules.get(name, _MISSING) for name in MODULES}
    previous_path = list(sys.path)
    try:
        sys.modules.update(modules)
        yield
    finally:
        for name, module in previous.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.path[:] = previous_path


def bound(function, modules, *, native=False):
    @functools.wraps(function)
    def call(*args, **kwargs):
        if native:
            os.environ.update(JAX_PLATFORMS='cpu', MUJOCO_GL='egl')
        with activated(modules):
            return function(*args, **kwargs)
    return call


def check_pldm(cfg):
    n = cfg['cube_count']
    assert n in (1, 4) and cfg['group_id'] == f'pldm_cube{n}'
    assert cfg['encoder_binding']['family'] == 'pldm' and cfg['encoder_binding']['latent_dim'] == 512
    assert cfg['input_contract'] == dict(current_dim=531,goal_dim=512+3*n,latent_dim=512,zero_padding=False)
    assert cfg['native_max_episode_steps'] == {1:200,4:1000}[n]
    assert cfg['fresh'] and cfg['reward'] == dict(rule='ALL(each cube L2<=0.4 scaled=4cm)',values=[-1,0],mask='1-success')


def cache_pins(cfg):
    """Metadata-only closure/identity checks, including original asset code pins."""
    check_pldm(cfg)
    read = lambda path: json.loads(Path(path).read_text())
    binding = cfg['cache_plan_binding']
    path = Path(binding['path'])
    assert sha(path) == binding['sha256']
    plan = read(path)
    encoder = cfg['encoder_binding']
    assert plan['cube_count'] == cfg['cube_count'] and plan['encoder_family'] == 'pldm' and plan['latent_dim'] == 512
    assert plan['checkpoint_sha256'] == encoder['checkpoint_sha256']
    assert plan['encoder_spec'] == encoder['spec_path'] and plan['encoder_spec_sha256'] == encoder['spec_sha256']
    assert plan['encoder_adapter'] == encoder['adapter']['path'] and plan['encoder_adapter_sha256'] == encoder['adapter']['sha256']
    assert plan['inference_precision'] == 'fp32' and plan['encoder_preprocessing_device'] == 'cpu'
    pins = {str(Path(__file__).resolve()):sha(__file__), str(path):binding['sha256']}
    for relative, digest in plan['source_pins'].items():
        rel = PurePosixPath(relative)
        assert not rel.is_absolute() and '..' not in rel.parts
        pins[str(ROOT.joinpath(*rel.parts))] = digest
    delivery_binding = cfg['delivery_binding']
    delivery_path = Path(delivery_binding['path'])
    assert sha(delivery_path) == delivery_binding['sha256']
    delivery = read(delivery_path)
    assert delivery['purpose'] == 'formal' and delivery['cube_count'] == cfg['cube_count']
    assert delivery['encoder_binding'] == encoder and delivery['feature_loader_arguments'] == cfg['feature_loader_arguments']
    assert delivery['camera'] == cfg['camera'] and delivery['cache_audit'] == cfg['cache_audit']
    audit = read(delivery['cache_audit']['path'])
    assert audit['actual_exit_code'] == 0 and audit['episodes_verified'] == sum(plan['split_episodes'].values())
    pins[str(delivery_path)] = delivery_binding['sha256']
    pins[delivery['cache_audit']['path']] = delivery['cache_audit']['sha256']
    # Both groups must have completed the same original six-phase embedding job.
    control = ROOT/'pldm_cube14_embeddings_v1/control/full'
    closure_path = control/'runtime/closure.json'
    closure = read(closure_path)
    assert closure['actual_exit_code'] == 0 and closure['completed_phases'] == 6
    assert closure['training_updates'] == closure['SR_episodes'] == 0
    pins[str(closure_path)] = sha(closure_path)
    for end in (control/'runtime/outer_exit_code',control/'wrapper_exit_code'):
        assert end.read_text().strip() == '0'
        pins[str(end)] = sha(end)
    for candidate, digest in pins.items():
        assert sha(candidate) == digest, candidate
    return pins


def bootstrap(*, pldm=True):
    """Return compatible modules; loading this API creates no model or optimizer."""
    modules = {}
    env = {name:os.environ.get(name, _MISSING) for name in ('JAX_PLATFORMS','MUJOCO_GL')}
    try:
        with activated(modules):
            for name in MODULES:
                path = ROOT/'foundation_visual_gciql_v1/functional'/f'{name}.py'
                module = types.ModuleType(('_pldm14_' if pldm else '_original14_') + name)
                module.__file__ = str(path)
                modules[name] = module
                sys.modules[name] = sys.modules[module.__name__] = module
                exec(compile(effective_source(name, pldm=pldm),str(path),'exec'),module.__dict__)
                if name == 'prepare':
                    module.ROOT = ROOT
                    if pldm:
                        module.CASE = ROOT/'pldm_cube14_gciql_v1'
    finally:
        for name, value in env.items():
            if value is _MISSING:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if pldm:
        prepare, runtime, audit = (modules[name] for name in ('prepare','runtime','audit_native'))
        original_config, original_commands = prepare.make_config, prepare.commands
        def make_config(template, delivery):
            cfg = original_config(template, delivery)
            check_pldm(cfg)
            return cfg
        def commands(config_path, cfg):
            result = original_commands(config_path, cfg)
            for key, argv in result.items():
                result[key] = [str(Path(__file__).resolve()), 'runtime' if Path(argv[0]).name == 'runtime.py' else 'native', *argv[1:]]
            result['native_audit'] = [str(Path(__file__).resolve()),'audit',*[arg for arg in result['native'][2:] if arg != '--gate-smoke']]
            return result
        prepare.make_config, prepare.commands = make_config, commands
        original_check = runtime.check_config
        def check_config(cfg):
            original_check(cfg)
            check_pldm(cfg)
        runtime.check_config = check_config
        original_install = runtime.install
        def install(cfg):
            installed, boundary = original_install(cfg)
            original_prepare = installed.prepare
            def traced_prepare(config, shared_datasets=None):
                pins = cache_pins(config)
                samplers, resources, binding = original_prepare(config, shared_datasets)
                binding['code'].update(pins)
                binding['code'][str(Path(__file__).resolve())] = sha(__file__)
                binding['runtime']['functional_overlay'] = recipe()
                return samplers, resources, binding
            installed.prepare = bound(traced_prepare, modules)
            boundary.prepare, boundary.get_batch = installed.prepare, installed.get_batch
            return installed, boundary
        runtime.install = install
        original_binding = audit.source_binding
        def source_binding(cfg, config_path):
            pins = cache_pins(cfg)
            binding = original_binding(cfg, config_path)
            binding['source_pins'].update(pins)
            binding['functional_overlay'] = recipe()
            return binding
        audit.source_binding = source_binding
    # Main stays active for the entire original training/evaluation/audit call.
    for name in MODULES:
        if hasattr(modules[name], 'main'):
            modules[name].main = bound(modules[name].main, modules, native=name=='native_sr')
    modules['runtime'].install = bound(modules['runtime'].install, modules)
    modules['native_sr'].install_core = bound(modules['native_sr'].install_core, modules, native=True)
    return modules


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('mode', choices=('runtime','native','audit'))
    args, remaining = parser.parse_known_args(argv)
    modules = bootstrap()
    target = {'runtime':'runtime','native':'native_sr','audit':'audit_native'}[args.mode]
    sys.argv = [str(Path(__file__).resolve()),*remaining]
    return modules[target].main()


if __name__ == '__main__':
    raise SystemExit(main())

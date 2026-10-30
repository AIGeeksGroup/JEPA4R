"""PLDM512/CPU binding over the frozen foundation producer and full CPU audit."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
PRODUCER = 'pldm_cube14_embeddings_v1/cache_bridge.py'
FOUNDATION_SHA = {
    'producer': '2a366174041abd0a21cb6b299f9584601f27a32bfead141f3ced5a8228424e93',
    'audit': '9dfe4cd0fff48de33c6bc37e442fc631fb34a2724a432a04b89d7041a4a837d7',
}
COMMON_SHA = '061812303cc4b7ba74c11e817fc20cfc61c86e4f8828fe9e840c0a9fd2717f90'
PLDM2_ADAPTER = 'cube2_pldm_gciql_v1/cache_v1/encoder_adapter.py'
PLDM2_ADAPTER_SHA = '72161520d1ddd839c8266ad5fea7490611f21d3681d261f86f4b993a14430a06'
EXTRA_KEYS = ('prefetch', 'inference_precision', 'storage_dtype', 'encoder_preprocess')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_once(source, before, after):
    assert source.count(before) == 1, ('Unexpected frozen-source anchor', before)
    return source.replace(before, after, 1)


def foundation_api(mode, expanded=False):
    assert mode in ('producer', 'audit')
    directory = ROOT / 'foundation_visual_gciql_v1/cache'
    assert sha(directory/'common.py') == COMMON_SHA
    sys.path.insert(0, str(directory))
    common = importlib.import_module('common')
    assert Path(common.__file__).resolve() == (directory/'common.py').resolve()
    name = mode + '_bridge'
    path = directory / (name + '.py')
    assert sha(path) == FOUNDATION_SHA[mode]
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve() == path.resolve()
    if expanded:
        source = replace_once(path.read_text(), 'd in (768,1024)', 'd in (512,768,1024)')
        module = types.ModuleType('_pldm14_' + name)
        module.__file__ = str(path)
        exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def validate_plan(plan):
    assert plan['cube_count'] in (1, 4) and plan['latent_dim'] == 512
    assert plan['encoder_family'] == 'pldm'
    assert plan['renderer_workers'] == plan['prefetch'] == 2
    assert plan['encoder_batch'] == 32 and plan['torch_cpu_threads'] == 1
    assert plan['pixel_preprocess_device'] == 'shared_cpu'
    assert plan['encoder_preprocessing_device'] == 'cpu' and plan['host_tensor_dtype'] == 'uint8'
    assert plan['inference_precision'] == 'fp32' and plan['storage_dtype'] == 'float32'
    assert plan['training_updates'] == plan['SR_episodes'] == 0


def effective_source(plan, mode='producer'):
    if plan['latent_dim'] != 512:
        original = foundation_api(mode)
        return (original.effective_source if mode == 'producer' else original.source)(plan)
    validate_plan(plan)
    original = foundation_api(mode, expanded=True)
    source = (original.effective_source if mode == 'producer' else original.source)(plan)
    if mode == 'producer':
        edits = [
            ("assert policy == 'shared_gpu'", "assert policy == 'shared_cpu'"),
            ("if policy == 'shared_gpu':\n        activate('cuda', None)", "if policy == 'shared_cpu':\n        activate('cpu', None)"),
            ("assert selected_device == 'cuda'  # shared encoder API; no CPU/GPU speed selection", "assert selected_device == 'cpu'  # shared encoder API; no CPU/GPU speed selection"),
        ]
    else:
        edits = [
            ("assert policy == 'shared_gpu', 'Shared encoder GPU preprocessing required'", "assert policy == 'shared_cpu', 'Shared PLDM CPU preprocessing required'"),
            ("selected_device = 'cuda'", "selected_device = 'cpu'"),
            ("if policy not in ('cpu', 'shared_gpu'):", "if policy not in ('cpu', 'shared_cpu'):"),
        ]
    for before, after in edits:
        source = replace_once(source, before, after)
    return source


def load(plan, mode):
    validate_plan(plan)
    spec = json.loads(Path(plan['encoder_spec']).read_text())
    for key in ('cube_count', 'checkpoint', 'checkpoint_sha256'):
        assert spec[key] == plan[key], 'Encoder spec/plan binding mismatch: ' + key
    api = foundation_api(mode, expanded=True)
    if mode == 'producer':
        api.effective_source = lambda value: effective_source(value, mode)
    else:
        api.source = lambda value: effective_source(value, mode)
    module = api.load(plan)
    module.__file__ = str(Path(__file__).resolve())
    if mode == 'producer':
        # Keep the original renderer location and arrays_existing callback.
        module.HERE = ROOT / 'foundation_visual_gciql_v1/cache'
        module.source_arrays = foundation_api('producer').arrays_existing
        module.CONTRACT_KEYS += EXTRA_KEYS
    else:
        module.PRODUCER = PRODUCER
        module.PROTOCOL_KEYS += EXTRA_KEYS
    return module


def load_encoder(spec, device='cuda'):
    """Model construction occurs only through the original producer execute path."""
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text())
    assert spec['schema_version'] == 1 and spec['family'] == 'pldm'
    assert spec['cube_count'] in (1, 4) and spec['output_dim'] == 512
    assert spec['inference_precision'] == 'fp32' and spec['preprocess_device'] == 'cpu'
    assert str(device).startswith('cuda'), 'Frozen encoder requires the declared CUDA device'
    root = Path(spec['asset_root']).resolve()
    assert root.name == spec['asset_id'] == f"cube{spec['cube_count']}_pldm_impala512_epoch10_20261023_v1"
    assert sha(root/'MANIFEST.json') == spec['manifest_sha256']
    manifest = json.loads((root/'MANIFEST.json').read_text())
    assert manifest['asset_id'] == spec['asset_id'] and manifest['cube_count'] == spec['cube_count']
    assert manifest['embedding_dim'] == 512 and manifest['epochs'] == 10
    checkpoint = root/'weights/weights_epoch_10.pt'
    assert Path(spec['checkpoint']).resolve() == checkpoint
    assert sha(checkpoint) == spec['checkpoint_sha256'] == manifest['members']['weights/weights_epoch_10.pt']['sha256']
    addendum_path = root/'RUNTIME_ADDENDUM.json'
    addendum_sha = spec['runtime_addendum_sha256']
    assert (addendum_sha is None) == (spec['cube_count'] == 1)
    if addendum_sha is None:
        assert not addendum_path.exists(), 'Unbound runtime addendum'
        extra_files = {}
    else:
        assert sha(addendum_path) == addendum_sha
        extra_files = json.loads(addendum_path.read_text())['files']
    sys.path.insert(0, str(root))
    for name in ('encoder', 'py312_config_compat', 'encoder_py312'):
        filename = name + '.py'
        expected = extra_files.get(filename, manifest['members'][filename]['sha256'])
        assert sha(root/filename) == expected, 'Asset executable source differs: ' + filename
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve() == root/filename, 'Asset module namespace collision: ' + name
    model = sys.modules['encoder_py312'].FrozenPLDM(root).to(device).eval().requires_grad_(False)
    assert len(model.model.state_dict()) == 44
    model.obs_key = 'pixels'
    adapter_path = ROOT/PLDM2_ADAPTER
    assert sha(adapter_path) == spec.get('pldm2_adapter_sha256', PLDM2_ADAPTER_SHA) == PLDM2_ADAPTER_SHA
    imported = importlib.util.spec_from_file_location('_pldm14_shared_adapter', adapter_path)
    adapter = importlib.util.module_from_spec(imported)
    imported.loader.exec_module(adapter)
    # Preserve the accepted CPU AA65 preprocessing and encode implementation verbatim.
    model.encode_uint8 = types.MethodType(
        lambda current, pixels: adapter.encode_uint8(current, pixels, preprocess_device='cpu'), model)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('producer', 'audit'), required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    plan = json.loads(args.plan.read_text())
    module = load(plan, args.mode)
    sys.argv = [str(Path(__file__).resolve()), '--plan', str(args.plan), *remaining]
    return module.main()


if __name__ == '__main__':
    main()

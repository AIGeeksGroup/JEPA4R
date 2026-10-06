"""Release leaf contracts; optional real CPU comparison to the pinned f1cc loader."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'bench/cortexbench'
LEAF = HERE / 'lewm_adroit_20261006.py'


@pytest.mark.parametrize('mode,variable,function', [
    ('2d', 'PEN_20261006_RGB_CHECKPOINT', 'load_lewm'),
    ('pcd', 'PEN_20261006_PCD_CHECKPOINT', 'load_lewm_pcd'),
])
def test_versioned_config_uses_explicit_checkpoint_variable(mode, variable, function):
    path = HERE / f'conf/model/lewm_adroit_pen_{mode}_20261006.yaml'
    assert path.exists(), 'new config is missing'
    text = path.read_text()
    assert f'_target_: lewm_adroit_20261006.{function}' in text
    assert f'checkpoint_dir: ${{oc.env:{variable}}}' in text
    assert '/root/' not in text and 'random_init: true' not in text


def test_bridge_reuses_old_classes_without_editing_old_api():
    assert LEAF.exists(), 'new bridge is missing'
    tree = ast.parse(LEAF.read_text())
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)
    assert {'build_lewm_embedding', 'load_lewm', 'load_lewm_pcd'} <= {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    old = (HERE / 'lewm_cortex.py').read_text()
    assert 'def _remap_vit_state_dict_for_runtime' in old
    assert 'random_init: bool = False, seed: int = 0' in old


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    torch = pytest.importorskip('torch')
    reference_path = os.environ.get('PEN_20261006_REFERENCE_LOADER')
    if not reference_path:
        pytest.skip('set PEN_20261006_REFERENCE_LOADER to exact f1cc loader for real CPU gate')
    reference_bytes = Path(reference_path).read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(reference_bytes).hexdigest() == 'e2223dd386c362fff774ff07164dea6c78fac047d4ddb4aea1bbcfbda45b0778'
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.syspath_prepend(str(HERE))
    import lewm_cortex
    bridge = _module(LEAF, 'adroit_release_bridge_test')
    reference = _module(reference_path, 'adroit_release_f1cc_reference')
    # The exported reference file is outside its original checkout; share the
    # identical encoder class source instead of resolving its relative path there.
    reference._dexart_pointnet_class = lewm_cortex._dexart_pointnet_class
    assert not torch.cuda.is_initialized(), 'this test is CPU only'
    torch.set_num_threads(2)
    return torch, bridge, reference


@pytest.mark.parametrize('mode,variable,function', [
    ('rgb', 'PEN_20261006_RGB_CHECKPOINT', 'load_lewm'),
    ('pcd', 'PEN_20261006_PCD_CHECKPOINT', 'load_lewm_pcd'),
])
def test_real_checkpoint_matches_f1cc_state_outputs_and_cpu_rng(runtime, mode, variable, function):
    torch, bridge, reference = runtime
    checkpoint = os.environ.get(variable)
    if not checkpoint:
        pytest.skip(f'set {variable} to the verified epoch10 checkpoint')
    kwargs = {} if mode == 'rgb' else dict(center=[.075, -.02, .225], scale=4., use_rgb=True)
    torch.manual_seed(3917)
    before = torch.get_rng_state().clone()
    expected = getattr(reference, function)(checkpoint, **kwargs)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    actual = getattr(bridge, function)(checkpoint, **kwargs)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert actual[1] == expected[1] == 192 and actual[3] == expected[3]
    assert actual[0].state_dict().keys() == expected[0].state_dict().keys()
    for key, value in actual[0].state_dict().items():
        assert value.device.type == 'cpu' and torch.isfinite(value).all()
        assert torch.equal(value, expected[0].state_dict()[key]), key
    shape = (2, 3, 224, 224) if mode == 'rgb' else (2, 1200, 6)
    inputs = torch.linspace(-1., 1., int(__import__('math').prod(shape))).reshape(shape)
    with torch.no_grad():
        assert torch.equal(actual[0](inputs), expected[0](inputs))
    if mode == 'rgb':
        import numpy as np
        from PIL import Image
        image = Image.fromarray(np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3))
        assert torch.equal(actual[2](image), expected[2](image))
        assert torch.equal(before, expected_rng), 'RGB must preserve caller CPU RNG'
    else:
        assert torch.equal(actual[2](inputs[0]), expected[2](inputs[0]))
        assert not torch.equal(before, expected_rng), 'PCD must retain native construction RNG consumption'
    assert not torch.cuda.is_initialized()

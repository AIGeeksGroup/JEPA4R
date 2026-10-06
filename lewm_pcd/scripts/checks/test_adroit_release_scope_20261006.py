"""Exercise the actual release dispatch expression without simulator imports."""
import ast
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _stats_dispatch(env_name):
    source = ast.parse((ROOT/'eval.py').read_text())
    node = next(n for n in ast.walk(source) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'stats_dataset' for t in n.targets))
    dataset, train = object(), object()
    calls = []
    def get_dataset(cfg, path):
        calls.append(path)
        return train
    cfg = SimpleNamespace(world=SimpleNamespace(env_name=env_name),
                          dataset=SimpleNamespace(stats='train-only'))
    scope = dict(cfg=cfg, dataset=dataset, get_dataset=get_dataset)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'release_dispatch', 'exec'), scope)
    return scope['stats_dataset'], dataset, train, calls


def test_existing_metaworld_normalization_identity_is_unchanged():
    result, dataset, _, calls = _stats_dispatch('envs.metaworld:swm/MWBinPick-v0')
    assert result is dataset and calls == []


def test_existing_dmc_or_similar_namespace_is_not_selected():
    for name in ('dmc_finger-v1', 'envs.adroit:swm/AdroitPen-v0'):
        result, dataset, _, calls = _stats_dispatch(name)
        assert result is dataset and calls == []


def test_only_new_fragment_task_uses_training_statistics():
    result, dataset, train, calls = _stats_dispatch('envs.adroit:swm/AdroitPenFragment-v0')
    assert result is train and result is not dataset and calls == ['train-only']


def test_native_and_fragment_registrations_are_separate():
    source = (ROOT/'envs/adroit/__init__.py').read_text()
    assert 'swm/AdroitPen-v0' in source and 'swm/AdroitRelocate-v0' in source
    assert "entry_point='envs.adroit.env:AdroitEnv'" in source
    assert "entry_point='envs.adroit.fragment_env:PenFragmentEnv'" in source
    assert 'class AdroitEnv' in (ROOT/'envs/adroit/env.py').read_text()
    assert 'class PenFragmentEnv' in (ROOT/'envs/adroit/fragment_env.py').read_text()

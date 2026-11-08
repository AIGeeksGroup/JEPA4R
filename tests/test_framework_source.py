import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def adapter():
    path = ROOT / 'lewm_pcd/scripts/baselines/adroit/train_pldm.py'
    spec = importlib.util.spec_from_file_location('pldm_source_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundled_pinned_source_can_be_verified_without_fabricated_git():
    module = adapter()
    source = ROOT / 'lewm_pcd/third_party/stable-worldmodel'
    assert module.framework_revision(source) == module.FRAMEWORK_COMMIT


def test_bundled_source_manifest_rejects_tampered_bytes(tmp_path):
    module = adapter()
    (tmp_path / 'code.py').write_text('changed', encoding='utf-8')
    (tmp_path / 'SOURCE.json').write_text(json.dumps({
        'repository': 'https://github.com/galilai-group/stable-worldmodel',
        'commit': module.FRAMEWORK_COMMIT,
        'files': [{'path': 'code.py', 'bytes': 7, 'sha256': '0' * 64}],
    }), encoding='utf-8')
    try:
        module.framework_revision(tmp_path)
    except RuntimeError as error:
        assert 'source digest' in str(error).lower()
    else:
        raise AssertionError('Changed upstream source accepted')


def test_lejepa_loads_the_final_renamed_native_entry(monkeypatch):
    path = ROOT / 'lewm_pcd/scripts/baselines/adroit/train_lejepa.py'
    spec = importlib.util.spec_from_file_location('lejepa_entry_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = []
    def load(name):
        loaded.append(name)
        assert name == 'scripts.baselines.train_lejepa_relocate'
        return SimpleNamespace(__file__=str(module.NATIVE_SCRIPT))
    monkeypatch.setattr(module.importlib, 'import_module', load)
    assert module._load_native().__file__ == str(module.NATIVE_SCRIPT)
    assert len(loaded) == 1

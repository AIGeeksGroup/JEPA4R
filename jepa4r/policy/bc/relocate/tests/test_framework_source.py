# Moved from OGBench-cube@4fbb0c6:tests/test_framework_source.py (verbatim except: module names / paths of the moved code adapted)
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[5]


def adapter():
    path = ROOT / 'jepa4r/baselines/relocate/train_pldm.py'
    spec = importlib.util.spec_from_file_location('pldm_source_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundled_pinned_source_can_be_verified_without_fabricated_git():
    module = adapter()
    # the release bundled a copy with SOURCE.json (which no longer matched its stable_worldmodel/__init__.py); this
    # repository pins the 6f1e499 checkout of the PLDM framework instead (JEPA4R_SWM, README)
    source = Path(os.environ.get('JEPA4R_SWM') or ROOT / 'third_party/stable-worldmodel')
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
    path = ROOT / 'jepa4r/baselines/relocate/train_lejepa.py'
    spec = importlib.util.spec_from_file_location('lejepa_entry_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = []
    def load(name):
        loaded.append(name)
        assert name == 'jepa4r.baselines.relocate.train_lejepa_relocate'
        return SimpleNamespace(__file__=str(module.NATIVE_SCRIPT))
    monkeypatch.setattr(module.importlib, 'import_module', load)
    assert module._load_native().__file__ == str(module.NATIVE_SCRIPT)
    assert len(loaded) == 1

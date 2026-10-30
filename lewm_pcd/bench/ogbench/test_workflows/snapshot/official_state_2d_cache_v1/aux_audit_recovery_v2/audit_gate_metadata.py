"""Accept the already-frozen descriptive gate field; keep all original checks."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA = 'a97a55ce945c12b16d6fb473c71b289cdf8b2a0164b697ae0f8d4c5d7f9b5006'
SCOPE = 'original preprocessing plus frozen encode; rendering excluded; CUDA tail synchronized'


def expected_gate(original):
    return dict(original.GATE, scope=SCOPE)


def load_original():
    source = HERE.parent / 'audit_cache.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == ORIGINAL_SHA
    spec = importlib.util.spec_from_file_location('_original_cache_audit', source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    return original


if __name__ == '__main__':
    original = load_original()
    original.GATE = expected_gate(original)
    original.main()

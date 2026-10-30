"""Resource-metadata revision only; retain the frozen original preparation guards."""
import argparse
import hashlib
import json
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA = 'd4a1415be31368ea0bfd6d4910eaae462cc834e8dac496be22b8477e483f037c'


def load_original(case):
    path = HERE/'prepare.py'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA
    module = types.ModuleType('_P3_frozen_prepare_metadata_v2')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    original_read = module.read
    old_template = case/'functional_v1/configs/P3_cube3_pldm_rgb.TEMPLATE.json'
    new_template = case/'functional_v1/configs/P3_cube3_pldm_rgb.TEMPLATE_v2.json'
    def read(path, records, digest=None):
        path = Path(path)
        if path.resolve() == old_template.resolve():
            path = new_template
        return original_read(path, records, digest)
    def source_pins(case):
        frozen = json.loads((case/'functional_v1/SOURCE_FREEZE_v2.json').read_text())
        pins = {str(case/'functional_v1'/name):digest for name,digest in frozen['files'].items()}
        pins.update(frozen['original_pins'])
        for target, digest in pins.items():
            assert module.sha(target) == digest, target
        return pins
    module.read, module.source_pins = read, source_pins
    return module


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train','native'))
    parser.add_argument('--case', type=Path, required=True)
    args = parser.parse_args()
    case = args.case.resolve()
    assert case.name == 'cube3_pldm_gciql_v1'
    module = load_original(case)
    print(json.dumps((module.prepare_train if args.phase == 'train' else module.prepare_native)(case)))

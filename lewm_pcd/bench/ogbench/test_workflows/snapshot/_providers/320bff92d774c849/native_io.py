"""Only expose a DataLoader prefetch override around the unchanged entry."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def module(case, prefetch):
    path = case / 'source/train_lejepa_mw.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '64caecbd3f2282d38c159018cb3d6f204079bc60ccecfa41b231d00332937b51'
    spec = importlib.util.spec_from_file_location('native_lejepa_io', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    original = m.DataLoader

    class PrefetchLoader(original):
        def __init__(self, *args, **kwargs):
            kwargs['prefetch_factor'] = prefetch
            super().__init__(*args, **kwargs)
            receipt = os.environ.get('IO_PROFILE_RECEIPT')
            if receipt:
                with Path(receipt).open('x') as f:
                    json.dump({'workers': self.num_workers, 'prefetch_factor': self.prefetch_factor,
                               'batch': self.batch_size, 'persistent_workers': self.persistent_workers,
                               'pin_memory': self.pin_memory, 'drop_last': self.drop_last,
                               'dataset_frames': len(self.dataset), 'sampler': type(self.sampler).__name__}, f, indent=2)

    m.DataLoader = PrefetchLoader
    return m


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--prefetch', type=int, required=True)
    p.add_argument('native_args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    assert a.prefetch > 0
    m = module(a.case, a.prefetch)
    tail = a.native_args[1:] if a.native_args[:1] == ['--'] else a.native_args
    sys.argv = [m.__file__] + tail
    m.main()

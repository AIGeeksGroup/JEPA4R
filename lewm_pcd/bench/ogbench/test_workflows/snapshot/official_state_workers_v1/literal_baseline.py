"""One control without the diagnostic reader's per-batch RNG-state snapshot."""
from pathlib import Path
import sys
import measure


class LiteralInput:
    def __init__(self, dataset, mode='baseline', batch_size=1024):
        assert mode == 'baseline' and batch_size == 1024
        self.dataset = dataset

    def sample(self, batch_size=1024, *, prefetch_next=True):
        return self.dataset.sample(batch_size)

    def barrier(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


if __name__ == '__main__':
    assert sys.argv[sys.argv.index('--stage') + 1] == 'benchmark'
    original_identity = measure.identity

    def identity(args, cfg):
        return dict(**original_identity(args, cfg), literal_official_sampler_control=True,
                    control_wrapper_sha256=measure.sha(Path(__file__)))

    measure.identity = identity
    measure.OrderedInput = LiteralInput
    measure.main()

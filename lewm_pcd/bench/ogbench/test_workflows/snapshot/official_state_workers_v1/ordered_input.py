"""Default-off, one-batch ordered materialization for the official state sampler.

All sampling and NumPy RNG run on the caller thread. Background threads only
index frozen NumPy arrays. Call barrier BEFORE validation/evaluation/save or
any other external NumPy RNG use; it discards lookahead and restores host RNG.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
import threading

import numpy as np


ROLES = ('observations', 'next_observations', 'value_goals', 'actor_goals')
MODES = {'baseline': 0, 'prefetch1': 0, 'workers2': 2, 'workers4': 4}


def _same_rng(left, right):
    return left[0] == right[0] and left[2:] == right[2:] and np.array_equal(left[1], right[1])


@dataclass(frozen=True)
class _Gather:
    table: np.ndarray
    indices: np.ndarray

    def read(self):
        return self.table[self.indices]


class _Array:
    def __init__(self, table):
        self.table = table

    def __getitem__(self, indices):
        # Own indices: future caller mutation cannot change a queued batch.
        return _Gather(self.table, np.array(indices, copy=True))


class _DeferredDataset:
    """Reuse original Dataset methods, replacing only ndarray row indexing."""
    def __init__(self, host):
        self.host = host
        self._dict = {key: _Array(value) for key, value in host._dict.items()}

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __getitem__(self, key):
        return self._dict[key]

    def sample(self, batch_size, idxs=None):
        return type(self.host).sample(self, batch_size, idxs)

    def get_subset(self, idxs):
        return type(self.host).get_subset(self, idxs)


@dataclass
class _Pending:
    future: object
    before: tuple
    after: tuple


class OrderedInput:
    """Four opt-in modes; fixed B1024, one pending batch, no sampler rewrite.

    sample(prefetch_next=False) avoids unused work at a known final/boundary
    step. Otherwise barrier() waits for and discards exactly one future batch.
    While pending, host RNG is ahead by one sample; committed_rng_state is the
    state after the last consumed batch. barrier makes global RNG match it.
    """
    def __init__(self, dataset, mode='baseline', batch_size=1024):
        if mode not in MODES or batch_size != 1024:
            raise ValueError('Supported modes require the official B1024 contract')
        self.dataset = dataset
        self.mode = mode
        self.batch_size = batch_size
        self._owner = threading.get_ident()
        self._closed = False
        self._pending = None
        self._committed = np.random.get_state()
        self._materializer = self._readers = None
        if mode == 'baseline':
            return
        if dataset.config['frame_stack'] is not None or dataset.config['p_aug'] != 0:
            raise ValueError('Deferred indexing requires frame_stack=None and p_aug=0')
        if any(not isinstance(value, np.ndarray) or value.flags.writeable
               for value in dataset.dataset._dict.values()):
            raise ValueError('Deferred indexing requires flat frozen NumPy arrays')
        for key, width in (('observations', 46), ('actions', 5)):
            value = dataset.dataset[key]
            if value.dtype != np.float32 or value.shape != (dataset.size, width):
                raise ValueError(f'Unexpected official state field: {key}')
        self._deferred = copy.copy(dataset)
        self._deferred.dataset = _DeferredDataset(dataset.dataset)
        self._materializer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ordered-materialize')
        if MODES[mode]:
            self._readers = ThreadPoolExecutor(max_workers=MODES[mode], thread_name_prefix='ordered-rows')

    @property
    def pending(self):
        return self._pending is not None

    @property
    def committed_rng_state(self):
        return copy.deepcopy(self._committed)

    def _check_owner(self):
        if threading.get_ident() != self._owner or self._closed:
            raise RuntimeError('OrderedInput requires its open caller thread')

    def _check_rng(self):
        if self._pending is not None and not _same_rng(np.random.get_state(), self._pending.after):
            raise RuntimeError('External NumPy RNG use with lookahead; barrier was required first')

    def _materialize(self, batch):
        futures = {} if self._readers is None else {
            key: self._readers.submit(batch[key].read) for key in ROLES}
        # Only these four independent RAM role reads go to the read workers.
        # Actions and metadata retain their NumPy dtype and batch key order.
        return {key: futures[key].result() if key in futures else
                value.read() if isinstance(value, _Gather) else value
                for key, value in batch.items()}

    def _plan(self):
        before = np.random.get_state()
        try:
            batch = self._deferred.sample(self.batch_size)
            after = np.random.get_state()
            return _Pending(self._materializer.submit(self._materialize, batch), before, after)
        except BaseException:
            np.random.set_state(before)
            raise

    def sample(self, batch_size=None, *, prefetch_next=True):
        self._check_owner()
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError('The configured batch size cannot change')
        if self.mode == 'baseline':
            batch = self.dataset.sample(self.batch_size)
            self._committed = np.random.get_state()
            return batch
        self._check_rng()
        if self._pending is None:
            self._pending = self._plan()
        current = self._pending
        try:
            batch = current.future.result()
        except BaseException:
            np.random.set_state(current.before)
            self._pending = None
            raise
        self._pending = None
        try:
            following = self._plan() if prefetch_next else None
        except BaseException:
            # The caller has not received current yet, so retry must replay it.
            np.random.set_state(current.before)
            raise
        self._committed = current.after
        self._pending = following
        return batch

    def barrier(self):
        self._check_owner()
        self._check_rng()
        if self._pending is not None:
            current = self._pending
            try:
                current.future.result()
            finally:
                np.random.set_state(current.before)
                self._pending = None
        self._committed = np.random.get_state()

    def close(self):
        if self._closed:
            return
        try:
            self.barrier()
        finally:
            if self._materializer is not None:
                self._materializer.shutdown(wait=True, cancel_futures=True)
            if self._readers is not None:
                self._readers.shutdown(wait=True, cancel_futures=True)
            self._pending = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

"""Opt-in PLDM read32 tasks merged back into the unchanged logical B128 stream."""
import math
import time


class _ReadTaskBatchSampler:
    def __init__(self, parent, read_batch_size):
        self.parent, self.read_batch_size = parent, read_batch_size

    def __iter__(self):
        for batch in self.parent:
            for start in range(0, len(batch), self.read_batch_size):
                yield batch[start:start+self.read_batch_size]

    def __len__(self):
        full, tail = divmod(len(self.parent.sampler), self.parent.batch_size)
        return full*math.ceil(self.parent.batch_size/self.read_batch_size) + (
            math.ceil(tail/self.read_batch_size) if tail and not self.parent.drop_last else 0)


class _MergeIterator:
    def __init__(self, outer, inner_loader, inner_iterator):
        self._outer, self._inner_loader, self._inner_iterator = outer, inner_loader, inner_iterator
        self._index, self._closed = 0, False
        self._count = len(outer)
        self._stats = outer.readtask_statistics
        self._stats['active_iterators'] += 1

    def __iter__(self): return self

    def __next__(self):
        if self._closed: raise StopIteration
        if self._index == self._count:
            try:
                next(self._inner_iterator)
            except StopIteration:
                self.close(); raise
            self.close(); raise RuntimeError('inner returned extra read tasks')
        import torch
        target = self._outer.batch_size
        if not self._outer.drop_last and self._index == self._count-1:
            target = len(self._outer.sampler) % target or target
        merged, filled = None, 0
        measured = self._outer._readtask_measure
        try:
            while filled < target:
                before = time.perf_counter() if measured else None
                try: piece = next(self._inner_iterator)
                except StopIteration as error:
                    raise RuntimeError('inner ended inside an original batch') from error
                if measured: self._stats['read_wait_seconds'] += time.perf_counter()-before
                if not isinstance(piece, dict) or not {'pixels','actions','sample_id'} <= piece.keys():
                    raise ValueError('PLDM dictionary batch required')
                size = piece['sample_id'].shape[0]
                if not 0 < size <= 32 or filled+size > target:
                    raise ValueError('read task crossed an original batch boundary')
                before = time.perf_counter() if measured else None
                if merged is None:
                    if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' for t in piece.values()):
                        raise ValueError('only CPU tensor batch fields supported')
                    merged = {k:torch.empty((target,*t.shape[1:]),dtype=t.dtype,
                        device='cpu',pin_memory=self._stats['pinned_output']) for k,t in piece.items()}
                if piece.keys() != merged.keys(): raise ValueError('batch fields changed')
                for key, value in piece.items():
                    out = merged[key]
                    if value.dtype!=out.dtype or value.shape[0]!=size or value.shape[1:]!=out.shape[1:]:
                        raise ValueError('read task tensor contract changed')
                    out[filled:filled+size].copy_(value)
                if measured: self._stats['merge_seconds'] += time.perf_counter()-before
                self._stats['read_tasks'] += 1; filled += size
            self._index += 1
            self._stats['merged_batches'] += 1; self._stats['merged_samples'] += target
            return merged
        except BaseException:
            self.close(); raise

    def close(self):
        if not self._closed:
            self._closed = True
            shutdown = getattr(self._inner_iterator, '_shutdown_workers', None)
            try:
                if shutdown is not None: shutdown()
            finally:
                self._stats['active_iterators'] -= 1
                self._stats['closed_iterators'] += 1

    def __del__(self):
        try: self.close()
        except Exception: pass


class _ReadTaskLoaderMixin:
    def __iter__(self):
        import torch
        # Read current fields: Lightning may inject worker_init_fn after wrap_loader.
        kwargs = dict(dataset=self.dataset, batch_sampler=_ReadTaskBatchSampler(self.batch_sampler,32),
            num_workers=self.num_workers, prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn, worker_init_fn=self.worker_init_fn,
            generator=self.generator, timeout=self.timeout,
            multiprocessing_context=self.multiprocessing_context,
            persistent_workers=False, pin_memory=False)
        if hasattr(self,'in_order'): kwargs['in_order'] = self.in_order
        inner = torch.utils.data.DataLoader(**kwargs)
        # This is deliberately eager: native base_seed is drawn before sampler next().
        iterator = iter(inner)
        available = bool(self.pin_memory and torch.cuda.is_available())
        self.readtask_statistics.update(pin_requested=bool(self.pin_memory),
            pinned_output=available, pin_validation_available=available)
        return _MergeIterator(self, inner, iterator)


def wrap_loader(native_loader, *, workers, prefetch_factor, read_batch_size=32, measure=False):
    """Return the same DataLoader object; preserve dataset/sampler/B128 semantics."""
    import torch
    if not isinstance(native_loader,torch.utils.data.DataLoader): raise TypeError('DataLoader required')
    if getattr(native_loader.dataset,'method',None) != 'pldm': raise ValueError('only PLDM is supported')
    if native_loader.batch_size != 128 or read_batch_size != 32:
        raise ValueError('logical B128/read32 contract required')
    if workers <= 0 or prefetch_factor <= 0: raise ValueError('positive worker/prefetch values required')
    if native_loader.persistent_workers or not getattr(native_loader,'in_order',True):
        raise ValueError('native persistentFalse/in_orderTrue required')
    if hasattr(native_loader,'_readtask_measure'): raise ValueError('loader already split; do not wrap twice')
    if not isinstance(native_loader.batch_sampler,torch.utils.data.BatchSampler):
        raise ValueError('original native BatchSampler required')
    native_loader.num_workers, native_loader.prefetch_factor = int(workers), int(prefetch_factor)
    native_loader._readtask_measure = bool(measure)
    native_loader.readtask_statistics = dict(workers=int(workers),prefetch_factor=int(prefetch_factor),
        read_batch_size=32,logical_batch_size=128,merged_batches=0,merged_samples=0,read_tasks=0,
        merge_seconds=0.0 if measure else None,read_wait_seconds=0.0 if measure else None,
        pin_requested=bool(native_loader.pin_memory),pinned_output=False,
        pin_validation_available=False,active_iterators=0,closed_iterators=0)
    native_loader.__class__ = type('ReadTask32'+type(native_loader).__name__,
        (type(native_loader),),{'__iter__':_ReadTaskLoaderMixin.__iter__})
    return native_loader

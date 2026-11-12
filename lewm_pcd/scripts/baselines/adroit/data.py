"""Relocate HDF handle lifecycle only; native indices, arrays and RNG are preserved."""
from contextlib import contextmanager
from pathlib import Path


def close_parent_hdf(dataset):
    """Close inherited HDF handles without changing cached values or sampling."""
    visited, handles = set(), set()
    def close(value):
        if value is None or id(value) in visited:
            return
        visited.add(id(value))
        for key in ('dataset', 'ds', '_reader'):
            child = getattr(value, key, None)
            if child is not value:
                close(child)
        handle = getattr(value, 'h5_file', None)
        if handle is not None and callable(getattr(handle, 'close', None)):
            if id(handle) not in handles:
                handle.close()
                handles.add(id(handle))
            value.h5_file = None
    close(dataset)
    return len(handles)


@contextmanager
def dataset_handle_hook(swm_module, expected_path, recorder=None):
    """Scope handle cleanup to one HDF; restore the original loader in finally.

    PLDM's original normalizer and forward already handle its NaN action sentinel.
    LeJEPA reads every valid visual observation, including the terminal image.
    This adapter must not filter, reorder or replace either native index set.
    """
    target = Path(expected_path).expanduser().resolve()
    original = swm_module.data.load_dataset
    def load(*args, **kwargs):
        dataset = original(*args, **kwargs)
        path = getattr(dataset, 'h5_path', None)
        if path is not None and Path(path).expanduser().resolve() == target:
            closed = close_parent_hdf(dataset)
            lengths = getattr(dataset, 'lengths', ())
            receipt = {'dataset': str(target), 'episodes': len(lengths),
                       'source_rows': sum(int(x) for x in lengths),
                       'native_clip_count': len(dataset.clip_indices),
                       'num_steps': int(dataset.num_steps),
                       'frameskip': int(dataset.frameskip), 'span': int(dataset.span),
                       'native_indices_preserved': True,
                       'source_arrays_changed': False, 'random_draws_added': 0,
                       'parent_hdf_handles_closed': closed}
            if recorder is not None:
                recorder(receipt)
        return dataset
    swm_module.data.load_dataset = load
    try:
        yield
    finally:
        swm_module.data.load_dataset = original

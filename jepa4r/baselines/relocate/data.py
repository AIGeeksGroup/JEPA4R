"""The HDF handles of the Relocate training data in the PLDM / LeJEPA trainers."""

from contextlib import contextmanager
from pathlib import Path


def close_parent_hdf(dataset):
    """Close inherited HDF handles without changing cached values or sampling."""
    visited, handles = set(), set()

    def close(value):
        if value is None or id(value) in visited:
            return
        visited.add(id(value))
        for key in ("dataset", "ds", "_reader"):
            child = getattr(value, key, None)
            if child is not value:
                close(child)
        handle = getattr(value, "h5_file", None)
        if handle is not None and callable(getattr(handle, "close", None)):
            if id(handle) not in handles:
                handle.close()
                handles.add(id(handle))
            value.h5_file = None

    close(dataset)
    return len(handles)


@contextmanager
def dataset_handle_hook(swm_module, expected_path):
    """While the context is open, a dataset loaded from ``expected_path`` gets the HDF handles inherited from its
    parent closed (each loader worker opens its own); the loader is restored afterwards. Indices, arrays and sampling
    stay as the trainer makes them."""
    target = Path(expected_path).expanduser().resolve()
    original = swm_module.data.load_dataset

    def load(*args, **kwargs):
        dataset = original(*args, **kwargs)
        path = getattr(dataset, "h5_path", None)
        if path is not None and Path(path).expanduser().resolve() == target:
            close_parent_hdf(dataset)
        return dataset

    swm_module.data.load_dataset = load
    try:
        yield
    finally:
        swm_module.data.load_dataset = original

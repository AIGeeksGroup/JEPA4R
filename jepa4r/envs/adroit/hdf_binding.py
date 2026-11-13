# Moved from OGBench-cube@4fbb0c6:lewm_pcd/envs/adroit/hdf_binding.py (verbatim)
"""Read-only content binding for concrete HDF ExternalLink and VDS dependencies."""
import hashlib
import json
from pathlib import Path


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def hdf_dependencies(path):
    """Bind the header and its reachable files once, independent of storage paths.

    Logical edge names distinguish swapped backing files with the same set of
    hashes. Cycles, soft links, printf VDS sources and raw external storage fail
    closed because this reader does not infer their backing-file coverage.
    """
    import h5py
    records, active = {}, set()

    def node(value):
        source = Path(value).resolve(strict=True)
        if source in active:
            raise ValueError('HDF dependency cycle: ' + str(source))
        if source in records:
            return records[source]
        active.add(source)
        before = source.stat()
        edges, groups = [], set()

        def target(filename):
            filename = filename.decode('utf-8') if isinstance(filename, bytes) else str(filename)
            if filename in ('', '.') or '%' in filename:
                raise ValueError('unsupported self/printf HDF dependency source')
            resolved = Path(filename)
            return resolved if resolved.is_absolute() else source.parent / resolved

        def scan(group):
            identity = h5py.h5o.get_info(group.id).addr
            if identity in groups:
                return
            groups.add(identity)
            for name in sorted(group.keys()):
                link = group.get(name, getlink=True)
                logical_name = group.name.rstrip('/') + '/' + name
                if isinstance(link, h5py.ExternalLink):
                    edges.append(('external:' + logical_name + '->' + link.path, target(link.filename)))
                    continue
                if not isinstance(link, h5py.HardLink):
                    raise ValueError('unsupported HDF soft/unknown link: ' + logical_name)
                item = group[name]
                if isinstance(item, h5py.Group):
                    scan(item)
                elif isinstance(item, h5py.Dataset):
                    if item.external:
                        raise ValueError('unsupported HDF raw external storage')
                    if item.is_virtual:
                        for index, mapping in enumerate(item.virtual_sources()):
                            dataset_name = mapping.dset_name.decode('utf-8') if isinstance(mapping.dset_name, bytes) else mapping.dset_name
                            edges.append(('vds:' + logical_name + '[' + str(index) + ']->' + dataset_name,
                                          target(mapping.file_name)))

        with h5py.File(source, 'r') as handle:
            scan(handle)
        digest = _sha(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('HDF input changed while binding: ' + str(source))
        dependencies = []
        for via, child in sorted(edges, key=lambda edge: edge[0]):
            bound = node(child)
            dependencies.append(dict(via=via, path=bound['path'], content_sha256=bound['content_sha256']))
        semantic = dict(bytes=after.st_size, sha256=digest,
                        dependencies=[dict(via=row['via'], content_sha256=row['content_sha256'])
                                      for row in dependencies])
        content = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        record = dict(path=str(source), bytes=after.st_size, sha256=digest,
                      content_sha256=content, dependencies=dependencies)
        records[source] = record
        active.remove(source)
        return record

    try:
        root = node(path)
    except (OSError, KeyError, RuntimeError) as error:
        raise ValueError('cannot bind concrete HDF dependencies: ' + str(error)) from error
    return dict(schema='relocate-hdf-content-v1', root_path=root['path'], root_sha256=root['sha256'],
                content_sha256=root['content_sha256'], file_count=len(records),
                files=[records[key] for key in sorted(records, key=str)])

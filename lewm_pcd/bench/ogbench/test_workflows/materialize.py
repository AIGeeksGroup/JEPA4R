"""Restore the archived native layout into a fresh directory; execute nothing."""
import argparse
import json
from pathlib import Path

from .catalog import _portable_path, _read_record, load_sources, verify_sources


def materialize(destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('A fresh destination is required: '+str(destination))
    sources = load_sources()
    report = verify_sources()
    if report['hash_mismatches']:
        raise ValueError('Source verification failed before materialization')
    for logical in sources['files']:
        _portable_path(logical)
    destination.mkdir(parents=True, exist_ok=False)
    for logical in sources['files']:
        target = destination.joinpath(*logical.split('/'))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(_read_record(logical, sources['files'][logical]))
    result = dict(files=len(sources['files']), destination=str(destination.resolve()),
        original_native_root=sources['native_root'], original_source_bytes_preserved=True,
        absolute_paths_rewritten=False, external_assets_copied=False, executed=False)
    with (destination/'MATERIALIZATION_RECEIPT.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.destination), indent=2))


if __name__ == '__main__':
    main()

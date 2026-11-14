#!/usr/bin/env python3
"""List pinned checkpoint bundles; download only with explicit --download."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen


SCHEMA = 'jepa4r_released_checkpoints_v1'
REPOSITORY = Path(__file__).resolve().parents[1]


def _matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value)


def _relative(value):
    if not isinstance(value, str) or not value:
        raise ValueError('file path must be a nonempty POSIX-relative path')
    parts = value.split('/')
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {
        f'{prefix}{i}' for prefix in ('COM', 'LPT') for i in range(1, 10)}
    if any(part in ('', '.', '..') or part.endswith((' ', '.')) or
           part.split('.')[0].upper() in reserved or
           any(ord(char) < 32 or char in '\\<>:"|?*' for char in part) for part in parts):
        raise ValueError(f'unsafe POSIX-relative path: {value!r}')
    return parts


def _validate_release(release):
    if not isinstance(release, dict):
        raise ValueError('each release must be an object')
    for key in ('task', 'model'):
        if not _matches(r'[A-Za-z0-9_-]+', release.get(key)):
            raise ValueError(f'unsafe {key}')
    repo = release.get('repo_id', '')
    if not _matches(r'[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo) or '..' in repo:
        raise ValueError('unsafe Hugging Face repo_id')
    if not _matches(r'[0-9a-f]{40}', release.get('revision')):
        raise ValueError('revision must be an immutable 40-character lowercase commit SHA')
    if not isinstance(release.get('branch'), str) or not release['branch']:
        raise ValueError('informational branch is required')
    files = release.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('release files must be a nonempty list')
    paths = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError('each file must be an object')
        _relative(item.get('path'))
        key = item['path'].casefold()
        if key in paths:
            raise ValueError(f'duplicate file path: {item["path"]}')
        paths.add(key)
        if type(item.get('bytes')) is not int or item['bytes'] <= 0:
            raise ValueError('file bytes must be a positive integer')
        if not _matches(r'[0-9a-f]{64}', item.get('sha256')):
            raise ValueError('file sha256 must be 64 lowercase hexadecimal characters')
    if any('/'.join(item['path'].casefold().split('/')[:index]) in paths
           for item in files for index in range(1, len(item['path'].split('/')))):
        raise ValueError('release paths contain a file/directory conflict')
    for item in files:
        partial = str(Path(item['path']).with_name(
            f'.{Path(item["path"]).name}.{item["sha256"]}.part')).replace('\\', '/')
        if any((partial + suffix).casefold() in paths for suffix in ('', '.json', '.lock')):
            raise ValueError('release path conflicts with downloader staging files')
    wm, policies = release.get('world_model'), release.get('policies')
    if not isinstance(wm, dict) or not isinstance(policies, list):
        raise ValueError('world_model and policies are required')
    references = [wm.get(key) for key in ('weights', 'config', 'training_config')]
    for policy in policies:
        if (not isinstance(policy, dict) or type(policy.get('seed')) is not int or
                policy['seed'] < 0 or not isinstance(policy.get('kind'), str) or not policy['kind']):
            raise ValueError('policies require a nonnegative integer seed and kind')
        references.append(policy.get('path'))
        references.extend(policy[key] for key in ('run_state', 'manifest', 'agent_config') if key in policy)
    for path in references:
        _relative(path)
        if path not in {item['path'] for item in files}:
            raise ValueError(f'bundle reference is absent from files: {path}')


def load_registry(path):
    """Read and validate the immutable registry without network or filesystem writes."""
    registry = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(registry, dict) or registry.get('schema') != SCHEMA:
        raise ValueError(f'expected registry schema {SCHEMA}')
    if not isinstance(registry.get('releases'), list):
        raise ValueError('registry releases must be a list')
    seen = set()
    for release in registry['releases']:
        _validate_release(release)
        key = (release['task'].casefold(), release['model'].casefold())
        if key in seen:
            raise ValueError('duplicate task/model release')
        seen.add(key)
    return registry


def select_release(registry, task, model='jepa4r'):
    matches = [row for row in registry['releases'] if row['task'] == task and row['model'] == model]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one release for {task}/{model}')
    return matches[0]


def _local(base, *parts):
    path = base
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'symlink is forbidden in checkpoint destination: {path}')
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f'destination escapes checkpoint root: {path}')
    return path


def _verify(path, item):
    if not path.is_file() or path.stat().st_size != item['bytes']:
        raise ValueError(f'file size/integrity mismatch: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    if digest.hexdigest() != item['sha256']:
        raise ValueError(f'file SHA256 integrity mismatch: {path}')


def _download(release, item, final):
    url = ('https://huggingface.co/' + release['repo_id'] + '/resolve/' +
           release['revision'] + '/' + quote(item['path'], safe='/'))
    partial = _local(final.parent, f'.{final.name}.{item["sha256"]}.part')
    metadata = _local(final.parent, partial.name + '.json')
    lock = _local(final.parent, partial.name + '.lock')
    identity = dict(url=url, bytes=item['bytes'], sha256=item['sha256'])
    final.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('x', encoding='utf-8'):
        pass
    try:
        if metadata.exists():
            if json.loads(metadata.read_text(encoding='utf-8')) != identity:
                raise ValueError(f'partial download identity mismatch: {metadata}')
        elif partial.exists():
            raise ValueError(f'refusing unowned partial download: {partial}')
        else:
            with metadata.open('x', encoding='utf-8') as stream:
                json.dump(identity, stream)
        start = partial.stat().st_size if partial.exists() else 0
        if start > item['bytes']:
            raise ValueError(f'partial download exceeds expected size: {partial}')
        if start < item['bytes']:
            headers = {'Range': f'bytes={start}-'} if start else {}
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                if response.status == 206:
                    expected = f'bytes {start}-{item["bytes"] - 1}/{item["bytes"]}'
                    if response.headers.get('Content-Range') != expected:
                        raise ValueError('invalid Content-Range for pinned download')
                elif response.status == 200:
                    start = 0  # A server that ignores Range must replace only our owned partial.
                else:
                    raise ValueError(f'unexpected download HTTP status: {response.status}')
                mode = 'ab' if start else ('wb' if partial.exists() else 'xb')
                with partial.open(mode) as stream:
                    count = start
                    while block := response.read(min(1 << 20, item['bytes'] - count + 1)):
                        count += len(block)
                        if count > item['bytes']:
                            raise ValueError('download exceeds expected file size')
                        stream.write(block)
        _verify(partial, item)
        os.link(partial, final)  # Atomic no-replace publication, including concurrent final creation.
        partial.unlink()
        metadata.unlink()
    finally:
        lock.unlink()


def prepare_release(release, root, download=False):
    """Verify all bytes before returning a resolved bundle; never overwrite finals."""
    _validate_release(release)
    root = Path(root).expanduser().resolve()
    directory = _local(root, release['task'], release['model'])
    targets = [(item, _local(directory, *_relative(item['path']))) for item in release['files']]
    for item, path in targets:
        if path.exists():
            _verify(path, item)
    for item, path in targets:
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'missing checkpoint file: {path}; use --download explicitly')
            _download(release, item, path)
    for item, path in targets:
        _verify(path, item)
    resolved = copy.deepcopy(release)
    resolved.update(root=str(root), directory=str(directory), verified=True)
    for row, (_, path) in zip(resolved['files'], targets):
        row['local_path'] = str(path)
    return resolved


def main(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--registry', type=Path, default=REPOSITORY / 'released_checkpoints.json')
    known, _ = preliminary.parse_known_args(argv)
    registry = load_registry(known.registry)
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument('--task', choices=sorted({row['task'] for row in registry['releases']}))
    parser.add_argument('--model', default='jepa4r')
    parser.add_argument('--root', type=Path,
                        default=os.environ.get('JEPA4R_CHECKPOINTS', REPOSITORY / 'checkpoints'))
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--list', action='store_true', help='list releases (the default; no downloads)')
    action.add_argument('--download', action='store_true', help='download and verify the selected release')
    action.add_argument('--verify', action='store_true', help='verify existing files without network access')
    parser.add_argument('--output', type=Path, help='write resolved JSON to a new file only')
    args = parser.parse_args(argv)
    if not (args.download or args.verify):
        if args.output:
            parser.error('--output requires --download or --verify')
        result = [{key: row[key] for key in ('task', 'model', 'repo_id', 'revision')}
                  for row in registry['releases'] if not args.task or row['task'] == args.task]
    else:
        if not args.task:
            parser.error('--task is required for --download or --verify')
        if args.output and (args.output.exists() or args.output.is_symlink()):
            raise FileExistsError(f'resolved output already exists: {args.output}')
        result = prepare_release(select_release(registry, args.task, args.model), args.root, args.download)
        if args.output:
            with args.output.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, indent=2)
                stream.write('\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(1)

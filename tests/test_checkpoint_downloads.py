"""Real local HTTP and filesystem checks; no model or simulator dependencies."""
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/download_checkpoints.py'


def load_downloader():
    assert SCRIPT.is_file(), 'checkpoint download entry is missing'
    spec = importlib.util.spec_from_file_location('download_checkpoints', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckpointDownloads(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.payloads = {'jepa4r/world.pt': b'world model fixture' * 37,
                         'jepa4r/config.json': b'{"encoder": "fixture"}',
                         'jepa4r/train.yaml': b'task: cube1\n',
                         'jepa4r/policy/seed_0.bin': b'policy fixture' * 19}
        self.release = dict(task='cube1', model='jepa4r', repo_id='owner/checkpoints',
            revision='a' * 40, branch='cube1', files=[dict(path=path, bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest()) for path, data in self.payloads.items()],
            world_model=dict(weights='jepa4r/world.pt', config='jepa4r/config.json',
                             training_config='jepa4r/train.yaml'),
            policies=[dict(seed=0, kind='gcbc', path='jepa4r/policy/seed_0.bin')],
            profile={'source': 'original producer'})
        self.registry = self.root / 'registry.json'
        self.write_registry()

    def write_registry(self, release=None):
        self.registry.write_text(json.dumps(dict(schema='jepa4r_released_checkpoints_v1',
            releases=[release or self.release])), encoding='utf-8')

    def serve(self, module):
        fixture = self
        self.requests = []
        self.mode = 'good'

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                fixture.requests.append((self.path, self.headers.get('Range')))
                prefix = '/owner/checkpoints/resolve/' + 'a' * 40 + '/'
                if not self.path.startswith(prefix):
                    self.send_error(404)
                    return
                data = fixture.payloads[self.path[len(prefix):]]
                start = int(self.headers.get('Range', 'bytes=0-')[6:-1])
                ranged = start and fixture.mode != 'ignore_range'
                self.send_response(206 if ranged else 200)
                if ranged:
                    reported = start + (1 if fixture.mode == 'bad_range' else 0)
                    self.send_header('Content-Range',
                        f'bytes {reported}-{len(data) - 1}/{len(data)}')
                body = data[start:] if ranged else data
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                if fixture.mode == 'truncate':
                    body = body[:len(body) // 2]
                elif fixture.mode == 'bad_sha':
                    body = bytes([body[0] ^ 1]) + body[1:]
                self.wfile.write(body)
                self.close_connection = True

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 3)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def routed(request, **kwargs):
            self.assertTrue(request.full_url.startswith('https://huggingface.co/'))
            target = f'http://127.0.0.1:{server.server_port}' + urlsplit(request.full_url).path
            return urlopen(Request(target, headers=dict(request.header_items())), **kwargs)

        patcher = patch.object(module, 'urlopen', routed)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_registry_selection_and_real_download_verification(self):
        module = load_downloader()
        registry = module.load_registry(self.registry)
        selected = module.select_release(registry, 'cube1', 'jepa4r')
        self.serve(module)
        bundle = module.prepare_release(selected, self.root / 'checkpoints', download=True)
        self.assertTrue(bundle['verified'])
        self.assertEqual(bundle['profile'], self.release['profile'])
        self.assertEqual(Path(bundle['directory']), self.root / 'checkpoints/cube1/jepa4r')
        for row in bundle['files']:
            self.assertEqual(Path(row['local_path']).read_bytes(), self.payloads[row['path']])
            self.assertEqual(Path(row['local_path']), Path(bundle['directory']) / row['path'])
        self.assertEqual(len(self.requests), len(self.payloads))
        with patch.object(module, 'urlopen', side_effect=AssertionError('network during verify')):
            self.assertEqual(bundle, module.prepare_release(selected, self.root / 'checkpoints'))
        self.assertNotIn('local_path', self.release['files'][0])

    def test_missing_files_verify_is_read_only(self):
        module = load_downloader()
        missing = self.root / 'absent'
        with patch.object(module, 'urlopen', side_effect=AssertionError('network')):
            with self.assertRaises(FileNotFoundError):
                module.prepare_release(self.release, missing)
        self.assertFalse(missing.exists())

    def test_corrupt_existing_final_is_never_overwritten(self):
        module = load_downloader()
        final = self.root / 'checkpoints/cube1/jepa4r/jepa4r/world.pt'
        final.parent.mkdir(parents=True)
        final.write_bytes(b'previous experiment')
        with patch.object(module, 'urlopen', side_effect=AssertionError('network')):
            with self.assertRaisesRegex(ValueError, 'integrity|SHA256|size'):
                module.prepare_release(self.release, self.root / 'checkpoints', download=True)
        self.assertEqual(final.read_bytes(), b'previous experiment')

    def test_bad_download_sha_never_becomes_final(self):
        module = load_downloader()
        self.serve(module)
        self.mode = 'bad_sha'
        with self.assertRaisesRegex(ValueError, 'SHA256|integrity'):
            module.prepare_release(self.release, self.root / 'checkpoints', download=True)
        self.assertFalse((self.root / 'checkpoints/cube1/jepa4r/jepa4r/world.pt').exists())

    def test_interrupted_download_resumes_with_validated_content_range(self):
        module = load_downloader()
        self.serve(module)
        self.mode = 'truncate'
        with self.assertRaises(Exception):
            module.prepare_release(self.release, self.root / 'checkpoints', download=True)
        self.mode = 'good'
        result = module.prepare_release(self.release, self.root / 'checkpoints', download=True)
        self.assertTrue(result['verified'])
        self.assertTrue(any(value and value != 'bytes=0-' for _, value in self.requests))

    def test_invalid_content_range_keeps_partial_and_no_final(self):
        module = load_downloader()
        self.serve(module)
        self.mode = 'truncate'
        with self.assertRaises(Exception):
            module.prepare_release(self.release, self.root / 'checkpoints', download=True)
        self.mode = 'bad_range'
        with self.assertRaisesRegex(ValueError, 'Content-Range|range'):
            module.prepare_release(self.release, self.root / 'checkpoints', download=True)
        self.assertFalse((self.root / 'checkpoints/cube1/jepa4r/jepa4r/world.pt').exists())

    def test_ignored_range_restarts_owned_partial_safely(self):
        module = load_downloader()
        self.serve(module)
        self.mode = 'truncate'
        with self.assertRaises(Exception):
            module.prepare_release(self.release, self.root / 'checkpoints', download=True)
        self.mode = 'ignore_range'
        result = module.prepare_release(self.release, self.root / 'checkpoints', download=True)
        self.assertTrue(result['verified'])

    def test_registry_rejects_traversal_duplicate_and_mutable_revision(self):
        module = load_downloader()
        mutations = [('revision', 'main'), ('revision', 'A' * 40),
                     ('repo_id', 'https://evil.example/repo'), ('repo_id', '../repo'),
                     ('task', '../cube1'), ('model', 'jepa4r/root')]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(self.release)
                changed[key] = value
                self.write_registry(changed)
                with self.assertRaises(ValueError):
                    module.load_registry(self.registry)
        for path in ['../escape', '/absolute', 'C:/escape', 'a\\escape', 'a/../escape',
                     'a//escape', 'a:stream', 'a./file']:
            with self.subTest(path=path):
                changed = copy.deepcopy(self.release)
                changed['files'][0]['path'] = path
                with self.assertRaises(ValueError):
                    module.prepare_release(changed, self.root / 'absent', download=True)
        changed = copy.deepcopy(self.release)
        changed['files'].append(copy.deepcopy(changed['files'][0]))
        with self.assertRaises(ValueError):
            module.prepare_release(changed, self.root / 'absent')
        self.assertFalse((self.root / 'absent').exists())

    def test_registry_rejects_bad_sizes_hashes_and_dangling_references(self):
        module = load_downloader()
        for field, value in [('bytes', True), ('bytes', 0), ('sha256', 'z' * 64)]:
            changed = copy.deepcopy(self.release)
            changed['files'][0][field] = value
            self.write_registry(changed)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                module.load_registry(self.registry)
        changed = copy.deepcopy(self.release)
        changed['world_model']['weights'] = 'missing.pt'
        self.write_registry(changed)
        with self.assertRaises(ValueError):
            module.load_registry(self.registry)

    def test_registry_rejects_nonstring_identifiers_and_file_directory_conflicts(self):
        module = load_downloader()
        for key, value in [('task', None), ('model', []), ('repo_id', 17), ('revision', {})]:
            changed = copy.deepcopy(self.release)
            changed[key] = value
            self.write_registry(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.load_registry(self.registry)
        changed = copy.deepcopy(self.release)
        changed['files'].append(dict(path='jepa4r/world.pt/child', bytes=1, sha256='0' * 64))
        self.write_registry(changed)
        with self.assertRaises(ValueError):
            module.load_registry(self.registry)

    def test_unowned_partial_is_preserved_and_never_resumed(self):
        module = load_downloader()
        self.serve(module)
        self.mode = 'truncate'
        destination = self.root / 'checkpoints'
        with self.assertRaises(Exception):
            module.prepare_release(self.release, destination, download=True)
        partial = next(destination.rglob('*.part'))
        prefix = partial.read_bytes()
        Path(str(partial) + '.json').unlink()
        with patch.object(module, 'urlopen', side_effect=AssertionError('unowned resume')):
            with self.assertRaisesRegex(ValueError, 'unowned'):
                module.prepare_release(self.release, destination, download=True)
        self.assertEqual(partial.read_bytes(), prefix)

    def test_oversized_download_is_never_published(self):
        module = load_downloader()
        self.serve(module)
        self.payloads['jepa4r/world.pt'] += b'extra unexpected bytes'
        with self.assertRaisesRegex(ValueError, 'size'):
            module.prepare_release(self.release, self.root / 'checkpoints', download=True)
        self.assertFalse((self.root / 'checkpoints/cube1/jepa4r/jepa4r/world.pt').exists())

    def test_symlink_parent_cannot_escape_selected_root(self):
        module = load_downloader()
        outside = self.root / 'outside'
        outside.mkdir()
        checkpoints = self.root / 'checkpoints'
        checkpoints.mkdir()
        try:
            (checkpoints / 'cube1').symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f'local symlink privilege unavailable: {exc}')
        with self.assertRaises(ValueError):
            module.prepare_release(self.release, checkpoints, download=True)
        self.assertEqual(list(outside.iterdir()), [])

    def cli(self, *args, env=None):
        return subprocess.run([sys.executable, str(SCRIPT), '--registry', str(self.registry),
            *map(str, args)], capture_output=True, text=True, env=env, timeout=10)

    def test_cli_default_lists_without_download_or_training_imports(self):
        load_downloader()
        destination = self.root / 'checkpoints'
        environment = dict(os.environ, JEPA4R_CHECKPOINTS=str(destination))
        guard = self.root / 'sitecustomize.py'
        guard.write_text('import sys\n'
            'def guard(event, args):\n'
            '    if event == "import" and args[0].split(".")[0] in '
            '("torch", "numpy", "mujoco", "jax", "train"):\n'
            '        raise RuntimeError("training/runtime import forbidden")\n'
            '    if event in ("socket.connect", "socket.getaddrinfo"):\n'
            '        raise RuntimeError("network forbidden")\n'
            'sys.addaudithook(guard)\n', encoding='utf-8')
        environment['PYTHONPATH'] = str(self.root)
        result = self.cli(env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('cube1', result.stdout)
        self.assertFalse(destination.exists())
        unknown = self.cli('--task', 'not-registered')
        self.assertNotEqual(unknown.returncode, 0)

    def test_cli_verify_writes_only_new_resolved_output(self):
        module = load_downloader()
        self.serve(module)
        destination = self.root / 'checkpoints'
        module.prepare_release(self.release, destination, download=True)
        output = self.root / 'bundle.json'
        result = self.cli('--verify', '--task', 'cube1', '--root', destination, '--output', output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(output.read_text())['verified'])
        original = output.read_bytes()
        again = self.cli('--verify', '--task', 'cube1', '--root', destination, '--output', output)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()

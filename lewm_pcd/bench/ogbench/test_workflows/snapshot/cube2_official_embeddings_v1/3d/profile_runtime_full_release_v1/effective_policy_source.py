"""Resource-only overlay of the pinned producer; never edits the base producer."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

BASE_SHA = 'c8aec46ae0fec30b56a36071af02aa807af542a7e04881e6a43fe2a40ba70d9a'
SUBMIT = """                try:
                    job = next(iterator)
                    queue.append((job, pool.submit(render_episode, job)))
                except StopIteration:
                    pass
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ast_sha(source):
    return hashlib.sha256(ast.dump(ast.parse(source), include_attributes=False).encode()).hexdigest()


def overlay(source):
    """Resource, scheduling and receipt changes; reverse and infer AST are checked."""
    edits = [
        ("os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-8:])",
         "os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-_RESOURCE_PROFILE['affinity_cores']:])"),
        ('ProcessPoolExecutor(max_workers=2,', "ProcessPoolExecutor(max_workers=_RESOURCE_PROFILE['renderer_workers'],"),
        ('range(min(2, len(pending)))', "range(min(_RESOURCE_PROFILE['prefetch_jobs'], len(pending)))"),
        (SUBMIT, ''),
        ('                rendered = future.result()\n', '                rendered = future.result()\n' + SUBMIT),
        ("                save(epdir / 'receipt.json', record)",
         "                record['runtime_resource_profile'] = _RESOURCE_PROFILE\n                save(epdir / 'receipt.json', record)"),
    ]
    modified = source
    for before, after in edits:
        assert modified.count(before) == 1, repr(before)
        modified = modified.replace(before, after)
    restored = modified
    for before, after in reversed(edits):
        if after:
            assert restored.count(after) == 1
            restored = restored.replace(after, before)
        else:
            # The final while-loop statement immediately precedes the pilot branch.
            anchor = '            if pilot:\n                for split in'
            assert restored.count(anchor) == 1
            restored = restored.replace(anchor, SUBMIT + anchor)
    assert restored == source and ast_sha(restored) == ast_sha(source)
    def inference(text):
        return [ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.FunctionDef) and n.name == 'infer']
    assert inference(modified) == inference(source)
    return modified


def write_once(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == content, str(path)
    else:
        with path.open('xb') as stream:
            stream.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, required=True)
    args, original_args = parser.parse_known_args()
    profile = json.loads(args.profile.read_text())
    base = Path(profile['base_producer'])
    assert sha(base) == BASE_SHA == profile['base_producer_sha256']
    assert 1 <= profile['renderer_workers'] <= 16
    assert profile['renderer_workers'] <= profile['prefetch_jobs'] <= 2 * profile['renderer_workers']
    assert profile['affinity_cores'] == 16
    if profile['GPU_preprocess']:
        gate = profile['GPU_preprocess_gate']
        assert sha(gate['path']) == gate['sha256']
        accepted = json.loads(Path(gate['path']).read_text())
        assert accepted['actual_exit_code'] == 0 and accepted['prepared_bitexact'] and accepted['latent_bitexact']
        assert _PERFORMANCE_RELEASE['positive_gain_release'] is True and accepted['accepted'] is False and accepted['candidate_source_sha256'] == sha(Path(__file__).with_name('gpu_preprocess.py'))
        assert 0 < accepted['candidate_median_seconds'] < accepted['baseline_median_seconds'] and _PERFORMANCE_RELEASE['original_5percent_time_reduction_failed'] is True
        plan_file = Path(original_args[original_args.index('--plan')+1])
        original_plan = json.loads(plan_file.read_text())
        assert accepted['plan_sha256'] == sha(plan_file)
        assert accepted['checkpoint_sha256'] == original_plan['checkpoint_sha256']
    source = base.read_text()
    effective = overlay(source)
    identity = dict(profile_path=str(args.profile.resolve()), profile_sha256=sha(args.profile),
        wrapper_path=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__),
        base_producer_sha256=BASE_SHA, effective_AST_sha256=ast_sha(effective),
        renderer_workers=profile['renderer_workers'], prefetch_jobs=profile['prefetch_jobs'],
        affinity_cores=profile['affinity_cores'],
        next_job_submitted_before_infer=True, FIFO_commit_order=True,
        preprocess_device='cuda' if profile['GPU_preprocess'] else 'cpu',
        GPU_preprocess_source_sha256=sha(Path(__file__).with_name('gpu_preprocess.py')) if profile['GPU_preprocess'] else None,
        GPU_preprocess_gate=profile.get('GPU_preprocess_gate'), XYZ_dtype='float32', RGB_dtype='uint8',
        base_protocol_renderer_workers_is_historical=True)
    launch = dict(actual_argv=sys.argv, started=time.time(), identity=identity,
                  base_AST_sha256=ast_sha(source), reverse_AST_equals_base=True,
                  inference_AST_equals_base=True)
    write_once(args.runtime_dir / 'actual_launch.json', json.dumps(launch, indent=2).encode())
    write_once(args.runtime_dir / 'effective_source.py', effective.encode())
    sys.path.insert(0, str(base.parent))
    sys.argv = [str(base)] + original_args
    namespace = dict(__name__='_resource_profile_producer', __file__=str(base),
                     _RESOURCE_PROFILE=identity)
    exec(compile(effective, str(base), 'exec'), namespace)
    if profile['GPU_preprocess']:
        from gpu_preprocess import install
        original_load = namespace['load_source']
        def load_profiled(name, path):
            module = original_load(name, path)
            if name == '_cache_original_pcd_encoder':
                install(module)
            return module
        namespace['load_source'] = load_profiled
    namespace['main']()
    if '--execute' in original_args and '--pilot' not in original_args:
        plan = json.loads(Path(original_args[original_args.index('--plan')+1]).read_text())
        cache = Path(plan['cache_dir'])
        manifest = json.loads((cache / 'cache_manifest.json').read_text())
        rows = []
        for split, entries in manifest['splits'].items():
            for item in entries:
                receipt = Path(item['receipt_path'])
                assert sha(receipt) == item['receipt_sha256']
                record = json.loads(receipt.read_text())
                rows.append(dict(split=split, episode=item['episode'], receipt_path=str(receipt),
                    receipt_sha256=sha(receipt), runtime_resource_profile=record.get('runtime_resource_profile'),
                    profile_kind='resource_overlay' if 'runtime_resource_profile' in record else 'legacy_base_workers2_prefetch2'))
        write_once(cache / 'runtime_profiles_manifest.json', json.dumps(dict(actual_exit_code=0,
            base_protocol_is_mathematical_identity=True, base_manifest_sha256=sha(cache/'cache_manifest.json'),
            mixed_resource_profiles=True, episodes=rows), indent=2).encode())


if __name__ == '__main__':
    main()

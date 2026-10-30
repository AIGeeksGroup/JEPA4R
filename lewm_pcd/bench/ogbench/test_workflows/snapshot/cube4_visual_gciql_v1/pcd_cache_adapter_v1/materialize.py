"""Quad3D schema/provenance adapter over the accepted cache production chain.

No Torch, renderer, policy, array extraction or GPU is imported here. Only a new
workspace and plans are written; accepted arrays, sources and caches stay intact.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

MODULE = 'official_state_3d_cache_v1'
CHECKPOINT_SHA = '95e7980f665a0ad888a0723499f5d0771c8cf797065b1f1b39c3eae39fd951d8'
RGB_AUDITOR_SHA = '209abcbf3ab4e82120638f290a1683b611803a9b3fd04bacbef13b1aa84b5fe4'
NAMESPACE = 'quadruple_official_pcd_epoch10_v1'
EXPECTED = {
    MODULE + '/common.py': '51ee5de9557e7a31210757aa514213699aadf7f2fb9a27f130c880a2ccefda28',
    MODULE + '/pcd_renderer.py': 'c6d61eb1af47a5e71c9c933cbf506c35ce20b71be6e5a9c77ce83d8cfefa875b',
    MODULE + '/produce.py': 'c8aec46ae0fec30b56a36071af02aa807af542a7e04881e6a43fe2a40ba70d9a',
    MODULE + '/gate.py': '4fcab671b8a9ad9876c3a2470e7fbc89a375bfdea3d82be9783a3093f6104bd4',
    MODULE + '/audit_cache.py': 'c230f64373fb6f90f47d73b366688dafcf296fbc9361953070d92c905f8a554a',
    MODULE + '/acceleration_v1/v4/run_profile.py': '7cbb13be10f73a92f4958de5a5a6af209be6a36e4df50e3e90c1b6211ecc90d1',
    MODULE + '/acceleration_v1/v4/gpu_preprocess.py': '7929544aab09cb17cd25b269ef0830259456c9fa76fcd62cf07e5d19819e35bf',
    MODULE + '/acceleration_v1/v4/bench_gpu_preprocess.py': 'd6d2282ba63e4f0d48fff2dccaf89aa352b08b52b85d75086a75b6f401fd0d21',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(data)


def read_bound(item):
    path = Path(item['path'])
    assert sha(path) == item['sha256'], path
    return json.loads(path.read_text())


def build_sources(original, rgb_audit_sha=RGB_AUDITOR_SHA):
    assert len(rgb_audit_sha) == 64 and set(rgb_audit_sha) <= set('0123456789abcdef')
    sources, ledger = {}, []
    for name, expected in EXPECTED.items():
        payload = (original / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected, name
        sources[name] = payload

    def edit(name, changes):
        before = sources[name]; text = before.decode()
        for old, new, count in changes:
            assert text.count(old) == count, (name, old, count, text.count(old))
            text = text.replace(old, new)
        ast.parse(text); sources[name] = text.encode()
        ledger.append(dict(path=name, original_sha256=hashlib.sha256(before).hexdigest(),
            generated_sha256=hashlib.sha256(sources[name]).hexdigest(), replacements=changes,
            diff=''.join(difflib.unified_diff(before.decode().splitlines(True), text.splitlines(True),
                fromfile='accepted_Double/' + name, tofile='new_Quad/' + name))))

    edit(MODULE + '/common.py', [
        ('qpos_width=28, qvel_width=26', 'qpos_width=42, qvel_width=38', 1),
        ("{'train': 1000, 'val': 100}", "{'train': 5000, 'val': 500}", 1),
        ("[[.1, -.5, -.02], [.75, .5, .6]]", "[[-.25, -.5, -.02], [.75, .5, .9]]", 1),
        ('double_official_pcd_epoch10_v1', NAMESPACE, 2)])
    edit(MODULE + '/pcd_renderer.py', [
        ('qpos.shape == (28,) and qvel.shape == (26,)', 'qpos.shape == (42,) and qvel.shape == (38,)', 1),
        ('range(2)', 'range(4)', 1), ('% 2', '% 4', 1), ('nq=28, nv=26', 'nq=42, nv=38', 1)])
    edit(MODULE + '/produce.py', [
        ('completed == 1100 and rows == 1101100', 'completed == 5500 and rows == 5505500', 1),
        ('train_rows=1001000, val_rows=100100', 'train_rows=5005000, val_rows=500500', 1)])
    membership = """def checkpoint_config_key(workspace, config_path, pins):
    relative = 'checkpoints/cube4/3d_epoch10/config.json'
    assert relative in pins
    assert (workspace / relative).resolve() == config_path.resolve()
    return relative


"""
    edit(MODULE + '/audit_cache.py', [
        ("ORIGINAL_SHA = '75eb5137c30cac3628f0e781c7d2eb4a96b29eb7689a7fcae6ce1509dc303adc'", "ORIGINAL_SHA = '" + rgb_audit_sha + "'", 1),
        ("{'train': 1000, 'val': 100}", "{'train': 5000, 'val': 500}", 1),
        ('[[.1,-.5,-.02],[.75,.5,.6]]', '[[-.25,-.5,-.02],[.75,.5,.9]]', 1),
        ('double_official_pcd_epoch10_v1', NAMESPACE, 1),
        ("accepted['data_id'] == 'cube-double-play-v0' and accepted['cube_count'] == 2", "accepted['data_id'] == 'cube-quadruple-play-v0' and accepted['cube_count'] == 4", 1),
        ('qpos_width=28, qvel_width=26', 'qpos_width=42, qvel_width=38', 1),
        ("[('qpos',28), ('qvel',26)]", "[('qpos',42), ('qvel',38)]", 1),
        ('nq=28,nv=26', 'nq=42,nv=38', 1),
        ('def audit(cache_dir, plan_path, progress=None):', membership + 'def audit(cache_dir, plan_path, progress=None):', 1),
        ("    assert config_path.relative_to(workspace).as_posix() in pins\n    config, _ = read(config_path)",
         "    config_key = checkpoint_config_key(workspace, config_path, pins)\n    config, _ = read(config_path, pins[config_key])", 1)])
    producer_sha = hashlib.sha256(sources[MODULE + '/produce.py']).hexdigest()
    edit(MODULE + '/acceleration_v1/v4/run_profile.py', [(EXPECTED[MODULE + '/produce.py'], producer_sha, 1)])
    return sources, ledger


def make_plan(root, original, rgb_workspace, case, binding_path, binding_sha, sources):
    assert sha(binding_path) == binding_sha
    binding = json.loads(binding_path.read_text())
    assert binding['actual_exit_code'] == 0 and binding['cube_count'] == 4 and binding['data_id'] == 'cube-quadruple-play-v0'
    assert binding['checkpoint']['sha256'] == CHECKPOINT_SHA and binding['checkpoint']['bytes'] == 72241790
    assert Path(binding['checkpoint']['path']) == root / 'checkpoints/cube4/3d_epoch10/weights_epoch_10.pt'
    assert sha(binding['checkpoint']['path']) == CHECKPOINT_SHA
    accepted, frozen, arrays = (read_bound(binding[key]) for key in ('data_acceptance', 'source_stat_freeze', 'source_arrays_view'))
    assert accepted['actual_exit_code'] == frozen['actual_exit_code'] == 0 and accepted['cube_count'] == 4
    assert accepted['data_id'] == 'cube-quadruple-play-v0'
    assert frozen['acceptance_sha256'] == binding['data_acceptance']['sha256']
    rgb_plan = read_bound(binding['old_quad_RGB_producer_plan'])
    assert Path(binding['old_quad_RGB_producer_plan']['path']) == rgb_workspace / 'official_state_2d_cache_v1/producer_plan.json'
    assert binding['camera'] == rgb_plan['camera'] and binding['camera']['pullback'] == .10 and binding['camera']['crop'] is None
    assert sha(rgb_workspace / 'official_state_2d_cache_v1/audit_cache.py') == RGB_AUDITOR_SHA
    assert binding['split_episodes'] == {'train': 5000, 'val': 500} and binding['raw_rows_per_episode'] == 1001
    assert binding['source_dimensions'] == dict(state=55, qpos=42, qvel=38, action=5)
    assert Path(binding['source_arrays_view']['path']) == Path(binding['source_arrays_root']) / 'source_arrays.json'
    pcd = copy.deepcopy(json.loads((original / MODULE / 'producer_plan.json').read_text())['pcd'])
    pcd.update(crop_box=[[-.25, -.5, -.02], [.75, .5, .9]], sampling_namespace=NAMESPACE)
    for key in pcd:
        if key != 'sampling_policy': assert binding['pcd'][key] == pcd[key], key
    assert binding['preprocess']['center'] == [.425, 0., .2] and binding['preprocess']['scale'] == 4. and binding['preprocess']['use_rgb'] is True
    assert binding['precision'] == dict(dtype='float32', AMP=False, change_original_settings=False)
    expected_contract = dict(data_acceptance_sha256=binding['data_acceptance']['sha256'],
        source_stat_freeze_sha256=binding['source_stat_freeze']['sha256'], split_episodes=binding['split_episodes'],
        raw_rows_per_episode=1001, qpos_width=42, qvel_width=38,
        extractor_source_sha256=binding['old_quad_RGB_source_pins']['official_state_2d_cache_v1/produce.py'])
    assert arrays['contract'] == expected_contract
    assert arrays['protocol_sha256'] == hashlib.sha256(json.dumps(expected_contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    for split, count in binding['split_episodes'].items():
        item = arrays['splits'][split]; prior = frozen['datasets']['validation' if split == 'val' else 'train']
        assert item['source_npz'] == prior['path'] and item['accepted_stat'] == prior['accepted_stat'] and item['rows'] == count * 1001
        for key, width in [('qpos', 42), ('qvel', 38)]: assert item[key]['shape'] == [count * 1001, width] and item[key]['dtype'] == 'float32'
    plan = json.loads((original / MODULE / 'producer_plan.json').read_text())
    pins, references = {}, {}
    for name, expected in plan['source_pins'].items():
        name = name.replace('checkpoints/cube2/3d_epoch10/', 'checkpoints/cube4/3d_epoch10/')
        if name in sources: pins[name] = hashlib.sha256(sources[name]).hexdigest(); continue
        source = rgb_workspace / name if name.startswith('official_state_2d_cache_v1/') else (original / name if name.startswith(MODULE + '/') else root / name)
        if name.startswith('official_state_2d_cache_v1/'):
            expected = binding['old_quad_RGB_source_pins'][name]
        if name.startswith('checkpoints/cube4/3d_epoch10/'):
            expected = binding['checkpoint_config'][Path(name).name]['sha256']
        assert source.is_file() and sha(source) == expected, str(source)
        references[name] = source; pins[name] = expected
    for name, content in sources.items(): pins[name] = hashlib.sha256(content).hexdigest()
    pins['official_state_2d_cache_v1/audit_cache.py'] = RGB_AUDITOR_SHA
    pins['binding/PCD_ENCODING_BINDING_v1.json'] = binding_sha
    plan.update(workspace=str(case / 'workspace'), shared_state_root=binding['source_arrays_root'],
        data_acceptance=binding['data_acceptance']['path'], data_acceptance_sha256=binding['data_acceptance']['sha256'],
        source_stat_freeze=binding['source_stat_freeze']['path'], source_stat_freeze_sha256=binding['source_stat_freeze']['sha256'],
        checkpoint=str(case / 'workspace/checkpoints/cube4/3d_epoch10/weights_epoch_10.pt'), checkpoint_sha256=CHECKPOINT_SHA,
        source_arrays_receipt_sha256=binding['source_arrays_view']['sha256'], source_pins=pins, camera=binding['camera'],
        pcd=pcd, split_episodes=binding['split_episodes'], cache_dir=str(case / '3d/cache'), gate_dir=str(case / '3d/gate'))
    return plan, references


def resource_profile(plan_path, gate_path):
    plan = json.loads(plan_path.read_text()); gate = json.loads(gate_path.read_text())
    assert gate['actual_exit_code'] == 0 and gate['accepted'] is True and gate['prepared_bitexact'] and gate['latent_bitexact']
    assert gate['candidate_median_seconds'] <= gate['baseline_median_seconds'] * .95
    assert gate['checkpoint_sha256'] == plan['checkpoint_sha256'] == CHECKPOINT_SHA and gate['plan_sha256'] == sha(plan_path)
    assert gate['candidate_source_sha256'] == EXPECTED[MODULE + '/acceleration_v1/v4/gpu_preprocess.py']
    assert Path(gate['samples_path']) == Path(plan['gate_dir']) / 'real_PCD32.npz' and sha(gate['samples_path']) == gate['samples_sha256']
    assert gate['source_samples_count'] == 32 and gate['real_raw_ids'] == list(range(29)) + [999, 1000, 1001]
    return dict(base_producer=str(Path(plan['workspace']) / MODULE / 'produce.py'),
        base_producer_sha256=plan['source_pins'][MODULE + '/produce.py'], renderer_workers=16,
        prefetch_jobs=32, affinity_cores=16, GPU_preprocess=True,
        GPU_preprocess_gate=dict(path=str(gate_path), sha256=sha(gate_path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('materialize', 'prepare-profile'), required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--original-workspace', type=Path)
    parser.add_argument('--rgb-workspace', type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--expected-binding-sha256')
    args = parser.parse_args(); case = args.case.resolve()
    if args.phase == 'prepare-profile':
        plan = case / 'workspace' / MODULE / 'producer_plan.json'
        gate = case / '3d/GPU_preprocess_report.json'
        write(case / '3d/resource_profile.json', resource_profile(plan, gate))
    else:
        root, original, rgb = args.root.resolve(), args.original_workspace.resolve(), args.rgb_workspace.resolve()
        assert case.is_relative_to(root) and not original.is_relative_to(case) and not rgb.is_relative_to(case)
        workspace = case / 'workspace'; assert not workspace.exists(), 'Never overwrite a materialized workspace'
        sources, ledger = build_sources(original)
        plan, references = make_plan(root, original, rgb, case, args.binding, args.expected_binding_sha256, sources)
        # Every reference is checked before any new directory or symlink is created.
        for name in ('source', 'external', 'reference', 'env_runtime', 'checkpoint_encoder.py', 'checkpoints'):
            assert (root / name).exists(), name
        workspace.mkdir(parents=True)
        for name in ('source', 'external', 'reference', 'env_runtime', 'checkpoint_encoder.py', 'checkpoints'):
            target = root / name; (workspace / name).symlink_to(target, target_is_directory=target.is_dir())
        (workspace / 'official_state_2d_cache_v1').symlink_to(rgb / 'official_state_2d_cache_v1', target_is_directory=True)
        for name, content in sources.items(): write(workspace / name, content)
        # Preserved legacy test source is a source pin only, never executed here.
        for name, source in references.items():
            if name.startswith(MODULE + '/'): write(workspace / name, source.read_bytes())
        write(workspace / 'binding/PCD_ENCODING_BINDING_v1.json', args.binding.read_bytes())
        write(workspace / MODULE / 'producer_plan.json', plan)
        write(case / 'materialization_receipt.json', dict(actual_exit_code=0, binding_sha256=args.expected_binding_sha256,
            producer_plan_sha256=sha(workspace / MODULE / 'producer_plan.json'), modules=ledger,
            source_arrays_copied_or_rederived=False, original_sources_modified=False, NN_GPU_executed=False))
    print(json.dumps(dict(actual_exit_code=0, phase=args.phase, NN_GPU_executed=False)))


if __name__ == '__main__':
    main()

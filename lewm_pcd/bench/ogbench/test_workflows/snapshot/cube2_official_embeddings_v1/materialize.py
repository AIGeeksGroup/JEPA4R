"""Materialize only the Double schema/resource adapter over frozen Triple producers.

No downloads, model imports, rendering, encoding or dispatch occur in this tool.
Original source directories are never written. All changes have exact-count edits.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

P2 = 'official_state_2d_cache_v1'
P3 = 'official_state_3d_cache_v1'
V4 = P3 + '/acceleration_v1/v4'
CROP = [[.1, -.5, -.02], [.75, .5, .6]]
EXPECTED = {
    P2+'/produce.py': 'dbbce2407f11c42ee2c5aca7601c9a055a7ceb08593fb4e99b1b0cd7e5415b18',
    P2+'/render_adapter.py': '5df6dc694f7c9d38ccff69385080b1bad814f41ad6c4ae71e9def2e6ac71662f',
    P2+'/audit_cache.py': 'a97a55ce945c12b16d6fb473c71b289cdf8b2a0164b697ae0f8d4c5d7f9b5006',
    P3+'/common.py': '93ff5465c43706e59465eab1aad9764e05f1d3a6b0c54e6dd8659f1b15260950',
    P3+'/pcd_renderer.py': '0aefbcbe52b2a9a4b8136cb2fe6f6f57e8069f86cd79b874d731525a5d5bf676',
    P3+'/produce.py': '7e4c201bb259a005e6b3cc49e4a2df3b3963e347c4a6aa26b17ea22666dce4db',
    P3+'/gate.py': '4fcab671b8a9ad9876c3a2470e7fbc89a375bfdea3d82be9783a3093f6104bd4',
    P3+'/audit_cache.py': '8b1d14046cc8ea0e97d9fd3304902a00c42bc549861dfb5b7b3db9d8491d68f7',
    V4+'/run_profile.py': '402a53952cc1d8f304d97449ff9e2cf008c29eb6ffbcb5f07f7be06437f48f08',
    V4+'/gpu_preprocess.py': '7929544aab09cb17cd25b269ef0830259456c9fa76fcd62cf07e5d19819e35bf',
    V4+'/bench_gpu_preprocess.py': 'd6d2282ba63e4f0d48fff2dccaf89aa352b08b52b85d75086a75b6f401fd0d21',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data if isinstance(data, bytes) else (json.dumps(data, indent=2, allow_nan=False)+'\n').encode())


def build_sources(original):
    """Pure text adaptation; every replacement count is pinned and logged."""
    output, ledger = {}, []
    def edit(name, replacements):
        original_bytes = (original/name).read_bytes()
        assert hashlib.sha256(original_bytes).hexdigest() == EXPECTED[name], name
        source = original_bytes.decode()
        changes = []
        for before, after, count in replacements:
            assert source.count(before) == count, (name, before, count, source.count(before))
            source = source.replace(before, after)
            changes.append(dict(before=before, after=after, count=count))
        ast.parse(source)
        output[name] = source.encode()
        ledger.append(dict(path=name, original_sha256=EXPECTED[name],
            generated_sha256=hashlib.sha256(output[name]).hexdigest(), replacements=changes,
            diff=''.join(difflib.unified_diff(original_bytes.decode().splitlines(True), source.splitlines(True), fromfile='original/'+name, tofile='cube2/'+name))))
    counts = [("{'train': 3000, 'val': 300}", "{'train': 1000, 'val': 100}", 1)]
    totals = [('3303300','1101100',1), ('3300','1100',1), ('3003000','1001000',1), ('300300','100100',1)]
    edit(P2+'/produce.py', counts + totals + [
        ("('qpos', 35), ('qvel', 32)", "('qpos', 28), ('qvel', 26)", 1),
        ('qpos_width=35, qvel_width=32','qpos_width=28, qvel_width=26',1),
        ('shape[1] == 46','shape[1] == 37',1), ('46 * 4','37 * 4',2), ('range(3)','range(2)',1)])
    edit(P2+'/render_adapter.py', [('cube-triple-v0','cube-double-v0',1),
        ('(35, 32, 7)','(28, 26, 7)',1),('range(3)','range(2)',2),('% 3','% 2',1),
        ('nq=35, nv=32','nq=28, nv=26',1)])
    edit(P2+'/audit_cache.py', counts + [
        ("accepted['stage'] == 'complete'", "accepted['data_id'] == 'cube-double-play-v0' and accepted['cube_count'] == 2",1),
        ('qpos_width=35, qvel_width=32','qpos_width=28, qvel_width=26',1),
        ("('qpos', 35), ('qvel', 32)","('qpos', 28), ('qvel', 26)",1),
        ('nq=35, nv=32','nq=28, nv=26',1),
        ('minimum_gain=.05, warmup_batches=3, timed_batches_per_path=20)',
         "minimum_gain=.05, warmup_batches=3, timed_batches_per_path=20,\n            scope='original preprocessing plus frozen encode; rendering excluded; CUDA tail synchronized')",1),
        ('Require full official3000/300 cache','Require full official1000/100 Double cache',1)])
    edit(P3+'/common.py', counts + [
        ('qpos_width=35, qvel_width=32','qpos_width=28, qvel_width=26',1),
        ('triple_official_pcd_epoch10_v1','double_official_pcd_epoch10_v1',2),
        ('[[-.25, -.5, -.02], [.75, .5, .9]]','[[.1, -.5, -.02], [.75, .5, .6]]',1)])
    edit(P3+'/pcd_renderer.py', [('Original Triple AA4','Original Double AA4',1),
        ('qpos.shape == (35,) and qvel.shape == (32,)','qpos.shape == (28,) and qvel.shape == (26,)',1),
        ('range(3)','range(2)',1), ('nq=35, nv=32','nq=28, nv=26',1),('% 3','% 2',1)])
    edit(P3+'/produce.py', totals)
    edit(P3+'/gate.py', [])
    two_audit_sha = hashlib.sha256(output[P2+'/audit_cache.py']).hexdigest()
    edit(P3+'/audit_cache.py', counts + [
        ("accepted['stage'] == 'complete'", "accepted['data_id'] == 'cube-double-play-v0' and accepted['cube_count'] == 2",1),
        ('a97a55ce945c12b16d6fb473c71b289cdf8b2a0164b697ae0f8d4c5d7f9b5006',two_audit_sha,1),
        ('qpos_width=35, qvel_width=32','qpos_width=28, qvel_width=26',1),
        ("('qpos',35), ('qvel',32)","('qpos',28), ('qvel',26)",1),
        ('nq=35,nv=32','nq=28,nv=26',1),
        ('[[-.25,-.5,-.02],[.75,.5,.9]]','[[.1,-.5,-.02],[.75,.5,.6]]',1),
        ('triple_official_pcd_epoch10_v1','double_official_pcd_epoch10_v1',1)])
    three_source_sha = hashlib.sha256(output[P3+'/produce.py']).hexdigest()
    edit(V4+'/run_profile.py', [('7e4c201bb259a005e6b3cc49e4a2df3b3963e347c4a6aa26b17ea22666dce4db',three_source_sha,1)])
    edit(V4+'/gpu_preprocess.py', [])
    edit(V4+'/bench_gpu_preprocess.py', [])
    return output, ledger


def plans(original, case, shadow):
    acceptance, freeze = case/'data_acceptance.json', case/'stat-freeze.json'
    accepted = json.loads(acceptance.read_text())
    frozen = json.loads(freeze.read_text())
    assert accepted['data_id'] == 'cube-double-play-v0' and accepted['cube_count'] == 2
    assert accepted['actual_exit_code'] == frozen['actual_exit_code'] == 0
    assert sha(acceptance) == '1bcafcafe815298041e42781ec8e6c25b49a9263e81d3eb1a29860e25c837cf0'
    assert sha(freeze) == '53e5bd43e3054287ab9c6ad8a7ac79e954e2e028b984ea8cf0a8a54c1ec59360'
    assert frozen['acceptance_sha256'] == sha(acceptance)
    assert Path(frozen['datasets']['train']['path']).name == 'cube-double-play-v0.npz'
    assert Path(frozen['datasets']['validation']['path']).name == 'cube-double-play-v0-val.npz'
    result = {}
    for mode, module in [('2d',P2),('3d',P3)]:
        plan = copy.deepcopy(json.loads((original/module/'producer_plan.json').read_text()))
        plan.update(workspace=str(shadow), shared_state_root=str(case/'source_arrays'),
            checkpoint=str(shadow/f'checkpoints/cube2/{mode}_epoch10/weights_epoch_10.pt'),
            data_acceptance=str(acceptance), data_acceptance_sha256=sha(acceptance),
            source_stat_freeze=str(freeze), source_stat_freeze_sha256=sha(freeze),
            split_episodes=dict(train=1000,val=100), cache_dir=str(case/mode/'cache'))
        plan['checkpoint_sha256'] = sha(plan['checkpoint'])
        expected_ckpt = {'2d':'995a99f4904d87e71424b2de7910e7e0cf79556de8c2cebd08ac582d70880190',
                         '3d':'f391f047aa73fa08b2b4534e6018fde6a219cc3ae6ba4464b907e2b9f9138a46'}
        assert plan['checkpoint_sha256'] == expected_ckpt[mode]
        old_pins = plan['source_pins']
        plan['source_pins'] = {}
        for relative in old_pins:
            relative = relative.replace(f'checkpoints/{mode}_epoch10/',f'checkpoints/cube2/{mode}_epoch10/')
            plan['source_pins'][relative] = sha(shadow/relative)
        if mode == '3d':
            plan['gate_dir'] = str(case/mode/'gate')
            plan['pcd']['crop_box'] = CROP
            plan['pcd']['sampling_namespace'] = 'double_official_pcd_epoch10_v1'
            plan['source_arrays_receipt_sha256'] = None
            for relative in (V4+'/run_profile.py',V4+'/gpu_preprocess.py'):
                plan['source_pins'][relative] = sha(shadow/relative)
        result[mode] = plan
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root',type=Path,required=True)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--phase',choices=('materialize','prepare-arrays','finalize-3d','profile'),default='materialize')
    args = parser.parse_args()
    original, case = args.original_root.resolve(), args.case.resolve()
    shadow = case/'workspace'
    assert case != original and case.is_relative_to(original), 'New case must be a strict child of original workspace'
    if args.phase == 'materialize':
        assert not shadow.exists(), 'Never overwrite a previous materialization'
        generated, ledger = build_sources(original)
        shadow.mkdir(parents=True)
        for name in ('source','external','reference','env_runtime','checkpoint_encoder.py','checkpoints'):
            assert (original/name).exists(), name
            (shadow/name).symlink_to(original/name,target_is_directory=(original/name).is_dir())
        for name, content in generated.items():
            write(shadow/name,content)
        all_plans = plans(original,case,shadow)
        write(shadow/P2/'producer_plan.json',all_plans['2d'])
        write(case/'3d/producer_plan.template.json',all_plans['3d'])
        write(case/'materialization_receipt.json',dict(actual_exit_code=0,original_root=str(original),
            shadow_root=str(shadow),cube_count=2,raw_rows_per_episode=1001,split_episodes=dict(train=1000,val=100),
            modules=ledger,original_files_modified=False,model_or_GPU_executed=False,
            twoD_plan_sha256=sha(shadow/P2/'producer_plan.json'),
            threeD_plan_pending_source_arrays=True))
    elif args.phase == 'prepare-arrays':
        sys.path.insert(0,str(shadow/P2))
        spec = importlib.util.spec_from_file_location('_Double_original_array_extractor',shadow/P2/'produce.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plan = json.loads((shadow/P2/'producer_plan.json').read_text())
        file_sha, stat, save = module.writer_helpers(str(shadow))
        contract = dict(data_acceptance_sha256=plan['data_acceptance_sha256'],
            source_stat_freeze_sha256=plan['source_stat_freeze_sha256'],
            split_episodes=plan['split_episodes'],raw_rows_per_episode=1001,
            qpos_width=28,qvel_width=26,extractor_source_sha256=file_sha(shadow/P2/'produce.py'))
        module.source_arrays(plan,module.object_sha(contract),file_sha,stat,save,contract)
    elif args.phase == 'finalize-3d':
        plan = json.loads((case/'3d/producer_plan.template.json').read_text())
        arrays_path = case/'source_arrays/source_arrays.json'
        arrays = json.loads(arrays_path.read_text())
        assert arrays['contract']['split_episodes'] == dict(train=1000,val=100)
        assert arrays['contract']['qpos_width'] == 28 and arrays['contract']['qvel_width'] == 26
        assert arrays['contract']['data_acceptance_sha256'] == plan['data_acceptance_sha256']
        plan['source_arrays_receipt_sha256'] = sha(arrays_path)
        write(shadow/P3/'producer_plan.json',plan)
    else:
        plan = shadow/P3/'producer_plan.json'
        gate_path = case/'3d/GPU_preprocess_report.json'
        gate = json.loads(gate_path.read_text())
        assert gate['actual_exit_code'] == 0 and gate['accepted'] is True
        assert gate['prepared_bitexact'] and gate['latent_bitexact']
        assert gate['plan_sha256'] == sha(plan)
        assert gate['candidate_source_sha256'] == sha(shadow/V4/'gpu_preprocess.py')
        profile = json.loads((original/P3/'acceleration_v1/chosen_profile_v1.json').read_text())
        profile.update(base_producer=str(shadow/P3/'produce.py'),base_producer_sha256=sha(shadow/P3/'produce.py'),
            status='new_Double_profile_new_checkpoint_and_plan_GPU_gate_accepted',
            GPU_preprocess_gate=dict(path=str(gate_path),sha256=sha(gate_path)),
            legacy199_receipt_SHA_preservation=False)
        profile.pop('renderer_selection',None)
        profile['resource_profile_reused_from_Triple'] = True
        profile['Double_renderer_speed_not_claimed_before_actual_run'] = True
        write(case/'3d/chosen_profile.json',profile)
    print(json.dumps(dict(actual_exit_code=0,phase=args.phase,shadow=str(shadow),GPU_or_encoder_executed=False)))


if __name__ == '__main__':
    main()

"""CPU contracts for the opt-in official-task initialization wrapper."""
import hashlib
import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np


LEWM_ROOT = Path(__file__).resolve().parents[2]
SOURCE = LEWM_ROOT / 'rh56_adapter/official_init.py'


def wrapper():
    assert SOURCE.exists(), 'official initialization wrapper is not implemented'
    spec = importlib.util.spec_from_file_location('official_init_collect_tested', SOURCE)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, 'path', [str(LEWM_ROOT), *sys.path]):
        spec.loader.exec_module(module)
    return module


def manifest(tmp_path):
    source = tmp_path / 'original_manifest.json'
    source.write_text('{"source": "official light demonstrations"}')
    records = [dict(demo_index=i, obj_pos=[i / 1000, .1, .035],
                    target_pos=[.1, -.1, .25], original_init_sha256='a' * 64,
                    source_id=f'demo_{i:04d}.pickle') for i in range(100)]
    value = dict(schema='official-task-init-v1', source_manifest=str(source),
                 source_sha=hashlib.sha256(source.read_bytes()).hexdigest(), records=records)
    path = tmp_path / 'task_initials.json'
    path.write_text(json.dumps(value))
    return path, value


def check_manifest_keeps_exact_official_order_and_seed_mapping(tmp_path):
    module = wrapper()
    path, _ = manifest(tmp_path)
    payload, records = module.load_initials(path, 64500000)
    assert list(records) == list(range(64500000, 64500100))
    assert [r['demo_index'] for r in records.values()] == list(range(100))
    assert payload['schema'] == 'official-task-init-v1'


def check_manifest_rejects_ambiguous_or_changed_input(tmp_path, change):
    module = wrapper()
    path, payload = manifest(tmp_path)
    if change == 'reordered':
        payload['records'][0], payload['records'][1] = payload['records'][1], payload['records'][0]
    elif change == 'nonfinite':
        payload['records'][0]['obj_pos'][0] = float('nan')
    else:
        Path(payload['source_manifest']).write_text('changed')
    path.write_text(json.dumps(payload))
    with unittest.TestCase().assertRaises(ValueError):
        module.load_initials(path, 64500000)


def check_reset_changes_only_task_positions_and_returns_fresh_observation():
    module = wrapper()

    class Native:
        def reset(self, seed=None):
            self.seed = seed
            data = SimpleNamespace(qpos=np.array([1., 2.]), qvel=np.array([.1, .2]),
                                   ctrl=np.array([3.]), act=np.array([4.]), xpos=np.zeros((2, 3)),
                                   site_xpos=np.zeros((2, 3)))
            model = SimpleNamespace(body_pos=np.zeros((2, 3)), site_pos=np.zeros((2, 3)))
            self.env = SimpleNamespace(data=data, model=model, obj_bid=1, target_sid=1,
                                       p=np.array([5.]), v=np.array([6.]), q_cmd=np.array([7.]))
            return np.zeros(6), {'seed': seed}

        def observation(self):
            return np.r_[self.env.data.xpos[1], self.env.data.site_xpos[1]]

    def forward(model, data):
        data.xpos[:] = model.body_pos
        data.site_xpos[:] = model.site_pos

    row = dict(demo_index=3, obj_pos=[.12, -.04, .035], target_pos=[-.1, .2, .3],
               source_id='demo3', original_init_sha256='b' * 64)
    receipts = []
    bridge = module.make_bridge(Native, {64500003: row}, forward, receipts.append)()
    obs, info = bridge.reset(seed=64500003)
    np.testing.assert_array_equal(obs, [.12, -.04, .035, -.1, .2, .3])
    np.testing.assert_array_equal(bridge.env.data.qpos, [1., 2.])
    np.testing.assert_array_equal(bridge.env.data.qvel, [.1, .2])
    np.testing.assert_array_equal(bridge.env.data.ctrl, [3.])
    np.testing.assert_array_equal(bridge.env.q_cmd, [7.])
    assert info == {'seed': 64500003}
    assert receipts[0]['demo_index'] == 3
    assert receipts[0]['robot_and_drive_unchanged'] is True
    with unittest.TestCase().assertRaisesRegex(ValueError, 'not in'):
        bridge.reset(seed=64500999)

    def wrong_forward(model, data):
        forward(model, data)
        data.qpos[0] += 1

    wrong_bridge = module.make_bridge(Native, {64500003: row}, wrong_forward, receipts.append)()
    with unittest.TestCase().assertRaises(AssertionError):
        wrong_bridge.reset(seed=64500003)


def check_failed_official_source_is_reported_without_replacement_or_filtering():
    module = wrapper()
    records = [dict(seed=64500000, success=True), dict(seed=64500001, success=False)]
    result = module.summarize_attempts(records, [64500000, 64500001])
    assert result['attempted'] == 2
    assert result['failed_seeds'] == [64500001]
    assert result['all_requested_sources_succeeded'] is False
    assert records[1]['success'] is False


class OfficialInitializationTests(unittest.TestCase):
    def test_order_and_seed_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            check_manifest_keeps_exact_official_order_and_seed_mapping(Path(temp))

    def test_changed_source_order_and_nonfinite(self):
        for change in ('reordered', 'nonfinite', 'changed_source'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                check_manifest_rejects_ambiguous_or_changed_input(Path(temp), change)

    def test_physical_reset_boundary(self):
        check_reset_changes_only_task_positions_and_returns_fresh_observation()

    def test_failures_are_kept(self):
        check_failed_official_source_is_reported_without_replacement_or_filtering()

    def test_bc_manifest_preserves_all100_successes_and_profile_case(self):
        module=wrapper()
        sources={64500000+i:dict(demo_index=i,source_id=f'demo{i}',original_init_sha256='a'*64)
                 for i in range(100)}
        attempts=[dict(seed=seed,success=True,npz=f'bc_demo_{seed}.npz') for seed in sources]
        original=json.loads(json.dumps(attempts))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def build(rows,seeds):
                return module.build_bc_manifest(rows,seeds,root/'episodes',sources,
                    init_source=root/'init_source.json',init_source_sha256='b'*64,
                    producer_chain={'path':str(root/'producer_chain.json'),'sha256':'c'*64})
            result=build(attempts,list(sources))
            self.assertEqual(result['status'],'READY_FORMAL_BC')
            self.assertTrue(result['formal_bc_ready'])
            self.assertEqual([r['seed'] for r in result['selected']],list(sources))
            self.assertEqual([r['official_demo_index'] for r in result['selected']],list(range(100)))
            self.assertTrue(all(Path(r['npz']).is_absolute() for r in result['selected']))
            self.assertEqual(result['selected'],result['all_attempts'])
            self.assertEqual(attempts,original)
            profile=build(attempts[:1],list(sources)[:1])
            self.assertEqual(profile['status'],'READY_PROFILE_BC')
            self.assertTrue(profile['profile_bc_ready'])
            self.assertFalse(profile['formal_bc_ready'])

    def test_bc_manifest_rejects_any_failure_without_selecting_success_subset(self):
        module=wrapper()
        sources={64500000+i:dict(demo_index=i,source_id=f'demo{i}',original_init_sha256='a'*64)
                 for i in range(100)}
        attempts=[dict(seed=seed,success=(i != 37),npz=f'bc_demo_{seed}.npz')
                  for i,seed in enumerate(sources)]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            result=module.build_bc_manifest(attempts,list(sources),root/'episodes',sources,
                init_source=root/'init_source.json',init_source_sha256='b'*64,
                producer_chain={'path':str(root/'producer_chain.json'),'sha256':'c'*64})
            self.assertEqual(result['status'],'REJECTED_FAILED_OFFICIAL_SOURCE')
            self.assertEqual(result['selected'],[])
            self.assertEqual(len(result['all_attempts']),100)
            self.assertFalse(result['all_attempts'][37]['success'])
            self.assertFalse(result['formal_bc_ready'])
            with self.assertRaisesRegex(ValueError,'source order'):
                module.build_bc_manifest(attempts[::-1],list(sources),root/'episodes',sources,
                    init_source=root/'init_source.json',init_source_sha256='b'*64,producer_chain={})
            attempts[0]['npz']='../outside.npz'
            with self.assertRaisesRegex(ValueError,'inside'):
                module.build_bc_manifest(attempts,list(sources),root/'episodes',sources,
                    init_source=root/'init_source.json',init_source_sha256='b'*64,producer_chain={})


def package_module(name):
    path=LEWM_ROOT/'rh56_adapter'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('_rh56_test_'+name,path)
    module=importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, 'path', [str(LEWM_ROOT), *sys.path]):
        spec.loader.exec_module(module)
    return module


class CollectionBoundaryTests(unittest.TestCase):
    def test_default_package_import_loads_no_simulator_or_training_stack(self):
        env=dict(os.environ,PYTHONPATH=str(LEWM_ROOT),PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-c',
            "import sys; import rh56_adapter; assert not any(k in sys.modules for k in ('mujoco','torch','train'))"],
            cwd=LEWM_ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_native_paths_use_repo_and_require_an_external_expert(self):
        path=LEWM_ROOT/'rh56_adapter/native.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        nodes=[node for node in tree.body if
            isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('ROOT','EXPERT','BASELINE')
                                               for t in node.targets)
            or isinstance(node,ast.FunctionDef) and node.name=='expert_files']
        def scope(environ):
            namespace={'__file__':str(path),'Path':Path,'os':SimpleNamespace(environ=environ)}
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
            return namespace
        missing=scope({})
        self.assertEqual(missing['BASELINE'],LEWM_ROOT)
        with self.assertRaisesRegex(ValueError,'RH56_EXPERT_ROOT'):missing['expert_files']()
        with tempfile.TemporaryDirectory() as temp:
            expert=Path(temp)
            for name in ('env.py','ppo.py','experts/relocate_expert.pt'):
                p=expert/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'external fixture')
            bound=scope({'RH56_EXPERT_ROOT':str(expert)})
            self.assertEqual(bound['expert_files']()['env.py'],expert.resolve()/'env.py')
            (expert/'ppo.py').unlink()
            with self.assertRaisesRegex(FileNotFoundError,'ppo.py'):bound['expert_files']()

    def test_plans_require_distinct_real_seed_labels_and_expert_auxiliaries(self):
        module=package_module('collect')
        valid={'episodes':[{'split':'train','seed':61000000,'coverage':'protocol'},
                           {'split':'bc_demo','seed':64500000,'coverage':'expert'}]}
        self.assertEqual(module.validate_plan(valid),[('train',61000000,'protocol'),('bc_demo',64500000,'expert')])
        for bad in ({'split':'train','seed':61000000,'coverage':'expert'},
                    {'split':'bc_demo','seed':64500001,'coverage':'protocol'},
                    {'split':'train','seed':True,'coverage':'protocol'}):
            with self.subTest(row=bad),self.assertRaises(ValueError):
                module.validate_plan({'episodes':[valid['episodes'][0],bad]})

    def test_original_mix_is_unforced_and_auxiliary_expert_has_no_noise(self):
        module=package_module('collect');seen=[]
        def original(config,rng):
            seen.append(config)
            return 0,.125,-1,-1
        result=module.draw_collection_plan(original,'protocol',object())
        self.assertIsNone(seen[-1].force_type)
        self.assertEqual(seen[-1].mix,(.5,.3,.2))
        self.assertEqual(result[:2],(0,.125))
        self.assertEqual(module.draw_collection_plan(original,'expert',object())[:2],(0,0.))
        self.assertEqual(seen[-1].force_type,0)

    def test_goal_receipt_keeps_native_flags_separate(self):
        module=package_module('collect')
        geometry=np.r_[np.ones(30,dtype=bool),np.zeros(10,dtype=bool)]
        native=geometry.copy();native[0]=False
        receipt=module.goal_receipt(geometry,native,True,False)
        self.assertTrue(receipt['success_consistent'])
        self.assertEqual(receipt['goal_steps'],30)
        self.assertEqual(receipt['native_goal_steps'],29)
        self.assertEqual(receipt['goal_flag_mismatch_steps'],[0])
        self.assertFalse(module.goal_receipt(np.ones(25),np.ones(25),False,False)['success'])
        self.assertFalse(module.goal_receipt(geometry,native,False,True)['success'])

    def test_registry_rejects_duplicate_physical_initials_even_with_different_seeds(self):
        module=package_module('data_tools')
        rows=[{'seed':1,'split':'train','initial_state_sha256':'a'*64},
              {'seed':2,'split':'val','initial_state_sha256':'a'*64}]
        with self.assertRaisesRegex(ValueError,'physical initial'):module.check_registry(rows)


def direct_batch(root,split,seed):
    """A tiny exact-shape source fixture, not a simulated or qualified episode."""
    import h5py
    module=package_module('data_tools')
    root.mkdir()
    initial={key:[[float(seed)]] for key in
             ('integration','body_pos','body_quat','site_pos','site_quat','p','v','q_cmd')}
    initial_hash=hashlib.sha256(json.dumps(initial,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    pixels=np.zeros((2,256,256,3),np.uint8);xyz=np.zeros((2,1200,3),np.float32)
    colors=np.zeros((2,1200,3),np.uint8);actions=np.zeros((1,12),np.float32)
    observation=np.zeros((2,56),np.float32);proprio=np.zeros((2,13),np.float32)
    qpos=np.zeros((2,25));qvel=qpos.copy();trace=np.zeros((1,9));rewards=np.zeros(1)
    camera={'mode':'opposite','position':[0.,1.2,1.2],'fovy':45.}
    npz=root/f'{split}_{seed}.npz'
    np.savez_compressed(npz,pixels=pixels,xyz=xyz,rgb=colors,actions=actions,
        observations=observation,proprio=proprio,qpos=qpos,qvel=qvel,trace=trace,rewards=rewards,
        integration=np.zeros((2,1)),initial_state_json=np.array(json.dumps(initial)),
        initial_state_sha256=np.array(initial_hash),seed=np.int64(seed),success=np.bool_(False),
        native_success=np.bool_(False),camera_json=np.array(json.dumps(camera)))
    config={'camera':'opposite'};config_hash=hashlib.sha256(
        json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    provenance={'producer_sha256':'a'*64,'config_sha256':config_hash}
    record={'split':split,'seed':seed,'episode_index':0,'initial_state_sha256':initial_hash,
        'rows':2,'steps':1,'npz':npz.name,'npz_sha256':module.sha(npz),
        'producer_sha256':provenance['producer_sha256'],'config_sha256':config_hash,
        'camera':camera,'sample_type':0,'branch_t':-1,'branch_len':-1,'noise_std':0.,
        'success':False,'native_success':False,'success_consistent':True,'requested_coverage':'protocol'}
    for name,value in [('state.json',{'phase':'complete','exit_code':0}),
                       ('config.json',{'config':config,'provenance':provenance}),
                       ('manifest.json',{'episodes':[record],'auxiliary':[],'provenance':provenance})]:
        (root/name).write_text(json.dumps(value))
    shared={'action':np.r_[actions,np.full((1,12),np.nan)].astype(np.float32),
            'observation':observation,'proprio':proprio,'qpos':qpos,'qvel':qvel,
            'episode_seed':np.full(2,seed,np.int64),'step_idx':np.arange(2,dtype=np.int32),
            'rand_action':np.zeros(2,np.int8)}
    for route in ('rgb','pcd'):
        with h5py.File(root/f'paired_{route}.h5','w') as f:
            f['ep_len']=np.array([2],np.int32);f['ep_offset']=np.array([0],np.int64)
            for key,value in shared.items():f[key]=value
            for key,value in ({'pixels':pixels} if route=='rgb' else {'pcd_xyz':xyz,'pcd_rgb':colors}).items():
                f[key]=value
    return root


@unittest.skipUnless(importlib.util.find_spec('h5py'),'h5py optional for direct paired-data tests')
class DirectPairedDataTests(unittest.TestCase):
    def test_validate_publish_preserves_payloads_and_terminal_sentinel(self):
        import h5py
        module=package_module('data_tools')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=direct_batch(root/'train','train',61000000);b=direct_batch(root/'val','val',62000000)
            for batch in (a,b):module.validate_batch(batch)
            before={p:module.sha(p) for batch in (a,b) for p in batch.iterdir() if p.is_file()}
            out=root/'published';result=module.publish_dataset([a,b],out,{'train':1,'val':1})
            self.assertEqual(result['status'],'PASS_VDS')
            for p,digest in before.items():self.assertEqual(module.sha(p),digest)
            with h5py.File(out/'paired_rgb.h5','r') as f:
                np.testing.assert_array_equal(f['ep_len'][:],[2,2])
                np.testing.assert_array_equal(f['ep_offset'][:],[0,2])
                self.assertTrue(np.isnan(f['action'][[1,3]]).all())
            with self.assertRaises(FileExistsError):module.validate_batch(a)
            with self.assertRaises(FileExistsError):module.publish_dataset([a,b],out,{'train':1,'val':1})

    def test_terminal_action_and_modified_accepted_payload_are_rejected(self):
        import h5py
        module=package_module('data_tools')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=direct_batch(root/'bad_terminal','train',61000000)
            for route in ('rgb','pcd'):
                with h5py.File(a/f'paired_{route}.h5','a') as f:f['action'][-1]=0.
            with self.assertRaisesRegex(ValueError,'terminal NaN'):module.validate_batch(a)
            b=direct_batch(root/'accepted','train',61000001);module.validate_batch(b)
            with next(b.glob('*.npz')).open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'accepted payload changed'):module.verify_receipt(b)


if __name__=='__main__':unittest.main()

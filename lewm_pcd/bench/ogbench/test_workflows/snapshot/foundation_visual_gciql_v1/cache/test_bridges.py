"""CPU-only counted-patch/contract regression over all supported N and widths."""
import ast
import json
import pickle
import unittest
import tempfile
from pathlib import Path
import numpy as np
import producer_bridge
import audit_bridge
import render_adapter
from common import COUNTS
from common import save,sha,stat
import deliver
from prepare_plans import namespace

class Bridges(unittest.TestCase):
    def test_random_namespace_isolation(self):
        family='vjepa21_vitl1024'
        official,random=namespace(family,dict(family=family))
        diagnostic=dict(family=family,init='random',diagnostic_only=True)
        isolated,is_random=namespace(family,diagnostic,'random_init_v1_',True)
        self.assertFalse(random);self.assertTrue(is_random);self.assertNotEqual(official,isolated)
        self.assertEqual(namespace(family,dict(family=family,initialization='random',diagnostic_only=True),
                                   'random_init_v1_',True),(isolated,True))
        for prefix,pilot in (('',True),('random_init_v1_',False)):
            with self.assertRaises(AssertionError):namespace(family,diagnostic,prefix,pilot)
        with self.assertRaises(AssertionError):namespace(family,dict(family=family,init='random'),'random_init_v1_',True)
    def test_all_counted_variants_compile(self):
        for n in COUNTS:
            for d in (768,1024):
                plan=dict(cube_count=n,latent_dim=d)
                for text in (producer_bridge.effective_source(plan),audit_bridge.source(plan)):
                    ast.parse(text)
                m=producer_bridge.load(plan)
                np.testing.assert_array_equal(m.canonical_ids(17),np.arange(17017,18018,dtype=np.int64))
                a=audit_bridge.load(plan)
                self.assertEqual(a.EXPECTED_EPISODES,COUNTS[n]);self.assertEqual(a.DIM,d)
                self.assertEqual(m.CONTRACT_KEYS[-len(producer_bridge.EXTRA):],producer_bridge.EXTRA)
                self.assertNotIn('def choose_preprocessing',producer_bridge.effective_source(plan))
                self.assertIn("activate('cuda', None)",producer_bridge.effective_source(plan))
    def test_spawn_pickle_uses_importable_bridge(self):
        for f in (render_adapter.init_worker,render_adapter.render_rows,render_adapter.render_episode):
            self.assertIs(pickle.loads(pickle.dumps(f)),f)
    def test_camera_check_and_negative(self):
        for n in COUNTS:
            from pathlib import Path
            planned=json.loads((Path(__file__).parent/'data'/f'cube{n}'/'camera.json').read_text())
            actual=dict(planned,source_quaternion=planned['base_quaternion'],arm_material_alpha={'arm':1.})
            audit_bridge.check_camera(actual,planned)
            actual['position']=[0,0,0]
            with self.assertRaises(AssertionError):audit_bridge.check_camera(actual,planned)
    def test_actual_source_receipt_fields_and_pilot_delivery(self):
        # Verify actual original syntax has these zero-exit receipt fields.
        tree=ast.parse(producer_bridge.BASE.read_text())
        committed=[];pilot=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
                keys={k.arg:k.value for k in node.keywords}
                if 'status' in keys and isinstance(keys['status'],ast.Constant):
                    if keys['status'].value=='episode_committed':committed.append(keys)
                    if keys['status'].value=='pilot_complete_pending_outer_and_audit':pilot.append(keys)
        self.assertEqual(len(committed),1);self.assertEqual(len(pilot),1)
        self.assertEqual(ast.literal_eval(committed[0]['actual_exit_code']),0)
        self.assertEqual(ast.literal_eval(pilot[0]['actual_exit_code']),0)
        with tempfile.TemporaryDirectory(prefix='CPU_synthetic_fixture_',dir=Path(__file__).parent) as tmp:
            root=Path(tmp);cache=root/'cache';d=768
            camera=json.loads((Path(__file__).parent/'data/cube1/camera.json').read_text())
            plan=dict(cache_dir=str(cache),cube_count=1,latent_dim=d,encoder_family='synthetic_CPU_fixture',
                encoder_spec='fixture_spec',encoder_spec_sha256='a'*64,encoder_adapter='fixture_adapter',
                encoder_adapter_sha256='b'*64,checkpoint_sha256='c'*64,camera=camera)
            data=dict(cube_count=1,data_id='CPU_synthetic_fixture',splits={})
            for split in ('train','val'):
                p=root/f'{split}.npz';term=np.zeros(2002,np.bool_);term[[1000,2001]]=True
                np.savez(p,observations=np.zeros((2002,28),np.float32),actions=np.zeros((2002,5),np.float32),
                    qpos=np.zeros((2002,21),np.float32),qvel=np.zeros((2002,20),np.float32),terminals=term)
                data['splits'][split]=dict(path=str(p),accepted_stat=stat(p),episodes=2,raw_rows=2002)
                path=cache/split/'episode_000000/latent.npy';path.parent.mkdir(parents=True)
                np.save(path,np.zeros((1001,d),np.float32))
                save(path.parent/'receipt.json',dict(actual_exit_code=0,latent=dict(path=str(path),sha256=sha(path)),
                    raw_row_start=0,terminal_raw_id=1000,camera=dict(camera,source_quaternion=camera['base_quaternion'],
                    arm_material_alpha={'arm':1.})))
            save(root/'plan.json',plan);save(root/'data_binding.json',data)
            save(cache/'pilot_receipt.json',dict(actual_exit_code=0,not_full_cache=True))
            save(cache/'batch1_alignment.json',dict(actual_exit_code=0,passed=True,encoder_spec_sha256='a'*64))
            save(root/'producer_exit.json',dict(actual_exit_code=0,residual_owned_members=[]))
            deliver.publish(root/'plan.json',pilot=True,producer_exit_file=root/'producer_exit.json')
            result=json.loads((root/'pilot_fixture/delivery.json').read_text())
            self.assertEqual(result['purpose'],'pilot')
            self.assertEqual(result['data_binding']['splits']['train']['raw_rows'],1001)
            self.assertEqual(result['feature_loader_arguments']['val']['expected_rows'],1001)
            with np.load(root/'pilot_fixture/data/train.npz') as fixture:
                np.testing.assert_array_equal(np.flatnonzero(fixture['terminals']),[1000])

if __name__=='__main__':unittest.main()

"""Two fake-metadata recovery cases; no model, cache, NN or GPU operations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

HERE=Path(__file__).resolve().parent


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


recovery=module(HERE/'recovery.py','_recovery_tested')
oldfixture=module(HERE.parent/'functional_v1/test_prepare_G12.py','_frozen_fixture')
dispatch=module(HERE.parent/'control_v2/after_cache_G12.py','_dispatch_tested')
write=oldfixture.write


def fixture(root,complete=True):
    q,pcd=oldfixture.fixture(root,True)
    (q/'functional_v2/configs').mkdir(parents=True)
    (q/'functional_v1/prepare_G12.py').write_bytes((HERE.parent/'functional_v1/prepare_G12.py').read_bytes())
    (q/'functional_v2/recovery.py').write_bytes((HERE/'recovery.py').read_bytes())
    for relative in ['after_cache_G12.py','run_role.py','run_G11.sh','G11/launch_plan.json']:
        p=q/'control_v1'/relative;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((HERE.parent/'control_v1'/relative).read_bytes())
    (q/'control_v2').mkdir()
    write(pcd/'runtime_v1/outer_exit_code','1\n');write(q/'control_v1/PCD/wrapper_exit_code','1\n')
    write(pcd/'runtime_v1/status.json',dict(actual_exit_code=1,stage='failed'))
    write(pcd/'runtime_v2/status.json',dict(actual_exit_code=0,stage='all_embedding_phases_completed'))
    write(pcd/'runtime_v2/closure.json',dict(actual_exit_code=0,completed_phases=4,cube_count=4,expected_features_per_modality=5505500,data_id='cube-quadruple-play-v0',training_updates=0,SR_episodes=0))
    for name in recovery.NEW_PHASES:write(pcd/'runtime_v2'/f'{name}_exit.json',dict(actual_exit_code=0,residual_owned_members=[]))
    if complete:
        write(pcd/'runtime_v2/outer_exit_code','0\n');write(q/'control_v2/PCD/wrapper_exit_code','0\n')
    gate=pcd/'3d/GPU_preprocess_report.json'
    write(gate,dict(actual_exit_code=0,accepted=False,prepared_bitexact=True,latent_bitexact=True,model_frozen_exact=True,prepared_max_abs_diff=0.,latent_max_abs_diff=0.,training_updates=0,render_frames=0,checkpoint_sha256=recovery.MODEL,baseline_median_seconds=1.,candidate_median_seconds=.96))
    decision=pcd/'performance_release_v2/release_decision.json'
    write(decision,dict(actual_exit_code=0,original_engineering_5percent_gate_passed=False,positive_gain_released=True,unchanged_numeric_tolerances=True,gate_report_sha256=recovery.sha(gate),positive_gain_fraction=(1.-.96)))
    plan=q/'control_v2/PCD/launch_plan.json';write(plan,dict(phases=[{'name':n} for n in recovery.NEW_PHASES]))
    binding=q/'control_v2/RECOVERY_BINDING_v2.json'
    write(binding,dict(schema_version=2,q=str(q.resolve()),pcd_case=str(pcd.resolve()),old_GPU_report={'path':str(gate),'sha256':recovery.sha(gate)},engineering_decision={'path':str(decision),'sha256':recovery.sha(decision)},remaining_launch_plan={'path':str(plan),'sha256':recovery.sha(plan)}))
    return q.resolve(),pcd.resolve(),binding.resolve(),recovery.sha(binding)


class RecoveryTests(unittest.TestCase):
    def test_failed_or_missing_new_closure_cannot_publish_or_launch(self):
        with tempfile.TemporaryDirectory() as name:
            q,pcd,binding,digest=fixture(Path(name),False)
            with self.assertRaises(FileNotFoundError):recovery.publish_recovery(q,pcd,binding,digest)
            self.assertFalse((pcd/'recovery_receipt_v2.json').exists())
            loaded=dispatch.load(q,pcd,binding,digest)
            self.assertFalse(loaded.cache_ready(q,pcd))
            write(pcd/'runtime_v2/outer_exit_code','1\n')
            with self.assertRaises(RuntimeError):loaded.cache_ready(q,pcd)
            self.assertFalse((q/'functional_v2/configs/G12_cube4_pcd.v3.json').exists())

    def test_complete_recovery_preserves_old_failure_and_dispatches_exact_G12(self):
        with tempfile.TemporaryDirectory() as name:
            q,pcd,binding,digest=fixture(Path(name))
            receipt=recovery.publish_recovery(q,pcd,binding,digest)
            self.assertEqual(receipt['chain']['logical_completed_phases'],8)
            self.assertEqual((pcd/'runtime_v1/outer_exit_code').read_text().strip(),'1')
            self.assertFalse(json.loads((pcd/'3d/GPU_preprocess_report.json').read_text())['accepted'])
            self.assertFalse((q/'functional_v2/configs/G12_cube4_pcd.v3.json').exists())
            loaded=dispatch.load(q,pcd,binding,digest);calls=[]
            def invoke(argv,**kwargs):
                calls.append(argv)
                if argv[0]!='bash':
                    self.assertEqual(argv[3],'prepare')
                    recovery.prepare(q,pcd,binding,digest)
                else:
                    role=q/'control_v2/G12'
                    write(role/'wrapper_exit_code','0\n');write(role/'runtime/outer_exit_code','0\n')
                    write(role/'runtime/status.json',dict(actual_exit_code=0,completed_phases=4,trained_groups=1,accepted_groups=1))
                return types.SimpleNamespace(returncode=0)
            loaded.run(q,pcd,poll_seconds=0,invoke=invoke)
            cfg=json.loads((q/'functional_v2/configs/G12_cube4_pcd.v3.json').read_text())
            plan=json.loads((q/'control_v2/G12/launch_plan.json').read_text())
            original=json.loads((q/'control_v1/G11/launch_plan.json').read_text())
            self.assertTrue(cfg['executable']);self.assertEqual(cfg['group_id'],'G12')
            self.assertEqual((cfg['batch_size'],cfg['total_updates'],len(cfg['checkpoint_steps'])),(1024,1000000,11))
            self.assertEqual([p['action'] for p in plan['phases']],['gate','reload','train','accept'])
            self.assertEqual([p['cpu'] for p in plan['phases']],[p['cpu'] for p in original['phases']])
            self.assertEqual([p['checks'][0]['equals'] for p in plan['phases']],[p['checks'][0]['equals'] for p in original['phases']])
            self.assertTrue(all(p['argv'][p['argv'].index('--config')+1]==str(q/'functional_v2/configs/G12_cube4_pcd.v3.json') for p in plan['phases']))
            self.assertEqual(len(calls),2)
            self.assertEqual((q/'control_v2/after_cache_G12_runtime_v2/outer_exit_code').read_text().strip(),'0')
            with self.assertRaises(FileExistsError):loaded.run(q,pcd,invoke=invoke)


if __name__=='__main__':unittest.main()

"""Two narrow metadata/subprocess fakes; never execute a trainer or GPU call."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_after_cache_G12_tested',HERE/'after_cache_G12.py')
relay=importlib.util.module_from_spec(spec);spec.loader.exec_module(relay)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value,dict) else value)


def fixture(root,closed):
    q=root/'cube4_visual_gciql_v1';pcd=root/'cube4_official_embeddings_v1/pcd_training_camera_v1'
    for relative in ['run_G11.sh','run_role.py','G11/launch_plan.json']:
        dest=q/'control_v1'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((HERE/relative).read_bytes())
    functional=HERE.parent/'functional_v1'
    for relative in ['prepare_G12.py','configs/G11_cube4_rgb.json','configs/G12_cube4_pcd.TEMPLATE.json']:
        dest=q/'functional_v1'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((functional/relative).read_bytes())
    if closed:
        write(pcd/'runtime_v1/outer_exit_code','0\n');write(q/'control_v1/PCD/wrapper_exit_code','0\n')
        write(pcd/'runtime_v1/status.json',dict(actual_exit_code=0,stage='all_embedding_phases_completed'))
        write(pcd/'runtime_v1/closure.json',dict(actual_exit_code=0,completed_phases=8))
        for name in relay.PCD_PHASES:write(pcd/'runtime_v1'/f'{name}_exit.json',dict(actual_exit_code=0,residual_owned_members=[]))
    return q.resolve(),pcd.resolve()


class AfterCacheTests(unittest.TestCase):
    def test_unclosed_cache_generates_nothing_and_real_failure_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as name:
            q,pcd=fixture(Path(name),False);calls=[]
            self.assertFalse(relay.cache_ready(q,pcd))
            with self.assertRaises(AssertionError):relay.activate(q,pcd,q/'unused',invoke=lambda *a,**k:calls.append(a))
            self.assertEqual(calls,[]);self.assertFalse((q/'control_v1/G12').exists())
            self.assertFalse((q/'functional_v1/configs/G12_cube4_pcd.v2.json').exists())
            write(pcd/'runtime_v1/outer_exit_code','1\n')
            with self.assertRaises(RuntimeError):relay.cache_ready(q,pcd)

    def test_closed_fixture_prepares_real_cfg_then_clones_four_phases_and_runs_original_wrapper(self):
        with tempfile.TemporaryDirectory() as name:
            q,pcd=fixture(Path(name),True);calls=[]
            old_plan=json.loads((q/'control_v1/G11/launch_plan.json').read_bytes())
            old_script=(q/'control_v1/run_G11.sh').read_bytes()
            def invoke(argv,**kwargs):
                calls.append(argv)
                if len(calls)==1:
                    self.assertEqual(Path(argv[2]).name,'prepare_G12.py')
                    cfg=json.loads((q/'functional_v1/configs/G12_cube4_pcd.TEMPLATE.json').read_bytes())
                    cfg.update(executable=True,output=str(q/'runs/G12_cube4_pcd'),cache_binding={'modality':'pcd'},feature_loader_arguments={s:{'cache_dir':str(pcd/'3d/cache')} for s in ['train','val']})
                    config=q/'functional_v1/configs/G12_cube4_pcd.v2.json';write(config,cfg)
                    write(q/'functional_v1/G12_CONFIG_PREPARED_v2.json',dict(actual_exit_code=0,config_path=str(config),config_sha256=relay.sha(config),group_id='G12',metadata_input_files={}))
                else:
                    self.assertEqual(argv,['bash',str(q/'control_v1/run_G12.sh')])
                    write(q/'control_v1/G12/wrapper_exit_code','0\n');write(q/'control_v1/G12/runtime/outer_exit_code','0\n')
                    write(q/'control_v1/G12/runtime/status.json',dict(actual_exit_code=0,completed_phases=4,trained_groups=1,accepted_groups=1))
                return types.SimpleNamespace(returncode=0)
            relay.run(q,pcd,invoke=invoke)
            self.assertEqual(len(calls),2)
            actual=json.loads((q/'control_v1/G12/launch_plan.json').read_bytes());config=q/'functional_v1/configs/G12_cube4_pcd.v2.json'
            self.assertEqual(actual['groups'],['G12']);self.assertTrue(actual['G12_executable'])
            self.assertEqual(actual['source_pins'][str(config)],relay.sha(config))
            self.assertEqual([p['action'] for p in actual['phases']],['gate','reload','train','accept'])
            for before,after in zip(old_plan['phases'],actual['phases']):
                self.assertEqual(before['cpu'],after['cpu']);self.assertEqual(before['action'],after['action'])
                self.assertEqual(after['group_id'],'G12')
                self.assertEqual(after['argv'][after['argv'].index('--config')+1],str(config))
                self.assertNotIn('TEMPLATE',' '.join(after['argv']))
                self.assertEqual([c['equals'] for c in before['checks']],[c['equals'] for c in after['checks']])
            self.assertEqual((q/'control_v1/run_G11.sh').read_bytes(),old_script)
            self.assertIn('RUN="$CASE/control_v1/G12"',(q/'control_v1/run_G12.sh').read_text())
            state=json.loads((q/'control_v1/after_cache_G12_runtime_v1/status.json').read_bytes())
            self.assertEqual(state['actual_exit_code'],0)
            self.assertEqual((q/'control_v1/after_cache_G12_runtime_v1/outer_exit_code').read_text().strip(),'0')
            with self.assertRaises(FileExistsError):relay.run(q,pcd,invoke=invoke)


if __name__=='__main__':unittest.main()

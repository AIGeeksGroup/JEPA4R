"""Two fake-metadata CPU cases; never read a model or feature payload."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_prepare_G12_tested',HERE/'prepare_G12.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value,dict) else value)


def fixture(root,closed):
    q=root/'cube4_visual_gciql_v1';pcd=root/'cube4_official_embeddings_v1/pcd_training_camera_v1'
    for name in ['quad_visual.py','configs/G11_cube4_rgb.json','configs/G12_cube4_pcd.TEMPLATE.json']:
        path=q/'functional_v1'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((HERE/name).read_bytes())
    if not closed:return q,pcd
    write(q/'control_v1/PCD/wrapper_exit_code','0\n')
    write(pcd/'runtime_v1/outer_exit_code','0\n')
    write(pcd/'runtime_v1/status.json',dict(stage='all_embedding_phases_completed',actual_exit_code=0))
    write(pcd/'runtime_v1/closure.json',dict(actual_exit_code=0,cube_count=4,data_id='cube-quadruple-play-v0',completed_phases=8,expected_features_per_modality=5505500,training_updates=0,SR_episodes=0))
    for name in prepare.PCD_PHASES:write(pcd/'runtime_v1'/f'{name}_exit.json',dict(actual_exit_code=0,residual_owned_members=[]))
    write(q/'control_v1/G11/runtime/outer_exit_code','0\n');write(q/'control_v1/G11/wrapper_exit_code','0\n')
    write(q/'control_v1/G11/runtime/status.json',dict(actual_exit_code=0,completed_phases=4,trained_groups=1,accepted_groups=1))
    for name in prepare.G11_PHASES:
        write(q/'control_v1/G11/runtime'/name/'exit.json',dict(actual_exit_code=0,residual_owned_members=[]))
        write(q/'control_v1/G11/runtime'/name/'actual_exit_code','0\n')
    g11=json.loads((q/'functional_v1/configs/G11_cube4_rgb.json').read_bytes())
    write(q/'acceptance/G11_full_acceptance.json',dict(actual_exit_code=0,full153_state_exact=True,state={'leaves':153,'sha256':'c'*64},Adam_count=1000000,network_step=1000001,CSV_validation_records=200,actual_current_dim=211,actual_goal_dim=204,synthetic_zero_padding=False,checkpoint_transactions=[{'counter':n} for n in prepare.COUNTERS],final_checkpoint=g11['output']+'/checkpoints/update_001000000'))
    cache=pcd/'3d/cache';plan_path=pcd/'workspace/official_state_3d_cache_v1/producer_plan.json'
    arrays=pcd/'metadata/source_arrays.json'
    write(arrays,dict(splits={s:dict(source_npz=v['path'],accepted_stat=v['accepted_stat'],rows=v['raw_rows']) for s,v in g11['data_binding']['splits'].items()}))
    protocol=dict(checkpoint_sha256=prepare.PCD_CHECKPOINT_SHA,split_episodes={'train':5000,'val':500},raw_rows_per_episode=1001)
    protocol_sha=prepare.object_sha(protocol)
    write(plan_path,dict(checkpoint_sha256=prepare.PCD_CHECKPOINT_SHA,cache_dir=str(cache),split_episodes={'train':5000,'val':500},raw_rows_per_episode=1001,encoder_batch=32,camera={'pullback':.1,'crop':None},pcd={'points':1200},source_arrays_receipt_sha256=prepare.sha(arrays)))
    manifest=dict(actual_exit_code=0,protocol=protocol,protocol_sha256=protocol_sha,source_arrays_receipt={'path':str(arrays),'sha256':prepare.sha(arrays)},checkpoint_sha256=prepare.PCD_CHECKPOINT_SHA,source_encoder_state_sha256='a'*64,latent_dim=192,dtype='float32',training_updates=0,SR_episodes=0,splits={s:[{'episode':i} for i in range(n)] for s,n in [('train',5000),('val',500)]})
    write(cache/'cache_manifest.json',manifest)
    write(cache/'cache_seal.json',dict(actual_exit_code=0,ready_for_downstream=False,manifest_sha256=prepare.sha(cache/'cache_manifest.json'),protocol_sha256=protocol_sha,train_rows=5005000,val_rows=500500,full_raw_ID_coverage=True))
    audit=pcd/'3d/independent_audit'
    write(audit/'actual_exit_code','0\n')
    write(audit/'report.json',dict(actual_exit_code=0,plan_sha256=prepare.sha(plan_path),manifest_sha256=prepare.sha(cache/'cache_manifest.json'),producer_seal_sha256=prepare.sha(cache/'cache_seal.json'),protocol_sha256=protocol_sha,checkpoint_sha256=prepare.PCD_CHECKPOINT_SHA,source_arrays_receipt_sha256=prepare.sha(arrays),episodes_verified=5500,raw_rows_verified=5505500,main_rows_verified=5500000,latent_dim=192,dtype='float32',all_new_latent_SHA_shape_dtype_finite=True,all_canonical_ID_and_terminal_next_contracts=True,tracked_stats_stable=True))
    return q,pcd


class PrepareG12Tests(unittest.TestCase):
    def test_missing_real_closure_is_rejected_without_config_publication(self):
        with tempfile.TemporaryDirectory() as name:
            q,pcd=fixture(Path(name),False)
            with self.assertRaises((FileNotFoundError,AssertionError)):prepare.prepare_config(q,pcd)
            self.assertFalse((q/'functional_v1/configs/G12_cube4_pcd.v2.json').exists())

    def test_closed_metadata_creates_only_actual_PCD_configuration_no_RGB_rebind(self):
        with tempfile.TemporaryDirectory() as name:
            q,pcd=fixture(Path(name),True)
            template=q/'functional_v1/configs/G12_cube4_pcd.TEMPLATE.json';old=hashlib.sha256(template.read_bytes()).hexdigest()
            record=prepare.prepare_config(q,pcd);cfg=json.loads(Path(record['config_path']).read_bytes())
            g11=json.loads((q/'functional_v1/configs/G11_cube4_rgb.json').read_bytes())
            self.assertEqual(cfg['group_id'],'G12');self.assertTrue(cfg['executable'])
            self.assertEqual(cfg['agent_config'],g11['agent_config']);self.assertEqual(cfg['reward'],g11['reward'])
            self.assertEqual(cfg['input_contract'],g11['input_contract'])
            self.assertEqual((cfg['batch_size'],cfg['total_updates'],len(cfg['checkpoint_steps'])),(1024,1000000,11))
            for split in ['train','val']:
                args=cfg['feature_loader_arguments'][split]
                self.assertNotIn('source_rebind',args);self.assertEqual(args['checkpoint_sha256'],prepare.PCD_CHECKPOINT_SHA)
                self.assertEqual(Path(args['cache_dir']).resolve(),(pcd/'3d/cache').resolve())
            self.assertEqual(hashlib.sha256(template.read_bytes()).hexdigest(),old)
            with self.assertRaises(FileExistsError):prepare.prepare_config(q,pcd)


if __name__=='__main__':unittest.main()

"""Two metadata cases: closed cache builds real widths; failed audit cannot release."""
import copy
import json
import os
from pathlib import Path
import unittest

import prepare

HERE = Path(__file__).resolve().parent
REFERENCE = Path(os.environ.get('P3_REFERENCE_ROOT', str(HERE/'reference_source')))


def metadata():
    plan = dict(workspace='/frozen/root',source_pins={'encoder_adapter.py':'1'*64},
        checkpoint_sha256=prepare.CHECKPOINT_SHA, split_episodes=dict(train=3000,val=300),
        raw_rows_per_episode=1001, data_acceptance='accepted.json', data_acceptance_sha256='a'*64,
        source_stat_freeze='stats.json', source_stat_freeze_sha256='b'*64)
    manifest = dict(actual_exit_code=0,checkpoint_sha256=prepare.CHECKPOINT_SHA,
        protocol=dict(encoder_family='pldm',latent_dim=512,pixel_preprocess_device='cpu'),
        splits={s:[{'episode':i} for i in range(n)] for s,n in plan['split_episodes'].items()})
    seal = dict(actual_exit_code=0,train_rows=3003000,val_rows=300300,full_raw_ID_coverage=True)
    audit = dict(actual_exit_code=0,checkpoint_sha256=prepare.CHECKPOINT_SHA,episodes_verified=3300,
        raw_rows_verified=3303300,main_rows_verified=3300000,latent_dim=512,dtype='float32',
        all_new_latent_SHA_shape_dtype_finite=True,all_raw_ID_and_true_terminal_next_contracts=True,
        tracked_stats_stable=True,batch1_alignment_passed=True,batch1_alignment_sha256='c'*64)
    selection = dict(actual_exit_code=0,selected_device='cpu')
    return plan,manifest,seal,audit,selection


class PreparationTests(unittest.TestCase):
    def test_closed_metadata_keeps_agent_reward_and_explicit_CPU_selection(self):
        plan,manifest,seal,audit,selection = metadata()
        prepare.check_cache_metadata(plan,manifest,seal,audit,selection)
        original = json.loads((REFERENCE/'cube23_distance_gciql_v1/five_group_extension_v1/configs/G08_cube3_rgb.json').read_bytes())
        template = copy.deepcopy(original)
        template.update(group_id=prepare.GROUP,role=prepare.GROUP,executable=False,
            encoder_binding=dict(family='pldm',latent_dim=512,checkpoint_sha256=prepare.CHECKPOINT_SHA))
        template['input_contract'].update(current_dim=531,next_dim=531,goal_dim=521,latent_dim=512)
        facts = dict(plan=plan,cache='new_PLDMonly',manifest_sha256='d'*64,manifest_path='manifest.json',
            seal_path='seal.json',audit_path='audit.json',selection_path='selection.json',selection_sha256='e'*64,
            selected_device='cpu',arrays_ref=dict(path='arrays.json',sha256='f'*64),
            arrays=dict(splits={s:dict(source_npz=v['path'],rows=n*1001,accepted_stat={'current':True})
                for (s,n),v in zip(plan['split_episodes'].items(),template['data_binding']['splits'].values())}))
        cfg = prepare.config_from_cache(template,facts,Path('/new/cube3_pldm_gciql_v1'))
        self.assertTrue(cfg['executable'])
        self.assertEqual(cfg['agent_config'],original['agent_config'])
        self.assertEqual(cfg['reward'],original['reward'])
        self.assertEqual((cfg['input_contract']['current_dim'],cfg['input_contract']['goal_dim']),(531,521))
        self.assertEqual(cfg['checkpoint_steps'],prepare.COUNTERS)
        self.assertEqual(cfg['encoder_binding']['selected_preprocess_device'],'cpu')
        for args in cfg['feature_loader_arguments'].values():
            self.assertEqual((args['encoder_family'],args['feature_dim']),('pldm',512))
            self.assertNotIn('source_rebind',args)
        self.assertFalse(template['executable'])

    def test_failed_or_missing_real_proof_cannot_pass_release_guard(self):
        for key,value in [('actual_exit_code',1),('batch1_alignment_passed',False),('latent_dim',192)]:
            plan,manifest,seal,audit,selection = metadata()
            audit[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                prepare.check_cache_metadata(plan,manifest,seal,audit,selection)


if __name__ == '__main__':
    unittest.main()

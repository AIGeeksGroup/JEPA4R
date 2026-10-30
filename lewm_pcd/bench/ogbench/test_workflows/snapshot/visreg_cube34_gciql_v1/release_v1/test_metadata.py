"""Metadata-only positive/negative contracts; never import the NN runtime."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
try:
    import common as m
except ModuleNotFoundError:
    m = None


class MetadataContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(m, 'metadata implementation is not present yet')

    def test_space_paid_once_and_boundary(self):
        self.assertEqual(m.space_contract(4029280256, 0)['required_free_bytes'], 3636870912)
        self.assertEqual(m.space_contract(3029280256, 1000000000)['required_free_bytes'], 2636870912)
        m.space_contract(2636870912, 1000000000)
        with self.assertRaises(AssertionError):
            m.space_contract(2636870911, 1000000000)
        self.assertEqual(m.space_contract(536870912, 4000000000)['required_free_bytes'], 536870912)

    def test_config_copy_changes_only_three_IO_fields(self):
        original = {'seed': 0, 'purpose': 'pilot', 'output': '/old/runs/g',
                    'execution_root': '/old', 'commands': {'gate': ['old']},
                    'nested': {'reward': [-1, 0]}, 'group_id': 'g'}
        before = copy.deepcopy(original)
        changed = m.remap_config(original, '/new/config.json', '/new',
                                 lambda p, c: {'gate': [str(p), c['output']]})
        self.assertEqual(original, before)
        self.assertEqual({k for k in original if original[k] != changed[k]},
                         {'execution_root', 'output', 'commands'})
        self.assertEqual(Path(changed['output']), Path('/new/runs/g'))

    def reports(self, full=False):
        g = 'visreg_vitb768_cube3'
        binding = dict(actual_exit_code=0, group_id=g, config_sha256='cfg',
                       encoder_family='visreg_vitb768', latent_dim=768,
                       mode='full' if full else 'tiny', gate_smoke=not full)
        report = dict(actual_exit_code=0, group_id=g, mode='full' if full else 'diagnostic',
                      episodes=250 if full else 1, unique_seeds=250 if full else 1)
        audit = dict(actual_exit_code=0, group_id=g, mode='full' if full else 'tiny',
                     config_sha256='cfg', source_binding_sha256='binding',
                     episodes=report['episodes'], unique_seeds=report['unique_seeds'],
                     actual_step_arrays_checked=True,
                     diagnostic_excluded_from_formal_SR=not full)
        return g, binding, report, audit

    def test_native_receipts_require_real_audit_and_current_binding(self):
        for full in (False, True):
            g, b, r, a = self.reports(full)
            m.native_contract(g, 'cfg', 'binding', b, r, a, full)
            for key, bad in [('actual_step_arrays_checked', False),
                             ('config_sha256', 'stale'), ('episodes', 249)]:
                broken = copy.deepcopy(a); broken[key] = bad
                with self.assertRaises(AssertionError):
                    m.native_contract(g, 'cfg', 'binding', b, r, broken, full)
            b['gate_smoke'] = full
            with self.assertRaises(AssertionError):
                m.native_contract(g, 'cfg', 'binding', b, r, a, full)

    def test_pilot_full153_RNG_nextbatch_and_actual_kernels(self):
        cfg = dict(group_id='visreg_vitb768_cube3', input_contract=dict(current_dim=787, goal_dim=777))
        gate = dict(actual_exit_code=0, role=cfg['group_id'], updates=2, batch_size=1024,
                    network_step=3, adam_count=2, initial_state={'leaves':153},
                    saved_state={'leaves':153,'sha256':'state'},
                    first_layer_kernels={'a':[1564,512], 'b':[2,1569,512]},
                    formal_training_started=False, encoder_calls=0, WM_dynamics_calls=0)
        reload = dict(actual_exit_code=0, role=cfg['group_id'], full153_state_exact=True,
                      next_batch_and_rawIDs_exact=True, state=gate['saved_state'],
                      applied_host_RNG=['numpy','python','torchCPU'], no_updates=True,
                      finite_actions=True, CPU_action_forward=True, backend='cpu', fresh_process=True)
        m.pilot_contract(cfg, gate, reload)
        for key, bad in [('updates',1), ('network_step',2), ('first_layer_kernels', {'a':[1564,512]})]:
            broken = copy.deepcopy(gate); broken[key] = bad
            with self.assertRaises(AssertionError):m.pilot_contract(cfg, broken, reload)
        reload['next_batch_and_rawIDs_exact'] = False
        with self.assertRaises(AssertionError):m.pilot_contract(cfg, gate, reload)

    def test_formal_1M_153_11CP_and_actual_widths(self):
        cfg = dict(group_id='visreg_vitb768_cube4', input_contract=dict(current_dim=787,goal_dim=780))
        accept = dict(actual_exit_code=0, role=cfg['group_id'], full153_state_exact=True,
                      state={'leaves':153}, Adam_count=1000000, network_step=1000001,
                      CSV_validation_records=200, checkpoint_transactions=[{'counter':v} for v in m.COUNTERS],
                      actual_current_dim=787, actual_goal_dim=780, synthetic_zero_padding=False,
                      finite_actions=True, CPU_action_forward=True, no_optimizer_updates=True,
                      entire_1M_host_sampler_replay_performed=False, SR_performed=False)
        m.formal_contract(cfg, accept)
        for key,bad in [('Adam_count',999999),('actual_goal_dim',777),('full153_state_exact',False),
                        ('checkpoint_transactions',accept['checkpoint_transactions'][1:])]:
            broken=copy.deepcopy(accept);broken[key]=bad
            with self.assertRaises(AssertionError):m.formal_contract(cfg,broken)

    def test_closure_requires_drained_zero_exits_and_plan_identity(self):
        plan=dict(cube_counts=[3],families=['visreg_vitb768'],training_updates=0,SR_episodes=0,
                  phases=[dict(name='gate',argv=['p'],cwd='/r')])
        closure=dict(actual_exit_code=0,completed_phases=1,cube_counts=[3],families=['visreg_vitb768'],training_updates=0,SR_episodes=0)
        exits={'gate':dict(actual_exit_code=0,residual_owned_members=[])}
        argv={'gate':dict(argv=['p'],cwd='/r')}
        m.closure_contract(plan,closure,exits,argv,0,0,{'actual_exit_code':0},'hash','hash')
        exits['gate']['residual_owned_members']=[{'PID':1}]
        with self.assertRaises(AssertionError):
            m.closure_contract(plan,closure,exits,argv,0,0,{'actual_exit_code':0},'hash','hash')
        exits['gate']['residual_owned_members']=[]
        with self.assertRaises(AssertionError):
            m.closure_contract(plan,closure,exits,argv,0,0,{'actual_exit_code':0},'old','hash')

    def test_owned_bytes_does_not_follow_links_or_double_pay_hardlinks(self):
        fixture = HERE/'validation';fixture.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=fixture) as tmp:
            root=Path(tmp);(root/'one').write_bytes(b'12345')
            import os
            os.link(root/'one',root/'two')
            self.assertEqual(m.owned_bytes(root),5)

    def test_actual_original_make_config_and_all_four_stage_selections(self):
        from unittest.mock import patch
        fixture=HERE/'validation';fixture.mkdir(exist_ok=True)
        foundation=m.F if (m.F/'functional/prepare.py').exists() else HERE.parent.parent/'foundation_visual_gciql_31435_20261024_v1'
        spec=importlib.util.spec_from_file_location('_original_prepare_fixture',foundation/'functional/prepare.py')
        prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
        expected=json.loads((HERE/'expected_cache_bindings_v1.json').read_text())
        with tempfile.TemporaryDirectory(dir=fixture) as tmp:
            root=Path(tmp);f=root/'foundation_visual_gciql_v1';case=root/'visreg_cube34_gciql_v1'
            oldcase=prep.CASE;prep.CASE=f
            try:
                locations=dict(ROOT=root,F=f,CASE=case,STORE=root/'store',PY=root/'env_runtime/bin/python',
                               TEMPLATE=root/'template.json',SMOKE=f/'control/smoke/launch_plan.json',FORMAL=f/'control/formal_w8/launch_plan.json')
                with patch.multiple(m,**locations):
                    agent=dict(lr=.0003,alpha=1.,discount=.99,expectile=.9,tau=.005,
                               layer_norm=True,actor_hidden_dims=[512]*3,value_hidden_dims=[512]*3)
                    m.write_new(m.TEMPLATE,dict(agent_config=agent));configs={};phase_sets={'smoke':[],'formal':[]}
                    for n in (3,4):
                        g=f'visreg_vitb768_cube{n}';entry=expected['groups'][g]
                        dp=foundation/'closed_embedding_metadata_v1'/g/'group/delivery.json'
                        if not dp.exists():dp=foundation/'cache/w8_groups'/g/'delivery.json'
                        d=json.loads(dp.read_text())
                        m.write_new(m.delivery_path(n),d);configs[n]={}
                        for purpose in ('pilot','formal'):
                            original=m.original_config(n,purpose,prep)
                            cfg=m.remap_config(original,m.cfgpath(n,purpose),m.home(purpose),prep.commands)
                            m.config_contract(cfg,n,purpose,prep);configs[n][purpose]=cfg
                            self.assertEqual(cfg['feature_loader_arguments'],entry['field_assertions']['feature_loader_arguments'])
                        for full in (False,True):
                            purpose='formal' if full else 'pilot';cfg=configs[n][purpose]
                            labels=('05_train','06_accept','07_native','08_native_audit') if full else ('04_hooks','05_gate','06_reload','07_native','08_native_audit')
                            for label in labels:
                                cuda=label in ('05_train','05_gate','07_native');jaxcuda=label in ('05_train','05_gate')
                                script=f/'functional'/('audit_native.py' if label.endswith('audit') else 'native_sr.py' if label=='07_native' else 'runtime.py')
                                prefix=[str(m.PY),'-B']
                                if cuda:prefix=[str(m.PY),'-B',str(root/'cube4_visual_gciql_v1/control_v1/gpu_lease.py'),'--',str(m.PY),'-B']
                                phase_sets['formal' if full else 'smoke'].append(dict(name=g+'_'+label,argv=prefix+[str(script),'old'],cwd=str(root),checks=[],environment={'CUDA_VISIBLE_DEVICES':'0' if cuda else '', 'JAX_PLATFORMS':'cuda' if jaxcuda else 'cpu'}))
                    for name,path in [('smoke',m.SMOKE),('formal',m.FORMAL)]:m.write_new(path,dict(phases=phase_sets[name]))
                    for n in (3,4):
                        for full in (False,True):
                            phases=m.selected_phases(n,full,configs[n]);self.assertEqual(len(phases),4 if full else 5)
                            payload=json.dumps(phases);self.assertNotIn('vjepa',payload);self.assertNotIn('encode',payload);self.assertNotIn('backup',payload)
                            self.assertTrue(all(str(m.cfgpath(n,'formal' if full else 'pilot')) in p['argv'] for p in phases))
                    corrupted=copy.deepcopy(configs[3]['formal']);corrupted['batch_size']=2048
                    with self.assertRaises(AssertionError):m.config_contract(corrupted,3,'formal',prep)
            finally:prep.CASE=oldcase


if __name__ == '__main__':unittest.main(verbosity=2)

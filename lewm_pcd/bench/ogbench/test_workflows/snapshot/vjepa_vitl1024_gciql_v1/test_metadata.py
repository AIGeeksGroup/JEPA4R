"""Local metadata/queue tests only; fixtures stay in this task's D scratch."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
TASK = HERE.parent if HERE.name == 'release_v1' else HERE
REMOTE = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
REFERENCE = TASK / 'preflight_v1/reference_sources'
if not REFERENCE.exists():
    REFERENCE = REMOTE
FOUNDATION = REMOTE/'foundation_visual_gciql_v1'
if not (FOUNDATION/'functional/prepare.py').exists():
    FOUNDATION = Path('D:/LeWM_Artifacts/coordination/foundation_visual_gciql_31435_20261024_v1')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    path = HERE / (name + '.py')
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location('_vj_test_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.prep, self.run = load('prepare'), load('run')

    def available(self):
        self.assertIsNotNone(self.prep, 'VJ metadata preparer missing')
        self.assertIsNotNone(self.run, 'VJ queue wrapper missing')

    @contextmanager
    def fixture(self):
        self.available()
        scratch = TASK / 'scratch'
        scratch.mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='metadata_', dir=scratch))
        case, base, store = root/'vj_case', root/'foundation_visual_gciql_v1', root/'outputs'
        case.mkdir()
        source = base/'functional/prepare.py'
        source.parent.mkdir(parents=True)
        source.write_bytes((FOUNDATION/'functional/prepare.py').read_bytes())
        frozen = root/'frozen.py'; frozen.write_text('# metadata fixture\n')
        env = root/'pldm_cube14_gciql_v1/baseline_environments.json'
        env.parent.mkdir(parents=True)
        env.write_bytes((REFERENCE/'pldm_cube14_gciql_v1/baseline_environments.json').read_bytes())
        adapter = base/'vj_precision_switch_v1/precision_bridge.py'
        adapter.parent.mkdir(parents=True); adapter.write_text('# fixture bridge\n')
        specpath = adapter.parent/'specs/vjepa21_vitl1024_fp16_reference_v1.json'
        spec = dict(family='vjepa21_vitl1024',output_dim=1024,inference_precision='fp16',
            batch_reference_policy='same_batch_official_fp16_reference_v1',checkpoint='fixture.pt',
            checkpoint_sha256='7'*64,source_root=str(root/'encoder_source'),source_files_sha256={})
        write(specpath,spec)
        py = root/'env_runtime/bin/python'
        values = dict(ROOT=root, CASE=case, BASE=base, STORE=store, PY=py,
            BASE_ENV=env, SPEC_SHA=digest(specpath), ADAPTER_SHA=digest(adapter), CHECKPOINT_SHA='7'*64)
        with patch.multiple(self.prep, **values), patch.multiple(self.run,ROOT=root,CASE=case), patch.dict(sys.modules,{'prepare':self.prep}):
            for n in (1,2,3,4):
                home=base/'cache/fp16_bulk_w8_groups'/f'vjepa21_vitl1024_cube{n}'
                cache=home/'cache'; arrays=base/'cache/data'/f'cube{n}'
                split_counts=self.prep.COUNTS[n]
                sources={}
                for split in ('train','val'):
                    item={}
                    for key in ('source_npz','qpos','qvel'):
                        p=arrays/(split+'_'+key+'.fixture');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
                        if key=='source_npz': item.update(source_npz=str(p),accepted_stat=self.prep.stat(p))
                        else: item[key]=dict(path=str(p),stat=self.prep.stat(p))
                    sources[split]=item
                write(arrays/'source_arrays.json',dict(splits=sources,contract=dict(extractor_source_sha256='e'*64)))
                plan=dict(cube_count=n,encoder_family='vjepa21_vitl1024',latent_dim=1024,
                    split_episodes=split_counts,raw_rows_per_episode=1001,checkpoint='fixture.pt',checkpoint_sha256='7'*64,
                    encoder_spec=str(specpath),encoder_spec_sha256=digest(specpath),encoder_adapter=str(adapter),
                    encoder_adapter_sha256=digest(adapter),inference_precision='fp16',output_dtype='float32',
                    batch_reference_policy='same_batch_official_fp16_reference_v1',encoder_batch=32,
                    shared_state_root=str(arrays),source_arrays_receipt_sha256=digest(arrays/'source_arrays.json'),
                    source_arrays_extractor_sha256='e'*64,source_pins={'frozen.py':digest(frozen)},cache_dir=str(cache))
                write(home/'plan.json',plan)
                write(cache/'cache_manifest.json',dict(actual_exit_code=0))
                write(cache/'cache_seal.json',dict(actual_exit_code=0))
                same=lambda b:dict(wrapper_official_bitwise_equal=True,finite=True,max_abs_difference=0.,embedding_shape=[b,1024])
                write(cache/'precision_reference_alignment.json',dict(actual_exit_code=0,
                    batch_reference_policy=plan['batch_reference_policy'],inference_precision='fp16',
                    encoder_spec_sha256=digest(specpath),offline_encoder_batch=32,online_reference_batch=1,
                    same_batch_reference={'B32':same(32),'B1':same(1)},legacy_diagnostic={'used_for_acceptance':False}))
                episodes=sum(split_counts.values())
                audit=dict(actual_exit_code=0,episodes_verified=episodes,raw_rows_verified=episodes*1001,
                    main_rows_verified=episodes*1000,latent_dim=1024,dtype='float32',all_new_latent_SHA_shape_dtype_finite=True,
                    all_raw_ID_and_true_terminal_next_contracts=True,tracked_stats_stable=True,
                    plan_sha256=digest(home/'plan.json'),manifest_sha256=digest(cache/'cache_manifest.json'),producer_seal_sha256=digest(cache/'cache_seal.json'))
                write(home/'full_audit_v1/report.json',audit);(home/'full_audit_v1/actual_exit_code').write_text('0\n')
                binding=dict(family='vjepa21_vitl1024',latent_dim=1024,checkpoint_sha256='7'*64,
                    spec_path=str(specpath),spec_sha256=digest(specpath),adapter={'path':str(adapter),'sha256':digest(adapter)})
                delivery=dict(purpose='formal',group_id=f'vjepa21_vitl1024_cube{n}',cube_count=n,latent_dim=1024,
                    encoder_binding=binding,data_binding={'data_id':f'official-fixture-{n}','splits':{s:{'episodes':c,'raw_rows':c*1001} for s,c in split_counts.items()}},
                    feature_loader_arguments={s:dict(cache_dir=str(cache),split=s,manifest_sha256=audit['manifest_sha256'],checkpoint_sha256='7'*64,source_npz=sources[s]['source_npz'],expected_rows=c*1001) for s,c in split_counts.items()},
                    camera={'fixture_cube':n},cache_audit={'path':str(home/'full_audit_v1/report.json'),'sha256':digest(home/'full_audit_v1/report.json')})
                write(home/'delivery.json',delivery)
            control=base/'control/fp16_bulk_w8_v1'
            write(control/'runtime/closure.json',dict(actual_exit_code=0,completed_phases=12,cube_counts=[1,2,3,4],families=['vjepa21_vitl1024'],training_updates=0,SR_episodes=0))
            (control/'runtime/outer_exit_code').write_text('0\n');(control/'wrapper_exit_code').write_text('0\n')
            for n in (1,2,3,4):
                for suffix in ('01_encode_full','02_cache_audit','03_deliver'):
                    write(control/'runtime'/f'vjepa21_vitl1024_cube{n}_{suffix}_exit.json',dict(actual_exit_code=0,residual_owned_members=[]))
            write(control/'launch_plan.json',dict(source_pins={str(frozen):digest(frozen)}))
            yield root,case,base,store

    def configs(self):
        accepted=self.prep.cache_admission(); result={}
        for n,item in accepted.items():
            for purpose in ('pilot','formal'):
                path=self.prep.CASE/'configs'/purpose/f'vjepa21_vitl1024_cube{n}.json'
                cfg=self.prep.make_config({'agent_config':{'frozen':'fixture'}},item,purpose,path)
                write(path,cfg);result[n,purpose]=(path,cfg)
        return accepted,result

    def test_real_factory_keeps_full_cache_and_original_cli(self):
        with self.fixture():
            _,configs=self.configs()
            for n in (1,2,3,4):
                for purpose in ('pilot','formal'):
                    path,cfg=configs[n,purpose]
                    self.assertEqual(cfg['input_contract'],dict(current_dim=1043,goal_dim=1024+3*n,latent_dim=1024,zero_padding=False))
                    self.assertEqual(cfg['data_binding']['splits']['train']['episodes'],self.prep.COUNTS[n]['train'])
                    self.assertEqual(cfg['durability']['pilot_output'],str(self.prep.CASE/'pilot/gates'/cfg['group_id']))
                    self.assertEqual(cfg['durability']['formal_output'],str(self.prep.CASE/'runs'/cfg['group_id']))
                    for argv in cfg['commands'].values():self.assertTrue(Path(argv[0]).is_relative_to(self.prep.BASE/'functional'))
                    self.assertIn('native_audit',cfg['commands']);self.assertEqual('train' in cfg['commands'],purpose=='formal')
                    self.assertEqual('diagnostic_scope' in cfg,purpose=='pilot')
            shell=(HERE/'run.sh').read_text()
            self.assertIn('set -o noclobber',shell)

    def test_cache_admission_rejects_wrong_precision_and_unclosed_phase(self):
        with self.fixture():
            self.assertEqual(list(self.prep.cache_admission()),[1,2,3,4])
            p=self.prep.BASE/'cache/fp16_bulk_w8_groups/vjepa21_vitl1024_cube2/plan.json'
            value=read(p);value['inference_precision']='fp32';write(p,value)
            with self.assertRaises(AssertionError):self.prep.cache_admission()
        with self.fixture():
            p=self.prep.BASE/'control/fp16_bulk_w8_v1/runtime/vjepa21_vitl1024_cube4_03_deliver_exit.json'
            write(p,dict(actual_exit_code=1,residual_owned_members=[]))
            with self.assertRaises(AssertionError):self.prep.cache_admission()

    def test_queue_exact_24_and_20_phases_with_backup_barriers(self):
        with self.fixture():
            _,configs=self.configs()
            runtime=self.prep.CASE/'control/smoke/runtime'
            write(runtime/'closure.json',dict(actual_exit_code=0,completed_phases=24,cube_counts=[1,2,3,4],families=[self.prep.FAMILY]))
            (runtime/'outer_exit_code').write_text('0\n');(runtime.parent/'wrapper_exit_code').write_text('0\n')
            for n in self.prep.COUNTS:
                path,cfg=configs[n,'pilot'];group=cfg['group_id'];home=Path(cfg['execution_root'])
                for i,label in enumerate(('verify_hooks','gate','backup_gate','reload','native_tiny','tiny_audit'),1):
                    write(runtime/f'{group}_{i:02d}_{label}_exit.json',dict(actual_exit_code=0,residual_owned_members=[]))
                write(home/'gates'/group/'report.json',dict(actual_exit_code=0,actual_backend='gpu',updates=2,batch_size=1024,network_step=3,adam_count=2,encoder_calls=0,WM_dynamics_calls=0))
                write(home/'gates'/(group+'_reload.json'),dict(actual_exit_code=0,full153_state_exact=True,finite_actions=True))
                write(runtime.parent/'durability_reports'/(group+'.json'),dict(actual_exit_code=0,stage='gate',backup_stage='pilot',group_id=group,config_sha256=digest(path),checkpoint_count=2,checkpoints=[{'counter':1},{'counter':2}]))
                write(home/'native_sr'/group/'report.json',dict(actual_exit_code=0,episodes=1,current_dim=1043,goal_dim=1024+3*n,max_episode_steps=4,encoding_protocol='fresh_current_and_goal_each_policy_call_2N'))
                write(home/'native_sr'/group/'independent_audit.json',dict(actual_exit_code=0,episodes=1,config_sha256=digest(path),actual_step_arrays_checked=True,diagnostic_excluded_from_formal_SR=True))
            for mode,count in [('smoke',24),('full',20)]:
                q=self.prep.build_queue(configs,mode,{})
                self.assertEqual(len(q['phases']),count)
                self.assertEqual(q['training_updates'],0 if mode=='smoke' else 4000000)
                for i,step in enumerate(q['phases']):
                    self.assertNotIn('visreg',step['name']);self.assertNotIn('pldm',step['name'])
                    if 'backup_gate' in step['name']:
                        self.assertIn('--report',step['argv']);self.assertEqual(step['environment']['CUDA_VISIBLE_DEVICES'],'')
                self.run.validate(q)
                changed=copy.deepcopy(q);changed['phases'][0]['name']='visreg_wrong'
                with self.assertRaises(AssertionError):self.run.validate(changed)
            p=runtime.parent/'durability_reports/vjepa21_vitl1024_cube4.json'
            value=read(p);value['config_sha256']='bad';write(p,value)
            with self.assertRaises(AssertionError):self.run.validate(self.prep.build_queue(configs,'full',{}))

    def test_full_cannot_release_before_smoke_closure(self):
        with self.fixture():
            _,configs=self.configs();q=self.prep.build_queue(configs,'full',{})
            with self.assertRaises((AssertionError,FileNotFoundError)):self.run.require_smoke()

    def test_remaining_budget_credits_only_current_store(self):
        with self.fixture() as (_,_,_,store):
            store.mkdir();(store/'closed.fixture').write_bytes(b'x'*1234)
            free=8000000000
            with patch.object(self.prep.shutil,'disk_usage',return_value=types.SimpleNamespace(free=free)):
                admission=self.prep.space_admission()
            self.assertEqual(admission['credited_owned_artifact_bytes'],1234)
            self.assertEqual(admission['required_now'],6800000000-1234+(512<<20))


if __name__ == '__main__':
    unittest.main()

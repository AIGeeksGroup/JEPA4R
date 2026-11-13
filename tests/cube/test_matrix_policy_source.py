# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_matrix_policy_source.py (verbatim except: imports /
# paths adapted to the package; lines wrapped at 120 columns)
"""Policy origin must bind the actual producer case, config and encoder."""
import copy,json
from pathlib import Path
import tempfile,unittest
from jepa4r.policy.gciql import gate_checks as gate
from jepa4r.policy.gciql.workflow_downstream import group_contract


class PolicyOriginTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.producer=self.root/'actual-producer';self.producer.mkdir()
        self.runtime = dict(
            model='lewm_pcd',
            cube_count=3,
            encoder_binding={'family': 'lewm', 'latent_dim': 192, 'checkpoint_sha256': 'a' * 64},
        )
        self.original = dict(
            cube_count=3,
            seed=0,
            input_contract=group_contract('lewm_pcd', 3)['input_contract'],
            agent_config={'lr': 0.0003},
        )
        self.origpin=self.write(self.producer/'original-config.json',self.original)
        self.inputs=self.write(self.root/'inputs.json',{'real':'input-byte-identity'})
        self.source=self.write(self.root/'source.json',self.runtime)
        checkpoint = self.producer / 'policy/params_0.pkl'
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b'owned policy bytes')
        self.checkpoint={'path':str(checkpoint.parent),'counter':0,'sha256':gate.sha(checkpoint)}
        oldcase = dict(
            model='lewm_pcd',
            cube_count=3,
            policy_config=self.origpin,
            policy_checkpoint=self.checkpoint,
            inputs=self.inputs,
            source_encoder_config=self.source,
        )
        ledger=self.write(self.producer/'SEALED_CASES.json',{'mode':'smoke','cases':[oldcase]})
        row = dict(
            model='lewm_pcd',
            cube_count=3,
            policy_sha256=self.checkpoint['sha256'],
            encoder_sha256='a' * 64,
            target_inputs={'inputs': self.inputs},
        )
        result=self.write(self.producer/'RESULT.json',{'actual_exit_code':0,'result':{'cases':[row]}})
        script = self.producer / 'observer.py'
        script.write_text('# actual frozen producer\n')
        scriptpin = {'path': str(script), 'sha256': gate.sha(script)}
        observer=self.write(self.producer/'OBSERVER.json',{'actual_exit_code':0,'observer':scriptpin})
        files={Path(pin['path']).name:{'sha256':pin['sha256']}for pin in [ledger,result,observer]}
        files['policy/params_0.pkl']={'sha256':self.checkpoint['sha256']}
        freeze = self.write(
            self.producer / 'REFERENCE_FREEZE.json', {'actual_exit_code': 0, 'producer': scriptpin, 'files': files}
        )
        effective = dict(
            self.original,
            model='lewm_pcd',
            method='lewm_pcd',
            role='lewm_pcd_cube3',
            encoder_binding=self.runtime['encoder_binding'],
            source_policy_config=self.origpin,
        )
        cfg=self.write(self.root/'effective-config.json',effective)
        manifest = dict(
            schema='cube_M2_policy_origin_v1',
            model='lewm_pcd',
            method='lewm_pcd',
            role='lewm_pcd_cube3',
            cube_count=3,
            policy_config=cfg,
            checkpoint=self.checkpoint,
            encoder_binding=self.runtime['encoder_binding'],
            inputs=self.inputs,
            producer_reference_freeze=freeze,
            producer_cases=ledger,
            producer_result=result,
            producer_observer=observer,
        )
        self.manifest=manifest;self.cfg=effective
        self.case = dict(
            model='lewm_pcd',
            cube_count=3,
            policy_config=cfg,
            policy_checkpoint=self.checkpoint,
            inputs=self.inputs,
            source_encoder_config=self.source,
            policy_origin=self.write(self.root / 'origin.json', manifest),
        )

    def tearDown(self):self.temp.cleanup()

    def write(self,path,value):
        path.write_text(json.dumps(value),encoding='utf8');return {'path':str(path),'sha256':gate.sha(path)}

    def test_real_cfg_origin_and_producer_case_are_linked(self):
        cfg,evidence=gate._matrix_policy_config(self.case,self.runtime)
        self.assertEqual(cfg['role'],'lewm_pcd_cube3');self.assertTrue(evidence['passed'])

    def test_same_dimension_wrong_method_and_role_reject(self):
        for fields in [{'model':'lewm_rgb','method':'lewm_rgb','role':'lewm_rgb_cube3'},{'role':'lewm_pcd_cube2'}]:
            cfg=dict(self.cfg,**fields);case=dict(self.case,policy_config=self.write(self.root/'wrong.json',cfg))
            with self.subTest(fields=fields),self.assertRaises(ValueError):gate._matrix_policy_config(case,self.runtime)

    def test_checkpoint_SHA_alone_does_not_override_producer_origin(self):
        wrong=copy.deepcopy(self.manifest);wrong['role']='lewm_rgb_cube3'
        case=dict(self.case,policy_origin=self.write(self.root/'wrong-origin.json',wrong))
        with self.assertRaises(ValueError):gate._matrix_policy_config(case,self.runtime)

    def test_cfg_science_cannot_change_while_only_metadata_is_enriched(self):
        cfg=copy.deepcopy(self.cfg);cfg['agent_config']['lr']=.001
        cfgpin=self.write(self.root/'changed-science.json',cfg);manifest=dict(self.manifest,policy_config=cfgpin)
        case=dict(self.case,policy_config=cfgpin,policy_origin=self.write(self.root/'changed-origin.json',manifest))
        with self.assertRaises(ValueError):gate._matrix_policy_config(case,self.runtime)

    def test_changed_producer_ledger_is_not_accepted_by_its_old_freeze(self):
        record = copy.deepcopy(self.manifest)
        path = Path(record['producer_cases']['path'])
        ledger = json.loads(path.read_text())
        ledger['cases'][0]['model'] = 'lewm_rgb'
        record['producer_cases']=self.write(path,ledger)
        case=dict(self.case,policy_origin=self.write(self.root/'changed-ledger-origin.json',record))
        with self.assertRaises(ValueError):gate._matrix_policy_config(case,self.runtime)


if __name__=='__main__':unittest.main()

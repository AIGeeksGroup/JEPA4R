# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_policy_reference.py (verbatim except: imports /
# paths adapted to the package; lines wrapped at 120 columns)
from pathlib import Path
import sys,tempfile,unittest,json,copy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from jepa4r.policy.gciql import policy_reference as reference
from jepa4r.policy.gciql.workflow_downstream import sha
from jepa4r.policy.gciql.gates.arrays import array_identity


class CpuReferenceTrustTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.gate=self.root/'gate';self.gate.mkdir()
        self.runtime={'seed':0,'agent_config':{'lr':.0003}}
        self.recipe=self.root/'runtime.json';self.recipe.write_text(json.dumps(self.runtime))
        self.code=self.root/'producer.py';self.code.write_text('# synthetic metadata trust test\n')
        self.pin=lambda p:dict(path=str(p.resolve()),sha256=sha(p))
        raw={f'leaf{i}':np.array(i,np.float32)for i in range(153)}
        np.savez(self.gate / 'full_state.npz', **raw)
        np.savez(self.gate / 'next_batch.npz', data=np.ones(3, np.float32))
        np.savez(self.gate / 'next_ids.npz', ids=np.arange(3))
        self.expected = dict(
            full_state_sha256=sha(self.gate / 'full_state.npz'),
            saved_state={'leaves': 153, 'sha256': 'a' * 64},
            actual_source_binding=dict(
                versions={'numpy': np.__version__},
                code={'original.py': sha(self.code)},
                runtime={'JAX_default_matmul_precision': 'None'},
            ),
        )
        (self.gate/'report.json').write_text(json.dumps(self.expected))
        self.actions=self.root/'actions.npy';np.save(self.actions,np.ones((1024,5),np.float32))
        self.meta = dict(
            schema='gciql_cpu_endpoint_reference_v1',
            scope='new_CPU_reference_from_real_terminal_NPZ_not_GPU_original_output',
            actual_exit_code=0,
            backend='cpu',
            storage='terminal_full_state_npz',
            updates=0,
            precision='fp32',
            config_digest=reference.config_digest(self.runtime),
            endpoint_fingerprint=self.expected['saved_state'],
            complete153_mapping={'passed': True},
            leaf_mapping=[
                dict(path=k, shape=list(v.shape), endpoint_dtype=str(v.dtype), sha256=array_identity(v))
                for k, v in raw.items()
            ],
            source_export={
                **{
                    k: self.pin(self.gate / n)
                    for k, n in [
                        ('producer_report', 'report.json'),
                        ('full_state', 'full_state.npz'),
                        ('next_batch', 'next_batch.npz'),
                        ('next_ids', 'next_ids.npz'),
                    ]
                },
                'runtime_recipe': self.pin(self.recipe),
            },
            versions={'numpy': np.__version__},
            math_sources={str(self.code): sha(self.code)},
            jax_default_matmul_precision='None',
            producer_source=self.pin(self.code),
            mapping_source=self.pin(self.code),
            producer_source_sha256=sha(self.code),
            actions={**self.pin(self.actions), 'array_sha256': array_identity(np.ones((1024, 5), np.float32))},
            action_forward=dict(key=0, temperature=0, full_agent_state_unchanged=True, host_rng_unchanged=True),
        )

    def validate(self,meta):
        path=self.root/'reference.json';path.write_text(json.dumps(meta))
        return reference.validate_cpu_reference(self.pin(path),self.runtime,self.gate,self.expected)

    def test_complete_input_leaf_source_and_action_chain_passes(self):
        actions,_=self.validate(self.meta);self.assertEqual(actions.shape,(1024,5))

    def test_backend_scope_input_and_leaf_dtype_mutations_are_rejected(self):
        for field in ('backend','scope','input','leaf'):
            meta=copy.deepcopy(self.meta)
            if field=='backend':meta['backend']='gpu'
            elif field=='scope':meta['scope']='original GPU training output'
            elif field=='leaf':meta['leaf_mapping'][76]['endpoint_dtype']='float64'
            else:meta['source_export']['next_batch']=meta['source_export']['next_ids']
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate(meta)

    def test_source_and_action_bytes_are_verified_not_only_labels(self):
        self.code.write_text('# changed\n')
        with self.assertRaises(ValueError):self.validate(self.meta)


if __name__=='__main__':unittest.main()

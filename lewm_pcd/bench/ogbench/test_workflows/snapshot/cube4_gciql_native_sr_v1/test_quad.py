"""Two focused fake-CPU cases; no agent, encoder, simulator or GPU creation."""
import json
from pathlib import Path
import tempfile
import unittest
import os
import numpy as np
from quad_bridge import load_core,camera_position,QUATERNION,POSITION
from audit import verify
from run_group import effective_source

HERE=Path(__file__).resolve().parent
REF=Path(os.environ.get('QUAD_REFERENCE_ROOT',str(HERE/'reference_source')))/'cube23_distance_gciql_v1/native_sr_visual_v1'


class QuadTests(unittest.TestCase):
    def test_only_N4_registration_pose_and_original_192_projection(self):
        core=load_core(REF/'visual_sr.py',HERE/'bindings.json')
        self.assertTrue(core.quad_registration_proof['inverse_source_exact'])
        self.assertIn("choices=('G11', 'G12')",effective_source(REF/'m5_restore_v1/run_group.py'))
        latent=np.arange(192,dtype=np.float32)/1000;state=np.arange(55,dtype=np.float32)/100
        current,goal=core.make_policy_inputs(latent,latent,state,state)
        self.assertEqual((current.shape,goal.shape),((211,),(204,)))
        np.testing.assert_array_equal(current[-19:],state[:19])
        np.testing.assert_array_equal(goal[-12:],state[[19,20,21,28,29,30,37,38,39,46,47,48]])
        np.testing.assert_allclose(camera_position(QUATERNION),POSITION,atol=1e-12,rtol=0)
        with self.assertRaises(ValueError):core.validate_latent(np.ones(193,np.float32),'invalid')

    def test_reused_PCD_audit_requires_fresh2N_and_Quad_camera(self):
        with tempfile.TemporaryDirectory() as name:
            out=Path(name);ep=out/'episodes/task1/episode_000';ep.mkdir(parents=True)
            cam=dict(position=POSITION,source_position=[1.287,0,.509],actual_quaternion=QUATERNION,pullback=.1,crop=None,aa_samples=4,geometry_offsamples=0)
            item=dict(actual_exit_code=0,group_id='G12',task_id=1,episode_index=0,seed=970100000,steps=4,
                current_goal_contract=dict(current_dim=211,goal_dim=204,synthetic_zero_padding=False),temperature=0.,gaussian_noise=None,
                goal_online_encoder_calls=4,online_encoder_calls=8,encoding_protocol='fresh_current_and_goal_each_policy_call_2N',
                original_evaluate_reused=True,success=0,camera=cam,goal_capture_proof=dict(physics_unchanged=True,env_rng_unchanged=True,geometry_offsamples=0),
                policy_input_contracts=[dict(current_dim=211,goal_dim=204,current_finite=True,goal_finite=True) for _ in range(4)])
            report=dict(actual_exit_code=0,stage='complete',group_id='G12',mode='diagnostic',episodes=1,unique_seeds=1,current_dim=211,goal_dim=204,
                max_episode_steps=4,encoding_protocol=item['encoding_protocol'],online_encoder=True,offline_embedding_cache_used_for_policy=False,
                synthetic_zero_padding=False,successes=0,success_rate=0.)
            for path,value in [(ep/'result.json',item),(out/'report.json',report),(out/'runtime_threads.json',dict(torch_num_threads=1,numpy_physics=dict(BLAS_threads=4)))]:path.write_text(json.dumps(value))
            (ep/'attempt_exit_code').write_text('0\n');(out/'actual_exit_code').write_text('0\n')
            result=verify('G12','smoke',out,REF/'m5_restore_v1/audit.py');self.assertEqual(result['actual_exit_code'],0)
            item['online_encoder_calls']=4;(ep/'result.json').write_text(json.dumps(item))
            with self.assertRaises(RuntimeError):verify('G12','smoke',out,REF/'m5_restore_v1/audit.py')


if __name__=='__main__':unittest.main()

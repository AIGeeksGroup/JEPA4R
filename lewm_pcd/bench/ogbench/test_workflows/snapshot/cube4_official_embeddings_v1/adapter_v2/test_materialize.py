"""Only training-camera geometry and metadata deltas; no model or renderer."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import materialize

ORIGINAL=Path(os.environ.get('ORIGINAL_2D_CACHE_WORKSPACE',str(HERE.parents[1]/'cube2_official_embeddings_101732Test_20261018_v1/effective_sources_v2')))


class CameraTests(unittest.TestCase):
    def test_approved_localZ_pullback_matches_training_pose_not_worldX(self):
        q=np.array([.5609588321928806,.4304941214288525,.4304941214288525,.5609588321928807])
        w,x,y,z=q
        local_Z=np.array([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)])
        np.testing.assert_allclose(np.array(materialize.SOURCE_POSITION)+.1*local_Z,
                                   materialize.TRAINING_POSITION,atol=1e-12,rtol=0)
        self.assertFalse(np.allclose(materialize.TRAINING_POSITION,[1.387,0.,.509]))

    def test_producer_math_unchanged_and_only_renderer_audit_camera_changes(self):
        sources,ledger=materialize.build_sources(ORIGINAL)
        self.assertEqual(hashlib.sha256(sources[materialize.MODULE+'/produce.py']).hexdigest(),
                         materialize.V1_EFFECTIVE[materialize.MODULE+'/produce.py'])
        renderer=sources[materialize.MODULE+'/render_adapter.py'].decode()
        auditor=sources[materialize.MODULE+'/audit_cache.py'].decode()
        self.assertIn('position + .10 * camera_rotation.reshape(3, 3)[:, 2]',renderer)
        self.assertIn('= render_position, actual, 45.',renderer)
        self.assertIn('pullback=.10, crop=None',renderer)
        self.assertIn("camera['source_position']",renderer)
        self.assertIn('pullback=.10, crop=None',auditor)
        self.assertEqual(len(ledger),3)

    def test_plan_uses_verified_training_camera_and_preserves_binding(self):
        plan_path=Path(os.environ.get('ORIGINAL_2D_PLAN_PATH',str(HERE.parents[1]/'cube2_official_embeddings_101732Test_20261018_v1/completed_metadata_v2/workspace/official_state_2d_cache_v1/producer_plan.json')))
        baseline=json.loads(plan_path.read_text())['camera']
        requested=copy.deepcopy(baseline)
        requested.update(source_position=materialize.SOURCE_POSITION,
                         position=materialize.TRAINING_POSITION,pullback=.10)
        binding={'camera':requested};original=copy.deepcopy(binding)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/materialize.MODULE).mkdir()
            (root/materialize.MODULE/'producer_plan.json').write_text(json.dumps({'camera':baseline}))
            with patch.object(materialize,'ORIGINAL_MAKE_PLAN',return_value={'camera':baseline,'unchanged':True}):
                plan=materialize.make_plan(root,root,root,binding,{})
                self.assertEqual(plan['camera'],requested)
                self.assertTrue(plan['unchanged'])
                self.assertEqual(binding,original)
                binding['camera']['pullback']=0.
                with self.assertRaises(AssertionError):materialize.make_plan(root,root,root,binding,{})


if __name__=='__main__':unittest.main()

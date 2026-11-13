# Moved from OGBench-cube@4fbb0c6:lewm_pcd/scripts/checks/test_encoder_freeze_identity.py (verbatim except: module names / paths of the moved code adapted)
"""All registered encoder tensors, including nonpersistent buffers, are frozen."""
import unittest
import torch
from jepa4r.policy.bc.relocate.encoder_identity import encoder_digest,assert_encoder_unchanged

class EncoderIdentityTests(unittest.TestCase):
    def model(self):
        model=torch.nn.Linear(2,2)
        model.register_buffer('cached_geometry',torch.tensor([1.,2.]),persistent=False)
        return model
    def test_nonpersistent_single_value_change_is_rejected(self):
        model=self.model();before=encoder_digest(model);old={k:v.clone() for k,v in model.state_dict().items()}
        model.cached_geometry[0]+=1.
        self.assertTrue(all(torch.equal(value,model.state_dict()[key]) for key,value in old.items()))
        with self.assertRaisesRegex(ValueError,'parameter or buffer'):assert_encoder_unchanged(model,before)
    def test_unchanged_and_independent_copy_match(self):
        import copy
        model=self.model();other=copy.deepcopy(model);before=encoder_digest(model)
        self.assertEqual(encoder_digest(other),before);assert_encoder_unchanged(model,before)
    def test_dtype_shape_and_name_are_part_of_identity(self):
        model=self.model();model.cached_geometry.zero_();before=encoder_digest(model)
        model.cached_geometry=model.cached_geometry.reshape(1,2)
        self.assertNotEqual(before,encoder_digest(model))
        model.cached_geometry=torch.zeros(2,dtype=torch.int32)
        self.assertNotEqual(before,encoder_digest(model))
        model.cached_geometry=torch.zeros(2);value=model.cached_geometry;delattr(model,'cached_geometry')
        model.register_buffer('renamed_geometry',value,persistent=False)
        self.assertNotEqual(before,encoder_digest(model))
    def test_policy_parameters_only_keeps_original_digest(self):
        import os,importlib.util
        from jepa4r.envs.twins.rh56.bc_bridge import _model_sha
        path=os.environ.get('BC_OLD_BRIDGE')
        if not path:self.skipTest('requires existing frozen old bridge source')
        spec=importlib.util.spec_from_file_location('old_bridge_policy_digest',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        model=self.model();self.assertEqual(_model_sha(model,parameters_only=True),old._model_sha(model,parameters_only=True))

if __name__=='__main__':unittest.main()

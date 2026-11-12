"""Read-only norm observation around the original clip_grad_norm_ call."""
import unittest
import torch
from bench.relocate.clip_evidence import observe_gradient_clipping
class ClipEvidenceTests(unittest.TestCase):
 def test_original_return_and_all_clipped_gradients_are_bitwise_unchanged(self):
  def parameters():
   a=torch.nn.Parameter(torch.ones(2));b=torch.nn.Parameter(torch.ones(2));a.grad=torch.tensor([3.,4.]);b.grad=torch.tensor([0.,12.]);return [a,b]
  expected=parameters();actual=parameters();original=torch.nn.utils.clip_grad_norm_;returned=original(expected,1.,foreach=False);records=[];rng=torch.get_rng_state()
  class Model:
   def named_parameters(self):return [('encoder.weight',actual[0]),('predictor.weight',actual[1])]
  with observe_gradient_clipping(records,model_provider=lambda:Model()):value=torch.nn.utils.clip_grad_norm_(iter(actual),1.,foreach=False)
  self.assertIs(torch.nn.utils.clip_grad_norm_,original);self.assertTrue(torch.equal(returned,value));self.assertTrue(torch.equal(torch.get_rng_state(),rng))
  for a,b in zip(expected,actual):self.assertTrue(torch.equal(a.grad,b.grad))
  self.assertEqual(len(records),1);self.assertEqual(records[0]['norm_before_original_return'],13.);self.assertLessEqual(records[0]['norm_after'],1.000001);self.assertEqual(records[0]['encoder_norm_before'],5.);self.assertGreater(records[0]['encoder_norm_after'],0.)
 def test_binding_restored_when_original_raises(self):
  original=torch.nn.utils.clip_grad_norm_
  with self.assertRaisesRegex(ValueError,'probe'):
   with observe_gradient_clipping([]):raise ValueError('probe')
  self.assertIs(torch.nn.utils.clip_grad_norm_,original)
if __name__=='__main__':unittest.main()

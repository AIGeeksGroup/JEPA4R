# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_pldm_gate_capture.py (verbatim except: imports /
# paths adapted to the package; lines wrapped at 120 columns)
"""The numeric observer reads the actual original PLDM forward-result once."""
from pathlib import Path
from types import SimpleNamespace
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from jepa4r.policy.gciql import gate_checks

class CaptureTests(unittest.TestCase):
    def test_original_result_and_arguments_are_unchanged(self):
        calls=[];encoded=object();prediction=object();actions=object()
        result = SimpleNamespace(
            backbone_output=SimpleNamespace(encodings=encoded),
            pred_output=SimpleNamespace(predictions=prediction),
            actions=actions,
        )
        def original(*args,**kwargs):calls.append((args,kwargs));return result
        model=SimpleNamespace(forward_result=original);captured={}
        gate_checks.observe_pldm_result(model,captured)
        value=model.forward_result('real RGB16','real A15',tag='fixed')
        self.assertIs(value,result);self.assertEqual(calls,[(('real RGB16','real A15'),{'tag':'fixed'})])
        self.assertIs(captured['prediction'],prediction);self.assertIs(captured['encoder_full'],encoded)
        self.assertIs(captured['actions_used'],actions)

if __name__=='__main__':unittest.main()

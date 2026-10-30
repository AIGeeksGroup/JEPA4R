"""Default-off FP16 canonical-N1 adapter; v1 and all original sources stay frozen."""
import argparse
import ast
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import types

V1=Path(__file__).with_name('precision_bridge.py')
V1_SHA='8ac34a03554671ccdbe5cdd63d6a7f8881850a25e31afeb7fbc3321f18766463'
assert hashlib.sha256(V1.read_bytes()).hexdigest()==V1_SHA
_spec=importlib.util.spec_from_file_location('_precision_v1_frozen_source',V1)
v1=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(v1)

def effective_source(precision='fp32'):
    assert precision in ('fp32','fp16')
    source=v1.effective_source(precision)
    if precision=='fp32':return source
    before='        device_type = next(self.encoder.parameters()).device.type'
    after=("        assert self.spec.get('canonical_microbatch') == 1\n"
           "        if len(raw) > 1:\n"
           "            with torch.autocast(device_type=next(self.encoder.parameters()).device.type, enabled=False):\n"
           "                return torch.cat([self.encode_uint8(raw[i:i+1]) for i in range(len(raw))], dim=0)\n"+before)
    assert source.count(before)==1
    result=source.replace(before,after);assert result.replace(after,before)==source
    ast.parse(result);return result

@lru_cache(maxsize=2)
def _api(precision):
    module=types.ModuleType('_frozen_canonical_encoder_'+precision);module.__file__=str(v1.BASE)
    exec(compile(effective_source(precision),str(v1.BASE),'exec'),module.__dict__)
    return module

def load_encoder(spec,device='cuda'):
    if isinstance(spec,(str,Path)):spec=json.loads(Path(spec).read_text())
    precision=spec.get('inference_precision','fp32');assert precision in ('fp32','fp16')
    if precision=='fp16':
        assert spec['family']=='vjepa21_vitl1024' and spec.get('canonical_microbatch')==1
        assert str(device).startswith('cuda')
    return _api(precision).load_encoder(spec,device=device)

def encode_uint8(model,pixels):return _api(model.spec.get('inference_precision','fp32')).encode_uint8(model,pixels)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-source',action='store_true',required=True);parser.parse_args()
    assert effective_source('fp32')==v1.BASE.read_text()
    print(json.dumps(dict(actual_exit_code=0,default_off_exact=True,half_v1_inverse_exact=True,NN_forward=False,
        effective_SHA256={mode:hashlib.sha256(effective_source(mode).encode()).hexdigest() for mode in ('fp32','fp16')})))

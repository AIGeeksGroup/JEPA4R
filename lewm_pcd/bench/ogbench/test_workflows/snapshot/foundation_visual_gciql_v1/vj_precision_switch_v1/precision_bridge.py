"""Default-off V-JEPA forward precision switch over the frozen complete adapter."""
import argparse
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import types

BASE=Path(__file__).resolve().parents[1]/'encoder/encoder_adapter.py'
BASE_SHA='3f5cb292728c809737ff0dbe851095cec8825883b0f009d913d315677f462bfe'
PRECISIONS=('fp32','fp16','bf16')

def effective_source(precision='fp32'):
    assert precision in PRECISIONS,'Explicit fp32/fp16/bf16 required'
    raw=BASE.read_bytes();assert hashlib.sha256(raw).hexdigest()==BASE_SHA
    original=raw.decode()
    if precision=='fp32':return original
    before='                tokens = self.encoder(x)'
    dtype={'fp16':'float16','bf16':'bfloat16'}[precision]
    after=(f"                assert self.spec.get('inference_precision') == '{precision}'\n"
           "                assert device_type == 'cuda', 'Reduced precision requires CUDA'\n"
           f"                with torch.autocast(device_type='cuda', dtype=torch.{dtype}):\n"
           "                    tokens = self.encoder(x)")
    pool_before='                features = tokens.mean(dim=1)'
    pool_after='                features = tokens.mean(dim=1).float()'
    assert original.count(before)==original.count(pool_before)==1
    result=original.replace(before,after).replace(pool_before,pool_after)
    assert result.replace(after,before).replace(pool_after,pool_before)==original
    ast.parse(result)
    return result

@lru_cache(maxsize=3)
def _api(precision):
    module=types.ModuleType('_frozen_encoder_adapter_'+precision);module.__file__=str(BASE)
    exec(compile(effective_source(precision),str(BASE),'exec'),module.__dict__)
    return module

def load_encoder(spec,device='cuda'):
    if isinstance(spec,(str,Path)):spec=json.loads(Path(spec).read_text())
    precision=spec.get('inference_precision','fp32');assert precision in PRECISIONS
    if precision!='fp32':
        assert spec['family']=='vjepa21_vitl1024','VISReg path is unchanged'
        assert str(device).startswith('cuda'),'No silent CPU precision fallback'
    return _api(precision).load_encoder(spec,device=device)

def encode_uint8(model,pixels):
    return _api(model.spec.get('inference_precision','fp32')).encode_uint8(model,pixels)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-source',action='store_true',required=True);parser.parse_args()
    original=BASE.read_text();proofs={}
    for precision in PRECISIONS:
        source=effective_source(precision);ast.parse(source)
        if precision=='fp32':assert source==original
        proofs[precision]=hashlib.sha256(source.encode()).hexdigest()
    try:effective_source('fp8')
    except AssertionError:pass
    else:raise AssertionError('Unsupported precision accepted')
    print(json.dumps(dict(actual_exit_code=0,NN_forward=False,default_off_source_exact=True,half_inverse_source_exact=True,effective_source_SHA256=proofs)))

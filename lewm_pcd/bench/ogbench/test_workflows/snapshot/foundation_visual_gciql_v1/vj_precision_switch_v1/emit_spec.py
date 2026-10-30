"""Write a fresh explicit compute-precision spec; original weight/protocol fields stay exact."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from precision_bridge import PRECISIONS

ORIGINAL_SPEC_SHA='c8323e5c917549115e29d9374d1f453e5c1fce2b69982a2aadd7ca247a217a99'

def make_spec(original,precision):
    assert precision in PRECISIONS
    assert original['family']=='vjepa21_vitl1024' and original['native_size']==384 and original['output_dim']==1024
    assert original['precision']=='FP32' and 'inference_precision' not in original
    assert original['constructor_kwargs']['img_temporal_dim_size']==1 and original['constructor_kwargs']['use_sdpa'] is True
    result=copy.deepcopy(original)
    result['inference_precision']=precision
    result['precision']='FP32' if precision=='fp32' else precision.upper()+'_autocast_FP32_storage'
    restored=copy.deepcopy(result);restored.pop('inference_precision');restored['precision']=original['precision']
    assert restored==original
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-spec',type=Path,required=True)
    p.add_argument('--precision',choices=PRECISIONS,required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();raw=args.source_spec.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SPEC_SHA
    result=make_spec(json.loads(raw),args.precision)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as out:json.dump(result,out,indent=2)
    print(json.dumps(dict(actual_exit_code=0,precision=args.precision,storage_dtype='float32',spec_only_no_NN=True,
        output=str(args.output) if args.output else None,original_fields_exact=True)))

if __name__=='__main__':main()

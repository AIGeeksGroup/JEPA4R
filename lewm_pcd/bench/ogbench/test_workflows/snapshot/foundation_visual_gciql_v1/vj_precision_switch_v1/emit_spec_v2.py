"""Explicit fresh FP16 canonical-N1 spec; preserve the complete original contract."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

V1=Path(__file__).with_name('emit_spec.py')
assert hashlib.sha256(V1.read_bytes()).hexdigest()=='8c94b290581fca22f5f2004ebb1ce14b0ac88a02e863a55309d29ee3c7d9a09a'
_spec=importlib.util.spec_from_file_location('_frozen_spec_v1',V1)
v1=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(v1)

def make_spec(original):
    result=v1.make_spec(original,'fp16');result['canonical_microbatch']=1
    restored=dict(result);restored.pop('canonical_microbatch');restored.pop('inference_precision');restored['precision']=original['precision']
    assert restored==original
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-spec',type=Path,required=True);parser.add_argument('--output',type=Path)
    args=parser.parse_args();raw=args.source_spec.read_bytes();assert hashlib.sha256(raw).hexdigest()==v1.ORIGINAL_SPEC_SHA
    spec=make_spec(json.loads(raw))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(spec,stream,indent=2)
    print(json.dumps(dict(actual_exit_code=0,spec_only_no_NN=True,compute_precision='fp16',canonical_microbatch=1,storage='float32',output=str(args.output))))

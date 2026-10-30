"""CPU audit resource identity adapter; preserve all original acceptance guards."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import audit_bridge

ORIGINAL_SOURCE=audit_bridge.source
BEFORE="PRODUCER = 'foundation_visual_gciql_v1/cache/producer_bridge.py'"
AFTER="PRODUCER = 'foundation_visual_gciql_v1/cache/resource_w8.py'"


def source(plan):
    assert plan['renderer_workers']==8 and plan['prefetch']==8 and plan['encoder_batch']==32
    original=ORIGINAL_SOURCE(plan)
    assert original.count(BEFORE)==1
    # The original auditor has no renderer==2 assertion: worker count is bound
    # through the declared protocol rather than a guessed extra source edit.
    assert "plan['renderer_workers'] == 2" not in original
    result=original.replace(BEFORE,AFTER)
    assert result.count(AFTER)==1 and result.replace(AFTER,BEFORE)==original
    ast.parse(result)
    return result


def load(plan):
    # Reuse load's complete camera and B1/B32 proof closure. The override is local
    # to this sequential construction, and the imported module is restored always.
    assert audit_bridge.source is ORIGINAL_SOURCE
    audit_bridge.source=source
    try:module=audit_bridge.load(plan)
    finally:audit_bridge.source=ORIGINAL_SOURCE
    module.PROTOCOL_KEYS=module.PROTOCOL_KEYS+('prefetch',)
    module.__file__=str(Path(__file__).resolve())
    return module


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path)
    parser.add_argument('--cache-dir',type=Path);parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-source',action='store_true');args=parser.parse_args()
    if args.check_source:
        assert args.plan is args.cache_dir is args.output_dir is None
        proofs=[]
        for dim in (768,1024):
            for cubes in (1,2,3,4):
                plan=dict(cube_count=cubes,latent_dim=dim,renderer_workers=8,prefetch=8,encoder_batch=32)
                text=source(plan);module=load(plan)
                assert module.PRODUCER=='foundation_visual_gciql_v1/cache/resource_w8.py'
                assert module.PROTOCOL_KEYS.count('prefetch')==1
                assert module.camera_check is audit_bridge.check_camera
                assert module.audit.__closure__ is not None
                proofs.append(dict(cube_count=cubes,latent_dim=dim,
                    effective_source_sha256=hashlib.sha256(text.encode()).hexdigest()))
        print(json.dumps(dict(actual_exit_code=0,prepared_only=True,NN_forward=False,
            exact_inverse_to_original=True,camera_and_alignment_closure_reused=True,proofs=proofs)))
        return
    assert args.plan is not None and args.cache_dir is not None,'--plan and --cache-dir are required'
    load(json.loads(args.plan.read_text())).main()


if __name__=='__main__':main()

"""Eight-renderer resource switch over the unchanged frozen foundation producer."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import producer_bridge


EDITS=(
    ("assert plan['renderer_workers'] == 2", "assert plan['renderer_workers'] == 8"),
    ('ProcessPoolExecutor(max_workers=2,', 'ProcessPoolExecutor(max_workers=8,'),
    ('for _ in range(min(2, len(pending))):', 'for _ in range(min(8, len(pending))):'),
)


def effective_source(plan):
    assert plan['renderer_workers']==8 and plan['prefetch']==8 and plan['encoder_batch']==32
    original=producer_bridge.effective_source(plan)
    source=original
    for before,after in EDITS:
        assert source.count(before)==1,before
        source=source.replace(before,after)
    restored=source
    for before,after in reversed(EDITS):
        assert restored.count(after)==1,after
        restored=restored.replace(after,before)
    assert restored==original,'Only the three declared resource edits are allowed'
    ast.parse(source)
    return source


def load(plan):
    module=producer_bridge.load(plan)
    source=effective_source(plan)
    path=Path(__file__).resolve()
    module.__file__=str(path)
    exec(compile(source,str(path),'exec'),module.__dict__)
    # Source initialization resets HERE and CONTRACT_KEYS; restore the same renderer
    # location and existing foundation contract additions, then bind prefetch too.
    module.__file__=str(path)
    module.HERE=path.parent
    module.CONTRACT_KEYS=module.CONTRACT_KEYS+producer_bridge.EXTRA+('prefetch',)
    module.source_arrays=producer_bridge.arrays_existing
    return module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--check-source',action='store_true')
    args=parser.parse_args()
    if args.check_source:
        assert not args.execute and args.plan is None and not args.pilot
        proofs=[]
        for dim in (768,1024):
            for cubes in (1,2,3,4):
                plan=dict(cube_count=cubes,latent_dim=dim,renderer_workers=8,prefetch=8,encoder_batch=32)
                source=effective_source(plan)
                proofs.append(dict(cube_count=cubes,latent_dim=dim,
                    effective_source_sha256=hashlib.sha256(source.encode()).hexdigest()))
        print(json.dumps(dict(actual_exit_code=0,prepared_only=True,NN_forward=False,
            exact_inverse_to_original=True,resource_edits=EDITS,proofs=proofs)))
        return
    assert args.plan is not None,'--plan is required'
    plan=json.loads(args.plan.read_text())
    module=load(plan)
    # Reuse the original CLI/readiness/execute/exception path without another loop.
    module.main()


if __name__=='__main__':main()

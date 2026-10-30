"""Keep full cache auditing; validate the explicit FP16 same-batch proof schema."""
import argparse
import inspect
import json
from pathlib import Path
import audit_bridge
from fp16_reference_producer_v1 import POLICY,PROOF,validate_reference_proof

ORIGINAL_SOURCE=audit_bridge.source

def source(plan):
    assert plan['batch_reference_policy']==POLICY and plan['inference_precision']=='fp16'
    original=ORIGINAL_SOURCE(plan)
    before="PRODUCER = 'foundation_visual_gciql_v1/cache/producer_bridge.py'"
    after="PRODUCER = 'foundation_visual_gciql_v1/cache/fp16_reference_producer_v1.py'"
    assert original.count(before)==1
    result=original.replace(before,after);assert result.replace(after,before)==original
    return result

def load(plan):
    assert audit_bridge.source is ORIGINAL_SOURCE
    audit_bridge.source=source
    try:module=audit_bridge.load(plan)
    finally:audit_bridge.source=ORIGINAL_SOURCE
    module.__file__=str(Path(__file__).resolve())
    module.PROTOCOL_KEYS=module.PROTOCOL_KEYS+('prefetch','inference_precision','batch_reference_policy')
    # The old wrapper adds only batch1_alignment checks. Reuse its captured full
    # original auditor (including all camera/source/NPY SHA/finite/ID checks), then
    # attach the versioned new proof. Never claim the old cross-batch gate passed.
    full_audit=inspect.getclosurevars(module.audit).nonlocals['original']
    def audit(cache_dir,plan_path,progress=None):
        report=full_audit(cache_dir,plan_path,progress)
        path=Path(cache_dir)/PROOF;proof=json.loads(path.read_text())
        manifest=json.loads((Path(cache_dir)/'cache_manifest.json').read_text())
        validate_reference_proof(proof,plan,manifest['source_encoder_state_sha256'])
        report['precision_reference_alignment']=dict(path=str(path),sha256=audit_bridge.sha(path),batch_reference_policy=POLICY)
        report['legacy_cross_batch_diagnostic']=proof['legacy_diagnostic']
        return report
    module.audit=audit
    return module

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();load(json.loads(args.plan.read_text())).main()

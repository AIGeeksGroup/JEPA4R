"""Bulk-FP16 producer: same-batch official reference proof; old FP32 gate untouched."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import resource_w8
import producer_bridge

POLICY='same_batch_official_fp16_reference_v1'
PROOF='precision_reference_alignment.json'

def validate_reference_proof(proof,plan,state_sha256):
    assert proof['schema_version']==1 and proof['actual_exit_code']==0 and proof['batch_reference_policy']==POLICY
    assert proof['inference_precision']=='fp16' and proof['storage_dtype']=='float32'
    assert proof['encoder_spec_sha256']==plan['encoder_spec_sha256'] and proof['frozen_encoder_state_sha256']==state_sha256
    assert len(state_sha256)==64
    for batch in ('B32','B1'):
        record=proof['same_batch_reference'][batch]
        assert record['wrapper_official_bitwise_equal'] is True and record['finite'] is True and record['max_abs_difference']==0.
        assert record['prepared_tensor_reused_exactly'] is True
        assert record['embedding_shape']==([32,1024] if batch=='B32' else [1,1024])
    legacy=proof['legacy_diagnostic']
    assert legacy['atol']==legacy['rtol']==1e-5 and legacy['used_for_acceptance'] is False
    assert isinstance(legacy['original_cross_batch_allclose'],bool)
    assert len(proof['raw_RGB_B32_sha256'])==len(proof['first_real_uint8_RGB_sha256'])==64
    return True

def prove_alignment(model,raw,plan,frozen_state_sha256):
    """One B32 wrapper call + same-input official ref, and the same pair for B1."""
    import torch
    assert tuple(raw.shape)==(32,3,224,224) and raw.dtype==torch.uint8
    assert plan['batch_reference_policy']==POLICY and plan['inference_precision']=='fp16'
    assert model.spec['inference_precision']=='fp16' and model.spec['batch_reference_policy']==POLICY
    assert model.spec['family']=='vjepa21_vitl1024'
    assert 'canonical_microbatch' not in model.spec
    def compare(value):
        capture={}
        hook=model.encoder.register_forward_pre_hook(lambda module,inputs:capture.update(x=inputs[0]))
        try:wrapper=model.encode_uint8(value)
        finally:hook.remove()
        x=capture['x'];assert x.dtype==torch.float32 and tuple(x.shape)==(len(value),3,1,384,384)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            tokens=model.encoder(x);official=tokens.mean(dim=1).float()
        assert tuple(tokens.shape)==(len(value),576,1024)
        finite=bool(torch.isfinite(wrapper).all() and torch.isfinite(official).all())
        equal=bool(torch.equal(wrapper,official));assert finite and equal,'Same-batch official FP16 reference differs'
        return wrapper,dict(wrapper_official_bitwise_equal=equal,finite=finite,max_abs_difference=float((wrapper-official).abs().max()),
            prepared_tensor_reused_exactly=True,prepared_dtype=str(x.dtype),tokens_dtype=str(tokens.dtype),embedding_shape=list(wrapper.shape))
    with torch.inference_mode():
        bulk,b32=compare(raw);single,b1=compare(raw[:1])
        original=bool(torch.allclose(single,bulk[:1],atol=1e-5,rtol=1e-5))
        images=raw.detach().cpu().permute(0,2,3,1).contiguous().numpy()
        proof=dict(schema_version=1,actual_exit_code=0,batch_reference_policy=POLICY,inference_precision='fp16',storage_dtype='float32',
            encoder_spec_sha256=plan['encoder_spec_sha256'],frozen_encoder_state_sha256=frozen_state_sha256,
            raw_RGB_B32_sha256=hashlib.sha256(images.tobytes()).hexdigest(),first_real_uint8_RGB_sha256=hashlib.sha256(images[0].tobytes()).hexdigest(),
            same_batch_reference=dict(B32=b32,B1=b1),offline_encoder_batch=32,online_reference_batch=1,
            legacy_diagnostic=dict(atol=1e-5,rtol=1e-5,original_cross_batch_allclose=original,
                max_abs_difference=float((single-bulk[:1]).abs().max()),used_for_acceptance=False))
    validate_reference_proof(proof,plan,frozen_state_sha256)
    return bulk,proof

def effective_source(plan):
    assert plan['batch_reference_policy']==POLICY and plan['inference_precision']=='fp16'
    original=resource_w8.effective_source(plan)
    begin=original.index('        latent = model.encode_uint8(tensor)')
    end=original.index('        return tensor, latent',begin)
    replacement='''        if not alignment_checked:
            alignment_path=out/'precision_reference_alignment.json'
            if alignment_path.exists():
                proof=json.loads(alignment_path.read_text())
                precision_validate_reference(proof, plan, frozen_hash)
                latent=model.encode_uint8(tensor)
            else:
                latent, proof = precision_prove_alignment(model, tensor, plan, frozen_hash)
                save(alignment_path, proof)
            alignment_checked=True
        else:
            latent=model.encode_uint8(tensor)
        assert latent.dtype == torch.float32 and tuple(latent.shape) == (32,plan['latent_dim'])
'''
    result=original[:begin]+replacement+original[end:]
    assert result.replace(replacement,original[begin:end],1)==original
    ast.parse(result);return result

def load(plan):
    module=resource_w8.load(plan);path=Path(__file__).resolve();module.__file__=str(path)
    exec(compile(effective_source(plan),str(path),'exec'),module.__dict__)
    module.__file__=str(path);module.HERE=path.parent
    module.CONTRACT_KEYS=module.CONTRACT_KEYS+producer_bridge.EXTRA+('prefetch','inference_precision','batch_reference_policy')
    module.source_arrays=producer_bridge.arrays_existing;module.precision_prove_alignment=prove_alignment
    module.precision_validate_reference=validate_reference_proof
    return module

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();assert not args.pilot,'New full-cache protocol only; do not route through old pilot B1 proof'
    load(json.loads(args.plan.read_text())).main()

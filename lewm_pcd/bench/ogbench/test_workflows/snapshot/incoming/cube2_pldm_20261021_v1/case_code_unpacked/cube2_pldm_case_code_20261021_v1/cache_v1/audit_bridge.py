"""Reuse the accepted raw1001 audit, changing only the PLDM encoder contract."""
import hashlib
from pathlib import Path
import types

from producer_bridge import ROOT,ADAPTER,ASSET,NEW_KEYS

ORIGINAL=ROOT/'cube2_official_embeddings_v1/workspace/official_state_2d_cache_v1/audit_cache.py'
ORIGINAL_SHA='75eb5137c30cac3628f0e781c7d2eb4a96b29eb7689a7fcae6ce1509dc303adc'


def effective_source(raw):
    assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA
    edits=[("extractor_source_sha256=plan['source_pins'][PRODUCER]","extractor_source_sha256=plan['source_arrays_original_extractor_sha256']",1),
      ("plan['source_pins']['source/lewm_pcd/utils.py']","plan['source_pins']["+repr(ASSET+'/encoder.py')+"]",1),
      ("plan['source_pins']['checkpoint_encoder.py']","plan['source_pins']["+repr(ADAPTER)+"]",1)]
    edits.append(("    assert manifest['source_arrays_receipt']['sha256'] == arrays_sha\n",
        "    assert manifest['source_arrays_receipt']['sha256'] == arrays_sha == plan['source_arrays_view_sha256']\n",1))
    source=raw.decode()
    for before,after,count in edits:assert source.count(before)==count;source=source.replace(before,after)
    return source,edits


def load_original():
    source,edits=effective_source(ORIGINAL.read_bytes())
    original=types.ModuleType('_PLDM_original_RGB_audit');original.__file__=str(Path(__file__).resolve())
    exec(compile(source,str(ORIGINAL),'exec'),original.__dict__)
    original.DIM=512;original.PRODUCER='cube2_pldm_gciql_v1/cache_v1/producer_bridge.py'
    original.PROTOCOL_KEYS=original.PROTOCOL_KEYS+NEW_KEYS
    # Exact descriptive field fix already accepted for the original LeWM audit.
    original.GATE=dict(original.GATE,scope='original preprocessing plus frozen encode; rendering excluded; CUDA tail synchronized')
    audit=original.audit
    def with_alignment(cache_dir,plan_path,progress=None):
        report=audit(cache_dir,plan_path,progress)
        import json
        plan=json.loads(Path(plan_path).read_text());cache=Path(cache_dir)
        assert plan['encoder_family']=='pldm' and plan['latent_dim']==512
        manifest=json.loads((cache/'cache_manifest.json').read_text());proof_path=cache/'batch1_alignment.json'
        proof=json.loads(proof_path.read_text())
        assert proof['actual_exit_code']==0 and proof['passed'] and proof['finite']
        assert proof['batch32_shape']==[32,512] and proof['batch1_shape']==[1,512]
        assert proof['checkpoint_sha256']==plan['checkpoint_sha256'] and proof['frozen_encoder_state_sha256']==manifest['source_encoder_state_sha256']
        assert proof['preprocessing_selection_sha256']==manifest['protocol']['preprocessing_selection_sha256']
        assert proof['preprocess_device']==manifest['protocol']['pixel_preprocess_device']
        assert proof['atol']==original.GATE['latent_atol'] and proof['rtol']==original.GATE['latent_rtol']
        report.update(encoder_family='pldm',batch1_alignment_passed=True,batch1_alignment_sha256=original.sha(proof_path))
        return report
    original.audit=with_alignment
    return original


if __name__=='__main__':load_original().main()

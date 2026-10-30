"""Counted in-memory PLDM family switch over the original RGB producer."""
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
ORIGINAL=ROOT/'official_state_2d_cache_v1/produce.py'
ORIGINAL_SHA='dbbce2407f11c42ee2c5aca7601c9a055a7ceb08593fb4e99b1b0cd7e5415b18'
ADAPTER='pldm_cube2_reaudit_20261029_v1/cache_v1/encoder_adapter.py'
ASSET='assets/baselines/cube2/cube2_pldm_impala512_epoch10_20261021_v1'
NEW_KEYS=('encoder_family','encoder_asset_id','encoder_asset_manifest_sha256','encoder_runtime_addendum_sha256','encoder_preprocess')


def effective_source(raw):
    assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA
    data=__import__('source_overlay').data_producer(raw)
    raw=data
    edits=[('latent_dim=192',"latent_dim=512",1),
      ("workspace / 'checkpoint_encoder.py'",'workspace / '+repr(ADAPTER),1),
      ('tuple(prepared.shape) == (32, 3, 224, 224)','tuple(prepared.shape) == (32, 3, 65, 65)',1),
      ('result.shape == (len(images), 192)','result.shape == (len(images), 512)',1),
      ('shape=[1001, 192]','shape=[1001, 512]',1),
      ('extractor_source_sha256=producer_hash',"extractor_source_sha256=plan['source_arrays_original_extractor_sha256']",1),
      ("plan['source_pins']['source/lewm_pcd/utils.py']","plan['source_pins']["+repr(ASSET+'/encoder.py')+"]",1),
      ("plan['source_pins']['checkpoint_encoder.py']","plan['source_pins']["+repr(ADAPTER)+"]",1)]
    edits.extend([
      ('    frozen_hash = model_hash()\n','    frozen_hash = model_hash()\n    check_prior_alignment(out, plan, selected_device, selection_path, frozen_hash)\n',1),
      ('    arrays_binding = dict(path=str(arrays_path), sha256=sha(arrays_path))\n',
       "    arrays_binding = dict(path=str(arrays_path), sha256=sha(arrays_path))\n    assert arrays_binding['sha256'] == plan['source_arrays_view_sha256']\n",1),
      ("        activate(record['selected_device'], sha(selection_path))\n",
       "        activate(record['selected_device'], sha(selection_path))\n        record_alignment(encoder, model, images, selected_device, gpu_latent if selected_device == 'cuda' else cpu_latent, plan, selection_path, frozen_hash, save, out)\n",1)])
    edits.append(("producer_hash, renderer_hash = sha(__file__), sha(HERE / 'render_adapter.py')",
                  "producer_hash, renderer_hash = sha(__file__), sha(Path(__file__).with_name('reference')/'render_adapter.py')",1))
    text=raw.decode()
    for before,after,count in edits:
        assert text.count(before)==count,(before,text.count(before));text=text.replace(before,after)
    return text,edits


def check_prior_alignment(out,plan,device,selection_path,frozen_hash):
    if not selection_path.exists():return
    proof=json.loads((out/'batch1_alignment.json').read_text())
    assert proof['actual_exit_code']==0 and proof['passed'] is True
    assert proof['checkpoint_sha256']==plan['checkpoint_sha256'] and proof['frozen_encoder_state_sha256']==frozen_hash
    assert proof['preprocess_device']==device and proof['preprocessing_selection_sha256']==hashlib.sha256(selection_path.read_bytes()).hexdigest()
    assert proof['atol']==plan['preprocess_gate']['latent_atol'] and proof['rtol']==plan['preprocess_gate']['latent_rtol']


def record_alignment(encoder,model,images,device,reference,plan,selection_path,frozen_hash,save,out):
    import torch
    assert torch.get_float32_matmul_precision()=='highest' and torch.backends.cuda.matmul.allow_tf32 is False
    tensor=torch.from_numpy(images[:1]).to(device)
    prepared=encoder.preprocess({'pixels':tensor},'2d')[model.obs_key].to('cuda')
    single=encoder.encode(model,prepared)
    finite=bool(torch.isfinite(single).all() and torch.isfinite(reference).all())
    gate=plan['preprocess_gate'];passed=finite and bool(torch.allclose(single,reference[:1],atol=gate['latent_atol'],rtol=gate['latent_rtol']))
    proof=dict(actual_exit_code=0 if passed else 1,passed=passed,finite=finite,
        checkpoint_sha256=plan['checkpoint_sha256'],frozen_encoder_state_sha256=frozen_hash,
        preprocess_device=device,preprocessing_selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        batch32_shape=list(reference.shape),batch1_shape=list(single.shape),
        same_real_frame_RGB_sha256=hashlib.sha256(memoryview(images[0])).hexdigest(),
        atol=gate['latent_atol'],rtol=gate['latent_rtol'],max_abs_difference=float((single-reference[:1]).abs().max()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),matmul_allow_tf32=False,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,extra_B1_forwards=1,extra_B32_forwards=0)
    save(out/'batch1_alignment.json',proof)
    assert passed,'PLDM B32/B1 mismatch; preserve proof and stop'


def load_original():
    text,edits=effective_source(ORIGINAL.read_bytes())
    sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
    module=types.ModuleType('_PLDM_original_RGB_producer');module.__file__=str(Path(__file__).resolve())
    exec(compile(text,str(ORIGINAL),'exec'),module.__dict__)
    module.HERE=ORIGINAL.parent
    module.CONTRACT_KEYS=module.CONTRACT_KEYS+NEW_KEYS
    module.check_prior_alignment=check_prior_alignment;module.record_alignment=record_alignment
    execute=module.execute
    def checked_execute(plan,pilot):
        assert plan['encoder_family']=='pldm' and plan['latent_dim']==512
        return execute(plan,pilot)
    module.execute=checked_execute
    return module


if __name__=='__main__':load_original().main()

"""Seeded CPU initialization for interface diagnostics; never pretrained results."""
import datetime,hashlib,importlib,json,pathlib,sys,traceback
ROOT=pathlib.Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE=ROOT/'foundation_visual_gciql_v1'
ENC=CASE/'encoder'
OUT=CASE/'diagnostics/random_v1/assets'
OFFICIAL_ASSET=ROOT/'assets/baselines/foundation/vjepa21_vitl1024_official_v1'
COMMIT='204698b45b3712590f06245fbfba32d3be539812'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 checkpoint=OUT/'random_seed0_vjepa21_vitl1024_ema_encoder.pt'
 spec_path=ENC/'specs/random_diagnostic_vjepa21_vitl1024.json'
 assert not checkpoint.exists() and not spec_path.exists(),'Refuse to overwrite an existing diagnostic asset/spec'
 receipt=json.loads((OFFICIAL_ASSET/'source_receipt.json').read_text());source=pathlib.Path(receipt['source_root']);assert receipt['commit']==COMMIT
 for name,record in receipt['files'].items():assert sha(source/name)==record['sha256'],name
 status={'status':'CPU_random_construction','started_at':now(),'exit_code':None,'diagnostic_only':True,'official_pretrained':False,'seed':0,'source_commit':COMMIT}
 atomic(OUT/'status.json',status)
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
 assert not torch.cuda.is_initialized()
 sys.path.insert(0,str(source));module=importlib.import_module('app.vjepa_2_1.models.vision_transformer')
 assert pathlib.Path(module.__file__).resolve()==(source/'app/vjepa_2_1/models/vision_transformer.py').resolve()
 kwargs={'patch_size':16,'img_size':[384,384],'num_frames':64,'tubelet_size':2,'use_sdpa':True,'use_rope':True,'img_temporal_dim_size':1,'interpolate_rope':True}
 model=module.vit_large(**kwargs).requires_grad_(False).eval()
 assert all(p.device.type=='cpu' and p.dtype==torch.float32 and not p.requires_grad for p in model.parameters())
 state=model.state_dict();assert list(state['patch_embed_img.proj.weight'].shape)==[1024,3,1,16,16]
 assert list(state['norms_block.3.weight'].shape)==[1024]
 status.update(status='serializing_CPU_random_weights',updated_at=now(),parameter_count=sum(p.numel() for p in model.parameters()));atomic(OUT/'status.json',status)
 with checkpoint.open('xb') as f:torch.save({'ema_encoder':state},f)
 digest=sha(checkpoint)
 saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
 assert set(saved)=={'ema_encoder'} and set(saved['ema_encoder'])==set(state)
 for name,value in state.items():assert torch.equal(saved['ema_encoder'][name],value),name
 assert not torch.cuda.is_initialized()
 spec={'family':'vjepa21_vitl1024','checkpoint':str(checkpoint),'checkpoint_sha256':digest,'source_root':str(source),'source_commit':COMMIT,'source_files_sha256':{p:r['sha256'] for p,r in receipt['files'].items()},'state_dict_key':'ema_encoder','encoder_module':'app.vjepa_2_1.models.vision_transformer','encoder_constructor':'vit_large','constructor_kwargs':kwargs,'native_size':384,'output_dim':1024,'feature':'last_normalized_image_patch_mean','preprocess':'uint8_BCHW224_fullFOV_bicubic_antialias_resize384_ImageNet_normalize_once','precision':'FP32','diagnostic_only':True,'initialization':'random','official_pretrained':False,'random_seed':0,'result_scope':'interface_smoke_only_not_pretrained_performance'}
 atomic(spec_path,spec)
 report={'status':'CPU_random_checkpoint_serialization_identity_verified_NN_gate_pending','exit_code':0,'at':now(),'checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,'checkpoint_sha256':digest,'spec':str(spec_path),'spec_sha256':sha(spec_path),'source_commit':COMMIT,'source_receipt_sha256':sha(OFFICIAL_ASSET/'source_receipt.json'),'encoder_adapter_sha256':sha(ENC/'encoder_adapter.py'),'diagnostic_only':True,'initialization':'random','official_pretrained':False,'seed':0,'model_instances':1,'forward_calls':0,'CUDA_initialized':False,'parameters_frozen':True,'training':False,'tensor_count':len(state),'parameter_count':status['parameter_count'],'strict_adapter_reload':'Root_owned_pending','script_sha256':sha(pathlib.Path(__file__))}
 atomic(OUT/'MANIFEST.json',report);status.update(status='CPU_random_asset_ready_NN_gate_pending',updated_at=now(),exit_code=0,manifest=report);atomic(OUT/'status.json',status);(OUT/'make_random_diagnostic.actual_exit_code').write_text('0\n')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  atomic(OUT/'failure.json',{'at':now(),'exit_code':1,'error':repr(e),'traceback':traceback.format_exc()});(OUT/'make_random_diagnostic.actual_exit_code').write_text('1\n');raise

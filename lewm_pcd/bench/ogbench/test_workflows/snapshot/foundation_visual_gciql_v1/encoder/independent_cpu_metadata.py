"""Independent asset identity and CPU tensor metadata; no model/GPU operations."""
import datetime,hashlib,json,pathlib,re,time,traceback,zipfile
R=pathlib.Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/encoder')
A=pathlib.Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/assets/baselines/foundation/vjepa21_vitl1024_official_v1')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
try:
 while not (R/'finalize.actual_exit_code').exists():time.sleep(5)
 assert (R/'finalize.actual_exit_code').read_text().strip()=='0'
 assert (R/'segments.actual_exit_code').read_text().strip()=='0'
 assert not (R/'assets_failure.json').exists() and not (R/'finalize_failure.json').exists()
 status=json.loads((R/'assets_status.json').read_text());manifest=json.loads((A/'MANIFEST.json').read_text());spec_path=R/'specs/vjepa21_vitl1024.json';spec=json.loads(spec_path.read_text())
 assert status['exit_code']==manifest['exit_code']==0
 checkpoint=pathlib.Path(spec['checkpoint']);assert checkpoint==A/'vjepa2_1_vitl_dist_vitG_384.pt'
 assert checkpoint.stat().st_size==manifest['bytes']==5151198524
 digest=sha(checkpoint);assert digest==spec['checkpoint_sha256']==manifest['sha256']
 assert manifest['etag']=='"75aac8d267d11f285683714d390dc0ed-615"'
 assert spec['source_commit']=='204698b45b3712590f06245fbfba32d3be539812'
 assert spec['family']=='vjepa21_vitl1024' and spec['output_dim']==1024 and spec['native_size']==384
 assert spec['state_dict_key']=='ema_encoder' and spec['encoder_constructor']=='vit_large'
 for name,d in spec['source_files_sha256'].items():assert sha(pathlib.Path(spec['source_root'])/name)==d,name
 with zipfile.ZipFile(checkpoint) as z:assert z.testzip() is None;members=len(z.infolist())
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
 assert isinstance(payload,dict) and isinstance(payload['ema_encoder'],dict)
 state={k.replace('module.','').replace('backbone.',''):v for k,v in payload['ema_encoder'].items()}
 shapes={k:list(v.shape) for k,v in state.items()};assert all(isinstance(v,torch.Tensor) and v.dtype==torch.float32 for v in state.values())
 blocks=sorted({int(m.group(1)) for k in state for m in [re.match(r'blocks\.(\d+)\.',k)] if m});assert blocks==list(range(24))
 assert shapes['patch_embed_img.proj.weight']==[1024,3,1,16,16]
 assert shapes['patch_embed.proj.weight']==[1024,3,2,16,16]
 assert shapes['norms_block.3.weight']==[1024]
 for i in blocks:assert shapes[f'blocks.{i}.attn.qkv.weight']==[3072,1024]
 assert not torch.cuda.is_initialized()
 record={'status':'PASSED_CPU_ASSET_IDENTITY_AND_METADATA_MODEL_GATE_PENDING','exit_code':0,'at':now(),'checkpoint':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':digest,'spec_path':str(spec_path),'spec_sha256':sha(spec_path),'source_commit':spec['source_commit'],'source_files_verified':len(spec['source_files_sha256']),'zip_crc_members_verified':members,'checkpoint_keys':list(payload),'EMA_tensor_count':len(state),'EMA_shapes':shapes,'block_indices':blocks,'output_dim_confirmed':1024,'image_tokenizer_temporal_kernel_confirmed':1,'CUDA_initialized':False,'model_instances':0,'forward_calls':0,'strict_model_load':'Root_owned_pending','encoder_adapter_sha256':sha(R/'encoder_adapter.py')}
 atomic(R/'independent_cpu_metadata.json',record);(R/'independent_cpu_metadata.actual_exit_code').write_text('0\n')
except BaseException as e:
 atomic(R/'independent_cpu_metadata_failure.json',{'at':now(),'exit_code':1,'error':repr(e),'traceback':traceback.format_exc()});(R/'independent_cpu_metadata.actual_exit_code').write_text('1\n');raise

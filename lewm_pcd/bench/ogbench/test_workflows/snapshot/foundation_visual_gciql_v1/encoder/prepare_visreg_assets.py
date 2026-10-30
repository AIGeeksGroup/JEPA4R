"""CPU identity and strict bare-backbone load for author-supplied VISReg ViT-B."""
import datetime,hashlib,importlib.metadata,json,pathlib,re,traceback,zipfile
ROOT=pathlib.Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
E=ROOT/'foundation_visual_gciql_v1/encoder'
A=ROOT/'assets/baselines/foundation/visreg_vitb768_official_v1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
try:
 checkpoint=A/'visreg-vit-b-inet1k.pth';download=json.loads((A/'DOWNLOAD_MANIFEST.json').read_text())
 assert download['exit_code']==0 and checkpoint.stat().st_size==download['bytes']==368495235
 digest=sha(checkpoint);assert digest==download['sha256']
 with zipfile.ZipFile(checkpoint) as z:assert z.testzip() is None;members=len(z.infolist())
 import torch,timm
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
 assert isinstance(payload,dict) and all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in payload.items())
 state={k:v for k,v in payload.items() if not k.startswith('proj.')};proj={k:list(v.shape) for k,v in payload.items() if k.startswith('proj.')}
 shapes={k:list(v.shape) for k,v in state.items()}
 assert shapes['cls_token']==[1,1,768] and shapes['pos_embed']==[1,197,768] and shapes['patch_embed.proj.weight']==[768,3,16,16] and shapes['norm.weight']==[768]
 blocks=sorted({int(m.group(1)) for k in state for m in [re.match(r'blocks\.(\d+)\.',k)] if m});assert blocks==list(range(12))
 for i in blocks:assert shapes[f'blocks.{i}.attn.qkv.weight']==[2304,768]
 assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
 model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0,dynamic_img_size=True).requires_grad_(False).eval()
 result=model.load_state_dict(state,strict=True);assert not result.missing_keys and not result.unexpected_keys
 assert model.num_features==768 and len(model.blocks)==12 and not torch.cuda.is_initialized()
 spec={'family':'visreg_vitb768','checkpoint':str(checkpoint),'checkpoint_sha256':digest,'architecture':'vit_base_patch16_224','timm_version':importlib.metadata.version('timm'),'native_size':224,'output_dim':768,'feature':'last_normalized_CLS','preprocess':'uint8_BCHW224_fullFOV_bicubic_antialias_resize224_ImageNet_normalize_once','precision':'FP32','diagnostic_only':False,'initialization':'official_pretrained','official_pretrained':True,'checkpoint_format':'flat_state_dict_backbone_plus_proj','projector_handling':'drop_only_proj_prefix_then_strict_backbone_load','source_folder':download['source_folder'],'source_file_id':download['source_file_id'],'source_url':download['source_url']}
 spec_path=E/'specs/visreg_vitb768.json';assert not spec_path.exists();atomic(spec_path,spec)
 report={'status':'PASSED_CPU_STRICT_VISREG_BACKBONE_LOAD_NN_FORWARD_PENDING','exit_code':0,'at':now(),'checkpoint':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':digest,'zip_crc_members_verified':members,'spec_path':str(spec_path),'spec_sha256':sha(spec_path),'timm_version':spec['timm_version'],'backbone_tensor_count':len(state),'projector_tensor_shapes':proj,'backbone_shapes':shapes,'blocks':blocks,'output_dim':768,'model_instances':1,'forward_calls':0,'CUDA_initialized':False,'strict_load_missing_keys':[],'strict_load_unexpected_keys':[],'encoder_adapter_sha256':sha(E/'encoder_adapter.py'),'source_evidence':download['source_evidence'],'license_from_historical_model_card':'CC-BY-NC-4.0','not_published_official_SHA':True}
 atomic(A/'MANIFEST.json',report);atomic(E/'visreg_cpu_metadata.json',report);(E/'visreg_cpu_metadata.actual_exit_code').write_text('0\n')
except BaseException as e:
 atomic(E/'visreg_cpu_metadata_failure.json',{'at':now(),'exit_code':1,'error':repr(e),'traceback':traceback.format_exc()});(E/'visreg_cpu_metadata.actual_exit_code').write_text('1\n');raise

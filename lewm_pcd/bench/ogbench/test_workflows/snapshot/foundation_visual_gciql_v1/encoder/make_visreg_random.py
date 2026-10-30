"""Seed-zero CPU bare ViT-B state for isolated VISReg interface diagnostics."""
import datetime,hashlib,json,pathlib,traceback
ROOT=pathlib.Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
E=ROOT/'foundation_visual_gciql_v1/encoder'
A=ROOT/'foundation_visual_gciql_v1/diagnostics/random_visreg_v1/assets'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
try:
 A.mkdir(parents=True,exist_ok=True);checkpoint=A/'random_seed0_visreg_vitb768_backbone.pt';spec_path=E/'specs/random_diagnostic_visreg_vitb768.json'
 assert not checkpoint.exists() and not spec_path.exists(),'Refuse overwrite'
 assert (E/'visreg_cpu_metadata.actual_exit_code').read_text().strip()=='0'
 official_spec_path=E/'specs/visreg_vitb768.json';spec=json.loads(official_spec_path.read_text())
 status={'status':'CPU_random_construction','started_at':now(),'exit_code':None,'diagnostic_only':True,'official_pretrained':False,'seed':0};atomic(A/'status.json',status)
 import torch,timm
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
 model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0,dynamic_img_size=True).requires_grad_(False).eval()
 assert model.num_features==768 and len(model.blocks)==12
 assert all(p.device.type=='cpu' and p.dtype==torch.float32 and not p.requires_grad for p in model.parameters())
 state=model.state_dict()
 with checkpoint.open('xb') as f:torch.save(state,f)
 saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
 assert set(saved)==set(state)
 for name,value in state.items():assert torch.equal(saved[name],value),name
 assert not torch.cuda.is_initialized()
 digest=sha(checkpoint)
 for key in ('source_folder','source_file_id','source_url'):spec.pop(key,None)
 spec.update(checkpoint=str(checkpoint),checkpoint_sha256=digest,diagnostic_only=True,initialization='random',official_pretrained=False,random_seed=0,result_scope='interface_smoke_only_not_pretrained_performance',checkpoint_format='flat_state_dict_backbone_only')
 atomic(spec_path,spec)
 report={'status':'CPU_RANDOM_VISREG_ASSET_IDENTITY_VERIFIED_NN_FORWARD_PENDING','exit_code':0,'at':now(),'checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,'checkpoint_sha256':digest,'spec':str(spec_path),'spec_sha256':sha(spec_path),'official_architecture_spec_sha256':sha(official_spec_path),'encoder_adapter_sha256':sha(E/'encoder_adapter.py'),'diagnostic_only':True,'initialization':'random','official_pretrained':False,'seed':0,'model_instances':1,'forward_calls':0,'CUDA_initialized':False,'parameters_frozen':True,'training':False,'parameter_count':sum(p.numel() for p in model.parameters()),'tensor_count':len(state),'timm_version':spec['timm_version'],'script_sha256':sha(pathlib.Path(__file__))}
 atomic(A/'MANIFEST.json',report);status.update(status='CPU_random_asset_ready_NN_gate_pending',exit_code=0,updated_at=now(),manifest=report);atomic(A/'status.json',status);(A/'make_visreg_random.actual_exit_code').write_text('0\n')
except BaseException as e:
 atomic(A/'failure.json',{'at':now(),'exit_code':1,'error':repr(e),'traceback':traceback.format_exc()});(A/'make_visreg_random.actual_exit_code').write_text('1\n');raise

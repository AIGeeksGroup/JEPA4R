"""Official asset transport only; does not import torch or run a model."""
import concurrent.futures, datetime, hashlib, json, os, pathlib, subprocess, traceback, urllib.request, zipfile

ROOT = pathlib.Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE = ROOT/'foundation_visual_gciql_v1/encoder'
ASSET = ROOT/'assets/baselines/foundation/vjepa21_vitl1024_official_v1'
COMMIT = '204698b45b3712590f06245fbfba32d3be539812'
URL = 'https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'
N = 5151198524
ETAG = '"75aac8d267d11f285683714d390dc0ed-615"'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def fetch(url):
    with urllib.request.urlopen(url,timeout=45) as r:return r.read()

def main():
    CASE.mkdir(parents=True,exist_ok=True);ASSET.mkdir(parents=True,exist_ok=True)
    status={'stage':'source','started_at':now(),'exit_code':None}
    write(CASE/'assets_status.json',status)
    source=ASSET/'source'; source.mkdir(exist_ok=True)
    files=['hubconf.py','src/hub/backbones.py','app/vjepa_2_1/models/vision_transformer.py',
           'app/vjepa_2_1/models/utils/modules.py','app/vjepa_2_1/models/utils/patch_embed.py',
           'app/vjepa_2_1/models/utils/pos_embs.py','app/vjepa_2_1/models/utils/masks_dist.py',
           'src/models/vision_transformer.py','src/models/predictor.py','src/models/ac_predictor.py',
           'src/models/utils/modules.py','src/models/utils/patch_embed.py','src/models/utils/pos_embs.py',
           'src/utils/tensors.py','src/utils/distributed.py','src/utils/logging.py','src/masks/utils.py',
           'README.md','LICENSE','APACHE-LICENSE','requirements.txt']
    records={}
    def get_source(p):
        url=f'https://raw.githubusercontent.com/facebookresearch/vjepa2/{COMMIT}/{p}'
        target=source/p;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(fetch(url))
        return p,{'sha256':sha(target),'url':url,'bytes':target.stat().st_size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for p,record in pool.map(get_source,files):records[p]=record
    write(ASSET/'source_receipt.json',{'commit':COMMIT,'source_root':str(source),'files':records})
    status.update(stage='checkpoint_download',updated_at=now());write(CASE/'assets_status.json',status)
    with urllib.request.urlopen(urllib.request.Request(URL,method='HEAD'),timeout=45) as r:
        assert int(r.headers['Content-Length'])==N and r.headers['ETag']==ETAG
        write(ASSET/'official_HEAD.json',{'url':URL,'status':r.status,'headers':dict(r.headers),'at':now()})
    final=ASSET/'vjepa2_1_vitl_dist_vitG_384.pt'; part=final.with_suffix('.pt.part')
    if not final.exists():
        command=['curl','--fail','--location','--retry','5','--retry-delay','5','--connect-timeout','30','--speed-time','120','--speed-limit','1024','--continue-at','-','--output',str(part),URL]
        write(CASE/'download_argv.json',command)
        with (CASE/'download.log').open('a') as log:rc=subprocess.call(command,stdout=log,stderr=log)
        (CASE/'download.actual_exit_code').write_text(str(rc)+'\n')
        if rc:raise RuntimeError('checkpoint curl failed '+str(rc))
        assert part.stat().st_size==N
        part.rename(final)
    assert final.stat().st_size==N
    status.update(stage='SHA_ZIP_CRC',updated_at=now());write(CASE/'assets_status.json',status)
    digest=sha(final)
    with zipfile.ZipFile(final) as z:assert z.testzip() is None
    receipt={'url':URL,'bytes':N,'sha256':digest,'etag':ETAG,'status':'download_size_sha_zipcrc_verified_model_load_pending','exit_code':0,'at':now(),'checkpoint':str(final),'source_commit':COMMIT,'not_published_official_sha':True}
    write(ASSET/'MANIFEST.json',receipt)
    spec={'family':'vjepa21_vitl1024','checkpoint':str(final),'checkpoint_sha256':digest,'source_root':str(source),'source_commit':COMMIT,'source_files_sha256':{p:r['sha256'] for p,r in records.items()},'state_dict_key':'ema_encoder','encoder_module':'app.vjepa_2_1.models.vision_transformer','encoder_constructor':'vit_large','constructor_kwargs':{'patch_size':16,'img_size':[384,384],'num_frames':64,'tubelet_size':2,'use_sdpa':True,'use_rope':True,'img_temporal_dim_size':1,'interpolate_rope':True},'native_size':384,'output_dim':1024,'feature':'last_normalized_image_patch_mean','preprocess':'uint8_BCHW224_fullFOV_bicubic_antialias_resize384_ImageNet_normalize_once','precision':'FP32'}
    write(CASE/'specs/vjepa21_vitl1024.json',spec)
    status.update(stage='assets_ready_model_load_and_forward_pending',updated_at=now(),exit_code=0,receipt=receipt);write(CASE/'assets_status.json',status)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(CASE/'assets_failure.json',{'at':now(),'exit_code':1,'error':repr(e),'traceback':traceback.format_exc()});raise

"""Counted small family/N switch over the approved bounded two-worker producer."""
import argparse
import json
from pathlib import Path
import sys
import types
from common import sha, stat, replace, COUNTS

BASE=Path(__file__).parent/'base/produce.py'
BASE_SHA='dbbce2407f11c42ee2c5aca7601c9a055a7ceb08593fb4e99b1b0cd7e5415b18'
EXTRA=('cube_count','latent_dim','encoder_spec_sha256','encoder_family','encoder_adapter_sha256',
       'encoder_preprocessing_device','host_tensor_dtype')

def effective_source(plan):
    assert sha(BASE)==BASE_SHA
    n,d=plan['cube_count'],plan['latent_dim']
    assert n in COUNTS and d in (768,1024)
    t=BASE.read_text()
    changes=[('latent_dim=192',f'latent_dim={d}'),
      ("assert policy in ('cpu', 'cuda', 'auto_validated')","assert policy == 'shared_gpu'"),
      ("if policy == 'cpu':\n        activate('cpu', None)","if policy == 'shared_gpu':\n        activate('cuda', None)"),
      ("assert plan['split_episodes'] == {'train': 3000, 'val': 300}",f"assert plan['split_episodes'] == {COUNTS[n]!r}"),
      ('qpos_width=35, qvel_width=32',f'qpos_width={14+7*n}, qvel_width={14+6*n}'),
      ('extractor_source_sha256=producer_hash',"extractor_source_sha256=plan['source_arrays_extractor_sha256']"),
      ("workspace / 'checkpoint_encoder.py'","Path(plan['encoder_adapter'])"),
      ("encoder.load_frozen(plan['checkpoint'], device='cuda')","encoder.load_encoder(plan['encoder_spec'], device='cuda')"),
      ('result.shape == (len(images), 192)',f'result.shape == (len(images), {d})'),
      ('shape=[1001, 192]',f'shape=[1001, {d}]'),
      ("plan['source_pins']['source/lewm_pcd/utils.py']","plan['encoder_adapter_sha256']"),
      ("plan['source_pins']['checkpoint_encoder.py']","plan['encoder_adapter_sha256']"),
      ('shape[1] == 46',f'shape[1] == {19+9*n}'),
      ('int(raw_id) * 46 * 4',f'int(raw_id) * {19+9*n} * 4'),
      ('stream.read(46 * 4)',f'stream.read({19+9*n} * 4)'),
      ('range(3)',f'range({n})'),
      ('completed == 3300 and total_rows == 3303300',f'completed == {sum(COUNTS[n].values())} and total_rows == {sum(COUNTS[n].values())*1001}'),
      ('train_rows=3003000, val_rows=300300',f'train_rows={COUNTS[n]["train"]*1001}, val_rows={COUNTS[n]["val"]*1001}'),
      ('assert Path(plan[\'workspace\']).resolve() == HERE.parent',"assert Path(plan['workspace']).resolve() == Path(plan['root_workspace'])"),
      ('    frozen_hash = model_hash()\n',"    frozen_hash = model_hash()\n    assert sha(plan['encoder_spec']) == plan['encoder_spec_sha256']\n    assert sha(plan['encoder_adapter']) == plan['encoder_adapter_sha256']\n")]
    for old,new in changes:t=replace(t,old,new)
    begin=t.index('    def prepared_and_encoded(images, device):')
    end=t.index('\n    def infer(images):',begin)
    t=t[:begin]+'''    alignment_checked = False
    def prepared_and_encoded(images, device):
        nonlocal alignment_checked
        # Renderer produces original BHWC uint8; the shared encoder API owns preprocessing.
        tensor = torch.from_numpy(images).permute(0,3,1,2).contiguous()
        assert tensor.dtype == torch.uint8 and tuple(tensor.shape) == (32,3,224,224)
        latent = model.encode_uint8(tensor)
        assert latent.dtype == torch.float32 and tuple(latent.shape) == (32,plan['latent_dim'])
        if not alignment_checked:
            alignment_path=out/'batch1_alignment.json'
            if alignment_path.exists():
                proof=json.loads(alignment_path.read_text())
                assert proof['passed'] is True and proof['actual_exit_code']==0
                assert proof['encoder_spec_sha256']==plan['encoder_spec_sha256']
                assert proof['frozen_encoder_state_sha256']==frozen_hash
            else:
                single=model.encode_uint8(tensor[:1])
                finite=bool(torch.isfinite(single).all() and torch.isfinite(latent).all())
                passed=finite and bool(torch.allclose(single,latent[:1],atol=1e-5,rtol=1e-5))
                save(alignment_path,dict(passed=passed,actual_exit_code=0 if passed else 1,
                    encoder_spec_sha256=plan['encoder_spec_sha256'],frozen_encoder_state_sha256=frozen_hash,
                    finite=finite,atol=1e-5,rtol=1e-5,max_abs_difference=float((single-latent[:1]).abs().max()),
                    first_real_uint8_RGB_sha256=hashlib.sha256(memoryview(images[0])).hexdigest(),
                    batch1_shape=list(single.shape),batch32_shape=list(latent.shape),extra_B1_forwards=1))
                assert passed,'Real B1/B32 alignment failed; preserve proof and stop'
            alignment_checked=True
        return tensor, latent
'''+t[end:]
    t=replace(t,"        if selected_device is None:\n            choose_preprocessing(np.ascontiguousarray(images[:32]))",
              "        assert selected_device == 'cuda'  # shared encoder API; no CPU/GPU speed selection")
    # Original pilot is exactly one train + one validation episode (2 EP total).
    return t

def arrays_existing(plan,protocol,sha_fn,stat_fn,save,contract):
    path=Path(plan['shared_state_root'])/'source_arrays.json'
    assert sha_fn(path)==plan['source_arrays_receipt_sha256']
    record=json.loads(path.read_text())
    assert record['protocol_sha256']==protocol and record['contract']==contract
    for item in record['splits'].values():
        assert stat(item['source_npz'])==item['accepted_stat']
        for key in ('qpos','qvel'):
            current=stat(item[key]['path'])
            assert current==item[key]['stat']
    return path,record

def load(plan):
    text=effective_source(plan)
    m=types.ModuleType('_foundation_original_producer');m.__file__=str(Path(__file__).resolve())
    exec(compile(text,str(BASE),'exec'),m.__dict__)
    m.HERE=Path(__file__).parent.resolve()
    m.CONTRACT_KEYS=m.CONTRACT_KEYS+EXTRA
    m.source_arrays=arrays_existing
    return m

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--pilot',action='store_true')
    a=p.parse_args();plan=json.loads(a.plan.read_text());module=load(plan)
    module.main()

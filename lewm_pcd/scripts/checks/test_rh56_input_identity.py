"""Before-update BC source and independent RH split producer contracts."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import copy,hashlib,importlib.util,json,sys,tempfile,unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from rh56_adapter import bc_bridge,wm_bridge
from rh56_adapter.input_identity import require_bc_source

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2));return path

class InputIdentityTests(unittest.TestCase):
 def fixture(self,root):
  batch=root/'batches/batch_000';data=root/'data';batch.mkdir(parents=True);data.mkdir();raw=batch/'paired_pcd.h5';raw.write_bytes(b'explicit fake raw HDF contract fixture');header=data/'paired_pcd.h5';header.write_bytes(b'explicit fake VDS header fixture')
  rows=[dict(seed=61000000+i,split='train',episode_index=i,initial_state_sha256=chr(97+i)*64,rows=6,steps=5,requested_coverage='protocol') for i in range(2)]+[dict(seed=62000000,split='val',episode_index=2,initial_state_sha256='c'*64,rows=6,steps=5,requested_coverage='protocol')]
  config=dict(episodes=[dict(seed=r['seed'],split=r['split'],coverage='protocol') for r in rows]);config_sha=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest();provenance=dict(config_sha256=config_sha,plan_sha256='1'*64,producer_sha256='2'*64)
  for row in rows:row.update(config_sha256=config_sha,producer_sha256='2'*64,npz='episode'+str(row['episode_index'])+'.npz',npz_sha256='3'*64)
  write(batch/'config.json',dict(config=config,provenance=provenance));write(batch/'manifest.json',dict(episodes=rows,auxiliary=[],provenance=provenance));write(batch/'state.json',dict(phase='complete',exit_code=0))
  report=dict(status='PASS_BATCH',deep_data_check=True,records=copy.deepcopy(rows),plan_sha256='1'*64,files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in [raw,batch/'manifest.json',batch/'config.json',batch/'state.json']]);write(batch/'batch_acceptance.json',report)
  manifest=dict(action_alignment='outgoing; last row NaN',source_acceptance_checked=True,source_receipts=[dict(batch=str(batch),acceptance_sha256=sha(batch/'batch_acceptance.json'))],datasets=[dict(route='pcd',path=str(header),sha256=sha(header),rows=18,sources=[dict(path='../batches/batch_000/paired_pcd.h5',rows=18,offset=0)])],episodes=[dict(row,source_dir=str(batch),source_episode_index=row['episode_index']) for row in rows])
  seeds=np.repeat([r['seed'] for r in rows],6);dataset=SimpleNamespace(h5_path=str(header),lengths=[6]*3,offsets=[0,6,12],clip_indices=[(e,t) for e in range(3) for t in (0,1)],get_col_data=lambda key:seeds)
  return dataset,manifest,batch

 def test_bc_default_rejects_same_path_changed_source_before_torch_or_output(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);eai=root/'eai';loop=eai/'cortexbench/mujoco_vc/visual_imitation/train_loop.py';loop.parent.mkdir(parents=True);loop.write_text('def bc_pvr_train_loop(config):\n return "L1 instead of frozen MSE"\n');out=root/'out';args=SimpleNamespace(eai_root=str(eai),out=str(out),baseline=str(ROOT),demo_cache_root=None,bc_recovery=False,mode='profile',epoch=1,train_seed=100,eval_seed_base=65100000,route='pcd')
   original=__import__('builtins').__import__
   def imported(name,*a,**kw):
    if name=='torch':raise AssertionError('Torch reached before default BC dependency SHA guard')
    return original(name,*a,**kw)
   with mock.patch('builtins.__import__',side_effect=imported):
    with self.assertRaisesRegex(ValueError,'BC.*SHA|EAI.*SHA'):bc_bridge.run_bc(args)
   self.assertFalse(out.exists())

 def test_accepted_independent_producer_and_dataset_contract(self):
  with tempfile.TemporaryDirectory() as temp:
   dataset,manifest,_=self.fixture(Path(temp));partition=wm_bridge.EpisodePartition(dataset,manifest);self.assertEqual(partition.seeds['train'],[61000000,61000001]);self.assertEqual(partition.seeds['val'],[62000000])

 def test_swapped_labels_keep_count_rows_but_reject_before_norm(self):
  with tempfile.TemporaryDirectory() as temp:
   dataset,manifest,_=self.fixture(Path(temp));manifest['episodes'][1]['split']='val';manifest['episodes'][2]['split']='train'
   with self.assertRaisesRegex(ValueError,'producer|partition|namespace'):wm_bridge.EpisodePartition(dataset,manifest)
   normalizer=mock.Mock();train=SimpleNamespace(get_column_normalizer=normalizer,spt=SimpleNamespace(data=SimpleNamespace(random_split=mock.Mock())))
   with wm_bridge.install_episode_split(train,dataset.h5_path,manifest,subset_factory=lambda data,ids:ids):
    with self.assertRaisesRegex(ValueError,'producer|partition|namespace'):train.get_column_normalizer(dataset,'action','action')
   normalizer.assert_not_called()

 def test_self_rehashed_producer_record_or_header_does_not_override_contract(self):
  for change in ['producer_state','header']:
   with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
    dataset,manifest,batch=self.fixture(Path(temp))
    if change=='header':Path(dataset.h5_path).write_bytes(b'other same-shape header/data')
    else:
     report=json.loads((batch/'batch_acceptance.json').read_text());report['records'][1]['initial_state_sha256']='f'*64;write(batch/'batch_acceptance.json',report);manifest['source_receipts'][0]['acceptance_sha256']=sha(batch/'batch_acceptance.json')
    with self.assertRaisesRegex(ValueError,'producer|dataset|HDF|header'):wm_bridge.EpisodePartition(dataset,manifest)

 def test_missing_independent_producer_contract_rejects(self):
  with tempfile.TemporaryDirectory() as temp:
   dataset,manifest,_=self.fixture(Path(temp));manifest.pop('source_receipts')
   with self.assertRaisesRegex(ValueError,'producer|publication'):wm_bridge.EpisodePartition(dataset,manifest)

 def test_BC_file_contract_accepts_only_actual_DEP_pin(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);loop=root/'eai/cortexbench/mujoco_vc/visual_imitation/train_loop.py';loop.parent.mkdir(parents=True);loop.write_bytes(b'MSE');deps=write(root/'DEPENDENCIES.json',{'eai_vc':{'train_loop_sha256':sha(loop)}});self.assertEqual(require_bc_source(root/'eai',deps)['sha256'],sha(loop));loop.write_bytes(b'L1_')
   with self.assertRaisesRegex(ValueError,'SHA'):require_bc_source(root/'eai',deps)

 def test_direct_accepted_batch_remains_supported(self):
  with tempfile.TemporaryDirectory() as temp:
   dataset,_,batch=self.fixture(Path(temp));dataset.h5_path=str(batch/'paired_pcd.h5');manifest=json.loads((batch/'manifest.json').read_text());self.assertEqual(wm_bridge.EpisodePartition(dataset,manifest).seeds['val'],[62000000])

 def rewrite_producer_seeds(self,dataset,manifest,batch,seeds):
  producer=json.loads((batch/'manifest.json').read_text());config=json.loads((batch/'config.json').read_text());report=json.loads((batch/'batch_acceptance.json').read_text())
  for row,seed in zip(producer['episodes'],seeds):row['seed']=seed
  for row,seed in zip(config['config']['episodes'],seeds):row['seed']=seed
  digest=hashlib.sha256(json.dumps(config['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest();config['provenance']['config_sha256']=digest;producer['provenance']=copy.deepcopy(config['provenance'])
  for row in producer['episodes']:row['config_sha256']=digest
  write(batch/'manifest.json',producer);write(batch/'config.json',config);report['records']=copy.deepcopy(producer['episodes'])
  for row in report['files']:
   path=batch/Path(row['path']).name;row.update(bytes=path.stat().st_size,sha256=sha(path))
  write(batch/'batch_acceptance.json',report);manifest['source_receipts'][0]['acceptance_sha256']=sha(batch/'batch_acceptance.json');manifest['episodes']=[dict(r,source_episode_index=r['episode_index']) for r in producer['episodes']];values=np.repeat(seeds,6);dataset.get_col_data=lambda key:values

 def test_explicit_legacy_namespace_passes_but_canonical_mislabel_rejects(self):
  for seeds,accepted in [([60000000,60000001,60001000],True),([61000000,62000001,62000000],False)]:
   with self.subTest(seeds=seeds),tempfile.TemporaryDirectory() as temp:
    dataset,manifest,batch=self.fixture(Path(temp));self.rewrite_producer_seeds(dataset,manifest,batch,seeds)
    if accepted:self.assertEqual(wm_bridge.EpisodePartition(dataset,manifest).seeds['train'],seeds[:2])
    else:
     with self.assertRaisesRegex(ValueError,'namespace'):wm_bridge.EpisodePartition(dataset,manifest)

if __name__=='__main__':unittest.main()

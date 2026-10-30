import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
import shared_data as d

def test_double_budget_and_episode_boundary():
    assert d.contract('pldm')==dict(frames=16,actions=15,per_episode=485,
        train_samples=4850000,validation_samples=485000,train_batches=37890,
        validation_batches=3790,validation_tail=8,total_updates=378900)
    assert d.location(484,'pldm',2)==(0,484)
    assert d.location(485,'pldm',2)==(1,0)
    assert d.location(-1,'pldm',2)==(1,484)
    with pytest.raises(IndexError):d.location(970,'pldm',2)

def test_reference_preprocessing_and_one_action_normalization():
    import torch
    from torchvision.transforms import v2
    class Reader:
        def __getitem__(self,index):
            return dict(pixels=torch.arange(16,dtype=torch.uint8)[:,None,None,None].expand(16,3,4,4),
                action=torch.arange(80,dtype=torch.float32).reshape(16,5))
        def close(self):pass
    x=d.VisualDataset('unused','pldm',reader=Reader(),action_stats=dict(mean=[1]*5,std=[2]*5))[484]
    assert set(x)=={'pixels','actions','sample_id'}
    assert x['sample_id']==484 and x['actions'].shape==(15,5)
    assert torch.equal(x['actions'],(torch.arange(75).reshape(15,5)-1)/(2+1e-6))
    torch.manual_seed(7)
    pixels=torch.randint(0,256,(2,16,3,224,224),dtype=torch.uint8)
    expected=v2.Resize((65,65),antialias=True)(pixels.float()/255.0)
    out=d.PLDMPreprocess()(dict(pixels=pixels,actions=x['actions'].unsqueeze(0).expand(2,-1,-1).contiguous()))
    assert torch.equal(out['pixels'],expected)
    with pytest.raises(ValueError):d.PLDMPreprocess()(out)

def test_double_scheduler_budget():
    import torch
    from methods.pldm import method
    sys.path.insert(0,str(Path(__file__).parents[1]/'vendor/pldm/source'))
    opt=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=.0007)
    scheduler=method.build_scheduler(opt,epochs=10,batch_steps=d.contract('pldm')['train_batches'])
    assert scheduler.adjust_learning_rate(0)==0
    assert scheduler.adjust_learning_rate(37890)==pytest.approx(.00035)
    assert scheduler.adjust_learning_rate(378900)==pytest.approx(.00000035)

def test_native_spt_batch_transfer_preprocesses_pixels_once_before_forward():
    import torch
    import inspect
    from types import SimpleNamespace
    from train import native_module
    from shared_data import PLDMPreprocess
    assert 'gpu_preprocess' not in inspect.signature(native_module).parameters
    class Model(torch.nn.Module):
        def forward(self,pixels,actions):
            assert tuple(pixels.shape)==(2,16,3,65,65)
            assert pixels.dtype==torch.float32
            assert tuple(actions.shape)==(2,15,5)
            assert actions.dtype==torch.float32
            return {'prediction':pixels.mean()+actions.mean()}
    preprocess=PLDMPreprocess()
    module=native_module('pldm',SimpleNamespace(),Model())
    trainer=SimpleNamespace(training=False,validating=True,sanity_checking=True,
        testing=False,predicting=False,
        val_dataloaders=[SimpleNamespace(dataset=SimpleNamespace(gpu_transform=preprocess))],
        datamodule=None)
    module._trainer=trainer
    batch={'pixels':torch.zeros(2,16,3,224,224,dtype=torch.uint8),
           'actions':torch.ones(2,15,5)}
    batch=module.on_after_batch_transfer(batch,dataloader_idx=0)
    assert batch['pixels'].dtype==torch.float32
    assert tuple(batch['pixels'].shape)==(2,16,3,65,65)
    assert preprocess.last_contract is not None
    result=module.forward(batch,'train')
    assert torch.isfinite(result['prediction'])

def test_sealed_double_vds_first_last_boundary_and_validation_tail():
    import os
    import numpy as np
    import h5py
    from shared_data import original_reader,VisualDataset,action_statistics,contract,TRAIN_VIEW,VAL_VIEW,validate_manifest,require_sources
    root=Path(os.environ['DOUBLE_DATA_ROOT'])
    assert validate_manifest(root)['actual_exit_code']==0
    source_root=Path(__file__).parents[1]
    assert require_sources(source_root,root)['data_root']==str(root.resolve())
    assert contract('pldm')['train_samples']==4_850_000
    assert contract('pldm')['validation_samples']==485_000
    train=original_reader(root/'data_views'/TRAIN_VIEW,'pldm',root)
    val=original_reader(root/'data_views'/VAL_VIEW,'pldm',root)
    try:
        assert len(train)==4_850_000 and len(val)==485_000
        stats=action_statistics(train)
        assert stats['count']==5_000_000 and stats['source_split']=='train'
        train_view=VisualDataset(root/'data_views'/TRAIN_VIEW,'pldm',reader=train,action_stats=stats)
        with h5py.File(root/'data_views'/TRAIN_VIEW,'r',swmr=True) as raw:
            for index,episode,start in ((0,0,0),(484,0,484),(485,1,0),(4_849_999,9999,484)):
                sample=train[index]
                begin=episode*500+start
                assert tuple(sample['pixels'].shape)==(16,3,224,224)
                assert sample['pixels'].dtype==__import__('torch').uint8
                assert np.array_equal(sample['pixels'][0].permute(1,2,0).numpy(),raw['pixels'][begin])
                assert np.array_equal(sample['action'][:15].numpy(),raw['action'][begin:begin+15])
        normalized=train_view[4_849_999]['actions']
        expected=(train[4_849_999]['action'][:15]-__import__('torch').tensor(stats['mean']))/(
            __import__('torch').tensor(stats['std'])+1e-6)
        assert __import__('torch').allclose(normalized,expected)
        with h5py.File(root/'data_views'/VAL_VIEW,'r',swmr=True) as raw:
            for index,episode,start in ((0,0,0),(484,0,484),(485,1,0),(484_992,999,477),(484_999,999,484)):
                sample=val[index];begin=episode*500+start
                assert tuple(sample['pixels'].shape)==(16,3,224,224)
                assert np.array_equal(sample['pixels'][0].permute(1,2,0).numpy(),raw['pixels'][begin])
                assert np.array_equal(sample['action'][:15].numpy(),raw['action'][begin:begin+15])
        tail=__import__('torch').utils.data.DataLoader(
            __import__('torch').utils.data.Subset(val,list(range(484_992,485_000))),
            batch_size=128,drop_last=False,num_workers=0)
        batches=list(tail)
        assert len(batches)==1 and batches[0]['pixels'].shape[0]==8
    finally:
        train.close();val.close()

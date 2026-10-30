"""Small deterministic CPU fixtures; never reads production data or runs a model."""
import importlib.util
from pathlib import Path
import pytest

HERE = Path(__file__).parent


def adapter():
    path = HERE / 'adapter.py'
    assert path.exists(), 'isolated adapter not implemented'
    spec = importlib.util.spec_from_file_location('split32_fixture_adapter', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class PlainBatchSampler:
    sampler = range(296)
    batch_size = 128
    drop_last = False
    def __iter__(self):
        for start in range(0,296,128): yield list(range(start,min(start+128,296)))
    def __len__(self): return 3


class SyntheticPLDM:
    method = 'pldm'
    gpu_transform = 'unchanged_fixture_sentinel'
    def __init__(self,count): self.count=count; self.marker=0
    def __len__(self): return self.count
    def __getitem__(self,index):
        import torch
        return dict(pixels=torch.full((16,3,4,4),index%251,dtype=torch.uint8),
                    actions=torch.full((15,5),index/1000,dtype=torch.float32),
                    sample_id=index,worker_marker=self.marker)


def initialize_worker(worker_id):
    import torch
    torch.utils.data.get_worker_info().dataset.marker=101


def test_split_sampler_preserves_boundaries_without_eager_draw():
    module=adapter(); parent=PlainBatchSampler(); split=module._ReadTaskBatchSampler(parent,32)
    assert len(split)==10
    chunks=list(split)
    assert [len(x) for x in chunks]==[32]*9+[8]
    assert [i for x in chunks for i in x]==list(range(296))


def test_large_lengths_and_original_objects_metadata_only():
    torch=pytest.importorskip('torch');torch.set_num_threads(1);module=adapter()
    for count,drop,expected in [(5850000,True,45703),(585000,False,4571)]:
        dataset=SyntheticPLDM(count);generator=torch.Generator().manual_seed(0)
        native=torch.utils.data.DataLoader(dataset,batch_size=128,drop_last=drop,shuffle=drop,
            generator=generator,num_workers=2,prefetch_factor=2,pin_memory=True)
        sampler,batch_sampler=native.sampler,native.batch_sampler
        wrapped=module.wrap_loader(native,workers=4,prefetch_factor=4)
        assert wrapped is native and isinstance(wrapped,torch.utils.data.DataLoader)
        assert wrapped.dataset is dataset and wrapped.sampler is sampler and wrapped.batch_sampler is batch_sampler
        assert wrapped.generator is generator and wrapped.batch_size==128 and wrapped.drop_last==drop
        assert wrapped.dataset.gpu_transform=='unchanged_fixture_sentinel' and len(wrapped)==expected
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('workers',[2,4])
def test_multiprocess_values_layout_ids_full_epoch_rng_and_tail40(workers):
    torch=pytest.importorskip('torch');torch.set_num_threads(1);module=adapter()
    for count,drop in [(413,True),(296,False)]:
        g1=torch.Generator().manual_seed(17);g2=torch.Generator().manual_seed(17)
        kwargs=dict(batch_size=128,drop_last=drop,shuffle=drop,num_workers=2,prefetch_factor=2,pin_memory=True)
        native=torch.utils.data.DataLoader(SyntheticPLDM(count),generator=g1,worker_init_fn=initialize_worker,**kwargs)
        candidate=torch.utils.data.DataLoader(SyntheticPLDM(count),generator=g2,**kwargs)
        candidate=module.wrap_loader(candidate,workers=workers,prefetch_factor=2)
        # Lightning changes this after construction; the adapter must forward the current value.
        candidate.worker_init_fn=initialize_worker
        torch.manual_seed(99);left=list(native);global1=torch.get_rng_state().clone()
        torch.manual_seed(99);right=list(candidate);global2=torch.get_rng_state().clone()
        assert len(left)==len(right)==3
        for a,b in zip(left,right):
            assert a.keys()==b.keys()
            for key in a:
                assert torch.equal(a[key],b[key]) and a[key].dtype==b[key].dtype
                assert a[key].shape==b[key].shape and a[key].stride()==b[key].stride()
            assert bool((b['worker_marker']==101).all())
        assert right[-1]['sample_id'].shape[0]==(128 if drop else 40)
        assert torch.equal(g1.get_state(),g2.get_state()) and torch.equal(global1,global2)
        i1=iter(native);i2=iter(candidate)
        try: assert torch.equal(next(i1)['sample_id'],next(i2)['sample_id'])
        finally: i1._shutdown_workers();i2.close()
        stats=candidate.readtask_statistics
        assert stats['pin_requested'] is True
        if not torch.cuda.is_available():
            assert stats['pin_validation_available'] is False and stats['pinned_output'] is False
            assert not right[0]['pixels'].is_pinned()
        assert stats['merge_seconds'] is None
    assert not torch.cuda.is_initialized()


def test_immediate_seed_consumption_early_close_and_worker_cleanup():
    torch=pytest.importorskip('torch');torch.set_num_threads(1);module=adapter()
    for workers in (2,4):
        g1=torch.Generator().manual_seed(31);g2=torch.Generator().manual_seed(31)
        kwargs=dict(batch_size=128,shuffle=True,num_workers=2,prefetch_factor=2,generator=g1)
        native=torch.utils.data.DataLoader(SyntheticPLDM(4096),**kwargs)
        kwargs['generator']=g2
        candidate=module.wrap_loader(torch.utils.data.DataLoader(SyntheticPLDM(4096),**kwargs),
                                     workers=workers,prefetch_factor=2)
        i1=iter(native);i2=iter(candidate)
        # Before next(): native base_seed and first permutation have already been consumed.
        assert torch.equal(g1.get_state(),g2.get_state())
        processes=list(i2._inner_iterator._workers)
        i1._shutdown_workers();i2.close();i2.close()
        assert all(not process.is_alive() for process in processes)
        assert candidate.readtask_statistics['active_iterators']==0
        assert torch.equal(g1.get_state(),g2.get_state())
    assert not torch.cuda.is_initialized()

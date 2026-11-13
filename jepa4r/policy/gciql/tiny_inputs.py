# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tiny_inputs.py (verbatim)
# ruff: noqa: F401  (moved verbatim: unused imports)
"""CPU-only fixed real augmented batches, using f437's own Frames pipeline."""
from pathlib import Path
import argparse,hashlib,json,os,sys,random

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf8')as f:json.dump(value,f,indent=2,allow_nan=False)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--collection-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    print(json.dumps(dict(executed=False,augmented_batches=3,batch_shape=[256,4,3,224,224],device='cpu',model_updates=0)))
    if not args.execute:return 0
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    source,raw,out=args.source_root.resolve(),args.collection_root.resolve(),args.output.resolve()
    existing=sys.modules.get('bench.ogbench')
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(source):
        raise RuntimeError('run input producer as a direct script against frozen source')
    sys.path[:0]=[str(source/'lewm_pcd'),str(source/'lewm_pcd/third_party/stable-worldmodel'),str(source/'lewm_pcd/third_party/ogbench')]
    from bench.ogbench.training_runtime import load_tiny_trainer
    from bench.ogbench.collection import merge_raw
    from bench.ogbench.fresh_data import _view
    original=load_tiny_trainer()
    if sha(original.__file__)!='7d6e15821eb2d5aa976b5cc5f86b2e258c7212dc36d3fcad9c8febf177ab8ff1':
        raise ValueError('original Tiny augmentation source is not f437')
    import torch,numpy as np,hdf5plugin,importlib.metadata
    receipt=json.loads((raw/'COLLECTION.json').read_text())
    sources=[]
    for index in (0,1):
        path=raw/f'shards/part_{index:03d}/datasets/pair_pix.h5'
        expected=[value for name,value in receipt['data_files'].items()if name.endswith(f'/shards/part_{index:03d}/datasets/pair_pix.h5')]
        if len(expected)!=1 or sha(path)!=expected[0]['sha256']:raise ValueError('original paired RGB shard differs')
        sources.append(dict(path=str(path),sha256=sha(path)))
    out.mkdir(parents=True,exist_ok=False)
    raw_descriptor=merge_raw([row['path']for row in sources],out/'train_rgb_raw_descriptor.h5')
    view=_view(raw_descriptor,out/'train_rgb_view.h5',200,2,'rgb',sha(raw/'COLLECTION.json'))
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    dataset=original.Frames(view['path'],4)
    if len(dataset)!=400:raise ValueError('native Tiny small-data input must contain400 outgoing-frame observations')
    generator=torch.Generator().manual_seed(0)
    captured=[]
    class ObservedSampler:
        def __len__(self):return len(dataset)
        def __iter__(self):
            for index in torch.utils.data.RandomSampler(dataset,generator=generator):
                captured.append(int(index));yield index
    loader=torch.utils.data.DataLoader(dataset,batch_size=256,sampler=ObservedSampler(),drop_last=True,num_workers=0)
    if len(loader)!=1:raise ValueError('native ten-epoch scheduler expects one batch per epoch for this input')
    batches,entries={},[]
    for epoch in range(3):
        captured.clear();batch=next(iter(loader))
        if batch.shape!=(256,4,3,224,224)or batch.dtype!=torch.float32 or not torch.isfinite(batch).all():
            raise ValueError('original real augmentation output failed')
        key=f'epoch{epoch}_batch0';batches[key]=batch.numpy()
        entries.append(dict(id=f'train:epoch{epoch}:batch0',array_key=key,epoch=epoch,batch_index=0,
            frame_indices=captured.copy(),episode_start_pairs=[list(map(int,dataset.ds.clip_indices[index]))for index in captured]))
    arrays=out/'real_augmented_batches.npz'
    with arrays.open('xb')as f:np.savez_compressed(f,**batches)
    recipe=dict(model='lejepa',seed=0,epochs=10,batch_size=256,views=4,lr=.002,lamb=.02,proj_dim=16,
                precision='bf16',loader_samples=400,loader_batches_per_epoch=1,
                native_source_sha256=sha(original.__file__))
    recipe_path=out/'native_recipe.json';write(recipe_path,recipe)
    cfg=dict(mode='smoke',model='lejepa',precision='bf16',seed=0,stop_after=2,
        source_root=str(source),source_manifest_sha256=sha(source/'SOURCE_MANIFEST.json'),
        dataset=view['path'],dataset_binding=dict(path=view['path'],sha256=sha(view['path'])),
        batch_arrays=dict(path=str(arrays),sha256=sha(arrays)),batches=entries,workers=1,
        loader_samples=400,loader_batches_per_epoch=1,
        identity=dict(model='lejepa',data=sha(arrays),config=sha(recipe_path),split='train',action_dim=5))
    write(out/'old_reference.json',cfg)
    write(out/'INPUT_PROVENANCE.json',dict(scope='real fixed augmented inputs only; no model construction/forward/update',
        original_source_revision='f437d1a3f27c3cb1f9dc1d7adfbf0afb68dbaa16',original_augmentation_source_sha256=sha(original.__file__),
        original_paired_rgb=sources,original_actions_modified=False,collection_receipt_sha256=sha(raw/'COLLECTION.json'),
        input_producer_workers=0,augmentation_seed=0,sampler_seed=0,input_stage='native Frames augmented normalized FP32 output',
        actual_model_updates=0,batches=entries,arrays_sha256=sha(arrays),
        library_versions={n:importlib.metadata.version(n)for n in ['torch','torchvision','numpy','stable-pretraining']}))
    for row in sources:
        if sha(row['path'])!=row['sha256']:raise ValueError('original paired RGB bytes changed')
    return 0
if __name__=='__main__':raise SystemExit(main())

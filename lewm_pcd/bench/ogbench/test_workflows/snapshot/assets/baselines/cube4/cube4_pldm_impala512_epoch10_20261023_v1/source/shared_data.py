"""QuadrupleCube RGB/action bridge over the sealed 600-row VDS reader."""
from pathlib import Path
import hashlib
import importlib
import json
import operator
import sys

MANIFEST_SHA = 'd5eb5ed129e7152e21568c0549242f57cd2661f2525dbfd804703cd913cadc14'
TRAIN_VIEW_SHA = 'c7cef0b9db5af7011bd01b08b58a0d5eabf43580f0f9350ff40f4b6d79746a14'
VAL_VIEW_SHA = '2cd9c77f105906fc4e5f0c659de9c396625658e6c67d62e2decbc6075b7efb8f'
TRAIN_VIEW = '2d_train_action_rows600.h5'
VAL_VIEW = '2d_validation_action_rows600.h5'


def require_sources(root,data_root):
    """One stdlib source guard, shared by waiting relay and direct child CLI."""
    root=Path(root).resolve();data_root=Path(data_root).resolve()
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    frozen=json.loads((root/'source_freeze_v1.json').read_text())
    if data_root!=Path(frozen['data_root']).resolve():raise ValueError('actual data_root differs')
    for name,h in frozen['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=h:raise ValueError('frozen shared source differs: '+name)
    validate_manifest(data_root)
    return frozen


def contract(method):
    if method != 'pldm': raise ValueError('method must be pldm')
    frames = 16
    per = 600 - frames + 1; train=10000*per; val=1000*per
    return dict(frames=frames, actions=15, per_episode=per,
                train_samples=train,validation_samples=val,train_batches=train//128,
                validation_batches=(val+127)//128,validation_tail=val%128,total_updates=10*(train//128))


def location(index, method, episodes):
    per=contract(method)['per_episode'];index=operator.index(index);size=episodes*per
    if index<0:index+=size
    if not 0<=index<size:raise IndexError(index)
    return divmod(index,per)


def validate_manifest(data_root):
    path=Path(data_root)/'training_view_index.json'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:raise ValueError('sealed QuadrupleCube VDS manifest differs')
    value=json.loads(raw)
    expected={TRAIN_VIEW:TRAIN_VIEW_SHA,VAL_VIEW:VAL_VIEW_SHA}
    if (value.get('actual_exit_code')!=0 or value.get('raw_data_unchanged') is not True
            or value.get('raw_rows_per_episode')!=601 or value.get('view_rows_per_episode')!=600
            or value.get('raw_terminal_row')!=600
            or value.get('split_counts')!={'train':10000,'validation':1000}
            or value.get('no_GT_model_input') is not True or value.get('no_physical_visual_merge') is not True
            or value.get('no_feature_cache') is not True):
        raise ValueError('sealed QuadrupleCube VDS collection contract differs')
    views={Path(item['path']).name:item for item in value.get('views',[])
           if item.get('keys')==['pixels','action']}
    if set(views)!={TRAIN_VIEW,VAL_VIEW}:
        raise ValueError('exact QuadrupleCube RGB/action VDS splits are required')
    for name,digest in expected.items():
        item=views[name];episodes=10000 if name==TRAIN_VIEW else 1000
        if (item.get('sha256')!=digest or item.get('episodes')!=episodes
                or item.get('rows')!=episodes*600 or item.get('bytes') is None):
            raise ValueError('QuadrupleCube view provenance differs: '+name)
    # The manifest window describes the four-frame LeWM view. PLDM intentionally
    # uses sixteen consecutive action-bearing rows and frameskip one.
    return value


def original_reader(view, method, data_root):
    code=Path(__file__).resolve().parent/'data_bridge/code_front600_v1/lewm_pcd'
    sys.path.insert(0,str(code))
    module=importlib.import_module('training_hdf5')
    if Path(module.__file__).resolve()!=code/'training_hdf5.py':raise ValueError('another reader was imported')
    if Path(view).name not in (TRAIN_VIEW,VAL_VIEW):raise ValueError('unrecognized QuadrupleCube split')
    digest=TRAIN_VIEW_SHA if Path(view).name==TRAIN_VIEW else VAL_VIEW_SHA
    return module.TrainingHDF5Dataset(path=str(view),num_steps=contract(method)['frames'],frameskip=1,
        keys_to_load=['pixels','action'],keys_to_cache=['action'],transform=None,
        compact_clip_indices=True,expected_sha256=digest)


def action_statistics(train_reader):
    """Exactly one train-only normalization; no image statistics or action clipping."""
    import numpy as np
    import torch
    values=torch.from_numpy(np.asarray(train_reader.get_col_data('action')))
    if tuple(values.shape)!=(6000000,5) or not bool(torch.isfinite(values).all()):
        raise ValueError('full finite train action rows600 required')
    return dict(mean=values.mean(0).tolist(),std=values.std(0,correction=1).tolist(),
                count=len(values),correction=1,epsilon=1e-6,source_split='train',
                manifest_sha256=MANIFEST_SHA,normalizations_per_sample=1)


class PLDMPreprocess:
    """Approved RGB domain seam; no ImageNet normalization and no encoder call."""
    def __call__(self,batch):
        import torch
        from torchvision.transforms import v2
        pixels=batch['pixels']
        actions=batch['actions']
        if pixels.dtype is not torch.uint8 or tuple(pixels.shape[1:])!=(16,3,224,224):
            raise ValueError('one uint8 RGB16 preprocessing pass required')
        if tuple(actions.shape[1:])!=(15,5) or not actions.is_floating_point():
            raise ValueError('normalized robot actions must be floating RGB16/A15 batches')
        batch['pixels']=v2.Resize((65,65),antialias=True)(pixels.float()/255.0)
        if self.last_contract is None:
            self.last_contract=dict(raw_pixel_dtype='uint8',raw_pixel_shape=list(pixels.shape),
                model_pixel_dtype='float32',model_pixel_shape=list(batch['pixels'].shape),
                pixel_scale='divide_by_255',resize='torchvision v2 Resize65 antialias=True',
                action_dtype=str(actions.dtype),action_shape=list(actions.shape),
                action_normalization='train-only sample std + 1e-6, performed once in dataset')
        return batch

    def __init__(self):
        self.last_contract=None


class VisualDataset:
    """Original scalar contiguous reads; model-specific output shapes only."""
    def __init__(self,view,method,*,data_root=None,reader=None,action_stats=None,augmentations=None,gpu_preproc=None):
        self.method=method;self.reader=reader if reader is not None else original_reader(view,method,data_root)
        self.action_stats=action_stats;self.augmentations=augmentations
        self.gpu_transform=PLDMPreprocess() if method=='pldm' else gpu_preproc

    def __len__(self):return len(self.reader)

    def __getitem__(self,index):
        sample=self.reader[index];pixels=sample['pixels']
        if self.method=='pldm':
            if self.action_stats is None:raise ValueError('shared train-only action statistics required')
            actions=sample['action'][:15]
            if tuple(actions.shape)!=(15,5) or pixels.shape[0]!=16:raise ValueError('RGB16/A15 contract differs')
            mean=actions.new_tensor(self.action_stats['mean']);std=actions.new_tensor(self.action_stats['std'])
            return dict(pixels=pixels,actions=((actions-mean)/(std+1e-6)).float(),sample_id=int(index))
        frame=pixels[0]
        if self.augmentations is None:raise ValueError('original eight augmentation callables required')
        global_t,local_t=self.augmentations
        return dict(global_views=[t(frame.clone()) for t in global_t],
                    local_views=[t(frame.clone()) for t in local_t],sample_id=int(index))

    def close(self):self.reader.close()


def loaders(method,data_root,method_module):
    import torch
    validate_manifest(data_root);root=Path(data_root)
    train=VisualDataset(root/TRAIN_VIEW,method,data_root=data_root)
    stats=action_statistics(train.reader);train.action_stats=stats
    val=VisualDataset(root/VAL_VIEW,method,data_root=data_root,action_stats=stats)
    kwargs=dict(batch_size=128,num_workers=20,prefetch_factor=2,pin_memory=True,persistent_workers=False)
    generator=torch.Generator().manual_seed(0)
    a=torch.utils.data.DataLoader(train,shuffle=True,drop_last=True,generator=generator,**kwargs)
    b=torch.utils.data.DataLoader(val,shuffle=False,drop_last=False,**kwargs)
    c=contract(method)
    if (len(train),len(val),len(a),len(b))!=(c['train_samples'],c['validation_samples'],c['train_batches'],c['validation_batches']):
        raise ValueError('actual complete epoch counts differ')
    return a,b,stats

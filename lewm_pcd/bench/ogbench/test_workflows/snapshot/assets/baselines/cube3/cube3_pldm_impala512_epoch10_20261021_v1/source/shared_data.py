"""Thin episode-safe access over the already accepted Triple600 VDS reader."""
from pathlib import Path
import hashlib
import importlib
import json
import operator
import sys

MANIFEST_SHA = '98dde8e9a49bfdd3c18591939cab9208bd9d3a2af7ae095b47e020cd02ba9c29'
READER_CORE_FREEZE_SHA = '86bc5cd1579f6f58aecb567dbd924e011a10d31b12a3b5f23b8d621680c7c331'
PROCESS_AUX_FREEZE_SHA = '5f9533276969329d28e1ab85800513f9e3367841e1f97819329225b2dd96e189'
REUSED_CORE_FILES = ('training_hdf5.py','triple_training_contract.py','utils.py',
    'third_party/stable-worldmodel/stable_worldmodel/data/formats/hdf5.py',
    'third_party/stable-worldmodel/stable_worldmodel/data/dataset.py')


def require_sources(root,data_root):
    """One stdlib source guard, shared by waiting relay and direct child CLI."""
    root=Path(root).resolve();data_root=Path(data_root).resolve()
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    frozen=json.loads((root/'source_freeze_v2.json').read_text())
    if data_root!=Path(frozen['data_root']).resolve():raise ValueError('actual data_root differs')
    for name,h in frozen['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or digest(p)!=h:raise ValueError('frozen shared source differs: '+name)
    core=data_root/'code_front600_v1/source_freeze_v1.json'
    if digest(core)!=READER_CORE_FREEZE_SHA or frozen['reader_core_freeze_sha256']!=READER_CORE_FREEZE_SHA:
        raise ValueError('accepted reader core freeze differs')
    pins=json.loads(core.read_text())['files'];code=core.parent/'lewm_pcd'
    for name in REUSED_CORE_FILES:
        if digest(code/name)!=pins[name]:raise ValueError('accepted reader body differs: '+name)
    aux=data_root/'training_serial_compact_relay_v1/source_freeze_v3.json'
    if digest(aux)!=PROCESS_AUX_FREEZE_SHA:raise ValueError('accepted process auxiliary freeze differs')
    pins=json.loads(aux.read_text())['files']
    if digest(aux.parent/'serial_relay.py')!=pins['serial_relay.py']:
        raise ValueError('accepted reused process helper body differs')
    return frozen


def contract(method):
    if method not in ('pldm','lejepa'): raise ValueError('method must be pldm or lejepa')
    frames = 16 if method == 'pldm' else 1
    per = 600 - frames + 1; train=10000*per; val=1000*per
    return dict(frames=frames, actions=15 if method=='pldm' else 0, per_episode=per,
                train_samples=train,validation_samples=val,train_batches=train//128,
                validation_batches=(val+127)//128,validation_tail=val%128,total_updates=10*(train//128))


def location(index, method, episodes):
    per=contract(method)['per_episode'];index=operator.index(index);size=episodes*per
    if index<0:index+=size
    if not 0<=index<size:raise IndexError(index)
    return divmod(index,per)


def validate_manifest(data_root):
    path=Path(data_root)/'data/views/front600_v1/training_view_index.json'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:raise ValueError('accepted Triple600 manifest differs')
    value=json.loads(raw);c=value['recipe']['contract']
    if value['actual_exit_code']!=0 or value['stage']!='passed' or (c['physical_cube_count'],c['horizon_actions'],c['train_episodes'],c['validation_episodes'])!=(3,600,10000,1000):
        raise ValueError('Triple600 production contract required')
    return value


def original_reader(view, method, data_root):
    code=Path(data_root)/'code_front600_v1/lewm_pcd'
    sys.path[:0]=[str(code),str(code/'third_party/stable-worldmodel')]
    module=importlib.import_module('training_hdf5')
    if Path(module.__file__).resolve()!=code/'training_hdf5.py':raise ValueError('another reader was imported')
    return module.TrainingHDF5Dataset(path=str(view),num_steps=contract(method)['frames'],frameskip=1,
        keys_to_load=['pixels','action'] if method=='pldm' else ['pixels'],
        keys_to_cache=['action'] if method=='pldm' else [],transform=None,compact_clip_indices=True)


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
        if pixels.dtype is not torch.uint8 or tuple(pixels.shape[1:])!=(16,3,224,224):
            raise ValueError('one uint8 RGB16 preprocessing pass required')
        batch['pixels']=v2.Resize((65,65),antialias=True)(pixels.float()/255.0)
        return batch


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
    validate_manifest(data_root);root=Path(data_root)/'data/views/front600_v1'
    aug=method_module.build_augmentations() if method=='lejepa' else None
    gpu=method_module.gpu_preproc if method=='lejepa' else None
    train=VisualDataset(root/'2d_train_action_rows600.h5',method,data_root=data_root,augmentations=aug,gpu_preproc=gpu)
    stats=action_statistics(train.reader) if method=='pldm' else None;train.action_stats=stats
    val=VisualDataset(root/'2d_validation_action_rows600.h5',method,data_root=data_root,action_stats=stats,augmentations=aug,gpu_preproc=gpu)
    kwargs=dict(batch_size=128,num_workers=20,prefetch_factor=2,pin_memory=True,persistent_workers=False)
    generator=torch.Generator().manual_seed(0)
    a=torch.utils.data.DataLoader(train,shuffle=True,drop_last=True,generator=generator,**kwargs)
    b=torch.utils.data.DataLoader(val,shuffle=False,drop_last=False,**kwargs)
    c=contract(method)
    if (len(train),len(val),len(a),len(b))!=(c['train_samples'],c['validation_samples'],c['train_batches'],c['validation_batches']):
        raise ValueError('actual complete epoch counts differ')
    return a,b,stats

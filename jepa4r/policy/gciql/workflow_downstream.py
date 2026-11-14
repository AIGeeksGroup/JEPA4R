"""Unified six-model Cube paper protocol over original GCIQL/Native functions.

Planning imports only the standard library. Execution belongs in a fresh child.
The explicit smoke uses real complete official episodes and the original
two-update boundary. Formal uses complete Play splits,1M updates and Native250.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

SCHEMA = 'cube_workflow_downstream_v1'
STAGES = ('encode', 'cache-audit', 'gate', 'reload', 'gciql', 'accept',
          'native-sr', 'native-audit')
MODELS = {'lewm_rgb': ('lewm', 192, 'rgb'), 'lewm_pcd': ('lewm', 192, 'pcd'),
          'pldm': ('pldm', 512, 'rgb'), 'lejepa': ('lejepa', 512, 'rgb'),
          'visreg': ('visreg', 768, 'rgb'), 'vjepa': ('vjepa', 1024, 'rgb')}
PREREQUISITES = {'encode': (), 'cache-audit': ('encode',),
                 'gate': ('cache-audit',), 'reload': ('gate',),
                 'gciql': ('reload',), 'accept': ('gciql',),
                 'native-sr': ('reload',), 'native-audit': ('native-sr',)}
PACKAGE = Path(__file__).resolve().parent
OFFICIAL_EPISODES = {1:(1000,100), 2:(1000,100), 3:(3000,300), 4:(5000,500)}
CACHE_SEED_BASES = {'train':0, 'val':1000000000}


def protocol_budget(config):
    counts = (1,1) if config['mode'] == 'smoke' else OFFICIAL_EPISODES[config['cube_count']]
    return dict(split_episodes=dict(zip(('train','val'),counts)),
                native_episodes=1 if config['mode']=='smoke' else 250,
                diagnostic_only=config['mode']=='smoke')


def cache_sampling_seed(config, split, raw_id):
    if split not in CACHE_SEED_BASES or type(raw_id) is not int or raw_id < 0:
        raise ValueError('Explicit split/nonnegative canonical raw ID required for cache seed')
    return config['cache_sampling_seed_base'][split] + raw_id


def prerequisites(stage, mode):
    return ('accept',) if stage=='native-sr' and mode=='formal' else PREREQUISITES[stage]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def group_contract(model, cube_count):
    if model not in MODELS or type(cube_count) is not int or cube_count not in (1, 2, 3, 4):
        raise ValueError('Expected one of six paper models and Cube1..4')
    family, dim, modality = MODELS[model]
    return dict(model=model, cube_count=cube_count, family=family, modality=modality,
                input_contract=dict(latent_dim=dim, current_dim=dim+19,
                                    goal_dim=dim+3*cube_count, zero_padding=False),
                native_max_episode_steps=(200, 500, 1000, 1000)[cube_count-1])


def validate_config(config):
    if config.get('schema') != SCHEMA or config.get('mode') not in ('smoke', 'formal'):
        raise ValueError('Explicit workflow schema and smoke/formal mode required')
    contract = group_contract(config.get('model'), config.get('cube_count'))
    encoder = config.get('encoder_binding', {})
    if (encoder.get('family'), encoder.get('latent_dim')) != (
            contract['family'], contract['input_contract']['latent_dim']):
        raise ValueError('Encoder family/output width differs from the paper group')
    for key in ('checkpoint', 'checkpoint_sha256', 'preprocessing'):
        if key not in encoder:
            raise ValueError('Explicit encoder binding required: '+key)
    if not Path(config.get('output_root', '')).is_absolute():
        raise ValueError('An absolute fresh output_root is required')
    if set(config.get('official_data', {})) != {'train', 'val'}:
        raise ValueError('Both official train and val NPZ bindings are required')
    if config.get('cache_sampling_seed_base') != CACHE_SEED_BASES:
        raise ValueError('Unified paper cache seed bases must be train0/val1000000000')
    camera = config.get('camera', {})
    if (camera.get('width'), camera.get('height'), camera.get('aa_samples')) != (224, 224, 4):
        raise ValueError('Explicit RGB224/AA4 camera binding required')
    for key in ('position', 'source_position'):
        if (not isinstance(camera.get(key), list) or len(camera[key]) != 3 or
                any(type(v) not in (float, int) or not math.isfinite(v) for v in camera[key])):
            raise ValueError('Explicit finite camera '+key+' required')
    quaternion = camera.get('source_quaternion')
    if (not isinstance(quaternion, list) or len(quaternion) != 4 or
            any(type(v) not in (float, int) or not math.isfinite(v) for v in quaternion) or
            abs(sum(v*v for v in quaternion)-1.) > 1e-6):
        raise ValueError('Explicit normalized source_quaternion camera orientation required')
    if (camera.get('source_name') not in ('front', 'front_pixels') or
            camera.get('render_name') != 'front_pixels' or
            type(camera.get('fov')) not in (float, int) or not 0 < camera['fov'] < 180 or
            camera.get('local_x_up_degrees') not in (0., 5.) or
            camera.get('pullback_m') not in (0., .1)):
        raise ValueError('Explicit original camera identity/rotation/FOV/pullback required')
    if contract['modality'] == 'pcd':
        crop = camera.get('pointcloud_crop_box')
        if (not isinstance(crop, list) or len(crop) != 2 or
                any(not isinstance(row, list) or len(row) != 3 for row in crop) or
                any(not math.isfinite(v) for row in crop for v in row) or
                any(a >= b for a, b in zip(*crop))):
            raise ValueError('An explicit finite PCD crop is required')
    return contract


def pinned(record):
    if not isinstance(record, dict) or not isinstance(record.get('sha256'), str):
        raise ValueError('An explicit path/SHA256 binding is required')
    path = Path(record.get('path', ''))
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError('Bound artifact is missing: '+str(path))
    def identity(value):
        return (value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_dev, value.st_ino)
    before = identity(path.stat())
    actual = sha(path)
    after = identity(path.stat())
    if actual != record['sha256'] or before != after:
        raise ValueError('Artifact SHA/hash or stat changed: '+str(path))
    return path


def _current_training_data(receipt,cube_count,mode):
    """Require the actual current full-horizon, episode-heldout seed0 binding."""
    if receipt.get('schema')!='cube_fresh_wm_acceptance_v1':
        raise ValueError('Current training data requires a current WM acceptance receipt')
    data=read(pinned(receipt.get('data_manifest')))
    horizon=(200,500,1000,600)[cube_count-1]
    counts=(2,1) if mode=='smoke' else (10000,1000)
    if (data.get('schema')!='cube_current_training_v1' or
            type(data.get('actual_exit_code')) is not int or data['actual_exit_code']!=0 or
            data.get('protocol')!='paper_cube_fixed_mix_v1' or
            data.get('collection_protocol',{}).get('name')!='paper_cube_fixed_mix_v1' or
            data.get('training_protocol')!='episode_heldout_v1'):
        raise ValueError('Current training data schema/protocol/episode split is required')
    if any(type(data.get(key)) is not int or data[key]!=0 for key in ('training_seed','collection_seed')):
        raise ValueError('Current training and collection seed must both be integer0')
    expected={'cube_count':cube_count,'raw_horizon':horizon,'training_rows_per_episode':horizon,
              'train_episodes':counts[0],'validation_episodes':counts[1]}
    if data.get('mode') != mode or any(
        type(data.get(key)) is not int or data[key] != value for key, value in expected.items()
    ):
        raise ValueError('Current training data Cube/mode/full-horizon/budget differs')
    return data


def _assets(config):
    contract = validate_config(config)
    encoder = config['encoder_binding']
    pinned(dict(path=encoder['checkpoint'], sha256=encoder['checkpoint_sha256']))
    for record in config['official_data'].values():
        pinned(record)
    receipt_path = pinned(config['encoder_receipt'])
    receipt = read(receipt_path)
    if (receipt.get('schema') not in ('cube_workflow_encoder_v1', 'cube_fresh_wm_acceptance_v1') or
            receipt.get('actual_exit_code') != 0 or
            receipt.get('model') != config['model'] or
            receipt.get('cube_count') != config['cube_count'] or
            receipt.get('mode') != config['mode'] or
            receipt.get('checkpoint') != dict(path=encoder['checkpoint'], sha256=encoder['checkpoint_sha256'])):
        raise ValueError('Encoder receipt does not bind this actual model/Cube/checkpoint/mode')
    if not (receipt.get('strict_reload_verified') is True or
            contract['family'] in ('visreg', 'vjepa') and receipt.get('released_weights_verified') is True):
        raise ValueError('Encoder receipt must verify strict reload or official released weights')
    if config['model'] not in ('visreg','vjepa'):
        _current_training_data(receipt,config['cube_count'],config['mode'])
    if config['mode']=='formal' and contract['family'] not in ('visreg','vjepa') and (
            receipt.get('actual_epochs') != 10 or receipt.get('formal_acceptance_verified') is not True):
        raise ValueError('Formal paper WM requires its actual ten-epoch formal acceptance')
    for path, digest in receipt.get('source_binding', {}).items():
        pinned(dict(path=path, sha256=digest))
    if receipt.get('schema') == 'cube_fresh_wm_acceptance_v1':
        if receipt.get('config') != encoder.get('factory_config') or receipt.get('actual_epochs') != encoder.get(
            'epoch'
        ):
            raise ValueError('Fresh WM receipt and actual factory/epoch binding differ')
        if 'data_manifest' in receipt:
            pinned(receipt['data_manifest'])
    if contract['family'] != 'lejepa':
        pinned(encoder['adapter'])
    if contract['family'] == 'pldm':
        pinned(encoder['factory_config'])
    elif 'factory_config' in encoder:
        pinned(encoder['factory_config'])
    pre = encoder['preprocessing']
    pinned(dict(path=pre['source_path'], sha256=pre['source_sha256']))
    if contract['family'] in ('visreg', 'vjepa'):
        pinned(encoder['spec'])
        spec = read(encoder['spec']['path'])
        expected_family = 'visreg_vitb768' if contract['family'] == 'visreg' else 'vjepa21_vitl1024'
        if (
            spec.get('family') != expected_family
            or spec.get('output_dim') != encoder['latent_dim']
            or spec.get('checkpoint') != encoder['checkpoint']
            or spec.get('checkpoint_sha256') != encoder['checkpoint_sha256']
        ):
            raise ValueError('Official encoder spec/family/checkpoint differs')
        if config['mode']=='formal' and spec.get('diagnostic_only',False) is not False:
            raise ValueError('Diagnostic official encoder spec cannot enter formal paper mode')
    for path, digest in config.get('implementation_sources', {}).items():
        pinned(dict(path=path, sha256=digest))
    return contract


def prepare_recipe(run_config, encoder_receipt_path, output_path):
    """Bind an actual accepted WM export; this explicit operation writes no models.

    run_config provides model/cube_count/mode/official_data/camera and either
    downstream_output_root or output_root (then a downstream child directory).
    Frozen official families additionally provide their explicit encoder_binding.
    """
    config = dict(
        schema=SCHEMA,
        model=run_config['model'],
        cube_count=run_config['cube_count'],
        mode=run_config['mode'],
        official_data=copy.deepcopy(run_config['official_data']),
        camera=copy.deepcopy(run_config['camera']),
        cache_sampling_seed_base=copy.deepcopy(run_config.get('cache_sampling_seed_base', CACHE_SEED_BASES)),
        output_root=str(Path(run_config['downstream_output_root']).resolve())
        if 'downstream_output_root' in run_config
        else str(Path(run_config['output_root']).resolve() / 'downstream'),
    )
    receipt_path = Path(encoder_receipt_path).resolve()
    receipt = read(receipt_path)
    config['encoder_receipt'] = dict(path=str(receipt_path), sha256=sha(receipt_path))
    model = config['model']
    contract = group_contract(model, config['cube_count'])
    if model in ('visreg', 'vjepa'):
        if 'encoder_binding' not in run_config:
            raise ValueError('Official foundation weights/spec/adapter must be explicitly supplied')
        config['encoder_binding'] = copy.deepcopy(run_config['encoder_binding'])
    else:
        _current_training_data(receipt,config['cube_count'],config['mode'])
        checkpoint, factory = receipt.get('checkpoint'), receipt.get('config')
        if not checkpoint or not factory:
            raise ValueError('Accepted WM receipt needs actual checkpoint and factory config')
        checkpoint_path, factory_path = pinned(checkpoint), pinned(factory)
        if factory_path != checkpoint_path.parent/'config.json':
            raise ValueError('Original checkpoint loader requires its actual sibling config.json')
        epoch = receipt.get('actual_epochs')
        if type(epoch) is not int or epoch < 1:
            raise ValueError('Receipt must specify actual positive exported epochs')
        if model == 'lejepa' and checkpoint_path.name != f'weights_epoch_{epoch}.pt':
            raise ValueError('The canonical Tiny loader requires its actual weights_epoch_N.pt filename')
        if model.startswith('lewm_') and checkpoint_path.suffix != '.pt':
            raise ValueError('The original LeWM local loader requires an actual .pt bare export')
        pre_source = PACKAGE.parents[1]/'data/clips.py'   # LeWM's utils.py preprocessors, moved
        binding = dict(family=contract['family'], latent_dim=contract['input_contract']['latent_dim'],
                       checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint['sha256'],
                       factory_config=factory, epoch=epoch)
        if model == 'lejepa':
            pre_source = PACKAGE.parent/'bc/jepa_baselines.py'
        elif model == 'pldm':
            cfg = read(factory_path)
            if (
                cfg.get('_target_') != 'jepa4r.baselines.cube.pldm.method.build_model'
                or cfg.get('pretrained') is not False
            ):
                raise ValueError('Cube PLDM requires the native Impala factory, not a ViT checkpoint')
            from jepa4r.baselines.cube.workflow_pldm_encoder import encoder_source_bindings
            encoder_source_bindings(cfg['official_source'])
            pre_source = PACKAGE.parents[1]/'baselines/cube/workflow_pldm_encoder.py'
            binding['adapter'] = dict(path=str(pre_source), sha256=sha(pre_source))
        else:
            cfg = read(factory_path)
            encoder_target = cfg.get('encoder', {}).get('_target_', '')
            expected = 'DexArtPointNet' if model == 'lewm_pcd' else 'ViTCLS'
            if cfg.get('_target_') != 'jepa.JEPA' or expected not in encoder_target:
                raise ValueError('Actual LeWM factory encoder family differs from this model')
            adapter = PACKAGE.parents[1]/'models/cube/workflow_lewm_encoder.py'
            binding['adapter'] = dict(path=str(adapter), sha256=sha(adapter))
        binding['preprocessing'] = dict(source_path=str(pre_source), source_sha256=sha(pre_source),
            device='cpu', input_dtype='float32' if model == 'lewm_pcd' else 'uint8',
            layout='XYZRGB1200' if model == 'lewm_pcd' else 'HWC224')
        if model == 'lewm_pcd':
            binding['preprocessing'].update(center=[.425,0.,.2], scale=4., use_rgb=True)
        config['encoder_binding'] = binding
    if 'cuda_visible_devices' in run_config:
        config['cuda_visible_devices'] = run_config['cuda_visible_devices']
    for flag in ('retain_raw_inputs','r4_mapping'):
        if flag in run_config:
            if type(run_config[flag]) is not bool or config['mode']!='smoke':
                raise ValueError('raw-input/R4 capture must be explicit bounded smoke')
            config[flag]=run_config[flag]
    sources = [Path(__file__), PACKAGE.parents[1]/'models/cube/workflow_lewm_encoder.py',
               PACKAGE.parents[1]/'baselines/cube/workflow_pldm_encoder.py', PACKAGE/'downstream/sampler.py',
               PACKAGE/'downstream/trainer.py', PACKAGE/'downstream/runtime.py',
               PACKAGE/'downstream/native.py', PACKAGE/'downstream/native_core.py',
               PACKAGE.parents[2]/'configs/policy/gciql/workflow_agent.json']
    config['implementation_sources'] = {str(path):sha(path) for path in sources}
    if config.get('r4_mapping'):
        config['implementation_sources'][str(PACKAGE/'d3_mapping.py')] = sha(PACKAGE/'d3_mapping.py')
    if model == 'pldm':
        config['implementation_sources'].update(encoder_source_bindings(read(factory_path)['official_source']))
    _assets(config)
    save_new(output_path, config)
    return config


def _stage_receipt(root, stage, config_sha):
    path = root/'stages'/(stage+'.json')
    if not path.is_file():
        raise FileNotFoundError('Missing workflow '+stage+' receipt: '+str(path))
    value = read(path)
    if value.get('actual_exit_code') != 0 or value.get('config_sha256') != config_sha:
        raise ValueError('Stale or failed workflow '+stage+' receipt')
    for record in value.get('artifacts', {}).values():
        pinned(record)
    for prior in prerequisites(stage, value.get('mode','smoke')):
        _stage_receipt(root, prior, config_sha)
    return value


def build_plan(config_path, stage):
    if stage not in STAGES:
        raise ValueError('Unknown downstream stage: '+str(stage))
    config_path = Path(config_path).resolve()
    config = read(config_path)
    contract = validate_config(config)
    if config['mode'] == 'smoke' and stage in ('gciql', 'accept'):
        raise ValueError('smoke uses gate/reload, never a formal million-update train/accept')
    _assets(config)
    root, digest = Path(config['output_root']).resolve(), sha(config_path)
    for record in (*config['official_data'].values(), config['encoder_receipt']):
        if Path(record['path']).resolve().is_relative_to(root):
            raise ValueError('Output namespace may not contain an input artifact')
    if root.exists():
        if not (root/'WORKFLOW.json').is_file():
            raise FileExistsError('Existing foreign output namespace: '+str(root))
        if read(root/'WORKFLOW.json').get('config_sha256') != digest:
            raise ValueError('Existing workflow namespace binds a different config')
        if (root/'drivers'/stage).exists() or (root/'stages'/(stage+'.json')).exists():
            raise FileExistsError('Stage output already exists; preserve prior attempt: '+stage)
    elif stage != 'encode':
        raise FileNotFoundError('encode must create the fresh workflow namespace first')
    for prior in prerequisites(stage, config['mode']):
        _stage_receipt(root, prior, digest)
    budget=protocol_budget(config)
    return dict(
        schema=SCHEMA,
        stage=stage,
        config=str(config_path),
        config_sha256=digest,
        output_root=str(root),
        execute=False,
        model=config['model'],
        cube_count=config['cube_count'],
        mode=config['mode'],
        execution_protocol='cube_paper_gciql_v1' if config['mode'] == 'formal' else 'cube_paper_gciql_smoke_v1',
        paper_reported_result_reproduced=False,
        diagnostic_only=budget['diagnostic_only'],
        formal_acceptance=False,
        formal_acceptance_required=config['mode'] == 'formal',
        split_episodes=budget['split_episodes'],
        training_updates=1000000,
        batch_size=1024,
        training_seed=0,
        raw_rows_per_episode=1001,
        gate_updates=2,
        native_episodes=budget['native_episodes'],
        cache_sampling_seed_base=config['cache_sampling_seed_base'],
        native_max_episode_steps=contract['native_max_episode_steps'],
        input_contract=contract['input_contract'],
    )


class FrozenOfficialEncoder:
    """Use the original frozen foundation adapter and its real preprocessing."""
    def __init__(self, binding):
        import torch
        import numpy as np
        adapter_path = pinned(binding['adapter'])
        spec = importlib.util.spec_from_file_location('_workflow_official_encoder', adapter_path)
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.model = self.adapter.load_encoder(binding['spec']['path'], device='cuda')
        self.binding, self.torch, self.np, self.calls = binding, torch, np, 0
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Official encoder is not frozen')

    def encode(self, value):
        np, torch = self.np, self.torch
        if value.shape != (224, 224, 3) or value.dtype != np.uint8:
            raise ValueError('Official encoder requires real uint8 HWC224')
        with torch.inference_mode():
            features = self.adapter.encode_uint8(self.model, torch.from_numpy(np.ascontiguousarray(value[None])))
        latent = features[0].detach().cpu().numpy()
        if latent.shape != (self.binding['latent_dim'],) or latent.dtype != np.float32 or not np.isfinite(latent).all():
            raise ValueError('Official encoder actual width/dtype/finite contract failed')
        self.calls += 1
        return latent


def _encoder(config):
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if config['model'] in ('visreg', 'vjepa'):
        return FrozenOfficialEncoder(config['encoder_binding'])
    from .downstream.native import OnlineEncoder
    return OnlineEncoder(dict(config, modality=MODELS[config['model']][2]))


def _core(config):
    from .downstream import native_core as core
    from .downstream._source import VENDOR, load_source
    import torch
    torch.set_num_threads(1)
    physics = load_source('_workflow_native_physics', 'physics.py')
    physics.configure_numpy_physics_runtime()
    contract = group_contract(config['model'], config['cube_count'])
    group = config['model']+'_cube'+str(config['cube_count'])
    core.ROOT = VENDOR
    core.LATENT_DIM = contract['input_contract']['latent_dim']
    core.GROUPS = {group: dict(cube_count=config['cube_count'], state_dim=19+9*config['cube_count'],
        current_dim=contract['input_contract']['current_dim'], goal_dim=contract['input_contract']['goal_dim'],
        mode=contract['modality'], max_episode_steps=contract['native_max_episode_steps'],
        camera=config['camera'], pcd_crop_box=config['camera'].get('pointcloud_crop_box'))}
    core.OnlineEncoder = lambda _group: _encoder(config)
    original_camera = getattr(core, '_workflow_original_configure_camera', core.configure_camera)
    core._workflow_original_configure_camera = original_camera
    def checked_camera(base, name):
        import mujoco
        import numpy as np
        camera = config['camera']
        source = mujoco.mj_name2id(base._model, mujoco.mjtObj.mjOBJ_CAMERA, camera['source_name'])
        if source < 0 or not np.allclose(base._model.cam_quat[source], camera['source_quaternion'], atol=1e-12, rtol=0):
            raise ValueError('Actual source camera quaternion differs from explicit binding')
        return original_camera(base, name)
    core.configure_camera = checked_camera
    return core, group


def _extract_official(config, root):
    """Fresh source arrays; formal keeps original NPZ, smoke writes a separate subset."""
    import numpy as np
    from .downstream.sampler import file_stat
    n, splits, arrays = config['cube_count'], {}, {}
    counts = protocol_budget(config)['split_episodes']
    for split, source in config['official_data'].items():
        path = pinned(source)
        before=file_stat(path)
        rows=counts[split]*1001
        original_rows=OFFICIAL_EPISODES[n][0 if split=='train' else 1]*1001
        with np.load(path, allow_pickle=False) as raw:
            expected={'observations':(original_rows,19+9*n),'actions':(original_rows,5),
                      'qpos':(original_rows,14+7*n),'qvel':(original_rows,14+6*n),'terminals':(original_rows,)}
            values={}
            for key, shape in expected.items():
                value=raw[key]
                if value.shape!=shape or value.dtype!=(np.bool_ if key=='terminals' else np.float32):
                    raise ValueError('Complete official NPZ shape/dtype failed: '+key)
                value=value[:rows].copy() if config['mode']=='smoke' else value
                if key!='terminals' and not np.isfinite(value).all():
                    raise ValueError('Nonfinite official data: '+key)
                values[key]=value
        if not np.array_equal(np.flatnonzero(values['terminals']), np.arange(counts[split])*1001+1000):
            raise ValueError('Official real1001 terminal/episode boundaries differ')
        target=path
        if config['mode']=='smoke':
            target = root/'data'/(split+'.npz')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                np.savez_compressed(stream, **values)
        splits[split] = dict(
            path=str(target),
            accepted_stat=file_stat(target),
            accepted_raw_sha256=source['sha256'] if target == path else sha(target),
            episodes=counts[split],
            raw_rows=rows,
            observation_shape=list(values['observations'].shape),
            action_shape=list(values['actions'].shape),
            original_official_source=source,
            subset='first complete real1001 episode' if config['mode'] == 'smoke' else None,
        )
        arrays[split]={}
        for key in ('qpos','qvel'):
            array_path=root/'source_arrays'/(split+'_'+key+'.npy')
            array_path.parent.mkdir(parents=True,exist_ok=True)
            with array_path.open('xb') as stream:
                np.save(stream,values[key],allow_pickle=False)
            arrays[split][key]=dict(path=str(array_path),sha256=sha(array_path),shape=list(values[key].shape),
                                   dtype='float32',stat=file_stat(array_path))
        if file_stat(path)!=before:
            raise ValueError('Original official NPZ changed during source extraction')
        del values
    data=dict(cube_count=n, data_id='cube-'+('single','double','triple','quadruple')[n-1]+'-play-v0', splits=splits)
    save_new(root/'source_arrays.json',dict(schema='cube_paper_source_arrays_v1',actual_exit_code=0,
        cube_count=n,mode=config['mode'],data_binding=data,splits=arrays,source_npz_modified=False))
    return data,arrays


def _runtime_config(config, root, data, manifest_sha):
    # Reuse the exact existing agent hyperparameters, not a parallel agent factory.
    template = read(PACKAGE.parents[2]/'configs/policy/gciql/workflow_agent.json')
    contract = group_contract(config['model'], config['cube_count'])
    result = dict(
        protocol='cube_all_distance_v1',
        model=config['model'],
        method=config['model'],
        role=config['model'] + '_cube' + str(config['cube_count']),
        cube_count=config['cube_count'],
        state_dim=19 + 9 * config['cube_count'],
        modality=contract['modality'],
        encoder_family=contract['family'],
        latent_dim=contract['input_contract']['latent_dim'],
        seed=0,
        batch_size=1024,
        total_updates=1000000,
        resume_from=None,
        validation_interval=5000,
        save_interval=100000,
        save_step1=True,
        heartbeat_interval=100,
        feature_lookup_workers=4,
        input_contract=contract['input_contract'],
        agent_config=template['agent_config'],
        output=str(root / 'policy'),
        data_binding=data,
        encoder_binding=config['encoder_binding'],
        native_protocol=dict(camera=config['camera']),
        feature_loader_arguments={},
        workflow_protocol='cube_paper_gciql_v1',
        mode=config['mode'],
        diagnostic_only=config['mode'] == 'smoke',
        formal_acceptance=False,
    )
    for split, source in data['splits'].items():
        result['feature_loader_arguments'][split] = dict(cache_dir=str(root/'cache'), split=split,
            manifest_sha256=manifest_sha, checkpoint_sha256=config['encoder_binding']['checkpoint_sha256'],
            source_npz=source['path'], expected_rows=source['raw_rows'], encoder_family=contract['family'],
            feature_dim=contract['input_contract']['latent_dim'])
    return result


def encode(config, root):
    import numpy as np
    # installs and verifies OGBench's impls on import
    importlib.import_module('jepa4r.policy.gciql.downstream.sampler')
    data,arrays = _extract_official(config, root)
    core, group = _core(config)
    encoder = _encoder(config)
    encoder_calls_before=encoder.calls
    contract = group_contract(config['model'], config['cube_count'])
    cache = root/'cache'
    cache.mkdir(exist_ok=False)
    entries = {}
    for split, source in data['splits'].items():
        seed=cache_sampling_seed(config,split,0)
        env, camera, proof = core.make_env(group, seed)
        wrapper = None
        try:
            if contract['modality'] == 'pcd':
                wrapper = core.OnlineVisualEnv(env, group, encoder, seed)
                wrapper.capture_proof = proof
                wrapper.reset(options={})
            base = env.unwrapped
            tables={key:np.load(record['path'],mmap_mode='r',allow_pickle=False)
                    for key,record in arrays[split].items()}
            entries[split]=[]
            for episode in range(source['episodes']):
                start=episode*1001
                features = np.empty((1001, contract['input_contract']['latent_dim']), dtype=np.float32)
                raw_rgb=np.empty((1001,224,224,3),np.uint8) if config.get('retain_raw_inputs') else None
                raw_pcd=np.empty((1001,1200,6),np.float32) if raw_rgb is not None and wrapper else None
                capture_hash=hashlib.sha256()
                for local_id,raw_id in enumerate(range(start,start+1001)):
                    base.set_state(tables['qpos'][raw_id], tables['qvel'][raw_id])
                    image = base.render(camera='front_pixels')
                    if wrapper:
                        # The declared offline protocol gives each raw ID its
                        # own original point-sampling RNG, shared across models.
                        current_seed=cache_sampling_seed(config,split,raw_id)
                        wrapper.frame.seed=current_seed
                        wrapper.frame.cores['current']._rng=np.random.default_rng(current_seed)
                    value = wrapper.capture(image, 'current') if wrapper else image
                    if raw_rgb is not None:
                        raw_rgb[local_id]=image
                        if raw_pcd is not None:
                            raw_pcd[local_id]=value
                    features[local_id] = encoder.encode(value)
                    if wrapper:
                        capture_hash.update(json.dumps(wrapper.frame.last_receipts['current'],sort_keys=True).encode())
                    if not (np.array_equal(base._data.qpos, tables['qpos'][raw_id]) and
                            np.array_equal(base._data.qvel, tables['qvel'][raw_id])):
                        raise ValueError('Static rendering changed official pose')
                directory = cache/split/f'episode_{episode:06d}'
                directory.mkdir(parents=True)
                latent_path = directory/'latent.npy'
                with latent_path.open('xb') as stream:
                    np.save(stream, features, allow_pickle=False)
                record = dict(
                    actual_exit_code=0,
                    split=split,
                    episode=episode,
                    raw_row_start=start,
                    raw_row_count=1001,
                    main_rows=1000,
                    terminal_raw_id=start + 1000,
                    canonical_raw_ids_sha256=hashlib.sha256(
                        np.arange(start, start + 1001, dtype='<i8').tobytes()
                    ).hexdigest(),
                    checkpoint_sha256=config['encoder_binding']['checkpoint_sha256'],
                    source_npz=dict(path=source['path'], accepted_stat=source['accepted_stat']),
                    latent=dict(
                        path=str(latent_path),
                        shape=list(features.shape),
                        dtype='float32',
                        bytes=latent_path.stat().st_size,
                        sha256=sha(latent_path),
                    ),
                    camera=camera,
                    diagnostic_only=config['mode'] == 'smoke',
                    formal_acceptance=False,
                    cache_sampling_seed=dict(
                        rule='base+canonical_raw_id; current private RNG reset per raw frame',
                        base=config['cache_sampling_seed_base'][split],
                        first=cache_sampling_seed(config, split, start),
                        last=cache_sampling_seed(config, split, start + 1000),
                    ),
                    pcd_capture_receipt_stream_sha256=capture_hash.hexdigest() if wrapper else None,
                )
                if raw_rgb is not None:
                    raw_path=directory/'raw_inputs.npz'
                    raw_arrays=dict(rgb=raw_rgb)
                    if raw_pcd is not None:raw_arrays['pcd']=raw_pcd
                    with raw_path.open('xb') as stream:np.savez_compressed(stream,**raw_arrays)
                    record['raw_inputs']=dict(path=str(raw_path),sha256=sha(raw_path),
                        semantics='actual same-row encoder input and RGB; no feature reuse',rows=1001)
                if config.get('r4_mapping'):
                    from .d3_mapping import build_episode
                    with np.load(source['path'],allow_pickle=False) as source_data:
                        mapping,metadata=build_episode(features,source_data['observations'][start:start+1001],
                            source_data['actions'][start:start+1001],source_data['terminals'][start:start+1001],
                            split=split,cube_count=config['cube_count'],episode=episode,raw_start=start,
                            source_sha256=source['original_official_source']['sha256'])
                    map_path,meta_path=directory/'row_mapping.npz',directory/'row_mapping.json'
                    with map_path.open('xb')as stream:np.savez(stream,**mapping)
                    save_new(meta_path,metadata)
                    record['row_mapping']=dict(arrays=dict(path=str(map_path),sha256=sha(map_path)),
                                               metadata=dict(path=str(meta_path),sha256=sha(meta_path)))
                receipt = directory/'receipt.json'
                save_new(receipt, record)
                entries[split].append(dict(episode=episode, receipt_path=str(receipt), receipt_sha256=sha(receipt)))
                progress=root/'drivers/encode/progress.json'
                temporary=progress.with_suffix('.tmp')
                temporary.write_text(
                    json.dumps(
                        dict(
                            split=split,
                            completed_episodes=episode + 1,
                            total_episodes=source['episodes'],
                            updated_at=time.time(),
                            encoder_calls=encoder.calls,
                        )
                    )
                    + '\n',
                    encoding='utf8',
                )
                temporary.replace(progress)
        finally:
            if wrapper:
                wrapper.close()
            else:
                env.close()
    pre = config['encoder_binding']['preprocessing']
    actual_encoder_calls=encoder.calls-encoder_calls_before
    expected_encoder_calls=sum(s['raw_rows']for s in data['splits'].values())
    if actual_encoder_calls!=expected_encoder_calls:
        raise ValueError('actual encoder calls differ from complete official raw rows')
    protocol = dict(encoder_family=contract['family'], latent_dim=contract['input_contract']['latent_dim'],
        camera=config['camera'], pixel_preprocess_device=pre['device'],
        source_pins={pre['source_path']: pre['source_sha256'], str(Path(__file__)): sha(__file__)})
    manifest = cache/'cache_manifest.json'
    save_new(manifest, dict(actual_exit_code=0, checkpoint_sha256=config['encoder_binding']['checkpoint_sha256'],
        protocol=protocol, splits=entries, diagnostic_only=config['mode']=='smoke', formal_acceptance=False,
        cache_sampling_seed_base=config['cache_sampling_seed_base']))
    runtime = root/'runtime_recipe.json'
    save_new(runtime, _runtime_config(config, root, data, sha(manifest)))
    return dict(
        artifacts=dict(
            runtime_recipe=dict(path=str(runtime), sha256=sha(runtime)),
            cache_manifest=dict(path=str(manifest), sha256=sha(manifest)),
            source_arrays=dict(path=str(root / 'source_arrays.json'), sha256=sha(root / 'source_arrays.json')),
        ),
        raw_rows_encoded=sum(s['raw_rows'] for s in data['splits'].values()),
        actual_encoder_calls=actual_encoder_calls,
        real_complete_episodes=sum(s['episodes'] for s in data['splits'].values()),
        encoder_calls=encoder.calls,
    )


def _install_runtime(root):
    from .downstream.runtime import install
    config = read(root/'runtime_recipe.json')
    trainer, boundary = install(config)
    return config, trainer, boundary


def _native(config, root):
    import numpy as np
    from .downstream._source import IMPLS
    runtime, trainer, boundary = _install_runtime(root)
    core, group = _core(config)
    if config['mode']=='formal':
        from .downstream.native import install_native
        final=root/'policy/checkpoints/update_001000000'
        policy={key:str(path) for key,path in dict(training=root/'runtime_recipe.json',
                acceptance=root/'accept.json',gate=root/'policy/final_val_probe.npz',final=final).items()}
        for key in ('training','acceptance','gate'):
            policy[key+'_sha256']=sha(policy[key])
        policy['manifest_sha256']=sha(final/'manifest.json')
        native=copy.deepcopy(runtime)
        native['native_protocol']=dict(evaluator='ogbench_native_v1',
            max_episode_steps=group_contract(config['model'],config['cube_count'])['native_max_episode_steps'],
            encoding_protocol='fresh_current_and_goal_each_policy_call_2N',mode='full',
            camera=config['camera'],policy=policy)
        native_path=root/'native_recipe.json'
        save_new(native_path,native)
        core,group=install_native(native,trainer,boundary,native_path)
        # Same original policy/env/evaluator, with the correctly bound official
        # or newly trained encoder factory; no alternative policy loop.
        core.OnlineEncoder=lambda _group:_encoder(config)
        counter=1000000
    else:
        core.GROUPS[group]['record_state_trace'] = True
        gate, expected = root/'gate', read(root/'gate/report.json')
        reload = read(root/'reload.json')
        def load_gate(_group):
            if expected['saved_state'] != reload['state'] or reload['actual_exit_code'] != 0:
                raise ValueError('Diagnostic policy must match the independent strict gate reload')
            directory, _ = boundary.restore_saved(runtime, gate, 2)
            with np.load(gate/'next_batch.npz', allow_pickle=False) as saved:
                example = {key:saved[key][:1].copy() for key in ('observations','actions','actor_goals')}
            agent = boundary.create_agent(boundary.model.GCIQLAgent, 0, example, boundary.agent_config(runtime))
            agent = boundary.restore_agent(agent, str(directory), 2)
            if boundary.jax.default_backend() != 'cpu' or boundary.fingerprint(agent) != reload['state']:
                raise ValueError('Diagnostic gate state/backend mismatch')
            return agent
        core.load_agent = load_gate
        core.GROUPS[group]['training'] = root/'runtime_recipe.json'
        core.EVALUATOR_REFERENCE = root/'native_evaluator_reference.json'
        evaluator = IMPLS/'utils/evaluation.py'
        save_new(core.EVALUATOR_REFERENCE, dict(source_pins={str(evaluator):sha(evaluator)}))
        counter=2
    core.main(group, 'full' if config['mode']=='formal' else 'diagnostic', root/'native')
    artifacts = {'native_report':dict(path=str(root/'native/report.json'),sha256=sha(root/'native/report.json'))}
    for index, path in enumerate(sorted((root/'native/episodes').rglob('*'))):
        if path.is_file():
            artifacts['episode_artifact_'+str(index)] = dict(path=str(path), sha256=sha(path))
    return dict(artifacts=artifacts,
                policy_checkpoint_counter=counter, diagnostic_only=config['mode']=='smoke',
                formal_acceptance=config['mode']=='formal')


def run_stage(config, root, stage):
    if stage == 'encode':
        return encode(config, root)
    if stage == 'cache-audit':
        import numpy as np
        from .downstream.sampler import load_cached_features
        cfg = read(root/'runtime_recipe.json')
        mapping_rows=raw_rows=0
        for split,arguments in cfg['feature_loader_arguments'].items():
            features, _ = load_cached_features(**arguments)
            if config.get('r4_mapping') or config.get('retain_raw_inputs'):
                manifest=read(Path(arguments['cache_dir'])/'cache_manifest.json')
                for entry in manifest['splits'][split]:
                    receipt=read(pinned(dict(path=entry['receipt_path'],sha256=entry['receipt_sha256'])))
                    count,start=receipt['raw_row_count'],receipt['raw_row_start']
                    if config.get('r4_mapping'):
                        from .d3_mapping import validate_episode
                        bound=receipt['row_mapping']
                        metadata=read(pinned(bound['metadata']))
                        with np.load(pinned(bound['arrays']),allow_pickle=False)as mapped, \
                                np.load(cfg['data_binding']['splits'][split]['path'],allow_pickle=False)as source:
                            arrays=dict(mapped)
                            validate_episode(features[start:start+count],arrays,metadata)
                            for name,key in [('states','observations'),('actions','actions'),('terminals','terminals')]:
                                np.testing.assert_array_equal(arrays[name],source[key][start:start+count],err_msg=name)
                        if (
                            metadata['source_sha256'] != config['official_data'][split]['sha256']
                            or metadata['split'] != split
                        ):
                            raise ValueError('R4 row mapping belongs to another source/split')
                        mapping_rows+=count
                    if config.get('retain_raw_inputs'):
                        with np.load(pinned(receipt['raw_inputs']),allow_pickle=False)as raw:
                            if raw['rgb'].shape!=(count,224,224,3)or raw['rgb'].dtype!=np.uint8:
                                raise ValueError('retained real RGB shape/dtype differs')
                            if config['model']=='lewm_pcd' and (raw['pcd'].shape!=(count,1200,6)or
                                    raw['pcd'].dtype!=np.float32 or not np.isfinite(raw['pcd']).all()):
                                raise ValueError('retained real PCD shape/dtype/finite differs')
                        raw_rows+=count
            del features
        budget=protocol_budget(config)
        return dict(real_complete_episodes=sum(budget['split_episodes'].values()),
                    raw_rows_verified=sum(budget['split_episodes'].values())*1001,
                    r4_mapping_rows_verified=mapping_rows,retained_input_rows_verified=raw_rows)
    if stage in ('gate', 'reload'):
        cfg, trainer, boundary = _install_runtime(root)
        if stage == 'gate':
            boundary.gate(cfg, root/'gate')
            path = root/'gate/report.json'
        else:
            path = root/'reload.json'
            boundary.reload_gate(cfg, root/'gate', path)
        return dict(artifacts={stage+'_report':dict(path=str(path),sha256=sha(path))}, diagnostic_only=True,
                    formal_acceptance=False, policy_checkpoint_counter=2)
    if stage in ('gciql','accept'):
        if config['mode']!='formal':
            raise ValueError('Only explicit formal mode may run1M GCIQL or its complete acceptance')
        cfg,trainer,boundary=_install_runtime(root)
        if stage=='gciql':
            result=trainer.run(str(root/'runtime_recipe.json'))
            if result!=0:
                raise RuntimeError('Original GCIQL trainer returned '+str(result))
            paths={'training_status':root/'policy/status.json','training_exit':root/'policy/trainer_exit_code',
                   'final_manifest':root/'policy/checkpoints/update_001000000/manifest.json'}
        else:
            boundary.accept(cfg,root/'policy',root/'accept.json')
            paths={'full_acceptance':root/'accept.json'}
        return dict(artifacts={key:dict(path=str(path),sha256=sha(path)) for key,path in paths.items()},
                    training_updates=1000000,formal_acceptance=stage=='accept',diagnostic_only=False)
    if stage == 'native-sr':
        return _native(config, root)
    if stage == 'native-audit':
        import numpy as np
        from .downstream.native_core import seed_for
        report = read(root/'native/report.json')
        episodes = sorted((root/'native/episodes').glob('task*/episode_*/result.json'))
        expected=250 if config['mode']=='formal' else 1
        if (
            report['actual_exit_code'] != 0
            or report['episodes'] != expected
            or report['unique_seeds'] != expected
            or report['mode'] != ('full' if config['mode'] == 'formal' else 'diagnostic')
            or not report['online_encoder']
            or report['offline_embedding_cache_used_for_policy']
            or len(episodes) != expected
        ):
            raise ValueError('Native count/encoder/mode contract failed')
        identities,seeds,successes=set(),set(),0
        horizon=group_contract(config['model'],config['cube_count'])['native_max_episode_steps']
        for path in episodes:
            result=read(path)
            if result['actual_exit_code']!=0 or (path.parent/'attempt_exit_code').read_text().strip()!='0':
                raise ValueError('Native episode actual exit failed')
            identity=(result['task_id'],result['episode_index'])
            identities.add(identity);seeds.add(result['seed']);successes+=result['success']
            if result['seed']!=seed_for(*identity):
                raise ValueError('Native actual seed differs from original seed_for')
            if config['mode']=='smoke':
                trace_path=path.parent/'trajectory.npz'
                if sha(trace_path)!=result.get('state_trace_sha256'):
                    raise ValueError('Native state trace checksum differs')
                with np.load(trace_path, allow_pickle=False) as trace:
                    count=result['steps']
                    if (trace['qpos'].shape[0]!=count+1 or trace['qvel'].shape[0]!=count+1 or
                            trace['observations'].shape[0]!=count+1 or
                            trace['executed_actions'].shape!=(count,5) or
                            trace['policy_current'].shape!=(count,report['current_dim']) or
                            trace['policy_goal'].shape!=(count,report['goal_dim']) or
                            not all(np.isfinite(trace[k]).all() for k in trace.files)):
                        raise ValueError('Native full state/action/input trace alignment differs')
            with np.load(path.parent/'steps.npz', allow_pickle=False) as values:
                actions=values['actions']
                if (actions.shape!=(result['steps'],5) or not 1<=len(actions)<=horizon or
                        not np.isfinite(actions).all() or np.any(np.abs(actions)>1.000001) or
                        not np.isfinite(values['rewards']).all() or
                        result['online_encoder_calls']!=2*len(actions) or
                        result['goal_online_encoder_calls']!=len(actions)):
                    raise ValueError('Native actions/horizon/online calls failed')
        wanted={(t,i) for t in range(1,6) for i in range(50)} if expected==250 else {(1,0)}
        if (
            identities != wanted
            or len(seeds) != expected
            or successes != report['successes']
            or report['success_rate'] != successes / expected
        ):
            raise ValueError('Native independent aggregate/identity audit failed')
        return dict(episodes_verified=expected, actual_successes=successes,success_rate=successes/expected,
                    paper_reported_result_reproduced=False,diagnostic_only=config['mode']=='smoke',
                    formal_acceptance=config['mode']=='formal')
    raise ValueError('Unknown stage: '+stage)


def execute(config_path, stage):
    if 'torch' in sys.modules or 'jax' in sys.modules:
        raise RuntimeError('Execute downstream stages in a fresh child before numerical imports')
    plan = build_plan(config_path, stage)
    config, root = read(config_path), Path(plan['output_root'])
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        save_new(root/'WORKFLOW.json', dict(schema=SCHEMA, config_sha256=plan['config_sha256'],
                                          model=config['model'], cube_count=config['cube_count'], mode=config['mode']))
    driver = root/'drivers'/stage
    driver.mkdir(parents=True, exist_ok=False)
    from .downstream.runtime import runtime_environment
    phase = 'gciql' if stage in ('gate','reload','gciql','accept') else 'native-sr'
    action = 'reload' if stage in ('reload','accept') else 'gate' if stage == 'gate' else 'train'
    os.environ.update(runtime_environment(phase, action, config.get('cuda_visible_devices','0')))
    os.environ['CUBE_WORKFLOW_FACTORY_CACHE'] = str(root/'factory_cache')
    os.environ.pop('JAX_DEFAULT_MATMUL_PRECISION', None)
    state = dict(stage=stage, started_at=time.time(), actual_exit_code=None, plan=plan)
    save_new(driver/'started.json', state)
    code = 1
    with (driver/'stdout_stderr.log').open('x', encoding='utf8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                result = run_stage(config, root, stage)
                state.update(actual_exit_code=0, completed_at=time.time(), result=result,mode=config['mode'],
                             config_sha256=plan['config_sha256'],diagnostic_only=config['mode']=='smoke',
                             formal_acceptance=result.get('formal_acceptance',False))
                save_new(root/'stages'/(stage+'.json'), dict(state, artifacts=result.get('artifacts', {})))
                code = 0
            except BaseException:
                state.update(actual_exit_code=1, completed_at=time.time(), error=traceback.format_exc())
                traceback.print_exc()
            finally:
                save_new(driver/'completed.json', state)
                (driver/'actual_exit_code').write_text(str(code)+'\n', encoding='utf8')
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    plan = build_plan(args.config, args.stage)
    print(json.dumps(plan, indent=2))
    return execute(args.config, args.stage) if args.execute else 0


if __name__ == '__main__':
    raise SystemExit(main())

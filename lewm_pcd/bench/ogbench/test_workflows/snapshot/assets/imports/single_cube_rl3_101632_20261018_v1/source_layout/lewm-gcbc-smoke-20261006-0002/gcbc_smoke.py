"""Two-update frozen-LeWM GCBC smoke, independent reload, native interface gate.

Train and verify are separate process invocations. No formal training or SR.
"""
import argparse
from collections import deque
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import traceback


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_hash(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class DataTransform:
    def __init__(self, modality, preprocessor, scaler):
        self.modality, self.preprocessor, self.scaler = modality, preprocessor, scaler

    def __call__(self, steps):
        import torch
        color = steps['pixels' if self.modality == 'pixels' else 'pcd_rgb']
        if color.dtype != torch.uint8:
            raise ValueError(f'Expected original uint8 color, got {color.dtype}')
        if self.modality == 'pcd' and not torch.isfinite(steps['pcd_xyz']).all():
            raise ValueError('Nonfinite raw point geometry')
        steps = self.preprocessor(steps)
        steps['action'] = self.scaler(steps['action'])
        return steps


def build_real_data(cfg, seed=42):
    import hydra
    import stable_worldmodel as swm
    from stable_worldmodel.data.normalization import get_scaler
    modality = str(cfg.model.obs_key)
    keys = ['pixels', 'action'] if modality == 'pixels' else ['pcd_xyz', 'pcd_rgb', 'action']
    raw = swm.data.load_dataset(str(cfg.data.dataset.name), num_steps=3, frameskip=1,
                               keys_to_load=keys, keys_to_cache=['action'],
                               cache_dir=os.environ.get('LOCAL_DATASET_DIR'), transform=None)
    scaler = get_scaler('zscore').fit(raw.get_col_data('action'))
    raw.transform = DataTransform(modality, hydra.utils.instantiate(cfg.data.preprocessor), scaler)
    # Transforms create pcd; raw column_names does not contain this new field.
    goal_key = 'goal_pixels' if modality == 'pixels' else 'goal_pcd'
    dataset = swm.data.GoalDataset(raw, goal_probabilities=(0., 0., 1., 0.),
                                   current_goal_offset=3,
                                   goal_keys={modality: goal_key}, seed=seed)
    return dataset, scaler, modality, goal_key


def verify_batch(batch, modality, goal_key):
    import torch
    for key in (modality, goal_key, 'action'):
        if key not in batch or not torch.isfinite(batch[key]).all():
            raise ValueError(f'Missing/nonfinite real batch key {key}')
    if batch[modality].shape[1] != 3 or batch[goal_key].shape[1] != 1 or batch['action'].shape[1:] != (3, 5):
        raise ValueError('Expected real three-frame context, one-frame goal, single-step action5')
    expected = (3, 224, 224) if modality == 'pixels' else (1200, 6)
    if tuple(batch[modality].shape[2:]) != expected or tuple(batch[goal_key].shape[2:]) != expected:
        raise ValueError('Actual modality tensor contract differs')


def predict(policy, batch, modality, goal_key):
    import torch
    with torch.inference_mode():
        action = policy.get_action({'pixels': batch[modality], 'goal': batch[goal_key]})
    if action.shape != (len(batch[modality]), 5) or not torch.isfinite(action).all():
        raise ValueError('Real batch predicted action shape/nonfinite failure')
    return action.cpu()


def train(args, output, state, persist):
    import torch
    import stable_worldmodel as swm
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from lewm_gcbc_adapter import create_policy
    source = Path(args.source_checkpoint).resolve()
    cfg = OmegaConf.load(source.parent / 'config.yaml')
    if cfg.history_size != 3 or cfg.data.dataset.frameskip != 5 or cfg.model.action_encoder.input_dim != 25:
        raise ValueError('Unexpected source WM training contract')
    source_model = swm.wm.utils.load_pretrained(str(source), cache_dir='/')
    # Strict reload uses the full original WM; only its encoder/projector are retained.
    original_state = torch.load(source, map_location='cpu', weights_only=True)
    strict_result = source_model.load_state_dict(original_state, strict=True)
    if strict_result.missing_keys or strict_result.unexpected_keys:
        raise ValueError('Source WM strict reload failed')
    factory_config = {'_target_': 'lewm_gcbc_adapter.create_policy',
                      'source_checkpoint': str(source), 'history_size': 3, 'action_dim': 5,
                      'predictor_config': {'depth': 6, 'heads': 16, 'mlp_dim': 2048,
                                           'dim_head': 64, 'dropout': 0.0, 'emb_dropout': 0.0}}
    # Source architectures consume different random draws during reconstruction.
    # Reset before the common action head so the two modality heads start equal.
    torch.manual_seed(42)
    policy = create_policy(world_model=source_model, history_size=3, action_dim=5,
                           predictor_config=factory_config['predictor_config']).to(args.device)
    del source_model, original_state
    dataset, scaler, modality, goal_key = build_real_data(cfg)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42), drop_last=True)
    frozen_before = state_hash(policy.encoder)
    head_before = state_hash(policy.action_predictor)
    optimizer = torch.optim.AdamW(policy.action_predictor.parameters(), lr=1e-4)
    policy.train()
    losses, records = [], []
    reference_batch = None
    for index, batch in enumerate(loader):
        if index >= 2:
            break
        verify_batch(batch, modality, goal_key)
        batch = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        if reference_batch is None:
            reference_batch = {key: batch[key].detach().cpu() for key in (modality, goal_key, 'action')}
        latent = policy.encode({'pixels': batch[modality]}, target='embedding')['embedding']
        goal_latent = policy.encode({'pixels': batch[goal_key]}, target='embedding')['embedding']
        if latent.shape != (2, 3, 1, 192) or goal_latent.shape != (2, 1, 1, 192):
            raise ValueError('Expected actual LeWM global192, exactly one token')
        if not torch.isfinite(latent).all() or not torch.isfinite(goal_latent).all():
            raise ValueError('Nonfinite real LeWM latent')
        means, _ = policy.predict_actions(latent, goal_latent)
        loss = torch.nn.functional.mse_loss(means, batch['action'])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite GCBC MSE loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad for p in policy.action_predictor.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(torch.count_nonzero(g) > 0 for g in gradients):
            raise ValueError('Head gradients nonfinite or all zero')
        if any(p.grad is not None or p.requires_grad for p in policy.encoder.parameters()):
            raise ValueError('Frozen trunk unexpectedly trainable/has gradients')
        lr = float(optimizer.param_groups[0]['lr'])
        if lr <= 0:
            raise ValueError('Actual optimizer LR is zero')
        optimizer.step()
        if state_hash(policy.encoder) != frozen_before:
            raise ValueError('Encoder/projector parameter or buffer changed')
        records.append({'update': index, 'mse': float(loss.detach()), 'actual_lr': lr,
                        'latent_shape': list(latent.shape), 'goal_latent_shape': list(goal_latent.shape),
                        'frozen_parameters_and_buffers_exact': True})
        losses.append(float(loss.detach()))
        state.update(stage='two_head_updates', updates=records, completed_updates=len(records))
        persist()
    if len(losses) != 2 or state_hash(policy.action_predictor) == head_before:
        raise ValueError('Two real head updates were not completed')
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise ValueError('Policy parameters nonfinite after updates')
    policy.eval()
    device_batch = {key: value.to(args.device) for key, value in reference_batch.items()}
    expected_actions = predict(policy, device_batch, modality, goal_key)
    torch.save(reference_batch, output / 'reference_batch.pt')
    torch.save(expected_actions, output / 'reference_actions.pt')
    torch.save(policy.state_dict(), output / 'policy.pt')
    (output / 'config.json').write_text(json.dumps(factory_config, indent=2) + '\n')
    protocol = {'scope': 'frozen LeWM encoder+projector to GCBC head; two-update smoke, no formal SR',
                'modality': modality, 'source_checkpoint': str(source), 'source_sha256': sha_file(source),
                'source_config_sha256': sha_file(source.parent / 'config.yaml'),
                'training_dataset': str(cfg.data.dataset.name),
                'dataset_sha256_status': 'source SHA manifest must be audited by launcher',
                'source_wm_frameskip': 5, 'source_wm_action_chunk_dim': 25,
                'policy_frameskip': 1, 'policy_history': 3, 'policy_action_dim': 5,
                'head_only_training': True, 'latent_dim': 192, 'num_tokens': 1,
                'goal_sampling': [0., 0., 1., 0.], 'seed': 42,
                'learning_rate': 1e-4, 'batch_size': 2, 'updates': records,
                'action_scaler': {'method': 'original SWM ZScoreScaler', 'dataset': str(cfg.data.dataset.name),
                                  'mean': scaler.mean.tolist(), 'std': scaler.std.tolist(), 'eps': scaler.eps},
                'frozen_state_sha256': frozen_before, 'head_before_sha256': head_before,
                'head_after_sha256': state_hash(policy.action_predictor),
                'policy_sha256': sha_file(output / 'policy.pt'),
                'factory_sha256': sha_file(Path(__file__).with_name('lewm_gcbc_adapter.py')),
                'runner_sha256': sha_file(__file__), 'train_pid': os.getpid(),
                'training_precision': 'float32', 'device': args.device}
    import h5py
    dataset_path = Path(str(cfg.data.dataset.name))
    with h5py.File(dataset_path, 'r') as raw:
        data_keys = ('pixels','action') if modality == 'pixels' else ('pcd_xyz','pcd_rgb','action')
        protocol['dataset_contract'] = {
            'bytes': dataset_path.stat().st_size,
            'columns': {key: {'shape':list(raw[key].shape),'dtype':str(raw[key].dtype),
                              'storage_chunks':list(raw[key].chunks) if raw[key].chunks else None}
                        for key in data_keys},
            'attributes': {str(key):str(value) for key,value in raw.attrs.items()},
        }
    manifest = dataset_path.with_name(dataset_path.name+'.manifest.json')
    if manifest.exists():
        protocol['dataset_manifest'] = {'path':str(manifest),'sha256':sha_file(manifest),
                                        'note':'manifest hash only; does not prove HDF5 content SHA'}
    (output / 'training_report.json').write_text(json.dumps(protocol, indent=2) + '\n')
    return protocol


def verify(args, output, state, persist):
    import torch
    import numpy as np
    import gymnasium as gym
    import stable_worldmodel as swm
    from stable_worldmodel.data.normalization import ZScoreScaler
    from omegaconf import OmegaConf
    import hydra
    from fair_reset import ENV_ID, register_fair_environment, SeededPointCloudWrapper
    protocol = json.loads((output / 'training_report.json').read_text())
    if protocol['train_pid'] == os.getpid():
        raise ValueError('Verification requires a fresh process')
    if sha_file(output / 'policy.pt') != protocol['policy_sha256']:
        raise ValueError('Saved policy differs from training report')
    if sha_file(protocol['source_checkpoint']) != protocol['source_sha256']:
        raise ValueError('Source WM checkpoint changed after training')
    policy = swm.wm.utils.load_pretrained(str((output / 'policy.pt').resolve()), cache_dir='/')
    strict_state = torch.load(output / 'policy.pt', map_location='cpu', weights_only=True)
    policy.load_state_dict(strict_state, strict=True)
    if state_hash(policy.encoder) != protocol['frozen_state_sha256'] or state_hash(policy.action_predictor) != protocol['head_after_sha256']:
        raise ValueError('Strict reload altered model parameter/buffer state')
    policy.to(args.device).eval().requires_grad_(False)
    modality, goal_key = protocol['modality'], 'goal_pixels' if protocol['modality'] == 'pixels' else 'goal_pcd'
    batch = torch.load(output / 'reference_batch.pt', weights_only=True)
    verify_batch(batch, modality, goal_key)
    batch = {key: value.to(args.device) for key, value in batch.items()}
    actual = predict(policy, batch, modality, goal_key)
    expected = torch.load(output / 'reference_actions.pt', weights_only=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    state.update(stage='fresh_process_reload_passed', strict_reload=True,
                 action_max_diff=float((actual - expected).abs().max()))
    persist()
    cfg = OmegaConf.load(Path(protocol['source_checkpoint']).parent / 'config.yaml')
    original_preproc = hydra.utils.instantiate(cfg.data.preprocessor)
    scaler_cfg = protocol['action_scaler']
    scaler = ZScoreScaler(mean=scaler_cfg['mean'], std=scaler_cfg['std'], eps=scaler_cfg['eps'])
    register_fair_environment()
    env = gym.make(ENV_ID, ob_type='states', env_type='single', width=224, height=224,
                   multiview=False, mode='task', max_episode_steps=10,
                   terminate_at_goal=False, visualize_info=False)
    if modality == 'pcd':
        env = SeededPointCloudWrapper(env, num_points=1200, cameras=('front_pixels',),
                                     foreground_only=True,
                                     crop_box=((0.10,-0.50,-0.02),(0.75,0.50,0.60)),
                                     voxel_size=0.005, cube_fraction=0.25)
    records = []
    state['native_records'] = records
    try:
        _, info = env.reset(seed=args.native_seed, options={'task_id': 1, 'render_goal': True})
        base = env.unwrapped
        if int(base.np_random_seed) != args.native_seed or int(base.cur_task_id) != 1:
            raise ValueError('Fair native reset contract failed')
        reset_record = {
            'seed':args.native_seed, 'actual_np_random_seed':int(base.np_random_seed),
            'initial_qpos_qvel_sha256':hashlib.sha256(base._data.qpos.tobytes()+base._data.qvel.tobytes()).hexdigest(),
            'initial_control_sha256':hashlib.sha256(base._data.ctrl.tobytes()).hexdigest(),
            'initial_mocap_sha256':hashlib.sha256(base._data.mocap_pos.tobytes()+base._data.mocap_quat.tobytes()).hexdigest(),
            'goal_rgb_sha256':hashlib.sha256(np.asarray(base._cur_goal_rendered).tobytes()).hexdigest(),
        }
        state['native_reset'] = reset_record
        persist()
        if args.reference_output:
            reference = json.loads((Path(args.reference_output)/'verify_status.json').read_text())
            if reference['status'] != 'passed' or reference['native_reset'] != reset_record:
                raise ValueError('Cross-modality native initial state/control/goal differs')
            state['cross_modality_reset_gate'] = 'passed'
            persist()

        def native_frame(info, goal=False):
            if modality == 'pixels':
                image = base._cur_goal_rendered if goal else base.render(camera='front_pixels')
                if image.shape != (224,224,3) or image.dtype != np.uint8:
                    raise ValueError('Native RGB dtype/shape contract')
                sample = {'pixels': torch.from_numpy(image.copy()).permute(2,0,1)[None]}
            else:
                prefix = 'goal_' if goal else ''
                xyz, rgb = info[prefix+'pcd_xyz'], info[prefix+'pcd_rgb']
                if rgb.dtype != np.uint8 or xyz.shape != (1200,3) or rgb.shape != (1200,3):
                    raise ValueError('Native PCD color/geometry contract')
                sample = {'pcd_xyz': torch.from_numpy(xyz.copy())[None],
                          'pcd_rgb': torch.from_numpy(rgb.copy())[None]}
            value = original_preproc(sample)[modality][0]
            if not torch.isfinite(value).all():
                raise ValueError('Native preprocessed frame nonfinite')
            return value

        goal = native_frame(info, goal=True)[None,None].to(args.device)
        history = deque([native_frame(info)], maxlen=3)
        with (output / 'native_actions.jsonl').open('x', buffering=1) as action_log:
            for step in range(4):
                if step < 2:
                    action = np.zeros(5, dtype=np.float32)
                    normalized = None
                else:
                    with torch.inference_mode():
                        normalized = policy.get_action({'pixels': torch.stack(list(history))[None].to(args.device), 'goal': goal})
                    if normalized.shape != (1,5) or not torch.isfinite(normalized).all():
                        raise ValueError('Native policy prediction shape/nonfinite failure')
                    action = np.asarray(scaler.inverse_transform(normalized.cpu().numpy())[0], dtype=np.float32)
                if action.shape != (5,) or not np.isfinite(action).all():
                    raise ValueError('Invalid physical native action')
                raw_in_bounds = bool(env.action_space.contains(action))
                sent = action.copy()
                if not np.array_equal(action, sent):
                    raise ValueError('Runner changed raw physical policy action')
                _, reward, terminated, truncated, info = env.step(sent)
                control = base._data.ctrl.copy()
                if control.shape != (7,) or not np.isfinite(control).all() or not np.isfinite(base._data.qpos).all() or not np.isfinite(base._data.qvel).all():
                    raise ValueError('Native actuator/state gate failed')
                event = {'step':step, 'phase':'zero_warmup' if step<2 else 'policy',
                         'normalized_action':normalized.cpu().tolist() if normalized is not None else None,
                         'action_raw':action.tolist(), 'action_sent':sent.tolist(),
                         'action_raw_in_bounds':raw_in_bounds, 'control_7d':control.tolist(),
                         'reward':float(reward), 'terminated':bool(terminated), 'truncated':bool(truncated)}
                records.append(event)
                action_log.write(json.dumps(event, allow_nan=False)+'\n')
                action_log.flush()
                os.fsync(action_log.fileno())
                history.append(native_frame(info))
                persist()
                if terminated or truncated:
                    raise ValueError('Native smoke ended before four requested interface steps')
    finally:
        env.close()
    return {'strict_reload':True, 'fresh_process':True, 'native_steps':4,
            'native_seed':args.native_seed, 'task_id':1, 'policy_steps':2,
            'warmup_steps':2, 'raw_bounds_diagnostic_only':True,
            'scope':'custom mixed-data frozen-LeWM GCBC interface smoke, no official SR'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['train','verify'])
    parser.add_argument('--source-checkpoint')
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--native-seed', type=int, default=93020001)
    parser.add_argument('--reference-output', help='Passed first modality smoke for exact reset comparison')
    args = parser.parse_args()
    if args.stage == 'train' and not args.source_checkpoint:
        parser.error('--source-checkpoint required for train')
    output = Path(args.output).resolve()
    if args.stage == 'train':
        output.mkdir(parents=True, exist_ok=False)
    elif not (output/'training_report.json').exists():
        parser.error('verify needs a completed train output')
    state = {'status':'running', 'stage':args.stage, 'exit_code':None,
             'started_at':datetime.now(timezone.utc).isoformat(), 'pid':os.getpid(),
             'arguments':vars(args), 'completed_updates':0, 'error':None}
    status_path = output/(args.stage+'_status.json')
    if status_path.exists():
        parser.error('Stage already has state; use a fresh output or preserve previous attempt')
    def persist():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        temp=status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
        temp.replace(status_path)
    persist()
    code=1
    with (output/(args.stage+'.log')).open('x',buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        handler=logging.StreamHandler(log)
        logging.getLogger().addHandler(handler)
        try:
            lewm=Path(args.repo_root).resolve()/'lewm_pcd'
            sys.path.insert(0,str(lewm))
            sys.path.insert(0,str(lewm/'third_party/stable-worldmodel'))
            import stable_pretraining
            import torch
            from omegaconf import OmegaConf
            if not OmegaConf.has_resolver('eval'):
                OmegaConf.register_new_resolver('eval',eval)
            torch.set_num_threads(2)
            torch.manual_seed(42)
            result=train(args,output,state,persist) if args.stage=='train' else verify(args,output,state,persist)
            state.update(status='passed',exit_code=0,result=result)
            code=0
        except Exception as exc:
            state.update(status='failed',exit_code=1,error=str(exc),traceback=traceback.format_exc())
            traceback.print_exc()
        finally:
            logging.getLogger().removeHandler(handler)
            persist()
            (output/(args.stage+'_exit_code')).write_text(str(code)+'\n')
    print(json.dumps({'stage':args.stage,'status':state['status'],'exit_code':code,'output':str(output)}))
    return code


if __name__=='__main__':
    raise SystemExit(main())

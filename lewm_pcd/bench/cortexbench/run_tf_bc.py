"""CortexBench's TriFinger imitation benchmark (eai-vc cortexbench/trifinger_vc, third_party/eai-vc, unchanged on disk)
with the TriFinger LeWM encoders (scripts/run_tf_pipeline.sh), the way run_bc.py runs the MuJoCo suites: the official
bc_experiments/train_bc.py (config bc_default; the trifinger_vc README's settings algo.freeze_pretrained_rep=true
rep_to_policy=none) with runtime patches.
 1. Encoders: algo.pretrained_rep=lewm_tf_pix / lewm_tf_pcd loads the LeWM encoder + projector (lewm_cortex.py;
    checkpoint TF_CKPT_PIX / TF_CKPT_PCD, default $STABLEWM_HOME/checkpoints/tf_cube_{pix,pcd_dexart}; epoch TF_EPOCH,
    default 10) instead of a vc_models config; the names join vc_models' model zoo. 2D transform: the BC dataset's
    [0, 1] float images and the rollouts' PIL images take one path: float tensor -> 224 x 224 bicubic (antialiased)
    -> ImageNet normalisation (LeWM's training input).
 2. Frozen encoder: it stays in eval mode. The official loop calls encoder.train() every epoch, which would make the
    projector's BatchNorm use and update batch statistics on the demos (VC-1 has no BatchNorm); the MuJoCo harness
    embeds with the encoder in eval mode, so this keeps the two benchmarks alike.
 3. Logging only, scores unchanged (and torch.load with weights_only=False for the demo pickles, as torch < 2.6): the
    per-rollout npz / png dumps and gifs are not written (GBs per run), and camera180 / camera300 are not rendered
    (zeros; only camera60 feeds the policy). Every evaluation appends its
    average and running-maximum scaled success (train / test demos) to metrics.jsonl in the run dir.
 4. 3D (lewm_tf_pcd): camera60's point cloud instead of its image (envs.trifinger.env.Camera60PCD, the code the LeWM
    data's point clouds come from: 1200 points, 5 mm voxels, 25 % on the cube, a 224 x 224 depth render), for the demos
    (each row's joint angles and cube pose set in CortexBench's MoveCubeEnv, cached as
    <demos>/../pcd/<task>/demo-XXXX.npy by `run_tf_bc.py prep` before any run) and in the rollouts (rendered from the
    current state at every observation; the 2D image is the platform's latest 10 fps camera frame, up to 0.1 s older).
    A demo row's cube pose is the one the demo recorded, from that camera frame, so in a demo's point clouds the cube
    can lag the fingers by up to 0.1 s (move only: reach's cube does not move).
`prep` first downloads the official demos when they are missing (train_utils.download_demos_if_needed, what every
official run calls). Run from the demo home (assets/bc_demos/*.json, assets/data/trifinger-demos/), as the official
script expects; run_tf_grid.sh sets it up and runs the grid. One run:
    cd $STABLEWM_HOME/datasets/cortexbench/trifinger && PYTHONPATH=<lewm_pcd>:<lewm_pcd>/bench/cortexbench:\
<lewm_pcd>/third_party/stable-worldmodel:<trifinger_vc>/src:<eai-vc>/vc_models/src $PY <this file> task=move_cube \
        algo.pretrained_rep=lewm_tf_pix seed=100 no_wandb=True algo.freeze_pretrained_rep=true rep_to_policy=none \
        hydra.run.dir=<run dir> hydra.job.chdir=False
"""

import json
import os
import runpy
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LEWM = os.path.normpath(os.path.join(HERE, '..', '..'))
TF_VC = os.path.join(LEWM, 'third_party', 'eai-vc', 'cortexbench', 'trifinger_vc')
HOME = os.environ['STABLEWM_HOME']
CKPT = {'lewm_tf_pix': os.environ.get('TF_CKPT_PIX', os.path.join(HOME, 'checkpoints', 'tf_cube_pix')),
        'lewm_tf_pcd': os.environ.get('TF_CKPT_PCD', os.path.join(HOME, 'checkpoints', 'tf_cube_pcd_dexart'))}
EPOCH = int(os.environ.get('TF_EPOCH', '10'))
PCD_AFFINE = dict(center=(0.0, 0.0, 0.12), scale=4.0, use_rgb=True)  # config/train/data/tf_cube_pcd.yaml

_torch_load = torch.load


def torch_load(*args, **kwargs):  # the official demos are pickled dicts of numpy arrays; torch >= 2.6 loads
    kwargs.setdefault('weights_only', False)  # weights only by default
    return _torch_load(*args, **kwargs)


torch.load = torch_load

import vc_models  # noqa: E402

vc_models.vc_model_zoo.extend(CKPT)  # model_utils.MODEL_NAMES is this list

import trifinger_simulation.camera as tf_camera  # noqa: E402
import trifinger_vc.utils.data_utils as d_utils  # noqa: E402
import trifinger_vc.utils.dataset as tv_dataset  # noqa: E402
import trifinger_vc.utils.encoder_model as tv_encoder  # noqa: E402
import trifinger_vc.utils.model_utils as m_utils  # noqa: E402
import trifinger_vc.utils.train_utils as t_utils  # noqa: E402
import trifinger_vc.algos.bc_finetune as bc_finetune  # noqa: E402
from trifinger_vc.trifinger_envs.cube_reach import CubeReachEnv  # noqa: E402
from trifinger_vc.trifinger_envs.gym_cube_env import MoveCubeEnv  # noqa: E402

PREP = sys.argv[1:2] == ['prep']  # `prep`: compute and cache every demo's point clouds (before parallel runs read them)
REP = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('algo.pretrained_rep=')), None)
IS_PCD = PREP or REP == 'lewm_tf_pcd'
SEED = int(next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('seed=')), '0'))


# ------------------------------------------------------------------------------------------------ 1. encoders
class Img224:
    """PIL image or [0, 1] float tensor (C, H, W) -> 224 x 224 ImageNet-normalised tensor, one path for both."""

    def __call__(self, img):
        import torchvision.transforms.functional as F
        if not torch.is_tensor(img):
            img = F.pil_to_tensor(img).float() / 255.0
        img = F.resize(img, [224, 224], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
        return F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def get_vc_model_and_transform(model_name, device='cpu', use_compression_layer=False):
    if model_name not in CKPT:
        return _get_vc(model_name, device=device, use_compression_layer=use_compression_layer)
    import lewm_cortex
    if model_name == 'lewm_tf_pcd':
        with torch.random.fork_rng(devices=[]):  # as load_lewm: loading leaves the run's global RNG alone (2D / 3D
            model, dim, transform, _ = lewm_cortex.load_lewm_pcd(CKPT[model_name], epoch=EPOCH, **PCD_AFFINE)  # paired)
    else:
        model, dim, _, _ = lewm_cortex.load_lewm(CKPT[model_name], epoch=EPOCH)
        transform = Img224()
    return model, transform, dim


_get_vc = m_utils.get_vc_model_and_transform
m_utils.get_vc_model_and_transform = get_vc_model_and_transform
tv_dataset.get_vc_model_and_transform = get_vc_model_and_transform


# ------------------------------------------------------------------------------------------ 2. frozen encoder
def encoder_train(self, mode=True):
    torch.nn.Module.train(self, mode)
    self.pretrained_rep_model.eval()
    return self


tv_encoder.EncoderModel.train = encoder_train


# --------------------------------------------------------------------------------------------- 3. logging only
t_utils.save_demo_to_file = lambda *args, **kwargs: None
d_utils.save_gif = lambda *args, **kwargs: None
_get_images = tf_camera.CameraArray.get_images


def get_images(self, *args, **kwargs):
    if not isinstance(self, tf_camera.TriFingerCameras):
        return _get_images(self, *args, **kwargs)
    if IS_PCD:  # the policy reads the point cloud (4.) instead
        return [np.zeros((c._height, c._width, 3), dtype=np.uint8) for c in self.cameras]
    first = self.cameras[0].get_image(*args, **kwargs)
    return [first] + [np.zeros_like(first) for _ in self.cameras[1:]]


tf_camera.CameraArray.get_images = get_images
_sim_rollout = bc_finetune.BCFinetune.sim_rollout


def sim_rollout(self, sim_dir, outer_i, max_demo_per_diff=10):
    out = _sim_rollout(self, sim_dir, outer_i, max_demo_per_diff=max_demo_per_diff)
    rec = {'epoch': int(outer_i)}
    for env_name, splits in out.items():
        for split, d in splits.items():
            for k in ('diff-110_avg_scaled_reward', 'diff-110_max_avg_scaled_reward'):
                if k in d:
                    rec[f'{env_name}/{split}/{k}'] = float(d[k])
    with open(os.path.join(os.path.dirname(sim_dir), 'metrics.jsonl'), 'a') as f:
        f.write(json.dumps(rec) + '\n')
    return out


bc_finetune.BCFinetune.sim_rollout = sim_rollout

# ----------------------------------------------------------------------------------------------- 4. 3D (point clouds)
if IS_PCD:
    import zlib

    from envs.trifinger.env import Camera60PCD, TriFingerEnv  # the point clouds of the LeWM data (TriFingerPCDWrapper)

    def demo_pcds(demo_dir, demo):
        """Every row's point cloud of a demo (cached): the row's joint angles and cube pose set in CortexBench's
        MoveCubeEnv (TriFingerEnv.set_state)."""
        task, name = demo_dir.rstrip('/').split('/')[-3:-1]
        cache = os.path.join('assets', 'data', 'pcd', task, f'{name}.npy')
        if os.path.exists(cache):
            return np.load(cache)
        if not PREP:  # a second platform in a run's process: computed once, by `prep`, before the runs
            sys.exit(f'{cache} missing: run `run_tf_bc.py prep` first')
        global _SCENE
        if '_SCENE' not in globals():
            _SCENE = TriFingerEnv(task='move')
        env = _SCENE
        cam = Camera60PCD(env._cid, env.finger_id, env.cube_id, env.moving_links,
                          np.random.default_rng(zlib.crc32(f'{task}/{name}'.encode())))
        out = []
        for i in range(len(demo['robot_pos'])):
            env.set_state(np.concatenate([demo['robot_pos'][i], demo['o_pos_cur'][i] / 100.0, demo['o_ori_cur'][i]]),
                          np.zeros(15))
            out.append(cam())
        out = np.stack(out).astype(np.float32)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = f'{cache}.{os.getpid()}.tmp.npy'
        np.save(tmp, out)
        os.replace(tmp, cache)
        return out

    _get_traj_list = t_utils.get_traj_list

    def get_traj_list(demo_root_dir, demo_stats_list, obj_state_type):
        demos = _get_traj_list(demo_root_dir, demo_stats_list, obj_state_type)
        for st, demo in zip(demo_stats_list, demos):
            demo['image_60'] = demo_pcds(os.path.join(demo_root_dir, 'data', 'trifinger-demos', st['path']), demo)
        return demos

    t_utils.get_traj_list = get_traj_list

    def add_new_traj(self, demo_dir):  # BCFinetuneDataset.add_new_traj with the demo's point clouds for its images
        path = os.path.join(demo_dir, 'downsample.pth')
        if not os.path.exists(path):
            print(f'{path} not found')
            return
        demo = torch.load(path)
        pcds = demo_pcds(demo_dir, demo)
        o_goal_pos = torch.FloatTensor(demo['o_pos_cur'][-1]).to(self.device)
        o_goal_pos_rel = o_goal_pos - torch.FloatTensor(demo['o_pos_cur'][0]).to(self.device)
        goal = self.preproc_transform(pcds[-1]).to(self.device)
        for i in range(demo['o_pos_cur'].shape[0] - 1):
            action = torch.FloatTensor(demo['delta_ftpos'][i])[: self.n_fingers_to_move * 3]
            for _ in range(self.times_to_use_demo):
                self.dataset.append({
                    'input': {'ft_state': torch.FloatTensor(demo['ft_pos_cur'][i]).to(self.device),
                              'rgb_img_preproc': self.preproc_transform(pcds[i]).to(self.device),
                              'rgb_img_preproc_goal': goal, 'o_goal_pos': o_goal_pos, 'o_goal_pos_rel': o_goal_pos_rel},
                    'output': {'action': action.to(self.device)},
                })

    tv_dataset.BCFinetuneDataset.add_new_traj = add_new_traj

    def encode_img(self, img):  # EncoderModel.encode_img for a (1200, 6) point cloud
        x = self.transform(img).unsqueeze(0).to(next(self.parameters()).device)
        return self.forward(x)[0].detach()

    tv_encoder.EncoderModel.encode_img = encode_img

    def with_pcd(create):
        def _create_observation(self, t):
            obs = create(self, t)
            if 'camera_observation' in obs:
                if getattr(self, '_pcd_cam', None) is None:
                    p = self.platform
                    self._pcd_cam = Camera60PCD(
                        p.simfinger._pybullet_client_id, p.simfinger.finger_id, p.cube._object_id,
                        p.simfinger.pybullet_link_indices + p.simfinger.pybullet_tip_link_indices,
                        np.random.default_rng(SEED))
                obs['camera_observation']['camera60']['image'] = self._pcd_cam()
            return obs
        return _create_observation

    MoveCubeEnv._create_observation = with_pcd(MoveCubeEnv._create_observation)
    CubeReachEnv._create_observation = with_pcd(CubeReachEnv._create_observation)

if __name__ == '__main__' and PREP:
    t_utils.download_demos_if_needed(os.path.join('assets', 'data'))
    for js in sorted(os.listdir(os.path.join('assets', 'bc_demos'))):
        info = json.load(open(os.path.join('assets', 'bc_demos', js)))
        for st in info['train_demo_stats'] + info['test_demo_stats']:
            d = os.path.join('assets', 'data', 'trifinger-demos', st['path'])
            demo_pcds(d, torch.load(os.path.join(d, 'downsample.pth')))
        print(js, len(info['train_demo_stats']) + len(info['test_demo_stats']), 'demos: point clouds cached', flush=True)
elif __name__ == '__main__':
    script = os.path.join(TF_VC, 'bc_experiments', 'train_bc.py')
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name='__main__')

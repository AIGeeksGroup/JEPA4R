"""Frozen LeWM latents of OGBench puzzle states (3x3, 4x4, 4x5), shared by extract_feats.py (the training set) and
run_dqc.py (the evaluation): the camera and point clouds of the LeWM data (envs/ogb_puzzle_env.py: front_pixels 224, the
opaque arm of ob_type='states'; 2D anti-aliased, 3D the same camera's depth with 1200 points) and the encoders as
CortexBench loads them (bench/cortexbench/lewm_cortex.py: 2D pixels / 255 normalised with ImageNet's mean / std; 3D
(xyz - center) * scale with rgb / 255), their output the projector's 192-d latent (LeWM's latent).
envs/ogb_puzzle_env.py is loaded by its path: the DQC code (third_party/dqc) has packages named envs and utils too and
comes first on run_dqc.py's sys.path (the module imports nothing from the repo).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

LEWM = Path(__file__).resolve().parents[2]  # lewm_pcd
sys.path.insert(0, str(LEWM / 'bench' / 'cortexbench'))  # lewm_cortex
if str(LEWM) not in sys.path:
    sys.path.append(str(LEWM))  # module.py (the projector), after the DQC code's packages
from lewm_cortex import IMAGENET, build_lewm_embedding, load_lewm_pcd  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PCD_CENTER, PCD_SCALE = (0.425, 0.0, 0.2), 4.0  # = config/train/data/ogb_pcd.yaml (puzzle_3x3_pcd.yaml keeps it)


def load_puzzle():
    """envs/ogb_puzzle_env.py as the module ogb_puzzle_env (by path, see the module docstring)."""
    if 'ogb_puzzle_env' not in sys.modules:
        spec = importlib.util.spec_from_file_location('ogb_puzzle_env', LEWM / 'envs' / 'ogb_puzzle_env.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules['ogb_puzzle_env'] = mod
        spec.loader.exec_module(mod)
    return sys.modules['ogb_puzzle_env']


class Encoders:
    """The frozen 2D / 3D LeWM encoders + projectors of the given checkpoint directories (either may be None)."""

    def __init__(self, ckpt_2d=None, ckpt_3d=None, epoch=10):
        self.m2 = build_lewm_embedding(ckpt_2d, epoch).to(DEV) if ckpt_2d else None
        self.m3, self.tf = None, None
        if ckpt_3d:
            m3, _, self.tf, _ = load_lewm_pcd(ckpt_3d, epoch, center=PCD_CENTER, scale=PCD_SCALE)  # tf: PcdTransform
            self.m3 = m3.to(DEV)
        self.mean = torch.tensor(IMAGENET[0], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET[1], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def z2d(self, pix, bs=256):
        out = []
        for i in range(0, len(pix), bs):
            x = torch.from_numpy(pix[i: i + bs]).to(DEV).permute(0, 3, 1, 2).float() / 255
            out.append(self.m2((x - self.mean) / self.std).float().cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def z3d(self, xyz, col, bs=64):
        out = []
        for i in range(0, len(xyz), bs):
            x = torch.stack([self.tf(np.concatenate([a, b], 1)) for a, b in zip(xyz[i: i + bs], col[i: i + bs])])
            out.append(self.m3(x.to(DEV)).float().cpu())
        return torch.cat(out).numpy()


class BaselineEncoder:
    """A JEPA-family baseline as CortexBench loads it (bench/cortexbench/jepa_baselines.py: the frozen model and its own
    transforms, Resize + ToTensor + Normalize applied to a PIL image as run_bc_scene.py does): vjepa21 (a checkpoint
    file, 768-d at 384), visreg (a checkpoint file, 768-d at 224), lejepa (a checkpoint directory, 512-d), pldm (a
    checkpoint directory, 192-d). prep() is the PIL
    resize of 224 frames to the encoder's input size (identity at 224), encode() the rest on the GPU (/ 255 =
    ToTensor, Normalize, the model)."""

    LOADERS = {'vjepa21': 'load_vjepa21', 'visreg': 'load_visreg', 'lejepa': 'load_lejepa', 'pldm': 'load_pldm'}
    FILE = ('vjepa21', 'visreg')  # loaders that take a checkpoint file (the others a directory and an epoch)

    def __init__(self, name, path, epoch=10):
        import jepa_baselines as jb  # bench/cortexbench, on sys.path above
        load = getattr(jb, self.LOADERS[name])
        model, self.dim, tf, _ = load(path) if name in self.FILE else load(path, epoch)
        self.model = model.to(DEV)
        resize = [t for t in tf.transforms if isinstance(t, T.Resize)][0]
        norm = [t for t in tf.transforms if isinstance(t, T.Normalize)][0]
        self.size = tuple(resize.size) if isinstance(resize.size, (list, tuple)) else (resize.size, resize.size)
        self.mean = torch.tensor(norm.mean, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(norm.std, device=DEV).view(1, 3, 1, 1)

    @staticmethod
    def prep_frames(pix, size):
        """uint8 [N, H, W, 3] -> the size the encoder takes, resized as its transforms resize a PIL image."""
        if tuple(pix.shape[1:3]) == tuple(size):
            return pix
        return np.stack([np.asarray(Image.fromarray(p).resize((size[1], size[0]), Image.BICUBIC)) for p in pix])

    @torch.no_grad()
    def encode(self, pix, bs=64):
        out = []
        for i in range(0, len(pix), bs):
            x = torch.from_numpy(np.ascontiguousarray(pix[i: i + bs])).to(DEV).permute(0, 3, 1, 2).float() / 255
            out.append(self.model((x - self.mean) / self.std).float().cpu())
        return torch.cat(out).numpy()


class BaselineObs:
    """The evaluation-time observation for a baseline encoder: the current frame of the LeWM data's camera (the 2D
    frame LeWMObs gives LeWM 2D) -> BaselineEncoder -> its latent, the true lights appended, as
    extract_feats_baselines.py builds the training set's observations."""

    def __init__(self, env, kind, path, epoch):
        torch.set_num_threads(2)
        self.pdat, self.u = load_puzzle(), env.unwrapped
        self.cam = self.pdat.Camera(self.u._model, self.u._num_buttons)
        self.enc = BaselineEncoder(kind, path, epoch)

    def __call__(self):
        self.cam.r2d.update_scene(self.u._data, camera=self.pdat.CAMERA)
        pix = BaselineEncoder.prep_frames(self.cam.r2d.render().copy()[None], self.enc.size)
        z = self.enc.encode(pix)[0]
        return np.concatenate([z, self.u._cur_button_states.astype(np.float32)])


class LeWMObs:
    """The evaluation-time observation: the current frame of the same camera -> the frozen encoder -> the 192-d latent,
    with the true lights appended (lights=True; 9 on puzzle-3x3, one per button), as the training set's observations
    (extract_feats.py)."""

    def __init__(self, env, kind, ckpt, epoch, lights=True):
        torch.set_num_threads(2)
        self.pdat, self.kind, self.lights = load_puzzle(), kind, lights
        self.u = env.unwrapped
        self.cam = self.pdat.Camera(self.u._model, self.u._num_buttons)
        self.enc = Encoders(ckpt_2d=ckpt if kind == 'lewm2d' else None, ckpt_3d=ckpt if kind == 'lewm3d' else None, epoch=epoch)
        self.rng = np.random.default_rng(12345)

    def __call__(self):
        cam, data = self.cam, self.u._data
        if self.kind == 'lewm2d':
            cam.r2d.update_scene(data, camera=self.pdat.CAMERA)
            z = self.enc.z2d(cam.r2d.render().copy()[None])[0]
        else:
            _, rgb, depth, seg, _, R, t = cam.render(data)
            xyz, col = self.pdat.point_cloud(cam, rgb, depth, seg, R, t, self.rng)
            z = self.enc.z3d(xyz[None], col[None])[0]
        if not self.lights:
            return z
        return np.concatenate([z, self.u._cur_button_states.astype(np.float32)])

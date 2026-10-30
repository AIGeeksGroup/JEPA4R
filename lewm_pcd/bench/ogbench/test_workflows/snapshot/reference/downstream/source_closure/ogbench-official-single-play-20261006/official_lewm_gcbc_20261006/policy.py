"""Frozen LeWM single-frame representation + official-GCBC-style mean MLP.

The head follows pinned OGBench572788d GCActor: GELU(tanh approximation),
fan-average uniform kernels (hidden scale1, mean scale.01), zero biases,
512x3 hidden layers, no normalization/dropout/tanh-squashed output.
Torch seed0 reproduces this Torch head; it is not Flax PRNG equivalence.
"""
import math
from collections.abc import Mapping

import torch
from torch import nn


class FrozenLeWMGCBC(nn.Module):
    def __init__(self,encoder,modality='pixels',seed=0):
        super().__init__()
        if modality not in ('pixels','pcd'): raise ValueError('Unknown real input modality')
        self.encoder=encoder; self.modality=modality
        self.history_size=1; self.action_dim=5; self.latent_dim=192; self.fixed_std=1.0
        # Local CPU head RNG is independent of encoder loading and restored on exit.
        with torch.random.fork_rng(devices=[]):
            generator=torch.Generator(device='cpu').manual_seed(int(seed))
            self.action_head=nn.Sequential(nn.Linear(384,512),nn.GELU(approximate='tanh'),
                nn.Linear(512,512),nn.GELU(approximate='tanh'),
                nn.Linear(512,512),nn.GELU(approximate='tanh'),nn.Linear(512,5))
            for index,layer in enumerate(self.action_head):
                if isinstance(layer,nn.Linear):
                    scale=.01 if index==6 else 1.0
                    # Xavier uniform gain sqrt(scale) is fan_avg variance scaling.
                    nn.init.xavier_uniform_(layer.weight,gain=math.sqrt(scale),generator=generator)
                    nn.init.zeros_(layer.bias)

    def _validate_frames(self,current,goal):
        rank=4 if self.modality=='pixels' else 3
        if not torch.is_tensor(current) or not torch.is_tensor(goal): raise ValueError('Preprocessed tensors required')
        if current.ndim!=rank or goal.ndim!=rank or current.shape!=goal.shape or len(current)==0:
            raise ValueError('Exactly one current and one goal frame, matching batch/shape, no time axis')
        if (self.modality=='pixels' and current.shape[1]!=3) or (self.modality=='pcd' and current.shape[-1]!=6):
            raise ValueError('RGB requires BCHW3; PCD requires BN6 real xyz+RGB features')
        if not current.is_floating_point() or not goal.is_floating_point() or not torch.isfinite(current).all() or not torch.isfinite(goal).all():
            raise ValueError('Finite preprocessed float frames required')

    def forward(self,current,goal):
        self._validate_frames(current,goal)
        # Two real forwards every call; features are never cached for training.
        x=self.encoder(current).logits; g=self.encoder(goal).logits
        if x.shape!=(len(current),192) or g.shape!=x.shape:
            raise ValueError('Actual encoder+projector must output global192')
        mean=self.action_head(torch.cat((x,g),dim=-1))
        if mean.shape!=(len(current),5) or not torch.isfinite(mean).all(): raise ValueError('Nonfinite/wrong action means')
        return mean

    def get_action(self,info):
        if not isinstance(info,Mapping) or 'pixels' not in info or 'goal' not in info:
            raise ValueError('Current pixels alias and real same-modality goal required')
        # Deterministic mean. Native official evaluate applies its original clip.
        return self(info['pixels'],info['goal'])


def create_policy(source_checkpoint=None,world_model=None,modality='pixels',latent_dim=192,
                  history_size=1,action_dim=5,seed=0):
    """Hydra factory for full-policy strict state_dict reload, source WM separate."""
    if int(latent_dim)!=192 or int(history_size)!=1 or int(action_dim)!=5:
        raise ValueError('Single-frame global192/current+goal/mean5 contract required')
    from lewm_gcbc_adapter import LeWMFrozenEncoder
    encoder=LeWMFrozenEncoder(source_checkpoint=source_checkpoint,world_model=world_model,latent_dim=192)
    return FrozenLeWMGCBC(encoder,modality=modality,seed=seed)


def unit_gaussian_nll(means,raw_actions):
    """Batch-mean NLL, action-dimension sum, fixed std1 in published Box units."""
    if means.ndim!=2 or means.shape[-1]!=5 or means.shape!=raw_actions.shape or len(means)==0:
        raise ValueError('Current normalized-Box means/actions must both be B5')
    if not torch.isfinite(means).all() or not torch.isfinite(raw_actions).all() or (raw_actions.abs()>1).any():
        raise ValueError('Finite original NPZ normalized[-1,1] actions required; no ZScore')
    # Trainer supplies float32 under AMP; float64 remains available for fixtures.
    return (.5*(means-raw_actions).square().sum(dim=-1)+2.5*math.log(2*math.pi)).mean()


def make_optimizer(policy):
    expected={id(p) for p in policy.action_head.parameters()}
    actual={id(p) for p in policy.parameters() if p.requires_grad}
    if expected!=actual: raise ValueError('Trainable parameters must be exactly action_head')
    return torch.optim.Adam(policy.action_head.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)

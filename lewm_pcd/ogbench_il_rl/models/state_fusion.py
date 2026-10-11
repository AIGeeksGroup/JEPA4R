"""Frozen original LeWM plus official28 state, using the original GCBC head."""
import math
from collections.abc import Mapping

import torch
from torch import nn
from . import policy as base
unit_gaussian_nll, make_optimizer = base.unit_gaussian_nll, base.make_optimizer


class FrozenStateVisualGCBC(base.FrozenLeWMGCBC):
    def __init__(self, encoder, modality='pixels', seed=0, goal_state_mode='full', latent_dim=192):
        if goal_state_mode not in ('full', 'object_xyz'):
            raise ValueError('Goal state mode must be full or object_xyz')
        if latent_dim not in (192, 384) or (latent_dim == 384 and modality != 'pixels'):
            raise ValueError('LeWM192 or RGB DINO384 latent required')
        super().__init__(encoder, modality=modality, seed=seed)
        self.goal_state_mode = goal_state_mode
        self.latent_dim = latent_dim
        self.state_dim = 28
        # Reinitialize the whole fresh head with the original local RNG/math.
        with torch.random.fork_rng(devices=[]):
            self.action_head[0] = nn.Linear(2 * (latent_dim + self.state_dim), 512)
            generator = torch.Generator(device='cpu').manual_seed(int(seed))
            for index, layer in enumerate(self.action_head):
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=math.sqrt(.01 if index == 6 else 1.), generator=generator)
                    nn.init.zeros_(layer.bias)

    def forward(self, current, goal, current_state, goal_state):
        self._validate_frames(current, goal)
        for state in (current_state, goal_state):
            if not torch.is_tensor(state) or state.shape != (len(current), 28) or state.dtype != torch.float32 or state.device != current.device or not torch.isfinite(state).all():
                raise ValueError('Same-device finite original float32 current/goal state B28 required')
        if self.goal_state_mode == 'object_xyz':
            goal_state = goal_state.clone()
            goal_state[:, :19] = 0
            goal_state[:, 22:] = 0
        x = self.encoder(current).logits; g = self.encoder(goal).logits
        if x.shape != (len(current), self.latent_dim) or g.shape != x.shape:
            raise ValueError('Actual frozen encoder must output the declared global latent')
        mean = self.action_head(torch.cat((x, current_state, g, goal_state), dim=-1))
        if mean.shape != (len(current), 5) or not torch.isfinite(mean).all():
            raise ValueError('Nonfinite/wrong action means')
        return mean

    def get_action(self, info):
        keys = ('pixels', 'goal', 'current_state', 'goal_state')
        if not isinstance(info, Mapping) or any(key not in info for key in keys):
            raise ValueError('Same-pair current/goal visual and official28 state required')
        return self(*(info[key] for key in keys))


def create_policy(source_checkpoint=None, world_model=None, modality='pixels', latent_dim=192,
                  history_size=1, action_dim=5, seed=0, goal_state_mode='full', model_folder=None):
    if model_folder is None:
        encoder = base.create_policy(source_checkpoint=source_checkpoint, world_model=world_model,
            modality=modality, latent_dim=latent_dim, history_size=history_size, action_dim=action_dim, seed=seed).encoder
    else:
        if source_checkpoint is not None or world_model is not None:
            raise ValueError('DINO local model_folder cannot mix with LeWM source')
        if modality != 'pixels' or latent_dim != 384 or history_size != 1 or action_dim != 5:
            raise ValueError('DINO requires single RGB frame, raw CLS384 and action5')
        from .frozen_dino import FrozenDinoCLS
        encoder = FrozenDinoCLS(model_folder)
    return FrozenStateVisualGCBC(encoder, modality=modality, seed=seed,
        goal_state_mode=goal_state_mode, latent_dim=latent_dim)

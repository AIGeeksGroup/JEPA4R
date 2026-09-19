"""Open-loop rollout of a world model in latent space, as its planner does it: the model sees the first
``context`` frames and then predicts every later latent1 from its own predictions and the recorded action blocks.

One model frame = ``frameskip`` env steps; action block k = the ``frameskip`` actions leaving frame k
(``action[r]`` moves dataset row r to r + 1), z-scored with the statistics of the world model's TRAINING dataset
(``utils.get_column_normalizer``), NaN -> 0 -- exactly what ``train.py`` feeds the action encoder."""

from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf


def training_setup(ckpt):
    """(frameskip, history_size, action mean, action std) of the run a world-model checkpoint belongs to."""
    path = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), ckpt)
    cfg = OmegaConf.load((path.parent if path.suffix == '.pt' else path) / 'config.yaml')
    train_set = swm.data.load_dataset(cfg.data.dataset.name, transform=None, keys_to_load=['action'], keys_to_cache=['action'])
    a = torch.from_numpy(np.array(train_set.get_col_data('action')))
    a = a[~torch.isnan(a).any(dim=1)]
    return int(cfg.data.dataset.frameskip), int(cfg.history_size), a.mean(0, keepdim=True), a.std(0, keepdim=True)


def action_blocks(actions, frameskip, mean, std):
    """actions (T, A) of one episode, raw -> (T // frameskip, frameskip * A) normalised blocks."""
    a = ((torch.as_tensor(actions) - mean) / std).float()
    k = a.size(0) // frameskip
    return torch.nan_to_num(a[:k * frameskip].reshape(k, -1), 0.0)


@torch.no_grad()
def open_loop(world_model, latents, blocks, context, history):
    """latents (E, K, D): latent1 of the real frames; blocks (E, >= K - 1, A). Returns (E, K, D): the first
    ``context`` entries are the real latents, every later entry k is predicted from entries < k only.

    One step = the model's own: ``JEPA`` predicts from the last ``history`` (latent, block) pairs and keeps the
    newest output (``JEPA.rollout``); ``AggJEPA`` aggregates its last ``window`` latents into latent2 and predicts
    with the newest block (``AggJEPA.rollout``)."""
    act = world_model.action_encoder(blocks)
    emb = list(latents[:, :context].unbind(1))
    for k in range(context - 1, latents.size(1) - 1):                  # predict frame k + 1 from frames <= k
        if hasattr(world_model, 'aggregate'):
            nxt = world_model.predict(world_model.aggregate(torch.stack(emb, 1)), act[:, k])
        else:
            lo = max(0, k + 1 - history)
            nxt = world_model.predict(torch.stack(emb[lo:], 1), act[:, lo:k + 1])[:, -1]
        emb.append(nxt)
    return torch.stack(emb, 1)

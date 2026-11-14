"""Open-loop imagination (Sec. 4.4): encode the context frames at env steps 0, 5, 10, roll g_phi forward over the
action blocks (frameskip 5) up to env step 200, and decode every predicted latent -> decoded 4D rollout.

The paper's rollout code: training_setup (frameskip, history, action statistics of the world
model's training data), action_blocks (z-scored blocks of frameskip actions), open_loop (the planner's rollout in
latent space). ``imagine`` decodes such a rollout.
"""
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

try:  # stable-worldmodel (third_party/stable-worldmodel) is needed to run, not to import this module
    import stable_worldmodel as swm
except ImportError:  # pragma: no cover
    swm = None


def training_setup(ckpt):
    """(frameskip, history_size, action mean, action std) of the run a world-model checkpoint belongs to."""
    path = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), ckpt)
    cfg = OmegaConf.load((path.parent if path.suffix == '.pt' else path) / 'config.yaml')
    train_set = swm.data.load_dataset(
        cfg.data.dataset.name, transform=None, keys_to_load=['action'], keys_to_cache=['action']
    )
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


@torch.no_grad()
def imagine(world_model, decoder, context_obs, action_blocks, history=3, num_points=1200):
    """world_model: FrozenLatent (jepa4r.decoder.model) of the frozen JEPA; context_obs: the encoder input of the
    context frames [C, ...] (env steps 0, 5, 10); action_blocks: normalised blocks [K - 1, frameskip x A] of the episode
    (``action_blocks``) -> decoded clouds [K - C, num_points, 6] of the predicted frames (env steps 5C, ..., 5(K - 1)),
    in the decoder's target frame (jepa4r.metrics.pointcloud.to_metres -> metres)."""
    ctx = world_model(context_obs[None])                                              # (1, C, D)
    k = action_blocks.size(0) + 1
    latents = torch.cat([ctx, ctx.new_zeros(1, k - ctx.size(1), ctx.size(2))], 1)    # open_loop reads only the context
    pred = open_loop(world_model.world_model, latents, action_blocks[None], ctx.size(1), history)[0, ctx.size(1):]
    return decoder.to_points(decoder(pred), num_points)

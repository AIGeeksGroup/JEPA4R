"""Open-loop imagination (Sec. 4.4): encode the context frames at env steps 0, 5, 10, roll g_phi forward over the
action blocks (frameskip 5) up to env step 200, and decode every predicted latent -> decoded 4D rollout.
"""


def imagine(world_model, decoder, context_obs, action_blocks):
    """-> decoded clouds [H, 1200, 6] for env steps 15, 20, ..., 200."""
    raise NotImplementedError("TODO(integrate)")

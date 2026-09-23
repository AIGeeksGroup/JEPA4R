"""3D reconstruction from a frozen world model (``train_recon.py``).

    checkpoint (2D / 3D / 4D) --frozen--> latent1 of one frame (B, 192) --PAGE-4D decoder--> point cloud

Every checkpoint form shares ``jepa.JEPA.encode``: encoder + projector -> ``emb``, the per-frame latent1
(``jepa_agg.AggJEPA`` inherits it unchanged; its latent2 is not used here). Only the decoder is trained.

    model.py      the frozen world model as ``observation -> latent1``, the input pipeline it was trained with,
                  the training forward
    data.py       frames for the encoder + the point cloud of the same simulator state (one dataset, or two row-aligned ones)
    decoders/     the feed-forward decoder: latent -> token grid -> PAGE-4D's frozen backbone and point head (+ LoRA)
    losses.py     Chamfer, sliced Wasserstein
"""

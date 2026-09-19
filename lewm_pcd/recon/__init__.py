"""3D reconstruction from a frozen world model (``train_recon.py``).

    checkpoint (2D / 3D / 4D) --frozen--> latent1 of one frame (B, 192) --decoder--> point cloud | 3D Gaussians

Every checkpoint form shares ``jepa.JEPA.encode``: encoder + projector -> ``emb``, the per-frame latent1
(``jepa_agg.AggJEPA`` inherits it unchanged; its latent2 is not used here). Only the decoder is trained.

    model.py      the frozen world model as ``observation -> latent1``, the input pipeline it was trained with,
                  the training forward
    data.py       frames for the encoder + the point cloud of the same simulator state (one dataset, or two row-aligned ones)
    decoders/     pluggable decoders (point cloud, Gaussians) on a shared trunk
    losses.py     Chamfer, sliced Wasserstein, mixture likelihood
"""

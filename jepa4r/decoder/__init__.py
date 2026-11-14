"""Latent-space reconstruction decoder D_psi (Sec. 4.4, Eq. 2): frozen 192-d latent -> 1,200-point xyzrgb cloud.

set_generator.py  256 learnable queries, 3 residual MLP blocks FiLM-modulated by z_t -> 16x16 x 1024 tokens
backbones/        frozen 3D reconstruction backbone; tokens enter after its patch embedding
                  (page4d default; vggt, depth_anything3 for Tab. 4a)
head.py           learnable affine map of the 224x224 point map + rgb branch + 1,200-pixel selection
lora.py           stage-B rank-4 LoRA on q / v of all attention blocks
losses.py         Chamfer + sliced W2^2 (xyz) + 0.1 * L1 rgb of NN-matched points
model.py          assembles the decoder;  train.py  stage A / stage B;  imagine.py  open-loop rollouts
Each backbone module holds the whole decoder with that backbone, as the paper's reconstruction code builds it (head.py:
where the head lives).
"""

"""SIGReg [balestriero2025lejepa] anti-collapse regulariser (Eq. 1): compares 1,024 random one-dimensional
projections of each time step's latents with a standard Gaussian; weight lambda = 0.09.

The module is LeWM's ``SIGReg`` moved unchanged (17 knots on [0, 3], Epps-Pulley statistic, projections drawn from
torch's global RNG at every call). The world model calls it on the latents ordered time-first, ``(T, B, D)``.
"""

import torch


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


def sigreg(z, num_projections=1024):
    """z: [B, T, D] latents -> scalar loss."""
    # The training forward (world_model.lejepa_forward) calls SIGReg on emb.transpose(0, 1); the 17 knots are
    # LeWM's value (loss.sigreg.kwargs.knots).
    return SIGReg(knots=17, num_proj=num_projections).to(z.device)(z.transpose(0, 1))

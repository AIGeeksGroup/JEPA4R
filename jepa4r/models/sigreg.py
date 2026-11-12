"""SIGReg [balestriero2025lejepa] anti-collapse regulariser (Eq. 1): compares 1,024 random one-dimensional
projections of each time step's latents with a standard Gaussian; weight lambda = 0.09.
"""


def sigreg(z, num_projections=1024):
    """z: [B, T, D] latents -> scalar loss."""
    raise NotImplementedError("TODO(integrate): SIGReg from LeWM / LeJEPA")

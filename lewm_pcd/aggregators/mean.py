"""Parameter-free control: latent2 = mean of the K frame latents."""

from torch import nn


class MeanAggregator(nn.Module):
    def __init__(self, dim=192, max_frames=3):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=1)

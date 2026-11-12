"""Pluggable temporal aggregators for the aggregated world model (jepa_agg.AggJEPA).

Contract:
    input  x : float tensor (B, K, D) -- the last K per-frame latents (latent1), K = 1..max_frames,
               oldest first, frames `frameskip` env steps apart
    output   : float tensor (B, D)     -- latent2, the aggregated state (same width as latent1)
Aggregators only look at time; spatial aggregation happens inside the encoder.
"""

from .mean import MeanAggregator
from .mlp import MLPAggregator
from .temporal import TemporalTransformerAggregator

__all__ = ['MeanAggregator', 'MLPAggregator', 'TemporalTransformerAggregator']

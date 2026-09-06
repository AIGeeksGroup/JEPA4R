# JEPA4R

## 4R-JEPA — based on ManiSkill2-Learn and LeWM

Implementation: [4r-jepa-maniskill2-lewm](4r-jepa-maniskill2-lewm/README.md).

Three stages: point-cloud PPO, a consecutive-frame JEPA world model, and PPO
with the trained encoder frozen through its temporal representation (latent2).
Upstream sources are fetched at pinned commits; datasets, environments and
checkpoints are not committed. See the implementation README for setup and
the current validation limits.

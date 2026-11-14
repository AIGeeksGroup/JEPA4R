"""World-model training support of the Adroit Relocate and RH56 workflows.

  config/train/             LeWM's Hydra configs for Relocate (RGB, point cloud with the DexArt PointNet)
  recovery.py               resumable workers=0 loading around the original DataLoader / Lightning loop
  wm_recovery.py            its wiring for the Adroit world-model runs
"""

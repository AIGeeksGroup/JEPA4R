# PLDM (vendored)

- Repository: https://github.com/vladisai/PLDM
- Commit: `1bd7e564ecd961205bc18b23067b19e9ca24ac90`
- License: MIT (`LICENSE`, unchanged; attribution in `ATTRIBUTION.md`)

Vendored here: the package `pldm/` (configs, encoders, predictors, objectives, schedulers), as the OGBench-cube branch
bundles it (`OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/pldm/vendor/source`). Used by the Cube 1-4 PLDM runs
(`jepa4r/baselines/cube/pldm`, `jepa4r/baselines/cube/workflow_pldm_encoder.py`) and by
`jepa4r/models/encoders/impala.py`. The sha256 of every file is pinned in
`configs/method/cube/pldm/source_manifest.json` (entries `vendor/source/...`).

## Local changes

Line ends normalized from CRLF (as in the branch's copy) to LF; Python reads both alike. No other change.

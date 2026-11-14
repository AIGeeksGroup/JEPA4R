# PLDM (vendored)

- Repository: https://github.com/vladisai/PLDM
- Commit: `1bd7e564ecd961205bc18b23067b19e9ca24ac90`
- License: MIT (`LICENSE`, unchanged; attribution in `ATTRIBUTION.md`)

Vendored here: the package `pldm/` (configs, encoders, predictors, objectives, schedulers), as the paper's Cube runs
used it. Used by the Cube 1-4 PLDM runs
(`jepa4r/baselines/cube/pldm`, `jepa4r/baselines/cube/workflow_pldm_encoder.py`) and by
`jepa4r/models/encoders/impala.py`.

## Local changes

Line ends normalized from CRLF (as in the copy the paper's runs used) to LF; Python reads both alike. No other change.

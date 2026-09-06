# Third-Party Notices

JEPA4R contains adapted portions of the following projects. The corresponding
license texts are retained under `LICENSES/`.

## Pointcept

- Project: [Pointcept](https://github.com/Pointcept/Pointcept)
- Components: serialized point ordering and Point Transformer V3 aggregation
- Local files: `models/ordering.py`, `models/aggregator.py`
- License: MIT, copyright 2023 Pointcept

## HMA

- Project: [HMA](https://github.com/liruiw/HMA)
- Components: action normalization/MLP, modulation, attention, and predictor blocks
- Local files: `models/action.py`, `models/predictor.py`
- License: Apache License 2.0, copyright 2024 1X Technologies Inc.
- The local files were modified for point-cloud latent regression, native PyTorch
  attention, ordered action tokens, and per-layer action cross-attention.

## LeWorldModel

- Project: [LeWorldModel](https://github.com/lucas-maes/le-wm)
- Components: SIGReg and the joint-embedding prediction objective
- Local file: `losses/lewm.py`
- License: MIT, copyright 2026 Lucas Maes
- The local implementation adds patch matching, optional motion weighting,
  correct-vs-mismatched action ranking, and chunked SIGReg computation.

These notices apply to the third-party portions only. They do not grant a
license for the remainder of JEPA4R.

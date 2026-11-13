# PLDM attribution

This project includes source files from [vladisai/PLDM](https://github.com/vladisai/PLDM), the implementation of *Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models*.

- Upstream commit: [`1bd7e564ecd961205bc18b23067b19e9ca24ac90`](https://github.com/vladisai/PLDM/tree/1bd7e564ecd961205bc18b23067b19e9ca24ac90).
- Vendored source directory: `lewm_pcd/bench/ogbench/pldm/vendor/source/pldm/`.
- Upstream license: [MIT License, original file `LICENSE` at that commit](https://github.com/vladisai/PLDM/blob/1bd7e564ecd961205bc18b23067b19e9ca24ac90/LICENSE).
- Original license copyright notice: **Copyright (c) 2025 Vlad Sobal**.
- Preserved original license text: [LICENSE](LICENSE), 1067 bytes.
- Original license SHA-256: `49f81b32a86d9de3d3f73af42d0b5fe5c8584d405311a658806d8a22d5862a3b`.
- Original license Git blob SHA-1: `15ac57796ea968407d5557f6f3f4a094d67b8f6e`.

The complete original license text accompanies this attribution. Its copyright and permission notices are preserved.

## Existing source attributions

The vendored [`pldm/models/encoders/resnet.py`](https://github.com/vladisai/PLDM/blob/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm/models/encoders/resnet.py#L1-L6) retains its existing Facebook, Inc. and affiliates copyright header and reference to the source-tree root license. That header is preserved in the source.

The vendored [`pldm/configs.py`](https://github.com/vladisai/PLDM/blob/1bd7e564ecd961205bc18b23067b19e9ca24ac90/pldm/configs.py#L52-L56) retains its existing statement that `DataclassArgParser` is closely based on Hugging Face's `HfArgumentParser` and extended to support recursive dataclasses. That source attribution is preserved in the docstring.

These statements reproduce the upstream's visible attributions. They do not identify Vlad Sobal as the sole author or copyright holder of every component. No additional upstream license or NOTICE is asserted here; the pinned upstream tree contains a root `LICENSE` and no separate `NOTICE` file.

## Local source differences

The 20 vendored source files retain the pinned upstream source text, with line-ending and empty-package-file normalization as the only differences identified in the source comparison:

- The 17 non-empty upstream files use CRLF in the local copy. Converting only CRLF to LF reproduces their pinned upstream Git blob hashes.
- `pldm/models/__init__.py`, `pldm/models/encoders/__init__.py`, and `pldm/optimizers/__init__.py` are empty upstream. Each local copy contains one CRLF newline.

The local vendored source files are therefore not claimed to be byte-for-byte identical to the upstream Git blobs. These source-normalization differences do not change the preserved original license file's bytes or hash. Project-specific adapters outside the vendored source directory are separate from this upstream source attribution.

This attribution records the upstream source, published license, and retained source notices. It does not determine the license of the project as a whole or independently resolve the historical rights provenance of upstream-derived components.

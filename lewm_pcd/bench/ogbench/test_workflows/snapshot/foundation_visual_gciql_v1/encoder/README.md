# Official encoder assets and adapter

Remote case: `/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/foundation_visual_gciql_v1/encoder`.

- `encoder_adapter.py`: `load_encoder(spec_dict_or_json_path, device)` returns a frozen eval module. `model.encode_uint8(uint8[B,3,224,224])` returns finite FP32 `[B,D]`; legacy `model(raw).logits` is the same path. Module-level `encode_uint8(model,uint8[B,224,224,3])` permutes to the same path.
- V-JEPA 2.1 ViT-L uses the official EMA checkpoint and pinned source `204698b45b3712590f06245fbfba32d3be539812`. Raw uint8 moves to the model device before floating-point conversion, full-FOV bicubic antialias resize to 384, and ImageNet normalization once. Its single actual frame enters the image tokenizer as B,C,T=1,H,W. Last normalized 576 image patch tokens are spatially averaged to 1024 dimensions. No duplicated frame, predictor, projector, L2, or PCA.
- VISReg is `SKIPPED_UNAVAILABLE`; see `visreg_availability.json` for actual current HTTP evidence. Earlier model card advertised public weights. Do not interpret this skip as a claim it was never released.

Transport state: the original slow single connection and four-range logs are preserved. Only those owned download processes were stopped. `download_segments.py` reuses the four saved prefixes and writes 16 disjoint remaining ranges into the same single part file. `segments_status.json` reports actual received bytes and throughput; preallocated file size is not download progress. Each response checks official ETag, 206 and exact Content-Range. Segment receipts permit retry.

`finalize_assets.py` waits for segment completion, then publishes the single file, computes SHA256 and ZIP CRC, and writes `specs/vjepa21_vitl1024.json`. Completion requires `finalize.actual_exit_code=0`, `assets_status.json` exit_code 0, the asset manifest, and the resulting spec together. These prove asset transport only. Root separately owns strict load, preprocessing equivalence, frozen-state, NN/GPU, cache, GCIQL, and Native gates.

No GPU or model was run by the encoder/assets owner during preparation. Only AST/compilation was checked before Root's model gate.

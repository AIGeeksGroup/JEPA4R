# Cube2 PLDM → GCIQL → Native SR

This release reuses the accepted Cube3 PLDM pipeline on `103032Test` and changes only the cube-count, data, checkpoint and width bindings.

- Checkpoint: `cube2_pldm_impala512_epoch10_20261021_v1`, SHA-256 `ada09ccd9508434f688455622491b5910f2ad2190189a5f67f60cac53d7a5ae5`.
- Source: `10205090` formal run `double_cube_pldm_seed0_20261020_v2/attempt_002`, 10 epochs, global step 378900.
- Data: official `cube-double-play-v0`, 1000 train and 100 validation episodes, including true terminal rows.
- Policy input: current/next = PLDM512 + proprio19 = 531; goal = PLDM512 + two-object XYZ6 = 518.
- GCIQL: fresh seed0, B1024, 1M updates, original alpha/optimizer/sampler/reward semantics.
- Native SR: tasks1-5 x 50, max500 steps, fresh goal/current encoding on every policy call.

The Cube3 checkpoint must never satisfy this case's identity check.

# 3Cube PLDM representation baseline

Host: `103032Test`; remote namespace: `/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube3_pldm_gciql_v1`.

The final epoch10 Impala512 asset is already migrated and CPU-accepted. This case adds official-data frozen embeddings, the existing distance-reward GCIQL training, and final1M Native SR. See `PLAN.md` for the frozen scientific contract and `root_status.json` for actual progress. Preparation or a template does not mean encoding, training, or SR has started.

Native input widths: current531 = PLDM512 + proprio19; goal521 = PLDM512 + three-objectXYZ9. There is no PCA, zero padding, or current-object truth in the policy input. Wider first-layer parameter counts are part of the native-feature comparison.

All 3,303,300 official raw observation rows, including each true terminal, are encoded. FP32 feature payload is 6,765,158,400 bytes before NPY headers. The raw official data and prior LeWM caches remain unchanged.

Only the encoder/family, feature-width boundary, new config/role namespace, and serial launch interface are adapted. GCIQL losses, sampling, hidden layers, optimizer and training budget reuse existing sources. The PLDM encoder's own full-field antialiased65 preprocessing is retained; LeWM ImageNet preprocessing is not applied.

Stages: GPU pilot → full embeddings → independent cache acceptance → B1024 gate/restore → fresh1M/11checkpoints → full CPU policy acceptance → Native tiny/full250/audit → closed final-policy and evidence backup. `actual_exit_code`, the original child exits, drained owned process groups, and accepted artifacts determine completion.

Temporary storage was explicitly approved by the user. Full feature arrays remain on the remote data disk unless separately copied; a small metadata/model backup does not mean those arrays are backed up locally. No prior task, model, checkpoint, or result is deleted or overwritten.

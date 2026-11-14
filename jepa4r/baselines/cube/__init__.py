"""PLDM and LeJEPA training and the VISReg / V-JEPA 2.1 loaders of the Cube 1-4 workflow (OGBench-cube@4fbb0c6,
lewm_pcd/bench/ogbench; its stages are driven by jepa4r/policy/gciql/workflow.py).

  pldm/                     Impala-512 PLDM on Cube (the PLDM authors' source: third_party/pldm)
  native/tiny_train.py      LeJEPA's Tiny/14 trainer
  workflow_pldm_encoder.py  the frozen PLDM encoder for the policy stages
  released_encoders.py, foundation_assets.py
                            the released VISReg / V-JEPA 2.1 weights (configs/policy/gciql/released_models.json)
                            and their preprocessing
"""

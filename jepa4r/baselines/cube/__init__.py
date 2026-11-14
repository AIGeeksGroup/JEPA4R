"""PLDM training and the VISReg / V-JEPA 2.1 loaders of the Cube 1-4 workflow (its stages are driven by
jepa4r/policy/gciql/workflow.py); LeJEPA trains with jepa4r/baselines/lejepa.py.

  pldm/                     Impala-512 PLDM on Cube (the PLDM authors' source: third_party/pldm)
  workflow_pldm_encoder.py  the frozen PLDM encoder for the policy stages
  released_encoders.py, foundation_assets.py
                            the released VISReg / V-JEPA 2.1 weights (configs/method/cube/released_models.json)
                            and their preprocessing
"""

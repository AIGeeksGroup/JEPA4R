"""World-model training of the Cube 1-4 workflow (its stages are driven by jepa4r/policy/gciql/workflow.py).

  fresh_train.py            the train stage: JEPA4R / LeWM (jepa4r/models/world_model.py with the recipes in
                            configs/method/cube/recipes/), PLDM and LeJEPA (jepa4r/baselines/cube)
  training_runtime.py       recipe loading and the training environment
  workflow_lewm_encoder.py  the frozen LeWM / JEPA4R encoder for the policy stages
"""

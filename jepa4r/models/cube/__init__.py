"""World-model training of the Cube 1-4 workflow (OGBench-cube@4fbb0c6, lewm_pcd/bench/ogbench; its stages are driven
by jepa4r/policy/gciql/workflow.py).

  fresh_train.py            the train stage: JEPA4R / LeWM (jepa4r/models/world_model.py with the recipes in
                            configs/policy/gciql/recipes/), PLDM and LeJEPA (jepa4r/baselines/cube)
  training_runtime.py       recipe loading and the sha256 pins of the training sources
                            (configs/policy/gciql/SOURCE_MANIFEST.json)
  retained_training.py, retained_encoder.py
                            the retained Cube1 inputs and export of the branch's smoke run
  workflow_lewm_encoder.py  the frozen LeWM / JEPA4R encoder for the policy stages
"""

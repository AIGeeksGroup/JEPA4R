"""Data of the Cube 1-4 workflow (its stages are driven by jepa4r/policy/gciql/workflow.py).

  collection/               paired RGB / point-cloud collection under the fixed-task protocol (collect stage)
  fresh_data.py             the training views over the collected HDF5 (views stage)
  official_data.py          OGBench's public play data for GCIQL (prepare-data stage;
                            configs/env/cube/official_sources.json)

collection/vendor/ holds frozen copies of the paper's collection scripts and point-cloud wrapper (their original paths
in configs/env/cube/collection/sources.json), unchanged apart from formatting and the OGBench env registration (written
as importlib.import_module).
"""

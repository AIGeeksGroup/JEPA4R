"""Data of the Cube 1-4 workflow (OGBench-cube@4fbb0c6, lewm_pcd/bench/ogbench; its stages are driven by
jepa4r/policy/gciql/workflow.py).

  collection/               paired RGB / point-cloud collection under the fixed-task protocol (collect stage)
  fresh_data.py             the training views over the collected HDF5 (views stage)
  official_data.py          OGBench's byte-pinned play data for GCIQL (prepare-data stage;
                            configs/policy/gciql/official_sources.json)

collection/vendor/ holds the branch's frozen copies of its collection scripts and point-cloud wrapper (their original
paths in configs/policy/gciql/collection/sources.json). Here they are wrapped at 120 columns, without unused imports
and with the OGBench env registration written as importlib.import_module; their code is otherwise unchanged (the same
AST).
The sha256 pins the copies hold of each other (pcd_geometry.py / native_pcd_frame.py: CORE_SHA, ADAPTER_SHA; double /
triple collection: FRAME_SHA) and configs/policy/gciql/collection/sources.json pin them as they are here.
"""

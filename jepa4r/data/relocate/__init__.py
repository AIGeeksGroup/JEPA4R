"""Data of the Adroit Relocate workflow (its stages are driven by jepa4r/policy/bc/relocate/adroit_workflow.py).

adroit_collect.py           mixed feedback-expert trajectories (collect stage), written by paired_writer.py
adroit_data.py              the official demonstrations replayed as RGB and point clouds (official-demos stage)
convert_adroit_demos_3d.py  the point-cloud copies of the official demonstrations
paired_writer.py            the paired RGB / point-cloud HDF5 writer (also used by the RH56 twin)
"""

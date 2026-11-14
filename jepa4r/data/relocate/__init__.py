"""Data of the Adroit Relocate workflow (its stages are driven by jepa4r/policy/bc/relocate/adroit_workflow.py).

adroit_collect.py         mixed feedback-expert trajectories (collect stage), written by paired_writer.py
adroit_data.py            data checks and the replay of the official demonstrations (data-audit, official-demos)
official_contract.py      the policy-critical content of the official Relocate demonstrations
convert_adroit_demos_3d.py  the point-cloud copies of the official demonstrations
"""

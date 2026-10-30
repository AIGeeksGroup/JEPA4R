"""Frozen G05 registration used only by the Cube2 PLDM Native adapter."""
import hashlib
import json
from pathlib import Path

PATH = Path("/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube23_distance_gciql_v1/five_group_extension_v1/configs/G05_cube2_rgb.json")
raw = PATH.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "8c3af1e965e3266965cfac9a1baedbfb081da1c5a15549de1574b3cdfbf8b65b"
GROUPS = {"G05": json.loads(raw)}

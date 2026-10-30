#!/usr/bin/env bash
set -eu
CASE='/root/autodl-tmp/workspace/users/lewm/coordination/double_cube_lejepa_match_3cube_20261026_v1'
exec '/root/autodl-tmp/conda/envs/jepa4r-smoke/bin/python' -B "$CASE/launch.py"

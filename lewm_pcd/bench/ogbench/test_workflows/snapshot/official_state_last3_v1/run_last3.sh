#!/usr/bin/env bash
set -u
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /root/miniconda3/bin/python -B "$HERE/supervise_last3.py"

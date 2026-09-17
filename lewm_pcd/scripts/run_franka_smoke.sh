#!/usr/bin/env bash
set -euo pipefail
P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$P/run_franka_pipeline.sh" verify "$@"

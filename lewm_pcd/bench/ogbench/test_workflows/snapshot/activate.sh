#!/usr/bin/env bash
_CUBE_DOWNSTREAM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_CUBE_DOWNSTREAM_ROOT/env_runtime/bin/activate"
export CUBE_DOWNSTREAM_ROOT="$_CUBE_DOWNSTREAM_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="$_CUBE_DOWNSTREAM_ROOT/runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _CUBE_DOWNSTREAM_ROOT

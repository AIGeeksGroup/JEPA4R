#!/usr/bin/env bash
# Candidate runtime only: inherit frozen original profile; add one compile-kernel-selection flag.
_LEWM_RUNTIME_BASE_PROFILE="/root/autodl-tmp/workspace/users/lewm/coordination/gciql_frozen_lewm_109_20261009_v1/environment/activate_existing_profile.sh"
if [[ "$(sha256sum "$_LEWM_RUNTIME_BASE_PROFILE" | cut -d ' ' -f 1)" != "e36f92e198ba0e51fce7a9c4806dccd29c9530b8067b983c1d8ecaf841f9eca2" ]]; then
    printf '%s\n' 'Frozen original runtime profile SHA mismatch' >&2
    return 1 2>/dev/null || exit 1
fi
source "$_LEWM_RUNTIME_BASE_PROFILE" || return 1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0"

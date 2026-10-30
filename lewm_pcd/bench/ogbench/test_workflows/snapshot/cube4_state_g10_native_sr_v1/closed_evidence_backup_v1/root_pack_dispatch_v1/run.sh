#!/usr/bin/env bash
set +e
umask 077
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python -B /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_state_g10_native_sr_v1/closed_evidence_backup_v1/backup.py --execute
code=$?
printf "%s\n" "$code" > /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube4_state_g10_native_sr_v1/closed_evidence_backup_v1/root_pack_dispatch_v1/outer_exit_code
exit "$code"

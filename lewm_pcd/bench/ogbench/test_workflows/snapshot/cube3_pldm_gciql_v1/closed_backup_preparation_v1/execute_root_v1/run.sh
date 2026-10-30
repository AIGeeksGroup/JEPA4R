#!/usr/bin/env bash
set +e
umask 077
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python -B /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube3_pldm_gciql_v1/closed_backup_preparation_v1/backup.py --execute
code=$?
printf "%s\n" "$code" > /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/cube3_pldm_gciql_v1/closed_backup_preparation_v1/execute_root_v1/outer_exit_code
exit "$code"

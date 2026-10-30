"""Accepted copy body; only the fresh archive output namespace changes."""
from pathlib import Path
import reuse

reuse.COPY_EDITS.append(("/closed_SR_backup_v1/completed_v1'","/closed_SR_backup_v1/completed_v2'"))

if __name__=='__main__':
    local=Path(__file__).resolve().parents[2]/'cube4_gciql_103032Test_20261020_v1/closed_backup_preparation_v2/copy_verified.py'
    reuse.load('copy',local if local.is_file() else None).main()

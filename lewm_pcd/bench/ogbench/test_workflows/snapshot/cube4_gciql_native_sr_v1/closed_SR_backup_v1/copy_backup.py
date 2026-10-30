"""One closed archive SCP and accepted scratch/member/CRC/project verification."""
from pathlib import Path
from reuse import load
if __name__=='__main__':
    local=Path(__file__).resolve().parents[2]/'cube4_gciql_103032Test_20261020_v1/closed_backup_preparation_v2/copy_verified.py'
    load('copy',local if local.is_file() else None).main()

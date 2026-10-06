"""Path/logging adapter only. The original main, sampler and GCBC remain unchanged."""
import argparse
from functools import partial
from pathlib import Path
import runpy
import sys
import os
import json
import tempfile
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--data', required=True)
args, original_flags = parser.parse_known_args()
sys.path[:0] = [str(Path(args.source).resolve()), str(Path(args.source).resolve() / 'impls')]
import ogbench
import utils.log_utils as logging_utils

# Original impls loader exposes no dataset_dir flag. Bind only its storage location.
ogbench.make_env_and_datasets = partial(ogbench.make_env_and_datasets, dataset_dir=args.data)
# Original setup_wandb defaults to explicit online. This changes logging transport only.
def setup_offline(entity=None, project='project', group=None, name=None, mode='online'):
    # Modern SDK removed old start_method/_disable_stats Settings fields.
    # Keep original run metadata and config; change logging transport/compatibility only.
    return wandb.init(config=logging_utils.get_flag_dict(),project=project,entity=entity,
        tags=[group] if group is not None else None,group=group,name=name,
        dir=tempfile.mkdtemp(),mode='offline',save_code=True)

logging_utils.setup_wandb = setup_offline
main = Path(args.source) / 'impls/main.py'
sys.argv = [str(main), *original_flags]
try:
    runpy.run_path(str(main), run_name='__main__')
finally:
    import wandb
    receipt = os.environ.get('OFFICIAL_ENTRY_RECEIPT')
    if receipt and wandb.run is not None:
        mode = str(wandb.run.settings.mode)
        assert mode == 'offline', f'logging mode was {mode}'
        Path(receipt).write_text(json.dumps({'actual_wandb_mode':mode,'run_id':wandb.run.id,
            'run_dir':wandb.run.dir,'dataset_dir_bound':args.data,'original_main_path':str(main),
            'logging_compatibility':'modern SDK; obsolete Settings start_method/_disable_stats omitted'},indent=2))

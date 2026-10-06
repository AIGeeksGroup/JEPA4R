"""Original Hydra/train entry with numerical evidence callbacks only."""
import argparse
from pathlib import Path
import sys
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--code-root', type=Path, required=True)
    args, overrides = parser.parse_known_args()
    code = args.code_root.resolve()
    if (code/'lewm_pcd').is_dir(): code = code/'lewm_pcd'
    sys.path.insert(0,str(code)); sys.argv = [sys.argv[0]]+overrides
    import hydra
    import train
    from stable_worldmodel.data.utils import get_cache_dir
    from wm_real_batch_gate import install_recorder, configure_spt_cache

    @hydra.main(version_base=None,config_path=str(code/'config/train'),config_name='lewm')
    def recorded(cfg):
        directory = Path(get_cache_dir(sub_folder='checkpoints'))/str(cfg.output_model_name)
        directory.mkdir(parents=True,exist_ok=True)
        configure_spt_cache(train,os.environ['SPT_CACHE_DIR'],directory)
        route = 'rgb' if cfg.model.obs_key == 'pixels' else 'pcd'
        recorder, original = install_recorder(train,directory,route,cfg.data.dataset.name)
        try: return train.run.__wrapped__(cfg)
        finally: train.pl.Trainer = original

    recorded()

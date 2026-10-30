"""One real CPU batch: thin HWC bridge equals unchanged CHW preprocessing."""
import json
from pathlib import Path
import numpy as np
import torch
from checkpoint_encoder import source_paths,preprocess

ROOT=Path(__file__).resolve().parent
source_paths()
from utils import get_img_preprocessor
torch.set_num_threads(1)
raw=np.load(ROOT/'incoming/smoke_paired_rows_v1.npz',allow_pickle=False)['pixels'][:2]
x=torch.from_numpy(raw)
actual=preprocess({'pixels':x},'2d')['pixels']
expected=get_img_preprocessor('pixels','pixels',224)({'pixels':x.permute(0,3,1,2)})['pixels']
assert actual.shape==(2,3,224,224) and torch.equal(actual,expected)
assert torch.isfinite(actual).all()
(ROOT/'smoke/rgb_layout_report.json').write_text(json.dumps(dict(actual_exit_code=0,
    shape=list(actual.shape),exact_to_original_CHW_preprocessor=True,
    encoder_calls=0,GPU_calls=0,source_normalization_changed=False),indent=2))

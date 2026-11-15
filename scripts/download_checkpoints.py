"""Third-party weights the paper uses, expected under checkpoints/ (or $JEPA4R_CHECKPOINTS) at the paths the configs
name (configs/method/{visreg,vjepa21}.yaml, configs/decoder/*.yaml: ``weights``).

    python scripts/download_checkpoints.py            # list, with the status of each file
    python scripts/download_checkpoints.py --download # fetch the missing public files, then check their sizes

Sizes are those of the files the paper's runs used; a file of another size is reported. VISReg's weights are not
public (shared by the authors on request): place the file yourself.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("JEPA4R_CHECKPOINTS") or Path(__file__).resolve().parents[1] / "checkpoints").expanduser()
CHECKPOINTS = {  # path under checkpoints/: (url or None, bytes, used by)
    "visreg/visreg-vit-b-inet1k.pth": (
        None,
        368495235,
        "VISReg ViT-B/16 (configs/method/visreg.yaml); not public, from the authors on request",
    ),
    "cortexbench_baselines/vjepa2_1_vitb_dist_vitG_384.pt": (
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        1664223428,
        "V-JEPA 2.1 ViT-B/16 (configs/method/vjepa21.yaml)",
    ),
    "page4d/checkpoint_nomask.pt": (
        "https://huggingface.co/datasets/zhouk777/PAGE4D/resolve/main/checkpoint_nomask.pt",
        6782314015,
        "PAGE-4D, decoder backbone (configs/decoder/page4d.yaml)",
    ),
    "vggt/model.pt": (
        "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        5026874952,
        "VGGT-1B, Tab. 4a (configs/decoder/vggt.yaml)",
    ),
    "da3/model.safetensors": (
        "https://huggingface.co/depth-anything/DA3-GIANT/resolve/main/model.safetensors",
        5422814644,
        "Depth Anything 3 Giant, Tab. 4a (configs/decoder/da3.yaml)",
    ),
    "da3/config.json": (
        "https://huggingface.co/depth-anything/DA3-GIANT/resolve/main/config.json",
        None,
        "its model config, read next to the weights (jepa4r/decoder/backbones/depth_anything3.py)",
    ),
}


def status(path, size):
    if not path.exists():
        return "missing"
    if size is not None and path.stat().st_size != size:
        return f"WRONG SIZE ({path.stat().st_size} bytes, expected {size})"
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="fetch the missing public files")
    args = ap.parse_args()
    bad = 0
    for rel, (url, size, what) in CHECKPOINTS.items():
        path = ROOT / rel
        if args.download and url and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            print(f"downloading {url} -> {path}", flush=True)
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        st = status(path, size)
        bad += st != "ok"
        print(f"{st:>12}  checkpoints/{rel:<55} {what}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

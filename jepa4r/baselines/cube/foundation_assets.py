"""The released VISReg / V-JEPA 2.1 weights of a Cube run (the prepare-encoder stage): the checkpoint (downloaded from
configs/method/cube/released_models.json when it has a public URL, else placed by hand or given with --weights), its
encoder_spec.json and ENCODER.json (the binding the policy stages load the encoder from).
usage: python -m jepa4r.baselines.cube.foundation_assets --model <visreg|vjepa> --cube-count <n> --assets-root <dir>
       --output <dir> [--weights <file>] [--download]"""

import argparse
import copy
import json
import os
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # this repository
VJEPA2 = Path(os.environ.get("JEPA4R_VJEPA2") or REPO / "third_party/vjepa2")  # V-JEPA 2 204698b checkout


def asset(model):
    registry = json.loads((REPO / "configs/method/cube/released_models.json").read_text(encoding="utf8"))["models"]
    if model not in ("visreg", "vjepa"):
        raise ValueError("the released encoders are visreg and vjepa")
    return copy.deepcopy(registry[model])


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")


def download(model, entry, target):
    """Download the weights to ``target``; a partial download (<target>.part) is resumed."""
    if not entry.get("url"):  # VISReg: the authors share the weights on request; there is no public download
        raise FileNotFoundError(
            f"{model} weights have no public download: place {entry['filename']} ({entry['bytes']} bytes) at {target} "
            "or pass --weights"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset < entry["bytes"]:
        headers = {"User-Agent": "JEPA4R-released-assets/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urllib.request.urlopen(urllib.request.Request(entry["url"], headers=headers), timeout=60) as response:
            if offset and getattr(response, "status", 200) != 206:
                raise ValueError("the server did not resume the partial download")
            with part.open("ab") as stream:
                for block in iter(lambda: response.read(8 << 20), b""):
                    stream.write(block)
    if part.stat().st_size != entry["bytes"]:
        raise ValueError(f"downloaded {part.stat().st_size} bytes of {model}, expected {entry['bytes']}")
    part.rename(target)
    return target


def prepare(model, cube_count, assets_root, output_root, weights=None, download_missing=False):
    entry = asset(model)
    path = Path(weights).resolve() if weights is not None else Path(assets_root).resolve() / model / entry["filename"]
    if not path.is_file():
        if not download_missing:
            raise FileNotFoundError(f"the released weights are missing: {path} (pass --weights, or --download)")
        download(model, entry, path)
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(f"{path}: {path.stat().st_size} bytes, expected {entry['bytes']}")
    spec = {key: copy.deepcopy(entry[key]) for key in ("family", "native_size", "feature", "inference_precision")}
    spec.update(checkpoint=str(path), output_dim=entry["latent_dim"])
    for key in ("source_commit", "source_files", "state_dict_key", "encoder_module", "encoder_constructor"):
        if key in entry:
            spec[key] = copy.deepcopy(entry[key])
    for key in ("constructor_kwargs", "timm_version"):
        if key in entry:
            spec[key] = copy.deepcopy(entry[key])
    if "source_commit" in entry:
        spec["source_root"] = str(VJEPA2)
    output = Path(output_root)
    spec_path = output / "encoder_spec.json"
    _save(spec_path, spec)
    adapter = HERE / "released_encoders.py"
    binding = dict(
        family=model,
        latent_dim=entry["latent_dim"],
        checkpoint=str(path),
        adapter=str(adapter),
        spec=str(spec_path),
        preprocessing=dict(source_path=str(adapter), device="cuda", input_dtype="uint8", layout="HWC224"),
    )
    _save(output / "ENCODER.json", dict(model=model, cube_count=cube_count, encoder_binding=binding))
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("visreg", "vjepa"), required=True)
    parser.add_argument("--cube-count", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    binding = prepare(args.model, args.cube_count, args.assets_root, args.output, args.weights, args.download)
    print(json.dumps(binding, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

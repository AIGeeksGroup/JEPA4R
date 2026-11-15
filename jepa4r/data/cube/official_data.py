"""OGBench's public play data of Cube 1-4 (cube-<n>-play-v0 and its -val split; URLs and sizes:
configs/env/cube/official_sources.json): downloaded once (a partial download is resumed), then recorded in
OFFICIAL_DATA.json. usage: python -m jepa4r.data.cube.official_data --cube-count <n> --output <dir>"""

import argparse
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent


def sources(cube_count):
    table = HERE.parents[2] / "configs" / "env" / "cube" / "official_sources.json"
    return json.loads(table.read_text(encoding="utf8"))[str(cube_count)]


def download(item, directory):
    """One file of the play data; a partial download (<name>.part) is resumed, a complete file is kept."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / item["name"]
    if not target.exists():
        partial = directory / (item["name"] + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset < item["bytes"]:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with urllib.request.urlopen(urllib.request.Request(item["url"], headers=headers), timeout=60) as response:
                if offset and response.status != 206:
                    raise ValueError("the server did not resume the partial download")
                with partial.open("ab") as output:
                    while block := response.read(1 << 20):
                        output.write(block)
        if partial.stat().st_size != item["bytes"]:
            raise ValueError(f"{item['name']}: downloaded {partial.stat().st_size} bytes, expected {item['bytes']}")
        partial.rename(target)
    if target.stat().st_size != item["bytes"]:
        raise ValueError(f"{target}: {target.stat().st_size} bytes, expected {item['bytes']}")
    return dict(path=str(target.resolve()), bytes=item["bytes"], episodes=item["episodes"])


def prepare(cube_count, output_root):
    output = Path(output_root)
    record = output / "OFFICIAL_DATA.json"
    if not record.exists():
        source = sources(cube_count)
        result = dict(cube_count=cube_count, source_config=source["source_config"])
        for split, item in source["files"].items():
            result[split] = download(item, output)
        record.write_text(json.dumps(result, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-count", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.cube_count, args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

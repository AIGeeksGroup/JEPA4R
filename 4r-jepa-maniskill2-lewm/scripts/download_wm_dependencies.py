"""Fetch the archived WM wheel resolution and verify its SHA-256 hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
WHEELHOUSE = ROOT / ".cache/wm-wheels"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wheel_info(package):
    info = package["download_info"]
    url = info["url"]
    parts = urlsplit(url)
    filename = unquote(parts.path.rsplit("/", 1)[-1])
    if parts.scheme != "https" or Path(filename).name != filename or not filename.endswith(".whl"):
        raise ValueError(f"Invalid wheel URL: {url}")
    expected = info["archive_info"]["hashes"]["sha256"]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"Invalid SHA-256 for {filename}")
    return url, filename, expected


def download(package):
    url, filename, expected = wheel_info(package)
    dest = WHEELHOUSE / filename
    if not dest.exists() or sha256(dest) != expected:
        with tempfile.NamedTemporaryFile(dir=WHEELHOUSE, suffix=".part", delete=False) as stream:
            partial = Path(stream.name)
        try:
            subprocess.run([
                "curl", "-fsSL", "--retry", "3", "--max-time", "600",
                "-o", str(partial), url,
            ], check=True)
            if sha256(partial) != expected:
                raise ValueError(f"Checksum mismatch: {filename}")
            partial.replace(dest)
        finally:
            partial.unlink(missing_ok=True)
    print(filename, flush=True)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-lock", action="store_true", help="Validate the lock without downloads or writes")
    parser.add_argument("--install", action="store_true", help="Install exactly these verified wheels into the current project WM venv")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    packages = json.loads((ROOT / "repro/wm_dependencies.json").read_text())["install"]
    filenames = [wheel_info(package)[1] for package in packages]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate wheels in lock")
    if args.check_lock:
        print(f"Validated {len(packages)} locked wheel URLs and SHA-256 values")
        return
    if args.install and Path(sys.prefix).resolve() != (ROOT / ".venv-worldmodel").resolve():
        parser.error("--install requires this project's .venv-worldmodel/bin/python")
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        wheels = list(pool.map(download, packages))
    if args.install:
        # --no-deps preserves the archived compatibility recipe. Pass the exact
        # lock list, never a glob that could include stale wheels.
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", *map(str, wheels)], check=True)
    print("All WM dependencies verified", flush=True)


if __name__ == "__main__":
    main()

"""Fetch pinned official sources without modifying existing checkouts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args):
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(path), *args], text=True
    ).strip()


def verify(path, source):
    # rev-parse accepts both ordinary .git directories and worktree .git files.
    if Path(git(path, "rev-parse", "--show-toplevel")).resolve() != path.resolve():
        raise RuntimeError(f"Not an independent checkout: {path}")
    actual = git(path, "rev-parse", "HEAD")
    if actual != source["commit"]:
        raise RuntimeError(f"{path.name}: expected {source['commit']}, got {actual}; existing checkout left untouched")
    if git(path, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"{path.name}: tracked source changes present; existing checkout left untouched")
    for relative, expected in source.get("files_sha256", {}).items():
        actual_hash = hashlib.sha256((path / relative).read_bytes()).hexdigest()
        if actual_hash != expected:
            raise RuntimeError(f"Source hash mismatch: {path.name}/{relative}")


def bootstrap(source, check_only):
    destination = ROOT / "vendor" / source["name"]
    if destination.exists():
        verify(destination, source)
        print(f"Verified {source['name']} @ {source['commit']}")
        return
    if check_only:
        raise FileNotFoundError(f"Missing {destination}; run bootstrap_sources.py without --check")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{source['name']}-", dir=destination.parent))
    try:
        subprocess.run(["git", "init", "--quiet", str(temporary)], check=True)
        git(temporary, "remote", "add", "origin", source["url"])
        git(temporary, "fetch", "--depth", "1", "origin", source["commit"])
        git(temporary, "checkout", "--detach", "FETCH_HEAD")
        verify(temporary, source)
        if destination.exists():
            raise RuntimeError(f"Destination appeared during fetch; leaving it untouched: {destination}")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Fetched {source['name']} @ {source['commit']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("rl", "wm", "all"), default="all")
    parser.add_argument("--check", action="store_true", help="Only verify existing repositories; never fetch or edit")
    parser.add_argument("--print-lock", action="store_true", help="Print selected sources without writes or network")
    args = parser.parse_args()
    sources = json.loads((ROOT / "repro/sources.lock.json").read_text())["sources"]
    selected = [source for source in sources if args.group == "all" or source["group"] == args.group]
    if args.print_lock:
        print(json.dumps(selected, indent=2))
        return
    for source in selected:
        bootstrap(source, args.check)


if __name__ == "__main__":
    main()

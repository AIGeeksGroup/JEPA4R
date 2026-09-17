"""Convert the final CEM metrics line into an atomic completion record."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path


SUCCESS_RATE_PATTERN = re.compile(
    r"['\"]success_rate['\"]\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_success_rate(text: str) -> float:
    for raw in reversed(text.splitlines()):
        matches = list(SUCCESS_RATE_PATTERN.finditer(raw))
        if not matches:
            continue
        rate = float(matches[-1].group(1))
        if not math.isfinite(rate) or not 0.0 <= rate <= 100.0:
            raise ValueError(f"invalid success_rate: {rate}")
        return rate
    raise ValueError("complete success_rate metric not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation-seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")

    rate = parse_success_rate(args.log.read_text(encoding="utf-8", errors="replace"))
    payload = {
        "completed": args.episodes,
        "episodes": args.episodes,
        "successes": int(round(rate / 100.0 * args.episodes)),
        "success_rate": rate,
        "checkpoint": args.checkpoint,
        "evaluation_seconds": args.evaluation_seconds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f"{args.out.name}.", suffix=".tmp", dir=args.out.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, args.out)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

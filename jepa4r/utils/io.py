"""Small I/O helpers shared by the stages and the report."""
import json
from datetime import datetime, timezone
from pathlib import Path


def _to_builtin(obj):
    """json.dumps fallback for numpy / torch scalars and arrays."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_metrics(out_dir, key: dict, metrics: dict) -> Path:
    """Write ``out_dir/metrics.json`` = run key (stage, env, method, decoder, tag) + metrics + timestamp."""
    record = {**key, "metrics": metrics, "timestamp": datetime.now(timezone.utc).isoformat()}
    path = Path(out_dir) / "metrics.json"
    path.write_text(json.dumps(record, indent=2, default=_to_builtin), encoding="utf-8")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

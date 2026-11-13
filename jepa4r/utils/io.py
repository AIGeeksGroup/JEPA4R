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


def logged(run_dir, tag) -> list:
    """[(step, value), ...] of one scalar that Lightning's default logger wrote under run_dir/lightning_logs/version_*/
    (TensorBoard event files where tensorboard is installed -- the paper runs' environments --, else metrics.csv)."""
    import csv

    out = []
    for version in sorted(Path(run_dir).glob('lightning_logs/version_*'), key=lambda p: int(p.name.split('_')[-1])):
        if list(version.glob('events.out.tfevents.*')):
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            acc = EventAccumulator(str(version), size_guidance={'scalars': 0})
            acc.Reload()
            if tag in acc.Tags()['scalars']:
                out += [(int(e.step), float(e.value)) for e in acc.Scalars(tag)]
        elif (version / 'metrics.csv').exists():
            rows = csv.DictReader(open(version / 'metrics.csv', encoding='utf-8'))
            out += [(int(r.get('step') or 0), float(r[tag])) for r in rows if r.get(tag) not in (None, '')]
    return out


def last_logged(run_dir, tag):
    """The last logged value of a scalar (None if it was never logged)."""
    values = logged(run_dir, tag)
    return values[-1][1] if values else None

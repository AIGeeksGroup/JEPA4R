"""Compare finished runs with the paper: ``jepa4r-report`` (or ``python -m jepa4r report``).

Reads every ``outputs/**/metrics.json`` (written by the stage commands) and every table of
``paper/expected/*.yaml``; writes ``outputs/report.md`` and ``outputs/report.csv``.

Expected-table format::

    tables:
      - name: tab1_ogbench
        title: ...
        stage: train_policy           # stage whose metrics.json holds the numbers
        fixed: {env: scene}           # run keys shared by all cells (tag defaults to "default")
        rows: method                  # run key (env | method | decoder | tag) or "metric";
        cols: env                     #   omit both for a flat {metric: value} table
        metric: success               # metric name when neither axis is "metric"
        metric_format: "{row}@{col}"  # optional: axes are labels, metric = format(row, col)
        values: {row: {col: number}}  # the paper's numbers; a number may be {value: x, op: "<="}
        derived: {col: {mean_of: [col, ...]}}                    # computed from the same row
        published: {row: {label, source, values: {col: number}}} # reference rows, never run
        labels: {name: label}         # display names (default: `display` of env/method/decoder)
        tol: 1.0                      # optional absolute tolerance -> ok / x per cell
        decimals: 2                   # optional; default = as many decimals as the paper value shows
"""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import REPO_ROOT, load_group, paths

EXPECTED_DIR = REPO_ROOT / "paper" / "expected"
RUN_AXES = ("env", "method", "decoder", "tag")


@dataclass
class Cell:
    paper: float | None
    ours: float | None = None
    op: str = "~"      # "~": compare by delta; "<=" / ">=": bound; "published": reference only

    @property
    def delta(self):
        if self.op != "~" or self.paper is None or self.ours is None:
            return None
        return self.ours - self.paper

    def ok(self, tol=None):
        if self.paper is None or self.ours is None:
            return None
        if self.op == "<=":
            return self.ours <= self.paper
        if self.op == ">=":
            return self.ours >= self.paper
        if self.op == "~" and tol is not None:
            return abs(self.ours - self.paper) <= tol
        return None


# --------------------------------------------------------------------------- expected tables
def load_expected(path) -> list[dict]:
    path = Path(path)
    tables = yaml.safe_load(path.read_text(encoding="utf-8"))["tables"]
    for t in tables:
        rows, cols = t.get("rows"), t.get("cols")
        if (rows is None) != (cols is None):
            raise ValueError(f"{path.name}: {t['name']}: give both rows and cols, or neither")
        if rows is not None and not t.get("metric_format"):
            for axis in (rows, cols):
                if axis != "metric" and axis not in RUN_AXES:
                    raise ValueError(f"{path.name}: {t['name']}: axis {axis!r} is not a run key")
            if "metric" not in (rows, cols) and not t.get("metric"):
                raise ValueError(f"{path.name}: {t['name']}: needs `metric`")
    return tables


def _expect(value):
    if isinstance(value, dict):
        v = value.get("value")
        return (None if v is None else float(v)), value.get("op", "~")
    return (None if value is None else float(value)), "~"


def iter_cells(table):
    """Yield (row, col, metric, want, paper, op) for every cell that a run must produce."""
    rows_key, cols_key, fmt = table.get("rows"), table.get("cols"), table.get("metric_format")
    fixed = dict(table.get("fixed") or {})
    if fmt or "tag" not in (rows_key, cols_key):
        fixed.setdefault("tag", "default")
    if rows_key is None:
        for metric, value in table["values"].items():
            yield (metric, None, metric, dict(fixed), *_expect(value))
        return
    derived = set(table.get("derived") or {})
    for row, cols in table["values"].items():
        for col, value in cols.items():
            if col in derived:
                continue
            want, metric = dict(fixed), table.get("metric")
            if fmt:
                metric = fmt.format(row=row, col=col)
            else:
                for axis, name in ((rows_key, row), (cols_key, col)):
                    if axis == "metric":
                        metric = name
                    else:
                        want[axis] = name
            yield (row, col, metric, want, *_expect(value))


def required_runs(table) -> set:
    """{(stage, ((key, value), ...))} -- the runs needed to fill a table."""
    return {(table["stage"], tuple(sorted(want.items()))) for _, _, _, want, _, _ in iter_cells(table)}


# --------------------------------------------------------------------------- runs
def load_records(runs_root) -> list[dict]:
    records = []
    for path in sorted(Path(runs_root).rglob("metrics.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        record["_path"] = str(path)
        records.append(record)
    return records


def _value(records, stage, want, metric):
    hits = [r for r in records
            if r.get("stage") == stage and all(r.get(k) == v for k, v in want.items())]
    if not hits:
        return None
    value = max(hits, key=lambda r: r.get("timestamp", "")).get("metrics", {}).get(metric)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def build_table(table, records):
    """Return (row_names, col_names, {(row, col): Cell}); flat tables use col_names == [None]."""
    stage = table["stage"]
    cells = {(row, col): Cell(paper, _value(records, stage, want, metric), op)
             for row, col, metric, want, paper, op in iter_cells(table)}
    if table.get("rows") is None:
        return list(table["values"]), [None], cells
    values = table["values"]
    cols = table.get("col_order") or list(dict.fromkeys(c for r in values.values() for c in r))
    rows = table.get("row_order") or list(values)
    for row in rows:
        for col, spec in (table.get("derived") or {}).items():
            parts = [cells.get((row, c)) for c in spec["mean_of"]]
            ours = None
            if parts and all(p is not None and p.ours is not None for p in parts):
                ours = sum(p.ours for p in parts) / len(parts)
            paper, op = _expect(values.get(row, {}).get(col))
            cells[(row, col)] = Cell(paper, ours, op)
    published = table.get("published") or {}
    for row, entry in published.items():
        for col, value in (entry.get("values") or {}).items():
            cells[(row, col)] = Cell(_expect(value)[0], None, "published")
    return rows + list(published), cols, cells


# --------------------------------------------------------------------------- rendering
def _fmt(x, like=None, decimals=None) -> str:
    """Format like the paper: ``decimals`` if given, else as many as the paper value ``like`` shows."""
    if x is None:
        return "--"
    if decimals is None:
        decimals = 2
        if like is not None:
            text = repr(float(like))
            decimals = len(text.split(".")[1]) if "." in text and "e" not in text else 2
    return f"{x:.{decimals}f}"


def _label(table, axis, name) -> str:
    labels = table.get("labels") or {}
    if name in labels:
        return str(labels[name])
    published = table.get("published") or {}
    if name in published:
        return f"{published[name].get('label', name)} (published)"
    if axis in ("env", "method", "decoder"):
        try:
            return str(load_group(axis, str(name)).get("display", name))
        except FileNotFoundError:
            pass
    return str(name)


def _mark(cell: Cell, tol) -> str:
    ok = cell.ok(tol)
    return "" if ok is None else (" ok" if ok else " x")


def _paper_text(cell: Cell, decimals=None) -> str:
    prefix = cell.op if cell.op in ("<=", ">=") else ""
    return prefix + _fmt(cell.paper, cell.paper, decimals)


def render_markdown(table, rows, cols, cells) -> str:
    tol, dec = table.get("tol"), table.get("decimals")
    lines = [f"### {table.get('title', table['name'])}", ""]
    if cols == [None]:
        lines += ["| metric | ours | paper | delta |", "|---|---:|---:|---:|"]
        for row in rows:
            c = cells[(row, None)]
            delta = "" if c.delta is None else f"{c.delta:+.2f}"
            lines.append(f"| {_label(table, 'metric', row)} | {_fmt(c.ours, c.paper, dec)}{_mark(c, tol)} "
                         f"| {_paper_text(c, dec)} | {delta} |")
    else:
        lines.append("| | " + " | ".join(_label(table, table["cols"], c) for c in cols) + " |")
        lines.append("|---" + "|---:" * len(cols) + "|")
        for row in rows:
            texts = []
            for col in cols:
                c = cells.get((row, col))
                if c is None:
                    texts.append("")
                elif c.op == "published":
                    texts.append(f"({_fmt(c.paper, c.paper, dec)})")
                else:
                    texts.append(f"{_fmt(c.ours, c.paper, dec)} / {_paper_text(c, dec)}{_mark(c, tol)}")
            lines.append(f"| {_label(table, table['rows'], row)} | " + " | ".join(texts) + " |")
        lines += ["", "cell = ours / paper; `--` = not run yet; (x) = published, not reproduced"]
    todo = [c for c in cells.values() if c.op != "published" and c.paper is not None]
    have = [c for c in todo if c.ours is not None]
    deltas = [abs(c.delta) for c in have if c.delta is not None]
    fails = [c for c in have if c.ok(tol) is False]
    summary = f"reproduced {len(have)}/{len(todo)} cells"
    if deltas:
        summary += f"; mean |delta| {sum(deltas) / len(deltas):.2f}, max {max(deltas):.2f}"
    if fails:
        summary += f"; {len(fails)} outside tolerance/bound"
    lines += ["", summary]
    return "\n".join(lines)


def csv_rows(table, cells) -> list[list]:
    out = []
    for (row, col), c in cells.items():
        out.append([table["name"], row, "" if col is None else col, c.paper, c.ours,
                    c.delta, c.op, c.ok(table.get("tol"))])
    return out


def report(expected=None, outputs=None, out=None, write=True) -> str:
    """Build the paper-vs-reproduction report from the runs under ``outputs`` (default: the outputs root)."""
    outputs_root = Path(outputs) if outputs else paths()["outputs"]
    out_dir = Path(out) if out else outputs_root
    files = ([Path(f) for f in expected] if expected
             else sorted(EXPECTED_DIR.glob("*.yaml"), key=lambda p: (not p.stem.startswith("tab"), p.stem)))
    records = load_records(outputs_root)
    parts = [f"# Paper vs. reproduction\n\n{len(records)} metrics.json files under `{outputs_root}`"]
    rows_csv = []
    for f in files:
        for table in load_expected(f):
            rows, cols, cells = build_table(table, records)
            parts.append(render_markdown(table, rows, cols, cells))
            rows_csv += csv_rows(table, cells)
    text = "\n\n".join(parts) + "\n"
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(text, encoding="utf-8")
        with open(out_dir / "report.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["table", "row", "col", "paper", "ours", "delta", "op", "ok"])
            writer.writerows(rows_csv)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jepa4r-report", description="Compare finished runs with the paper.")
    parser.add_argument("--expected", nargs="*", help="paper/expected/*.yaml files (default: all)")
    parser.add_argument("--outputs", help="runs to scan (default: $JEPA4R_OUTPUTS or ./outputs)")
    parser.add_argument("--out", help="directory for report.md / report.csv (default: the --outputs directory)")
    args = parser.parse_args(argv)
    print(report(args.expected, args.outputs, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

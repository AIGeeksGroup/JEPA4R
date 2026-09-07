"""Canonical portable report primitives. All values come from reviewed audit outputs."""
from __future__ import annotations


def markdown(block_id, body, source_id=None):
    block = {"id": block_id, "type": "markdown", "body": body}
    if source_id:
        block["sourceId"] = source_id
    return block


def chart(chart_id, title, dataset, x, y, *, source_id="distribution_snapshot",
          chart_type="horizontalBar", subtitle="", x_label=None, y_label=None,
          percent=False, tooltip=(), color=None, label=None):
    quantitative_x = chart_type in {"scatter", "histogram"}
    spec = {
        "id": chart_id, "title": title, "subtitle": subtitle,
        "type": chart_type, "dataset": dataset, "sourceId": source_id,
        "layout": "full", "palette": {"kind": "sequential", "name": "blue"},
        "encodings": {
            "x": {"field": x, "type": "quantitative" if quantitative_x else "nominal",
                  "label": x_label or x},
            "y": {"field": y, "type": "quantitative", "label": y_label or y,
                  "format": "percent" if percent else "number"},
        },
        "valueFormat": "percent" if percent else "number",
        "xAxisTitle": x_label or x, "yAxisTitle": y_label or y,
        "settings": {"sort": "none"},
        "surface": {"surface": "card", "viewMode": "visualization"},
    }
    if chart_type in {"bar", "horizontalBar"}:
        spec["labels"] = {"values": "all"}
    if tooltip:
        spec["encodings"]["tooltip"] = [{"field": field} for field in tooltip]
    if color:
        spec["encodings"]["color"] = {"field": color, "type": "nominal"}
        spec["palette"] = {"kind": "categorical"}
        spec["legend"] = {"position": "bottom"}
    if label:
        spec["encodings"]["label"] = {"field": label, "type": "nominal"}
    return spec


def table(table_id, title, dataset, columns, *, sort_by,
          source_id="distribution_snapshot", subtitle="", descending=False):
    return {
        "id": table_id, "title": title, "subtitle": subtitle,
        "dataset": dataset, "sourceId": source_id, "layout": "full",
        "columns": columns,
        "defaultSort": {"field": sort_by, "direction": "desc" if descending else "asc"},
    }


def chart_block(chart_id):
    return {"id": chart_id + "_block", "type": "chart", "chartId": chart_id}


def table_block(table_id):
    return {"id": table_id + "_block", "type": "table", "tableId": table_id}

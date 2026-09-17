from __future__ import annotations

import pytest

from scripts.eval.summarize_cem_result import parse_success_rate


def test_parse_success_rate_uses_final_metrics_dictionary() -> None:
    text = (
        "warmup\n"
        "{'success_rate': 12.0}\n"
        "metrics: {'success_rate': 37.0, "
        "'episode_successes': array([1., 0.]), 'seeds': array([1, 2])}\n"
    )
    assert parse_success_rate(text) == pytest.approx(37.0)


@pytest.mark.parametrize(
    "text",
    ["no metrics", "{'loss': 0.2}", "{'success_rate': nan}", "{'success_rate': 101}"],
)
def test_parse_success_rate_rejects_missing_or_invalid_metrics(text: str) -> None:
    with pytest.raises(ValueError):
        parse_success_rate(text)

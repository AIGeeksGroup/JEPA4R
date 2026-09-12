import numpy as np
import pytest

from scripts.eval_liftcube_contact_head import contact_classification_metrics


def test_contact_metrics_report_both_classes_and_tie_correct_auroc():
    result = contact_classification_metrics(
        np.array([0.1, 0.8, 0.8, 0.2]), np.array([0, 1, 0, 1])
    )

    assert result["positive"] == 2
    assert result["negative"] == 2
    assert result["tp"] == 1
    assert result["tn"] == 1
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["specificity"] == pytest.approx(0.5)
    assert result["auroc"] == pytest.approx(0.625)


def test_contact_metrics_reject_invalid_values():
    with pytest.raises(ValueError, match="binary"):
        contact_classification_metrics(np.array([0.2]), np.array([2]))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        contact_classification_metrics(np.array([1.2]), np.array([1]))

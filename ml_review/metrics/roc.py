"""ROC, precision-recall, and threshold review helpers."""
from __future__ import annotations

from typing import Any

from .._legacy import suppress_legacy_warning
from ..results import MulticlassRocResult, PrecisionRecallResult, RocResult

with suppress_legacy_warning():
    from extended_sklearn_metrics.roc_auc_analysis import (
        calculate_multiclass_roc_metrics,
        calculate_precision_recall_metrics,
        calculate_roc_metrics,
    )


def binary(*args: Any, **kwargs: Any) -> RocResult:
    """Return binary ROC review data for an estimator."""
    return RocResult(calculate_roc_metrics(*args, **kwargs))


def multiclass(*args: Any, **kwargs: Any) -> MulticlassRocResult:
    """Return multiclass one-vs-rest ROC review data."""
    return MulticlassRocResult(calculate_multiclass_roc_metrics(*args, **kwargs))


def precision_recall(*args: Any, **kwargs: Any) -> PrecisionRecallResult:
    """Return binary precision-recall review data."""
    return PrecisionRecallResult(calculate_precision_recall_metrics(*args, **kwargs))


__all__ = ["binary", "multiclass", "precision_recall"]

"""ROC plotting helpers."""
from __future__ import annotations

from typing import Any

from .._legacy import suppress_legacy_warning

with suppress_legacy_warning():
    from extended_sklearn_metrics.visualizations import (
        create_multiclass_roc_plot,
        create_precision_recall_plot,
        create_roc_curve_plot,
        create_threshold_analysis_plot,
    )


def curve(*args: Any, **kwargs: Any) -> Any:
    return create_roc_curve_plot(*args, **kwargs)


def precision_recall(*args: Any, **kwargs: Any) -> Any:
    return create_precision_recall_plot(*args, **kwargs)


def multiclass(*args: Any, **kwargs: Any) -> Any:
    return create_multiclass_roc_plot(*args, **kwargs)


def thresholds(*args: Any, **kwargs: Any) -> Any:
    return create_threshold_analysis_plot(*args, **kwargs)


__all__ = ["curve", "multiclass", "precision_recall", "thresholds"]

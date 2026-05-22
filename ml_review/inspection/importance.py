"""Feature-importance report and plotting helpers."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .._legacy import suppress_legacy_warning

with suppress_legacy_warning():
    from extended_sklearn_metrics.evaluation_reporting import (
        create_feature_importance_report,
    )
    from extended_sklearn_metrics.visualizations import create_feature_importance_plot


def report(review: Any) -> pd.DataFrame | None:
    """Return the available feature-importance measures."""
    return create_feature_importance_report(review)


def plot(review: Any, **kwargs: Any) -> Any:
    """Plot feature importance from a review result."""
    return create_feature_importance_plot(review, **kwargs)


__all__ = ["plot", "report"]

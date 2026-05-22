"""Fairness report helpers."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .._legacy import suppress_legacy_warning

with suppress_legacy_warning():
    from extended_sklearn_metrics.evaluation_reporting import create_fairness_report


def report(review: Any) -> pd.DataFrame | None:
    """Return group fairness metrics from a review result."""
    return create_fairness_report(review)


__all__ = ["report"]

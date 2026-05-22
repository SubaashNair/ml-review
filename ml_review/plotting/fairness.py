"""Fairness plotting helpers."""
from __future__ import annotations

from typing import Any

from .._legacy import suppress_legacy_warning

with suppress_legacy_warning():
    from extended_sklearn_metrics.visualizations import create_fairness_comparison_plot


def compare(*args: Any, **kwargs: Any) -> Any:
    return create_fairness_comparison_plot(*args, **kwargs)


__all__ = ["compare"]

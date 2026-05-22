"""Residual plotting helpers."""
from __future__ import annotations

from typing import Any

from .._legacy import suppress_legacy_warning

with suppress_legacy_warning():
    from extended_sklearn_metrics.visualizations import (
        create_residual_plots,
        create_residual_summary_plot,
    )


def plot(*args: Any, **kwargs: Any) -> Any:
    return create_residual_plots(*args, **kwargs)


def summary(*args: Any, **kwargs: Any) -> Any:
    return create_residual_summary_plot(*args, **kwargs)


__all__ = ["plot", "summary"]

"""Performance plotting helpers."""
from __future__ import annotations

from typing import Any

from .._legacy import suppress_legacy_warning

with suppress_legacy_warning():
    from extended_sklearn_metrics.visualizations import (
        create_model_comparison_plot,
        create_performance_radar_chart,
        create_performance_summary_plot,
        print_performance_report,
    )


def summary(*args: Any, **kwargs: Any) -> Any:
    return create_performance_summary_plot(*args, **kwargs)


def compare(*args: Any, **kwargs: Any) -> Any:
    return create_model_comparison_plot(*args, **kwargs)


def radar(*args: Any, **kwargs: Any) -> Any:
    return create_performance_radar_chart(*args, **kwargs)


def print_report(*args: Any, **kwargs: Any) -> Any:
    return print_performance_report(*args, **kwargs)


__all__ = ["compare", "print_report", "radar", "summary"]

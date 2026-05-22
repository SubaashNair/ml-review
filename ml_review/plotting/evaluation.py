"""Evaluation dashboard plotting helpers."""
from __future__ import annotations

from typing import Any

from .._legacy import suppress_legacy_warning

with suppress_legacy_warning():
    from extended_sklearn_metrics.visualizations import (
        create_comprehensive_evaluation_plots,
    )


def dashboard(*args: Any, **kwargs: Any) -> Any:
    return create_comprehensive_evaluation_plots(*args, **kwargs)


__all__ = ["dashboard"]

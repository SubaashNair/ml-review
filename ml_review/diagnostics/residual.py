"""Regression residual diagnostic helpers."""
from __future__ import annotations

from typing import Any

from .._legacy import suppress_legacy_warning
from ..results import ResidualResult

with suppress_legacy_warning():
    from extended_sklearn_metrics.residual_diagnostics import (
        calculate_residual_diagnostics,
    )


def calculate(*args: Any, **kwargs: Any) -> ResidualResult:
    """Calculate residual diagnostics and wrap them with report helpers."""
    return ResidualResult(calculate_residual_diagnostics(*args, **kwargs))


__all__ = ["calculate"]

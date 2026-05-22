"""Helpers for importing the legacy compatibility package internally."""
from __future__ import annotations

from contextlib import contextmanager
import warnings


LEGACY_WARNING_PATTERN = "extended_sklearn_metrics is now a compatibility package.*"


@contextmanager
def suppress_legacy_warning():
    """Avoid surfacing migration warnings from MLReview's internal adapters."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=LEGACY_WARNING_PATTERN,
            category=FutureWarning,
        )
        yield

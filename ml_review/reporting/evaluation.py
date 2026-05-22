"""Supervised evaluation report helpers."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .._legacy import suppress_legacy_warning
from .._utils import result_dict

with suppress_legacy_warning():
    from extended_sklearn_metrics.evaluation_reporting import (
        create_evaluation_report,
        print_evaluation_summary,
    )


def report(review: Any) -> pd.DataFrame:
    """Return a readable supervised evaluation report."""
    return create_evaluation_report(review)


def print_summary(review: Any) -> None:
    """Print the supervised evaluation summary."""
    print_evaluation_summary(result_dict(review))


__all__ = ["print_summary", "report"]

"""Supervised evaluation entrypoints for MLReview."""
from __future__ import annotations

from typing import Any

from ._legacy import suppress_legacy_warning
from .results import ReviewResult

with suppress_legacy_warning():
    from extended_sklearn_metrics.classification_evaluation import (
        evaluate_classification_model_with_cross_validation,
    )
    from extended_sklearn_metrics.comprehensive_evaluation import final_model_evaluation
    from extended_sklearn_metrics.model_evaluation import (
        CustomThresholds,
        evaluate_model_with_cross_validation,
    )


Thresholds = CustomThresholds
regression_cv = evaluate_model_with_cross_validation
classification_cv = evaluate_classification_model_with_cross_validation


def evaluate(*args: Any, **kwargs: Any) -> ReviewResult:
    """Evaluate a fitted supervised estimator and return a review result."""
    return ReviewResult(final_model_evaluation(*args, **kwargs))


__all__ = ["Thresholds", "classification_cv", "evaluate", "regression_cv"]

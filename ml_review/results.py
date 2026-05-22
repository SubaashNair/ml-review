"""DataFrame-first result objects for MLReview workflows."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Any

import pandas as pd


class MappingResult(Mapping[str, Any]):
    """Mapping wrapper that exposes the raw analysis payload when needed."""

    def __init__(self, data: Mapping[str, Any]):
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive copy of the structured result data."""
        return deepcopy(self._data)


class ReviewResult(MappingResult):
    """Rich supervised model review result."""

    @property
    def importance(self) -> "ImportanceView":
        return ImportanceView(self)

    @property
    def shap(self) -> "ShapView":
        return ShapView(self)

    @property
    def fairness(self) -> "FairnessView":
        return FairnessView(self)

    def report(self) -> pd.DataFrame:
        from .reporting import evaluation

        return evaluation.report(self)

    def print_summary(self) -> None:
        from .reporting import evaluation

        evaluation.print_summary(self)


class ImportanceView:
    """Feature-importance operations for a review result."""

    def __init__(self, review: ReviewResult):
        self._review = review

    def report(self) -> pd.DataFrame | None:
        from .inspection import importance

        return importance.report(self._review)

    def plot(self, **kwargs: Any) -> Any:
        from .inspection import importance

        return importance.plot(self._review, **kwargs)


class ShapView:
    """SHAP operations for a review result."""

    def __init__(self, review: ReviewResult):
        self._review = review

    def report(self) -> pd.DataFrame | None:
        from .inspection import shap

        return shap.report(self._review)

    def plot_importance(self, **kwargs: Any) -> Any:
        from .inspection import shap

        return shap.plot_importance(self._review, **kwargs)

    def plot_explanation(self, sample_index: int, **kwargs: Any) -> Any:
        from .inspection import shap

        return shap.plot_explanation(self._review, sample_index=sample_index, **kwargs)


class FairnessView:
    """Fairness operations for a review result."""

    def __init__(self, review: ReviewResult):
        self._review = review

    def report(self) -> pd.DataFrame | None:
        from .reporting import fairness

        return fairness.report(self._review)

    def plot(self, **kwargs: Any) -> Any:
        from .plotting import fairness

        return fairness.compare(self._review, **kwargs)


class RocResult(MappingResult):
    """Binary ROC analysis with DataFrame report helpers."""

    def thresholds(self, **kwargs: Any) -> pd.DataFrame:
        from ._legacy import suppress_legacy_warning

        with suppress_legacy_warning():
            from extended_sklearn_metrics.roc_auc_analysis import find_optimal_thresholds

        return find_optimal_thresholds(self, **kwargs)

    def report(self, pr_result: "PrecisionRecallResult | None" = None) -> pd.DataFrame:
        from ._legacy import suppress_legacy_warning

        with suppress_legacy_warning():
            from extended_sklearn_metrics.roc_auc_analysis import create_threshold_analysis_report

        return create_threshold_analysis_report(self, pr_result)

    def print_summary(self, pr_result: "PrecisionRecallResult | None" = None) -> None:
        from ._legacy import suppress_legacy_warning

        with suppress_legacy_warning():
            from extended_sklearn_metrics.roc_auc_analysis import print_roc_auc_summary

        print_roc_auc_summary(self, pr_result)


class MulticlassRocResult(MappingResult):
    """Multiclass ROC analysis with a compact report view."""

    def report(self) -> pd.DataFrame:
        rows = []
        for class_label, class_result in self["class_results"].items():
            rows.append(
                {
                    "Class": class_label,
                    "ROC_AUC": class_result["roc_auc"],
                    "Optimal_Threshold": class_result["optimal_threshold"],
                    "Optimal_TPR": class_result["optimal_tpr"],
                    "Optimal_FPR": class_result["optimal_fpr"],
                }
            )
        rows.extend(
            [
                {"Class": "micro", "ROC_AUC": self["micro_average"]["roc_auc"]},
                {"Class": "macro", "ROC_AUC": self["macro_average"]["roc_auc"]},
            ]
        )
        return pd.DataFrame(rows)


class PrecisionRecallResult(MappingResult):
    """Precision-recall analysis with a readable summary."""

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ["PR Analysis", "PR AUC", self["pr_auc"], "Average precision score"],
                [
                    "PR Analysis",
                    "Optimal Threshold",
                    self["optimal_threshold"],
                    "Threshold maximizing F1 score",
                ],
                ["PR Analysis", "Optimal Precision", self["optimal_precision"], ""],
                ["PR Analysis", "Optimal Recall", self["optimal_recall"], ""],
                ["PR Analysis", "Optimal F1", self["optimal_f1"], ""],
            ],
            columns=["Category", "Metric", "Value", "Description"],
        )


class ResidualResult(MappingResult):
    """Residual diagnostic analysis with report and plot helpers."""

    def report(self) -> pd.DataFrame:
        from ._legacy import suppress_legacy_warning

        with suppress_legacy_warning():
            from extended_sklearn_metrics.residual_diagnostics import (
                create_residual_summary_report,
            )

        return create_residual_summary_report(self)

    def print_report(self) -> None:
        from ._legacy import suppress_legacy_warning

        with suppress_legacy_warning():
            from extended_sklearn_metrics.residual_diagnostics import (
                print_residual_diagnostics_report,
            )

        print_residual_diagnostics_report(self)

    def plot(self, **kwargs: Any) -> Any:
        from .plotting import residual

        return residual.plot(self, **kwargs)

    def plot_summary(self, **kwargs: Any) -> Any:
        from .plotting import residual

        return residual.summary(self, **kwargs)

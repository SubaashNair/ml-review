"""SHAP report and plot helpers built from serialized review data."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._legacy import suppress_legacy_warning
from .._utils import result_data

with suppress_legacy_warning():
    from extended_sklearn_metrics._plotting_backend import get_matplotlib


def _shap_payload(review: Any) -> tuple[dict[str, Any], dict[str, Any]] | None:
    feature_importance = result_data(review).get("feature_importance", {})
    importance = feature_importance.get("shap_importance")
    explanations = feature_importance.get("shap_explanations")
    if not importance or not explanations:
        return None
    return importance, explanations


def report(review: Any) -> pd.DataFrame | None:
    """Return an explanation-first SHAP summary for a review result."""
    payload = _shap_payload(review)
    if payload is None:
        return None

    importance, explanations = payload
    values = importance["values"]
    features = importance["features"]
    top_indices = np.argsort(values)[-5:][::-1]
    rows = [
        [
            "How to read SHAP",
            "Baseline",
            "The baseline is the model output before feature contributions are added.",
        ],
        [
            "How to read SHAP",
            "Contribution sign",
            "Positive contributions raise the explained output; negative contributions lower it.",
        ],
        [
            "How to read SHAP",
            "Global vs local",
            "Global importance averages absolute contributions; local explanations describe one sampled prediction.",
        ],
        [
            "Sampling",
            "Rows explained",
            f"{importance['sample_count']} sampled rows are stored for local explanations.",
        ],
        [
            "Sampling",
            "Background rows",
            f"{importance['background_count']} rows define the SHAP background sample.",
        ],
        [
            "Model output",
            "Explained output",
            f"SHAP values explain `{explanations['model_output']}` output.",
        ],
    ]
    if explanations.get("output_names"):
        rows.append(
            [
                "Model output",
                "Classifier outputs",
                "Local plots default to the predicted class unless an output index is passed.",
            ]
        )

    rows.extend(
        [
            [
                "Global importance",
                f"Top {position}",
                f"{features[index]} ({values[index]:.4f})",
            ]
            for position, index in enumerate(top_indices, start=1)
        ]
    )
    return pd.DataFrame(rows, columns=["Category", "Topic", "Explanation"])


def plot_importance(
    review: Any,
    top_n: int = 15,
    figsize: tuple[int, int] = (9, 6),
) -> Any:
    """Plot mean absolute SHAP importance for available explained rows."""
    payload = _shap_payload(review)
    if payload is None:
        raise ValueError("No SHAP data is available. Run evaluate() with SHAP enabled.")

    importance, _ = payload
    values = np.asarray(importance["values"])
    features = np.asarray(importance["features"])
    top_indices = np.argsort(values)[-min(top_n, len(values)) :]

    plt = get_matplotlib()
    _, ax = plt.subplots(figsize=figsize)
    ax.barh(features[top_indices], values[top_indices], alpha=0.85)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Global SHAP Feature Importance")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()
    return ax


def _local_row_position(explanations: dict[str, Any], sample_index: int) -> int:
    sampled_indices = list(explanations["sample_indices"])
    if sample_index in sampled_indices:
        return sampled_indices.index(sample_index)
    if 0 <= sample_index < len(sampled_indices):
        return sample_index
    raise ValueError(
        f"Sample index {sample_index} is not available. "
        f"Stored sampled indices include: {sampled_indices[:10]}"
    )


def _base_value(base_values: np.ndarray, position: int, output_index: int | None) -> float:
    if base_values.ndim == 0:
        return float(base_values)
    if base_values.ndim == 1:
        return float(base_values[position])
    if output_index is None:
        return float(base_values[position, 0])
    return float(base_values[position, output_index])


def plot_explanation(
    review: Any,
    sample_index: int,
    output_index: int | None = None,
    top_n: int = 12,
    figsize: tuple[int, int] = (10, 7),
) -> Any:
    """Plot signed SHAP contributions for one stored sampled prediction."""
    payload = _shap_payload(review)
    if payload is None:
        raise ValueError("No SHAP data is available. Run evaluate() with SHAP enabled.")

    _, explanations = payload
    position = _local_row_position(explanations, sample_index)
    values = np.asarray(explanations["values"])
    if values.ndim == 3:
        if output_index is None:
            predictions = explanations.get("predicted_output_indices") or [0]
            output_index = predictions[position] if position < len(predictions) else 0
        contributions = values[position, :, output_index]
    else:
        contributions = values[position]

    features = np.asarray(explanations["feature_names"])
    feature_values = np.asarray(explanations["feature_values"], dtype=object)[position]
    top_indices = np.argsort(np.abs(contributions))[-min(top_n, len(contributions)) :]
    labels = [f"{features[i]} = {feature_values[i]}" for i in top_indices]
    colors = ["#1f77b4" if contributions[i] >= 0 else "#d62728" for i in top_indices]
    base_value = _base_value(np.asarray(explanations["base_values"]), position, output_index)

    plt = get_matplotlib()
    _, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, contributions[top_indices], color=colors, alpha=0.88)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution")
    ax.set_title(
        f"Local SHAP Explanation for Sample {explanations['sample_indices'][position]}"
        f"\nBaseline output: {base_value:.4f}"
    )
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()
    return ax


__all__ = ["plot_explanation", "plot_importance", "report"]
